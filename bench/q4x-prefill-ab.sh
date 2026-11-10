#!/usr/bin/env bash
# latent-flash P1 speed A/B, same job: decode step() over T prompt tokens vs prefill(ids) of the same T tokens,
# arms alternated per prompt, median over prompts (P4), every arm parameter read back into OUT/receipt.txt (P1).
# Rerun on the same OUT_DIR resumes: prompts with a complete log are skipped, the header is appended (one per job).
# usage: bench/q4x-prefill-ab.sh OUT_DIR T [N_PROMPTS] [SEQ]   prompts: .work/lf/prompts/pNN-T.prompt (mkprompts.sh)
set -euo pipefail
OUT=${1:?out dir}; T=${2:?T}; NP=${3:-20}; SEQ=${4:-2048}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
BIN=${Q4X_BIN:-.work/p8/q4x}
PD=${Q4X_PROMPT_DIR:-.work/lf/prompts}
if [ -z "${Q4X_AB_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout 14400 -- env Q4X_AB_INNER=1 Q4X_AB_DECODE_NP="${Q4X_AB_DECODE_NP:-}" Q4X_GGUF="$GGUF" Q4X_BIN="$BIN" Q4X_PROMPT_DIR="$PD" HOME="$HOME" bash "$REPO/bench/q4x-prefill-ab.sh" "$OUT" "$T" "$NP" "$SEQ"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
vram() { for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && awk -v t=$(cat $d/mem_info_vram_total) -v u=$(cat $d/mem_info_vram_used) 'BEGIN { if (t > 20e9) printf "%.2f", (t - u) / 2^30 }'; done; }
res() { fincore --bytes --noheadings --output RES "$1" | awk '{s+=$1} END {printf "%.1f", s/2**30}'; }
{
  echo "date $(date -Is)"; echo "commit $(git rev-parse HEAD)"; echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -c1-16)"
  echo "gguf $GGUF sha256 $(sha256sum "$GGUF" | cut -c1-16)"
  echo "T $T SEQ $SEQ NP $NP decode-arm prompts ${Q4X_AB_DECODE_NP:-$NP} engA=decode(step per token, Q4X_TMAX=8192) engB=prefill(Q4X_PREFILL=1 Q4X_SEQ=$SEQ Q4X_TMAX=8192)"
  echo "vram_free_gb_start $(vram)"; for f in $(dirname "$GGUF")/*.gguf; do echo "pagecache_resident_gb $(basename $f) $(res $f)"; done
} >> "$OUT/receipt.txt"
PROMPTS=$(ls $PD/p*-$T.prompt | head -n "$NP")
DNP=${Q4X_AB_DECODE_NP:-$NP}; i=0
for p in $PROMPTS; do
  b=$(basename "$p" .prompt); ids=$(jq -r 'join(" ")' "$p"); n=$(jq length "$p"); i=$((i + 1))
  [ "$n" = "$T" ] || { echo "FAIL $b has $n ids, want $T" | tee -a "$OUT/receipt.txt"; exit 1; }
  for arm in A B; do
    if [ $arm = A ] && [ $i -gt $DNP ]; then continue; fi
    if grep -qs '^prompt ' "$OUT/$b.$arm.log"; then continue; fi
    if [ $arm = A ]; then envs="Q4X_TMAX=8192"; else envs="Q4X_TMAX=8192 Q4X_PREFILL=1 Q4X_SEQ=$SEQ"; fi
    if ! env $envs /usr/bin/time -f 'wall %e s rss %M KB' "$BIN" "$GGUF" 1 $ids > "$OUT/$b.$arm.log" 2>&1; then
      echo "RETRY $b arm $arm: $(grep -m1 -E 'exception|Error' "$OUT/$b.$arm.log" | cut -c1-120)" | tee -a "$OUT/receipt.txt"; mv "$OUT/$b.$arm.log" "$OUT/$b.$arm.failed.log"
      env $envs /usr/bin/time -f 'wall %e s rss %M KB' "$BIN" "$GGUF" 1 $ids > "$OUT/$b.$arm.log" 2>&1 || { echo "FAIL $b arm $arm twice, log $OUT/$b.$arm.log" | tee -a "$OUT/receipt.txt"; exit 1; }
    fi
    grep -q '^ids' "$OUT/$b.$arm.log" && grep -q '^prompt ' "$OUT/$b.$arm.log" || { echo "FAIL $b arm $arm: no ids/prompt line" | tee -a "$OUT/receipt.txt"; exit 1; }
    echo "$b $arm env[$envs] $(grep '^prompt ' "$OUT/$b.$arm.log") | $(grep '^expert misses' "$OUT/$b.$arm.log") | $(grep '^wall ' "$OUT/$b.$arm.log") | vram_free_gb_after $(vram)" | tee -a "$OUT/receipt.txt"
  done
done
python3 - "$OUT/receipt.txt" <<'PY' | tee -a "$OUT/receipt.txt"
import sys, re, statistics
a, b = [], []
for line in open(sys.argv[1]):
    m = re.search(r'^(\S+) ([AB]) .*? ms/token (\S+) ', line)
    if m: (a if m.group(2) == 'A' else b).append(float(m.group(3)))
if a and b:
    print(f"MEDIAN ms/token decode {statistics.median(a):.3f} (n={len(a)}, min {min(a):.3f} max {max(a):.3f}) prefill {statistics.median(b):.3f} (n={len(b)}, min {min(b):.3f} max {max(b):.3f}) ratio {statistics.median(a)/statistics.median(b):.2f}x")
PY
