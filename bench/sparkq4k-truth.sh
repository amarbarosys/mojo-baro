#!/usr/bin/env bash
# Exact-model reference arm for vibethinker Q4_K_M (bench/sparkq4k-protocol.md, amendment A1): our engine
# teacher-forced on the f64 oracle's 64 greedy tokens (exchange/receipts/sparkq4k/truth-gen.py), q8 pack and
# K-quant pack, 20 prompts, one GPU job. llama.cpp's CPU arm reaches only 52/64 of p04 against this model.
# Usage: bench/sparkq4k-truth.sh TRUTHDIR OUT   (TRUTHDIR holds <prompt>.truth-tokens.txt and <prompt>.ids)
#   ACT_BF16=1 builds the engine with bf16 GEMV activations (the pre-A2 numerics, control arm); default f32.
set -euo pipefail
cd "$(dirname "$0")/.."
TR=${1:?usage: bench/sparkq4k-truth.sh TRUTHDIR OUT}; OUT=${2:?}; ACT_BF16=${ACT_BF16:-0}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ -x "$HOME/.local/bin/gpu-wait" ]; then
  mkdir -p "$OUT"; exec "$HOME/.local/bin/gpu-wait" run --timeout 1800 -- env ACT_BF16="$ACT_BF16" bash -c "'$0' '$TR' '$OUT' > '$OUT/job.log' 2>&1"
fi
mkdir -p "$OUT"; M=$HOME/Models/vibethinker-1.5b-gguf
. bench/min16k.sh; min16k "$(grep -oP "^comptime TMAX = \K[0-9]+" "$M/profile/profile.mojo")" 16000
BIN="$OUT/spark-engine"
./.venv/bin/mojo build serve/spark.mojo -D BARO_ACT_BF16="$ACT_BF16" -I "$M/profile" -I . -I kernels -I serve -I tools -o "$BIN" > "$OUT/build.log" 2>&1 \
  || { echo "FAIL build: $(grep -m1 error: "$OUT/build.log")"; exit 1; }
ACT=$([ "$ACT_BF16" = 1 ] && echo bfloat16 || echo float32)
{ echo "commit=$(git rev-parse --short HEAD) engine_sha=$(sha256sum "$BIN" | cut -c1-16) spec=0 gen=64 act=$ACT truth=$TR"
  for a in q8 kq; do p=$([ $a = q8 ] && echo "$M/pack" || echo .work/sparkq4k/packs/vibe-kq)
    echo "arm $a pack=$p index_sha=$(sha256sum "$p/index.txt" | cut -c1-16) pack_bytes=$(stat -c %s "$p/pack.bin")"; done; } | tee "$OUT/arm.txt"
echo "prompt q8_vs_truth kq_vs_truth" > "$OUT/results.txt"
for tf in "$TR"/*.truth-tokens.txt; do
  p=$(basename "$tf" .truth-tokens.txt); row="$p"
  for a in q8 kq; do pk=$([ $a = q8 ] && echo "$M/pack" || echo .work/sparkq4k/packs/vibe-kq)
    env BARO_PROMPT="$TR/$p.ids" BARO_PACK="$pk" BARO_GEN=64 BARO_SPEC=0 BARO_FORCE="$tf" "$BIN" > "$OUT/$p.$a.log" 2>&1 \
      || { echo "FAIL $p $a: engine exit $?, see $OUT/$p.$a.log"; exit 1; }
    grep -q "matrices: q8 $([ $a = q8 ] && echo '141 kquant 0' || echo '0 kquant 141')  act: $ACT" "$OUT/$p.$a.log" || { echo "FAIL $p $a: read-back is not the $a / $ACT arm"; exit 1; }
    if grep -qE 'NOT-RESIDENT|Error|error' "$OUT/$p.$a.log"; then echo "FAIL $p $a: fail word in $OUT/$p.$a.log"; exit 1; fi
    row="$row $(grep '^forced agreement:' "$OUT/$p.$a.log" | sed 's/forced agreement: //; s/ \/ /\//')"
  done
  echo "$row" >> "$OUT/results.txt"
done
python3 - "$OUT/results.txt" <<'PY' | tee "$OUT/SUMMARY.txt"
import sys
rows = [l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
for i, name in ((1, "q8 pack"), (2, "K-quant pack")):
    pct = [100 * int(r[i].split("/")[0]) / int(r[i].split("/")[1]) for r in rows]
    print(f"{name} vs exact model: min {min(pct):.1f}% mean {sum(pct)/len(pct):.1f}% over {len(pct)} prompts; p04 {[r[i] for r in rows if r[0].startswith('p04')]}")
PY
echo "PASS sparkq4k-truth: $OUT/results.txt"
