#!/usr/bin/env bash
# q4x MTP speculative decoding check (round 5), one gpu-wait job:
#   1. identity: base binary vs new binary, plain decode, Q4X_DUMP block sums of one prompt + 4 tokens
#   2. per prompt, arms alternated: A = new binary plain decode, B = new binary Q4X_SPEC=K with the MTP head;
#      greedy ids of B must equal A's (speculation is exact); gen ms/token, acceptance, misses recorded (P1 read-back)
# An arm that fails is retried once (the q4x router doorbell timeout is a known flake, see q4x-prefill-ab.sh).
# Prompts: bench/q4x-prompts/*.prompt (frozen 2026-10-09).
# B_SPEC=0 B_BIN=bin turns step 2 into a plain-decode A/B of NEW (A) against B_BIN (B), ids still compared.
# usage: bench/q4x-spec.sh OUT_DIR N_PROMPTS N_TOK [K]      env BASE=bin NEW=bin Q4X_WARM=list Q4X_SPEC_P=p B_BIN B_SPEC
set -euo pipefail
OUT=${1:?usage: q4x-spec.sh OUT N_PROMPTS N_TOK [K]}; NP=${2:?}; NT=${3:?}; K=${4:-3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
MTP=${MTP_PATH:-${Q4X_MTP:-$HOME/Models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}}
unset Q4X_MTP
BASE=${BASE:-$REPO/.work/r5/q4x-base}; NEW=${NEW:-$REPO/.work/r5/q4x}
WARM=${Q4X_WARM:-$REPO/exchange/receipts/q4x-strata/r3a-warm-l01.txt}
PMIN=${Q4X_SPEC_P:-0.5}
B_BIN=${B_BIN:-$NEW}; B_SPEC=${B_SPEC:-1}
if [ -z "${Q4X_SPEC_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout 3600 -- env Q4X_SPEC_INNER=1 Q4X_GGUF="$GGUF" MTP_PATH="$MTP" BASE="$BASE" NEW="$NEW" B_BIN="$B_BIN" B_SPEC="$B_SPEC" \
    Q4X_WARM="$WARM" Q4X_SPEC_P="$PMIN" HOME="$HOME" bash "$REPO/bench/q4x-spec.sh" "$OUT" "$NP" "$NT" "$K"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
vram() { for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && awk -v t=$(cat $d/mem_info_vram_total) -v u=$(cat $d/mem_info_vram_used) 'BEGIN { if (t > 20e9) printf "%.2f", (t - u) / 2^30 }'; done; }
res() { fincore --bytes --noheadings --output RES "$1" | awk '{s+=$1} END {printf "%.1f", s/2**30}'; }
{
  echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l)"
  echo "engA=$NEW plain sha256 $(sha256sum "$NEW" | cut -c1-16)"
  if [ "$B_SPEC" = 1 ]; then echo "engB=$B_BIN Q4X_MTP=$MTP Q4X_SPEC=$K Q4X_SPEC_P=$PMIN sha256 $(sha256sum "$B_BIN" | cut -c1-16)"
  else echo "engB=$B_BIN plain sha256 $(sha256sum "$B_BIN" | cut -c1-16)"; fi
  echo "base=$BASE sha256 $(sha256sum "$BASE" | cut -c1-16)"
  echo "gguf $GGUF warm $WARM sha256 $(sha256sum "$WARM" | cut -c1-16) NP $NP NT $NT"
  echo "vram_free_gb_start $(vram)"; for f in $(dirname "$GGUF")/*.gguf "$MTP"; do echo "pagecache_resident_gb $(basename $f) $(res $f)"; done
} >> "$R"
run() { # log binary prompt-file ntok env...
  local log=$1 bin=$2 pf=$3 nt=$4; shift 4
  local ids; ids=$(jq -r 'join(" ")' "$pf")
  if ! env Q4X_WARM="$WARM" "$@" /usr/bin/time -f 'wall %e s rss %M KB' "$bin" "$GGUF" "$nt" $ids > "$log" 2>&1; then
    echo "RETRY $(basename "$log"): $(grep -m1 -E 'exception|Error' "$log" | cut -c1-120)" | tee -a "$R"; mv "$log" "$log.failed"
    env Q4X_WARM="$WARM" "$@" /usr/bin/time -f 'wall %e s rss %M KB' "$bin" "$GGUF" "$nt" $ids > "$log" 2>&1 || fail run "$log failed twice"
  fi
  grep -q '^ids ' "$log" || fail run "$log has no ids line"
}
P1=$(ls bench/q4x-prompts/*.prompt | head -1)
run "$OUT/dump-base.log" "$BASE" "$P1" 4 Q4X_DUMP=1
run "$OUT/dump-new.log" "$NEW" "$P1" 4 Q4X_DUMP=1
grep '^D ' "$OUT/dump-base.log" > "$OUT/dump-base.d"; grep '^D ' "$OUT/dump-new.log" > "$OUT/dump-new.d"
nd=$(wc -l < "$OUT/dump-base.d")
[ "$nd" -gt 1000 ] || fail identity "only $nd block sums in the base dump"
cmp -s "$OUT/dump-base.d" "$OUT/dump-new.d" && echo "identity base vs new plain: $nd block sums identical" | tee -a "$R" \
  || fail identity "block sums differ base vs new ($(diff "$OUT/dump-base.d" "$OUT/dump-new.d" | grep -c '^<') of $nd)"
bad=0; badn=""; i=0
for pf in $(ls bench/q4x-prompts/*.prompt | head -"$NP"); do
  b=$(basename "$pf" .prompt); i=$((i + 1))
  if [ $((i % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for arm in $order; do
    if [ "$arm" = A ]; then run "$OUT/$b.A.log" "$NEW" "$pf" "$NT"
    elif [ "$B_SPEC" = 1 ]; then run "$OUT/$b.B.log" "$B_BIN" "$pf" "$NT" Q4X_MTP="$MTP" Q4X_SPEC="$K" Q4X_SPEC_P="$PMIN"
    else run "$OUT/$b.B.log" "$B_BIN" "$pf" "$NT"; fi
  done
  [ "$B_SPEC" != 1 ] || grep -q '^spec rounds' "$OUT/$b.B.log" || fail spec "$b arm B printed no spec line (MTP path not taken)"
  ia=$(grep '^ids ' "$OUT/$b.A.log"); ib=$(grep '^ids ' "$OUT/$b.B.log")
  ga=$(grep -o 'gen ms/token [0-9.]*' "$OUT/$b.A.log" | awk '{print $3}')
  gb=$(grep -o 'gen ms/token [0-9.]*' "$OUT/$b.B.log" | awk '{print $3}')
  [ -n "$gb" ] || gb=$(grep '^prompt ' "$OUT/$b.B.log" | awk '{print $13}')
  if [ "$B_SPEC" = 1 ]; then sp=$(grep '^spec ' "$OUT/$b.B.log" | sed 's/ gen ms.*//'); else sp="B plain"; fi
  ma=$(grep -o 'expert misses [0-9]* of [0-9]*' "$OUT/$b.A.log"); mb=$(grep -o 'expert misses [0-9]* of [0-9]*' "$OUT/$b.B.log")
  if [ "$ia" = "$ib" ]; then eq=EQUAL; else eq=DIFF; bad=$((bad + 1)); badn="$badn $b"; fi
  echo "$b order=$order ids=$eq A_gen_ms=$ga B_gen_ms=$gb | $sp | A $ma | B $mb | $(grep '^wall ' "$OUT/$b.B.log") | vram_free_gb_after $(vram)" | tee -a "$R"
done
python3 - "$R" <<'EOF' | tee -a "$R"
import re, statistics, sys
a=[]; b=[]; tr=[]
for l in open(sys.argv[1]):
    m=re.search(r'A_gen_ms=([\d.]+) B_gen_ms=([\d.]+)', l)
    if m: a.append(float(m[1])); b.append(float(m[2]))
    m=re.search(r'tokens/round ([\d.]+)', l)
    if m: tr.append(float(m[1]))
if a:
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"MEDIAN gen ms/token A {ma:.2f} B {mb:.2f} speedup {ma/mb:.3f} n={len(a)} tokens/round median {statistics.median(tr) if tr else 0:.2f}")
EOF
[ "$bad" -eq 0 ] || fail ids "$bad/$NP prompts differ:$badn"
echo "PASS q4x-spec ids $NP/$NP equal" | tee -a "$R"
