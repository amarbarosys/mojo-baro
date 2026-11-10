#!/usr/bin/env bash
# q4x decode trace (lane q4x-serve16k item 3 step 1): where the wall-minus-kernel gap goes.
# Two resident runs (Q4X_STDIN, one process, 20 prompts) of the same engine under rocprofv3 --kernel-trace --memory-copy-trace,
# generating N1 and N2 tokens per prompt; the difference per decode token cancels load, prefill and warm fill.
# usage: bench/q4x-trace.sh OUT_DIR      env ENGINE=bin EXTRA="K=V ..." N1=17 N2=49 TMAX=16384 NP=20 PROMPTS="glob of JSON id arrays" (runs itself under gpu-wait)
set -euo pipefail
OUT=${1:?usage: q4x-trace.sh OUT_DIR}
REPO=$(cd "$(dirname "$0")/.." && pwd); . "$REPO/bench/min16k.sh"
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
ENGINE=${ENGINE:?ENGINE}; N1=${N1:-17}; N2=${N2:-49}; TMAX=${TMAX:-16384}; NP=${NP:-20}; EXTRA=${EXTRA:-}
min16k "$TMAX" 16000
if [ -z "${Q4X_TRACE_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --timeout 2400 -- env Q4X_TRACE_INNER=1 ENGINE="$ENGINE" EXTRA="$EXTRA" PROMPTS="${PROMPTS:-}" N1="$N1" N2="$N2" TMAX="$TMAX" NP="$NP" Q4X_GGUF="$GGUF" HOME="$HOME" PATH="$PATH" bash "$REPO/bench/q4x-trace.sh" "$OUT"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
{ echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l)"
  echo "engine=$ENGINE sha256 $(sha256sum "$ENGINE" | cut -c1-16) extra='$EXTRA' tmax=$TMAX n1=$N1 n2=$N2 np=$NP"
  echo "rocprofv3 $(rocprofv3 --version 2>&1 | grep -m1 version)"; } | tee "$R"
P=$(ls ${PROMPTS:-bench/q4x-prompts/*.prompt} | head -n "$NP")
for n in $N1 $N2; do
  for pf in $P; do echo "$(basename "$pf" .prompt) $n $(jq -r 'join(" ")' "$pf")"; done > "$OUT/req-$n.txt"
  d=$OUT/trace-$n; rm -rf "$d"; mkdir -p "$d"
  env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX="$TMAX" $EXTRA rocprofv3 --kernel-trace --memory-copy-trace -f csv -d "$d" -o t -- "$ENGINE" "$GGUF" < "$OUT/req-$n.txt" > "$OUT/run-$n.log" 2> "$d/stderr.log" \
    || fail "trace $n" "run failed, $(grep -m1 -iE 'error|fail' "$OUT/run-$n.log" "$d/stderr.log" | cut -c1-200)"
  [ "$(grep -c '^out ' "$OUT/run-$n.log")" = "$NP" ] || fail "trace $n" "$(grep -c '^out ' "$OUT/run-$n.log") of $NP answers"
  echo "read-back n=$n: $(grep -m1 '^q4x: expert cache' "$OUT/run-$n.log" | cut -c1-120) | gen ms/token median under trace $(grep '^gen ms/token' "$OUT/run-$n.log" | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END {printf "%.3f", a[int((NR+1)/2)]}')" | tee -a "$R"
done
python3 tools/q4x-trace-gaps.py "$OUT/trace-$N1" $((NP * (N1 - 1))) "$OUT/trace-$N2" $((NP * (N2 - 1))) | tee -a "$R"
echo "job PASS" | tee -a "$R"
