#!/usr/bin/env bash
# usage: bench/state-roundtrip.sh ENGINE PACK P_TOKENS PQ_TOKENS OUTDIR
#
# Gate for the engine state file (BARO_STATE_SAVE / BARO_STATE_LOAD, LatentOS
# use 2): a state saved by one process and loaded by a fresh one must reproduce
# the cold output exactly, and must skip the saved prefix.
#   cold-p   prompt P, cold, saves the state at P's end
#   load-p   fresh process, loads it, prompt P      -> GENERATED == cold-p
#   cold-pq  prompt P+Q, cold
#   load-pq  fresh process, loads P's state, P+Q    -> GENERATED == cold-pq,
#            cached == len(P) - 1 (only Q is prefilled)
# Greedy, one-shot (BARO_SERVE=0). GPU through gpu-wait by the caller.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 pack=$2 p=$3 pq=$4 out=$5
mkdir -p "$out"
tmax=${BARO_TMAX:-8448}
run() { local prompt=$1 name=$2; shift 2
  env BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_PROMPT="$prompt" "$@" "$eng" > "$out/$name.log" 2>&1; }
run "$p" cold-p BARO_STATE_SAVE="$out/p.state"
run "$p" load-p BARO_STATE_LOAD="$out/p.state"
run "$pq" cold-pq
run "$pq" load-pq BARO_STATE_LOAD="$out/p.state"
gen() { grep '^GENERATED' "$out/$1.log" || echo "none-$1"; }
fail=0
[ "$(gen cold-p)" = "$(gen load-p)" ] && echo "PASS identity P" || { echo "FAIL identity P"; fail=1; }
[ "$(gen cold-pq)" = "$(gen load-pq)" ] && echo "PASS identity P+Q" || { echo "FAIL identity P+Q"; fail=1; }
np=$(wc -w < "$p")
grep -q "cached: $((np - 1)) " "$out/load-pq.log" && echo "PASS load-pq cached $((np - 1))" || { echo "FAIL load-pq did not reuse the saved prefix"; fail=1; }
for r in cold-p load-p cold-pq load-pq; do
  echo "$r: $(grep -oE 'cached: [0-9]+|replay rows: [0-9]+' "$out/$r.log" | tr '\n' ' ')$(grep -oE 'state (saved|loaded):.*' "$out/$r.log" | tr '\n' ' ') $(grep -oE '(tok/s_total|tok/s_gen|gpu_total_s|prefill_s|ttft_s): [0-9.]+' "$out/$r.log" | tr '\n' ' ')"
done
ls -la "$out/p.state" | awk '{print "state file bytes:", $5}'
exit $fail
