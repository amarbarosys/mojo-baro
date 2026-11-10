#!/usr/bin/env bash
# usage: bench/ailab/parity.sh CAND_DIR REF_DIR [REF_DIR...]
# Greedy identity per request: every response in CAND_DIR (load.sh output) carries the same generated token ids
# (choices[0].tokens) as the reference response for the same state AND the same restore point
# (timings.cached: how many prompt tokens came from a prefix checkpoint). The restore point is part of the key
# because the kq prefill (WMMA) and the row replay round differently: a request restored deeper prefills fewer
# rows and can flip a near-tie, on the single-sequence engine as on the multi-session one (docs/BASELINE.md, kq
# prefill). References come from every REF_DIR; two references for one key that disagree void the gate.
# Prints one line per mismatch or missing reference and "parity: E/T equal, X without a reference";
# exit 0 only when every candidate response has a reference and equals it.
set -euo pipefail
[ $# -ge 2 ] || { echo "FAIL usage: parity.sh CAND_DIR REF_DIR [REF_DIR...]"; exit 1; }
cand=$1; shift
declare -A want
for ref in "$@"; do
  while IFS=$'\t' read -r r b s code _; do
    [ "$code" = 200 ] || continue
    f=$ref/r$r-b$b.json
    k="$s@$(jq -r .timings.cached "$f")"; t=$(jq -c '.choices[0].tokens' "$f")
    if [ -n "${want[$k]:-}" ] && [ "${want[$k]}" != "$t" ]; then echo "FAIL ref: references disagree for state@cached $k ($ref)"; exit 1; fi
    want[$k]=$t
  done < "$ref/lat.tsv"
done
eq=0; tot=0; bad=0; noref=0
while IFS=$'\t' read -r r b s code _; do
  tot=$((tot + 1))
  f=$cand/r$r-b$b.json
  k="$s@$(jq -r .timings.cached "$f" 2>/dev/null || echo -)"
  got=$(jq -c '.choices[0].tokens' "$f" 2>/dev/null || echo null)
  if [ -z "${want[$k]:-}" ]; then noref=$((noref + 1)); echo "NOREF state@cached $k r$r b$b http=$code"
  elif [ "$code" = 200 ] && [ "$got" = "${want[$k]}" ]; then eq=$((eq + 1))
  else bad=$((bad + 1)); echo "DIFF state@cached $k r$r b$b http=$code ref_n=$(jq length <<<"${want[$k]}") got_n=$(jq length <<<"$got" 2>/dev/null || echo -)"; fi
done < "$cand/lat.tsv"
echo "parity: $eq/$tot equal, $noref without a reference, ${#want[@]} reference keys"
[ "$bad" -eq 0 ] && [ "$noref" -eq 0 ] && [ "$tot" -gt 0 ] || { echo "FAIL parity: $bad differ, $noref without a reference ($cand)"; exit 1; }
