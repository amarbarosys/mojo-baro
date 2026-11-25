#!/usr/bin/env bash
# CPU selftest (no GPU) that five bench gates exit non-zero on the input they exist to reject (P11), with stub engines:
#   force-ab-serve, force-ab: candidate at 0/64 forced agreement; ab-prompts (AB_KILL_AT=0): arms with other ids;
#   race-fp16: a bench printing "correct": false; claims-verify: a step filter that matches nothing.
# Each gate also runs once on a passing stub, so a gate that fails for every input shows too.
set -euo pipefail
cd "$(dirname "$0")/../.."
t=.work/gate-exit-selftest; rm -rf "$t"; mkdir -p "$t"
bad=0; n=0
stub() {  # name body: an executable stub, distinct bytes per name
  printf '#!/usr/bin/env bash\n%s\n# %s\n' "$2" "$1" > "$t/$1"; chmod +x "$t/$1"
}
want() {  # label expected-rc cmd...
  local label=$1 w=$2 rc=0; shift 2; n=$((n + 1))
  "$@" > "$t/$label.log" 2>&1 || rc=$?
  [ "$rc" = "$w" ] || { echo "  FAIL $label: rc $rc want $w, see $t/$label.log"; bad=$((bad + 1)); }
}
stub ref-serve 'echo "{\"ready\":true}"; while IFS= read -r l; do id=${l#*\"id\":}; id=${id%%,*}; echo "{\"id\":$id,\"tok\":11}"; echo "{\"id\":$id,\"done\":true}"; done'
stub ref-greedy 'echo "GENERATED: 1 2 3"'
stub cand-zero 'echo "forced agreement: 0 / 64"; echo "GENERATED: 1 2 3"'
stub cand-full 'echo "forced agreement: 64 / 64"; echo "GENERATED: 1 2 3"'
want force-ab-serve-zero 1 bench/force-ab-serve.sh "$t/ref-serve" "$t/cand-zero" "$t/o1"
want force-ab-serve-full 0 bench/force-ab-serve.sh "$t/ref-serve" "$t/cand-full" "$t/o2"
want force-ab-zero 1 bench/force-ab.sh "$t/ref-greedy" "$t/cand-zero" "$t/o3"
want force-ab-full 0 bench/force-ab.sh "$t/ref-greedy" "$t/cand-full" "$t/o4"
stub arm-a 'echo "GENERATED: 1 2 3"; echo "tok/s_gen: 100.0"'
stub arm-b 'echo "GENERATED: 1 2 4"; echo "tok/s_gen: 100.0"'
stub arm-a2 'echo "GENERATED: 1 2 3"; echo "tok/s_gen: 100.0"'
export AB_KILL_AT=0 AB_PROMPTS="$PWD/bench/mtp-prompts/p01-water.tokens $PWD/bench/mtp-prompts/p02-python-fib.tokens"
AB_ENGINE_B=$t/arm-b want ab-prompts-differ 1 bench/ab-prompts.sh "$t/arm-a" "$t/o5" A=1 B=1
AB_ENGINE_B=$t/arm-a2 want ab-prompts-same 0 bench/ab-prompts.sh "$t/arm-a" "$t/o6" A=1 B=1
unset AB_KILL_AT AB_PROMPTS
stub fp-ok 'echo "{\"gflops\": 1000.0, \"correct\": true}"'
stub fp-wrong 'echo "{\"gflops\": 1000.0, \"correct\": false}"'
export NUM_CU=96 SPREAD_MAX=0.5
want race-fp16-wrong 1 bench/race-fp16.sh 2 a="$PWD/$t/fp-wrong" b="$PWD/$t/fp-ok"
want race-fp16-ok 0 bench/race-fp16.sh 2 a="$PWD/$t/fp-ok" b="$PWD/$t/fp-ok"
want claims-verify-nomatch 1 bench/claims-verify.sh "$t/cv" all no-such-step
[ "$bad" = 0 ] || { echo "FAIL $bad/$n gate-exit cases"; exit 1; }
echo "PASS gate-exit selftest $n/$n (each gate fails on its known-bad stub and passes on the good one)"
