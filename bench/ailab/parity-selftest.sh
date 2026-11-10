#!/usr/bin/env bash
# usage: gpu-wait run --room cpu --cores 2 --mem 4 --timeout 600 -- env PATH=$PATH HOME=$HOME bash bench/ailab/parity-selftest.sh OUT
# CPU selftest of parity-llama.sh's re-exec path: DRY=1 QUEUE=stub re-execs the script through a fake queue with a CLEAN environment
# (env -i), so a knob missing from the forwarded list is dropped exactly as `gpu-wait run` drops it. Three stub runs into one OUT:
#   1 fresh, QUICK=3        -> "gen: 3 states" (QUICK crossed the queue), exit 3
#   2 RESUME=1, QUICK=3     -> a "RESUME:" line and NO gen line (RESUME crossed the queue)
#   3 STUB_FORWARD=0 (the old bug) with RESUME=1 -> no RESUME line, "gen: 20 states": the assertions of 2 must FAIL here, or they prove nothing
# Stand-ins only, no model. Prints PASS selftest or FAIL <step>: <why>, exit 0/1.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$(realpath -m "${1:?usage: parity-selftest.sh OUT}"); rm -rf "$out"; mkdir -p "$out"
run() { local n=$1; shift; env -u GPU_WAITING_ROOM_JOB DRY=1 QUEUE=stub "$@" bench/ailab/parity-llama.sh "$out/o" > "$out/run-$n.out" 2>&1 || :; }
run 1 QUICK=3
grep -q '^gen: 3 states' "$out/run-1.out" || { echo "FAIL run1: QUICK=3 did not cross the fake queue: $(grep -m1 '^gen:' "$out/run-1.out")"; exit 1; }
grep -q '^exit\|DRY RUN reached the verdict' "$out/run-1.out" || { echo "FAIL run1: no dry verdict, see $out/run-1.out"; exit 1; }
run 2 QUICK=3 RESUME=1
grep -q '^RESUME:' "$out/run-2.out" || { echo "FAIL run2: RESUME=1 did not cross the fake queue, see $out/run-2.out"; exit 1; }
! grep -q '^gen:' "$out/run-2.out" || { echo "FAIL run2: gen ran although RESUME reused it"; exit 1; }
run 3 QUICK=3 RESUME=1 STUB_FORWARD=0
if grep -q '^RESUME:' "$out/run-3.out"; then echo "FAIL run3: the negative control (nothing forwarded) still shows RESUME, the assertions prove nothing"; exit 1; fi
grep -q '^gen: 20 states' "$out/run-3.out" || { echo "FAIL run3: nothing forwarded should fall back to 20 states: $(grep -m1 '^gen:' "$out/run-3.out")"; exit 1; }
u=$(ls "$out/o/units"/*.ref.json | head -1); printf ' ' >> "$u"  # units regenerated behind the stage receipts
run 4 QUICK=3 RESUME=1
grep -q '^FAIL resume: stage' "$out/run-4.out" || { echo "FAIL run4: RESUME on changed units was not refused, see $out/run-4.out"; exit 1; }
! grep -q '^gen:' "$out/run-4.out" || { echo "FAIL run4: the pipeline ran on a corrupt mix"; exit 1; }
echo "PASS selftest: QUICK and RESUME cross the fake queue; a dropped knob is visible (negative control); RESUME refuses units changed since the stage receipts"
