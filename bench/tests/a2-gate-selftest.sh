#!/usr/bin/env bash
# CPU selftest of bench/a2-gate.sh (no GPU): mock engines, one 8k set of 5 prompts.
#   pass:     all 5 recorded, SCOPED PASS
#   failfast: a bad request at the 2nd prompt stops the candidate there, exit 1
#   resume:   both engines die after 2 prompts -> exit 3 INCOMPLETE; the rerun serves only the other 3
set -euo pipefail
cd "$(dirname "$0")/../.."
t=.work/a2-gate-selftest; rm -rf "$t" .work/a2-rec/*-selftest*; mkdir -p "$t/pack"
head -c 64 /dev/zero > "$t/pack/pack.bin"
mk() { { cat bench/tests/a2-gate-mock-engine.py; echo "# $1 $RANDOM$RANDOM"; } > "$t/$1"; chmod +x "$t/$1"; }
export GPU_WAITING_ROOM_JOB=selftest BARO_PACK=$t/pack SETS=8192 N=5 MIN_PCT=90
mk ref; mk cand
MOCK_LOG=$t/pass.served bench/a2-gate.sh "$t/ref" "$t/cand" "$t/pass" > "$t/pass.log" 2>&1 || { echo "FAIL selftest pass: see $t/pass.log"; exit 1; }
grep -q 'SCOPED PASS' "$t/pass.log" || { echo "FAIL selftest pass: no SCOPED PASS line"; exit 1; }
mk cand2; : > "$t/ff.served"
if MOCK_BAD=102 MOCK_LOG=$t/ff.served bench/a2-gate.sh "$t/ref" "$t/cand2" "$t/ff" > "$t/ff.log" 2>&1; then echo "FAIL selftest failfast: exit 0 on a bad prompt"; exit 1; fi
grep -q 'failfast' "$t/ff.log" || { echo "FAIL selftest failfast: no failfast line, see $t/ff.log"; exit 1; }
rd=$(grep '^records' "$t/ff/arm.txt" | grep -oE 'cand=[^ ]+' | cut -d= -f2); n=$(ls "$rd"/*.json | wc -l)
[ "$n" = 2 ] || { echo "FAIL selftest failfast: $n candidate records, want 2 (stop at the bad prompt)"; exit 1; }
mk ref3; mk cand3; : > "$t/r1.served"; : > "$t/r2.served"
rc=0; MOCK_DIE=2 MOCK_LOG=$t/r1.served bench/a2-gate.sh "$t/ref3" "$t/cand3" "$t/r1" > "$t/r1.log" 2>&1 || rc=$?
[ "$rc" != 0 ] || { echo "FAIL selftest resume: killed run exited 0"; exit 1; }
rc=0; MOCK_DIE=2 MOCK_LOG=$t/r1.served bench/a2-gate.sh "$t/ref3" "$t/cand3" "$t/r1" >> "$t/r1.log" 2>&1 || rc=$?
MOCK_LOG=$t/r2.served bench/a2-gate.sh "$t/ref3" "$t/cand3" "$t/r2" > "$t/r2.log" 2>&1 || { echo "FAIL selftest resume: final run failed, see $t/r2.log"; exit 1; }
grep -q 'SCOPED PASS' "$t/r2.log" || { echo "FAIL selftest resume: no PASS after resume"; exit 1; }
tot=$(( $(wc -l < "$t/r1.served") + $(wc -l < "$t/r2.served") ))
[ "$tot" = 10 ] || { echo "FAIL selftest resume: $tot prompt runs over 3 attempts, want 10 (5 ref + 5 cand, none repeated)"; exit 1; }
grep -q INCOMPLETE "$t/r1.log" || grep -q 'without a record' "$t/r1.log" || { echo "FAIL selftest resume: killed run did not say it was incomplete"; exit 1; }
echo "PASS a2-gate selftest (pass 5/5, failfast at 2, resume: 10 prompt runs over 3 killed/resumed attempts, none repeated)"
