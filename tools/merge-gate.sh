#!/usr/bin/env bash
# Post-merge gate (lanes tokenizer+server+prefill on main): build, tests, identity q4/q8, prefill parity, mtp 20/20, server suite
cd "$(dirname "$0")/.."
G=.work/merge-gate.txt; : > $G
bad=0
say() { echo "$*" | tee -a $G; case "$*" in *" exit "[1-9]*|*" FAIL "[1-9]*) bad=1;; esac; }
say "== merge gate $(date -Is)"
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine > .work/merge-build.log 2>&1; say "engine build exit $? : $(grep -m1 'error' .work/merge-build.log | cut -c1-200)"
./.venv/bin/mojo build kernels/test_prefill.mojo -I kernels -o .work/test_prefill > .work/merge-build-pf.log 2>&1; say "test_prefill build exit $? : $(grep -m1 'error' .work/merge-build-pf.log | cut -c1-200)"
./run-tests.sh > .work/merge-run-tests.log 2>&1; say "run-tests.sh exit $? : $(grep -E 'GEMM OK|orphan|PASS|FAIL' .work/merge-run-tests.log | tr '\n' ';' | cut -c1-200)"
tools/ci-checks.sh > .work/merge-ci.log 2>&1; say "ci-checks.sh exit $? : $(grep -E 'FAIL|passed|failed' .work/merge-ci.log | tr '\n' ';')"
./.work/test_prefill > .work/merge-test-prefill.log 2>&1; say "test_prefill exit $? : $(grep -E 'PASS|FAIL' .work/merge-test-prefill.log | tail -1)"
BARO_PACK=.work/engine-pack-q4 ./.work/engine > .work/merge-oneshot-q4.log 2>&1; say "one-shot q4 exit $? : $(tools/check-tokens.sh .work/engine-pack-q4/ref-tokens-64.txt .work/merge-oneshot-q4.log) $(grep -oE 'tok/s_gen: [0-9.]+' .work/merge-oneshot-q4.log) $(grep -E 'prefill rows|mega fail' .work/merge-oneshot-q4.log | tr '\n' ';')"
BARO_PACK=.work/engine-pack-q8 ./.work/engine > .work/merge-oneshot-q8.log 2>&1; say "one-shot q8 exit $? : $(tools/check-tokens.sh .work/engine-pack/ref-tokens-64.txt .work/merge-oneshot-q8.log) $(grep -oE 'tok/s_gen: [0-9.]+' .work/merge-oneshot-q8.log)"
BARO_PACK=.work/engine-pack-q4 BARO_PROMPT=bench/prefill-prompts/p0512.tokens ./.work/engine > .work/merge-pf512.log 2>&1; say "prefill p0512 exit $? : $(grep -E 'prefill rows|mega fail|prefill_s|tok/s_gen' .work/merge-pf512.log | tr '\n' ';')"
bench/mtp-prompts.sh .work/engine .work/merge-mtp 2 > .work/merge-mtp.log 2>&1; say "mtp-prompts.sh k=2 exit $? : identity PASS $(grep -c ' B 2 .* PASS$' .work/merge-mtp/results.txt)/20 FAIL $(grep -c ' FAIL$' .work/merge-mtp/results.txt)"
tools/test_server.sh .work/merge-server-test > .work/merge-test-server.log 2>&1; say "test_server.sh exit $? :"; cat .work/merge-server-test/SUMMARY.txt | tee -a $G; grep -q "ALL PASS" .work/merge-server-test/SUMMARY.txt || bad=1
say "== end $(date -Is)"
exit $bad
