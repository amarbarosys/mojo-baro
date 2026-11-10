#!/usr/bin/env bash
# Negative control for run.sh: strip every barrier from the lowered IR of the kernels that have
# one and rerun them on the card. The gate is only evidence for the barrier lowering if this FAILS
# there. Needs a prior run.sh (uses its .work/spirv-probe/out). Last line: PASS reverse-arm or FAIL.
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
W="$R/.work/spirv-probe/elementwise"
LAB="${LAB:-root@lab-host}"
KS="amar_rmsnorm amar_rmsnorm_cast amar_rmsnorm_cast2 amar_softmax_rows amar_argmax_pos amar_argmax_row amar_quantize_q8_rows"
rm -rf "$W/rev"; mkdir -p "$W/rev"
for n in $KS; do
    [ -f "$W/out/$n.ll" ] || { echo "FAIL reverse-arm: $W/out/$n.ll missing, run run.sh first"; exit 1; }
    grep -v "call spir_func void @_Z7barrierj" "$W/out/$n.ll" > "$W/rev/$n.ll"
    clang --target=spirv64 -O2 -c "$W/rev/$n.ll" -o "$W/rev/$n.spv" 2> "$W/rev/$n.log" || { echo "FAIL reverse-arm spirv $n: $W/rev/$n.log"; exit 1; }
done
ssh -o BatchMode=yes "$LAB" 'rm -rf /root/g0/rev && mkdir -p /root/g0/rev && test -x /root/g0/host' || { echo "FAIL reverse-arm: lab host binary missing, run run.sh first"; exit 1; }
scp -q "$W"/rev/*.spv "$LAB:/root/g0/rev/"
# shellcheck disable=SC2029
ssh -o BatchMode=yes "$LAB" "cd /root/g0 && RUSTICL_ENABLE=radeonsi ./host rev $KS 2>/dev/null" > "$W/rev/card.log" && rc=0 || rc=$?
cat "$W/rev/card.log"
bad="$(grep -c '^FAIL amar_' "$W/rev/card.log")" || bad=0
if [ "$rc" -eq 0 ] || [ "$bad" -eq 0 ]; then echo "FAIL reverse-arm: kernels pass with barriers stripped, the gate cannot see the barrier lowering"; exit 1; fi
echo "PASS reverse-arm: $bad/7 barrier kernels fail once barriers are stripped"
