#!/usr/bin/env bash
# Every amar_* elementwise kernel: Mojo -> Metal LLVM IR -> SPIR-V (OpenCL kernel) -> Mesa rusticl on
# the lab host's Radeon R5 M330, checked against CPU references. Last line: PASS N/N or FAIL k/N <names>.
# usage: run.sh          the 13 elementwise kernels (the G0 gate)
#        run.sh gemv     G1 scouting: the engine's m = 1 q4 GEMV and its reduce
# env:   DEFS="-D V=151936" / "-D N=2048 -D K=1024" bakes other shapes into the IR, TAG names the work dir,
#        BUILD_ONLY=1 stops after the validated .spv files exist (time.sh drives the card itself)
set -euo pipefail
SET="${1:-elementwise}"
case "$SET" in
    elementwise) PROBE=probe_all.mojo; NAMES="" ;;
    gemv) PROBE=probe_gemv.mojo; NAMES="amar_matmul_skinny_q4rowb amar_skinny_reduce" ;;
    *) echo "FAIL usage: run.sh [elementwise|gemv]"; exit 2 ;;
esac
R="$(cd "$(dirname "$0")/../.." && pwd)"
T="$R/tools/spirv-probe"
W="$R/.work/spirv-probe/${TAG:-$SET}"
LAB="${LAB:-root@lab-host}"
die() { echo "FAIL $1: $2"; exit 1; }
rm -rf "$W/ir" "$W/out"
mkdir -p "$W/ir" "$W/out"
cd "$W/ir"
# shellcheck disable=SC2086
"$R/.venv/bin/mojo" build ${DEFS:-} -I "$R/kernels" "$T/$PROBE" --target-accelerator apple-m1 --emit asm -o pa.s > "$W/mojo.log" 2>&1 || die mojo-emit "$W/mojo.log"
clang -O2 "$T/helpers_check.c" -o "$W/helpers_check" -lm 2> "$W/helpers_check.log" || die helpers-build "$W/helpers_check.log"
"$W/helpers_check" || die helpers "integer bf16/f16 conversions differ from native casts"
clang --target=spirv64 -O2 -emit-llvm -S "$T/helpers.c" -o "$W/helpers.ll" 2> "$W/helpers.log" || die helpers-ir "$W/helpers.log"
python3 "$T/inventory.py" "$W"/ir/*.ll > "$W/inventory.md" || die inventory "$W/inventory.md"
rc=0
python3 "$T/air2spv.py" "$W/helpers.ll" "$W/out" "$W"/ir/*.ll || rc=1
for ll in "$W"/out/amar_*.ll; do
    n="$(basename "$ll" .ll)"
    # optimize to IR, then codegen at -O0: clang 22 -O2 codegen drops i64 -> i32 truncs (invalid SPIR-V on the GEMV)
    if ! { clang --target=spirv64 -O2 -emit-llvm -S "$ll" -o "$W/out/$n.opt.ll" && clang --target=spirv64 -O0 -c "$W/out/$n.opt.ll" -o "$W/out/$n.spv"; } 2> "$W/out/$n.clang.log"; then echo "FAIL spirv $n: $W/out/$n.clang.log"; rm -f "$W/out/$n.spv"; rc=1; continue; fi
    if ! spirv-val "$W/out/$n.spv" > "$W/out/$n.val.log" 2>&1; then echo "FAIL spirv-val $n: $W/out/$n.val.log"; rm -f "$W/out/$n.spv"; rc=1; fi
done
if [ -n "${BUILD_ONLY:-}" ]; then
    [ "$rc" -eq 0 ] || die build "a kernel failed conversion or SPIR-V validation, see lines above"
    echo "BUILT $W/out"; exit 0
fi
ssh -o BatchMode=yes "$LAB" 'rm -rf /root/g0/spv && mkdir -p /root/g0/spv' || die lab-ssh "$LAB unreachable"
scp -q "$T/host.c" "$LAB:/root/g0/" || die lab-scp host.c
scp -q "$W"/out/*.spv "$LAB:/root/g0/spv/" || die lab-scp spv
scp -q "$T/divprobe.c" "$LAB:/root/g0/" || die lab-scp divprobe.c
ssh -o BatchMode=yes "$LAB" 'cd /root/g0 && gcc -O2 divprobe.c -o divprobe -lOpenCL && RUSTICL_ENABLE=radeonsi ./divprobe 2>/dev/null' | tee "$W/divprobe.log" || die divprobe "$W/divprobe.log"
ssh -o BatchMode=yes "$LAB" 'cd /root/g0 && gcc -O2 -Wall host.c -o host -lOpenCL -lm 2> gcc.log || { echo "FAIL host-build: $(head -5 gcc.log)"; exit 1; }; RUSTICL_ENABLE=radeonsi ./host spv '"$NAMES" | tee "$W/card.log"
[ "$rc" -eq 0 ] || die convert "a kernel failed conversion upstream yet the card run passed; see lines above"
