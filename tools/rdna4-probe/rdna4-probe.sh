#!/usr/bin/env bash
# rdna4-probe.sh [TARGET...]: compile one tiny kernel per AMD intrinsic that mojo-baro uses, per target, and print a
# table: OK (instruction selected) or FAIL (isel abort / compile error). CPU only, no GPU.
# Default targets gfx1100 gfx1200 gfx1201. Receipt: stdout; logs in .work/rdna4-probe/.
# Gotcha: an isel abort kills the whole build, so each probe is its own build.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo
D=.work/rdna4-probe; mkdir -p "$D"
T=("${@:-gfx1100 gfx1200 gfx1201}"); [ $# -gt 0 ] || T=(gfx1100 gfx1200 gfx1201)
declare -A P=(
  [sdot4]='r = sdot4(x, x + 1, r, False)'
  [sudot4]='r = sudot4(True, x, False, x + 1, r, False)'
  [perm]='r = Int32(perm(UInt32(x), UInt32(x + 7), UInt32(0x05040100)))'
  [setprio]='setprio[2](); r = x'
  [sendmsg_rtn_realtime]='r = Int32(realtime())'
  [readlane]='r = Int32(readlane(Float32(x), Int32(3)))'
  [wmma_f16_w16]='var a = SIMD[DType.float16, 16](Float16(x)); var c = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.f16", SIMD[DType.float32, 8]](a, a, SIMD[DType.float32, 8](0)); r = Int32(c[0])'
  [wmma_f16_w8]='var a = SIMD[DType.float16, 8](Float16(x)); var c = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.f16", SIMD[DType.float32, 8]](a, a, SIMD[DType.float32, 8](0)); r = Int32(c[0])'
  [wmma_bf16_w16]='var a = SIMD[DType.int16, 16](Int16(x)); var c = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16", SIMD[DType.float32, 8]](a, a, SIMD[DType.float32, 8](0)); r = Int32(c[0])'
  [wmma_bf16_w8]='var a = SIMD[DType.int16, 8](Int16(x)); var c = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16", SIMD[DType.float32, 8]](a, a, SIMD[DType.float32, 8](0)); r = Int32(c[0])'
  [wmma_iu8_w4]='var a = SIMD[DType.int32, 4](x); var c = llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu8", SIMD[DType.int32, 8]](True, a, True, a, SIMD[DType.int32, 8](0), False); r = c[0]'
  [wmma_iu8_w2]='var a = SIMD[DType.int32, 2](x); var c = llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu8", SIMD[DType.int32, 8]](True, a, True, a, SIMD[DType.int32, 8](0), False); r = c[0]'
)
names=$(printf '%s\n' "${!P[@]}" | sort)
run() { local t=$1 n=$2 f="$D/$n-$1.mojo"
  awk -v b="${P[$n]}" '{ if ($0 ~ /@@BODY@@/) print "    " b; else print }' tools/rdna4-probe/probe.mojo.in > "$f"
  if timeout 300 "$MOJO" build "$f" -I kernels --target-accelerator "$t" -o "$D/$n-$t.bin" > "$D/$n-$t.log" 2>&1; then echo "$t $n OK"
  elif grep -q "Cannot select" "$D/$n-$t.log"; then echo "$t $n FAIL isel"
  else echo "$t $n FAIL $(grep -m1 -o 'error: .*' "$D/$n-$t.log" | cut -c1-60)"; fi > "$D/$n-$t.res"; }
pids=()
for t in "${T[@]}"; do for n in $names; do while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n || true; done; run "$t" "$n" & pids+=($!); done; done
wait
printf '%-22s' probe; printf '%-10s' "${T[@]}"; echo
for n in $names; do printf '%-22s' "$n"; for t in "${T[@]}"; do printf '%-10s' "$(cut -d' ' -f3- "$D/$n-$t.res" | cut -c1-9)"; done; echo; done
