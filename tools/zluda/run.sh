#!/usr/bin/env bash
# Run a Mojo binary built with --target-accelerator sm_XX on the local AMD GPU through ZLUDA.
# usage: tools/zluda/run.sh BIN [ARGS...]   (ZLUDA release unpacked at .work/zluda/zluda)
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
z=${ZLUDA_DIR:-$root/.work/zluda/zluda}
s=$root/.work/zluda/nvml
[[ -f $z/libcuda.so.1 ]] || { echo "FAIL zluda: no $z/libcuda.so.1 (gh release download -R vosen/ZLUDA -p '*linux*')"; exit 1; }
mkdir -p "$s"
if [[ ! -f $s/libnvidia-ml.so.1 || $root/tools/zluda/nvml_shim.c -nt $s/libnvidia-ml.so.1 ]]; then
  cc -O2 -shared -fPIC -o "$s/libnvidia-ml.so.1" "$root/tools/zluda/nvml_shim.c" -ldl || { echo "FAIL zluda: nvml shim build"; exit 1; }
  ln -sfn libnvidia-ml.so.1 "$s/libnvidia-ml.so"
fi
w=$root/.work/zluda/wrap-$(printf %s "$z" | md5sum | cut -c1-8)
if [[ ! -f $w/libcuda.so.1 || $root/tools/zluda/gen_cuda_wrap.sh -nt $w/libcuda.so.1 ]]; then
  mkdir -p "$w"; "$root/tools/zluda/gen_cuda_wrap.sh" "$z" "$w/cuda_wrap.c"
  cc -O2 -shared -fPIC -Wl,-soname,libcuda.so.1 -o "$w/libcuda.so.1" "$w/cuda_wrap.c" -ldl || { echo "FAIL zluda: cuda wrapper build"; exit 1; }
  ln -sfn libcuda.so.1 "$w/libcuda.so"
fi
exec env LD_LIBRARY_PATH="$w:$s:$z${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
