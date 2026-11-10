#!/usr/bin/env bash
# Second-client load for the Raphael iGPU, used by bench/p4-soak.sh (P4_SOAK_LOAD_CMD) as the
# preemption arm: vkcube on the iGPU's Vulkan device, unthrottled (drm-engine-gfx), plus a 60 fps
# VAAPI scale through ffmpeg (measured: drm-engine-compute, about 5 percent duty). Both compete
# with the engine's KFD queues for the two CUs. vkcube opens a window on the desktop; it needs
# WAYLAND_DISPLAY and XDG_RUNTIME_DIR, which gpu-wait does not pass on.
set -euo pipefail
NODE=${P4_IGPU_RENDER:-/dev/dri/renderD129}
GPU_NUMBER=${P4_IGPU_VK_INDEX:-1}
vulkaninfo --summary 2>/dev/null | grep -A8 "^GPU$GPU_NUMBER:" | grep -q INTEGRATED_GPU \
  || { echo "FAIL p4-igpu-load: Vulkan GPU$GPU_NUMBER is not the integrated GPU"; exit 1; }
ffmpeg -hide_banner -loglevel error -re -vaapi_device "$NODE" -f lavfi -i testsrc2=size=1920x1080:rate=60 \
  -vf format=nv12,hwupload,scale_vaapi=w=1280:h=720 -f null - &
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
vkcube --gpu_number "$GPU_NUMBER" --present_mode 0 &
wait -n
echo "FAIL p4-igpu-load: a load process exited"
exit 1
