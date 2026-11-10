#!/usr/bin/env bash
# One OrcaSAQ-2-27B engine (int8 KV) behind baro-serve for interactive chat (omp provider orca27b, port 8330). No team, no system prompt: the client brings its own.
# usage: bench/orca27b-serve.sh      stop with Ctrl-C or kill the gpu-wait job
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
W=${W:-65536}
. bench/min16k.sh; min16k "$W" 16000
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority ${PRIO:-90} --vram 23 --timeout 28800 -- env GPU_WAITING_ROOM_JOB=1 W="$W" ENG="${ENG:-}" PATH="$PATH" "$PWD/bench/orca27b-serve.sh"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
eng=$(realpath ${ENG:-.work/crew/engine-kv8-fix2}); pack=$(realpath .work/bake27/verify3/pack); tok=$(realpath .work/crew/tok/tokenizer.json); tpl=$(realpath serve/templates/qwen35.jinja)
shim=$(realpath .work/shim-build/libamarbaro_shim.so)
used=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )); [ "$used" -lt 2560 ] || fail preflight "vram_used ${used} MiB"
echo "engine_sha=$(sha256sum "$eng" | cut -c1-16) window=$W template_sha=$(sha256sum "$tpl" | cut -c1-16)"
exec env BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=1 BARO_MEGA=1 BARO_TMAX="$W" BARO_SHIM_LIB="$shim" \
  serve/target/release/baro-serve --engine "$eng" --pack "$pack" --tokenizer "$tok" --chat-template-file "$tpl" --port 8330
