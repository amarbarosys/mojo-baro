#!/usr/bin/env bash
# Memory-clock sweep of LACT profiles at spec k=2: per arm set the profile, read the live top memory DPM back, then one
# bench/energy-run.sh (tok/s, GPU/CPU J/tok). Mirrored order cancels drift. A trap restores the starting profile.
# usage: gpu-wait run --timeout S -- bench/mclk-sweep.sh ENGINE OUT "PROFILE..."   (run it as the gpu-wait job itself)
set -euo pipefail
cd "$(dirname "$0")/.."
eng=${1:?engine}; out=${2:?out}; profs=${3:?profiles}
start=$(lact cli profile get)
trap 'lact cli profile set "$start" >/dev/null; echo "restored $(lact cli profile get)"' EXIT
mkdir -p "$out"
read -ra P <<< "$profs"
order=("${P[@]}"); for ((i=${#P[@]}-1; i>=0; i--)); do order+=("${P[i]}"); done
n=0
for p in "${order[@]}"; do
  n=$((n + 1))
  lact cli profile set "$p" > /dev/null || { echo "FAIL profile: set $p" >&2; exit 1; }
  sleep 3
  [ "$(lact cli profile get)" = "$p" ] || { echo "FAIL profile: read-back is not $p" >&2; exit 1; }
  top=$(grep -oE '[0-9]+Mhz' /sys/class/drm/card1/device/pp_dpm_mclk | tail -1)
  r=$(env GPU_WAITING_ROOM_JOB=1 REQ_SPEC=true bench/energy-run.sh "$out" "$eng" 1024 \
      "BARO_SPEC=1 BARO_SPEC_K=2 BARO_TMAX=16384 HSA_ENABLE_MWAITX=1" "${p}_$n" 2>&1 | tail -1)
  echo "$p mclk_top=$top $r"
done
