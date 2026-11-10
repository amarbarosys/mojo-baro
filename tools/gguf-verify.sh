#!/usr/bin/env bash
# One-command verification of a self-describing BARO gguf on THIS card:
#   1. rebuild the engine from the sources embedded in the file (tools/gguf-closure.sh)
#   2. gate the rebuilt engine's tokens against the reference (identity)
#   3. report this card and driver next to the baro.hw.* expectations the file carries,
#      and the rebuilt engine's tok/s_gen next to the embedded 20-prompt number.
# usage: tools/gguf-verify.sh MODEL-BARO-<sha>.gguf [OUTDIR]     (run under gpu-wait)
# Exit 0 = sources complete and tokens identical on this card; the tok/s line is a
# receipt for the hardware report, never a pass/fail (a different card has a different number).
set -uo pipefail
cd "$(dirname "$0")/.."
model=$1; out=${2:-.work/gguf-verify}; mkdir -p "$out"
kv=$(~/iTools/bin/gguf-kv "$model" 2>/dev/null || python3 tools/gguf-extract.py --kv "$model" 2>/dev/null)
get() { printf '%s\n' "$kv" | awk -v k="$1" '$1 == k {sub(/^[^ ]+ +/, ""); gsub(/^'"'"'|'"'"'$/, ""); print; exit}'; }
echo "file:      $model"
echo "commit:    $(get baro.kernel.commit)   files: $(get baro.kernel.files | tr ',' ' ' | wc -w)"
echo "expects:   card=$(get baro.hw.card) driver=$(get baro.hw.driver) rocm=$(get baro.hw.rocm) power_cap_w=$(get baro.hw.power_cap_w) tok_s_gen_20p=$(get baro.hw.tok_s_gen_20p) config=$(get baro.hw.config) protocol=$(get baro.hw.protocol)"
gfx=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'); card=$(cat /sys/class/drm/card*/device/product_name 2>/dev/null | grep -v '^$' | head -1)
cap=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
echo "this card: gfx=${gfx:-?} name=${card:-?} amdgpu=$(cat /sys/module/amdgpu/version 2>/dev/null || uname -r) rocm=$(cat /opt/rocm/.info/version 2>/dev/null) power_cap_w=$(( ${cap:-0} / 1000000 ))"
tools/gguf-closure.sh "$model" "" "$out" 2>&1 | tee "$out.closure.log" | grep -E "commit:|split layout|external|tok/s|PASS|FAIL|identical|mismatch|FAILED|error" 
rc=${PIPESTATUS[0]}
tps=$(grep -oE 'tok/s_gen: [0-9.]+' "$out/run.log" 2>/dev/null | tail -1 | cut -d' ' -f2)
echo "rebuilt engine tok/s_gen on this card (one prompt, a receipt not a bar): ${tps:-n/a}   embedded 20-prompt median: $(get baro.hw.tok_s_gen_20p)"
echo "closure exit: $rc"
exit $rc
