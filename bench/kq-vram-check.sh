#!/usr/bin/env bash
# usage: bench/kq-vram-check.sh MAIN_ENGINE NEW_ENGINE KQ_PACK OUTDIR
# Run 10 VRAM read-back (bench/llama-stint-protocol.md): both engines prefill bench/prefill-prompts/p1024.tokens with
# BARO_VRAM_LOG=1 (drm mem_info_vram_used per stage), one gpu-wait job alone on the card. Prints after_alloc_bufs and
# after_first_token for both and new - main; read-back per log: TMAX, pack kq trunk, prefill rows.
set -euo pipefail
cd "$(dirname "$0")/.."
main=$1 new=$2 pack=$3 out=$4
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 900 -- "$0" "$@"
fi
mkdir -p "$out"
p=bench/prefill-prompts/p1024.tokens
echo "main=$main sha=$(sha256sum "$main" | cut -c1-16) new=$new sha=$(sha256sum "$new" | cut -c1-16) pack=$pack load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
stage() { grep -oP "VRAM_LOG stage: $2 bytes: \K[0-9]+" "$1" | head -1; }
for arm in main new; do
  e=$main; [ $arm = main ] || e=$new; log=$out/$arm.log
  env BARO_VRAM_LOG=1 BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$p" "$e" > "$log" 2>&1 || { echo "FAIL $arm: engine exit"; tail -3 "$log"; exit 1; }
  grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL $arm: TMAX read-back"; exit 1; }
  grep -q 'pack kq trunk: True' "$log" || { echo "FAIL $arm: not a kq trunk"; exit 1; }
  grep -q ' prefill rows: 1023 ' "$log" || { echo "FAIL $arm: prefill rows read-back"; exit 1; }
  a=$(stage "$log" after_alloc_bufs); t=$(stage "$log" after_first_token)
  [ -n "$a" ] && [ -n "$t" ] || { echo "FAIL $arm: no VRAM_LOG stages"; exit 1; }
  echo "$arm after_alloc_bufs $a after_first_token $t prefill_s $(grep -oE 'prefill_s: [0-9.]+' "$log" | head -1 | awk '{print $2}')" | tee -a "$out/stages.txt"
done
awk '{ a[$1] = $3; t[$1] = $5 } END { printf "new - main: after_alloc_bufs %+.1f MB  after_first_token %+.1f MB\n", (a["new"] - a["main"]) / 1e6, (t["new"] - t["main"]) / 1e6 }' "$out/stages.txt" | tee "$out/summary.txt"
