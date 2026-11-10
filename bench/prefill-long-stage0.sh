#!/usr/bin/env bash
# Prefill-long stage 0: wall time (champion build, 3 runs) and the attention /
# SSM scan / GEMM / other split (BARO_PROFILE=1, one run) at 8k/16k/32k, then
# llama-bench pp at the same lengths on the Q4_0-pure gguf. Every run goes
# through the GPU waiting room at priority 20 (shared GPU, never preempts).
# usage: bench/prefill-long-stage0.sh <base-engine> <prof-engine> [lengths...]
set -u
cd "$(dirname "$0")/.."
BASE=${1:?base engine}; PROF=${2:?prof engine}; shift 2
LENS=${*:-8192 16384 32768}
L=.work/logs/stage0; mkdir -p "$L"; S=$L/stage0.txt
GGUF=~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf
E="env BARO_PACK=.work/engine-pack-q4 BARO_TMAX=32896"
cap=$(awk '{printf "%d", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
echo "stage0 $(date -Is) power_cap_W=$cap" >> "$S"
sha256sum "$BASE" "$PROF" | cut -c1-16 >> "$S"
rcpt() { grep -hoE 'prompt tokens: [0-9]+|pack q4 trunk: [A-Za-z]+|BARO_MEGA: [A-Za-z]+|BARO_SPEC: [A-Za-z]+|TMAX: [0-9]+|prefill chunk: [0-9]+|prefill rows: [0-9]+|prefill_s: [0-9.]+|tok/s_gen: [0-9.]+|mega fail word: [0-9]+' "$1" | tr '\n' ' '; echo "gen_md5=$(grep '^GENERATED' "$1" | md5sum | cut -c1-8)"; }
for n in $LENS; do
  p=bench/prefill-prompts/p$(printf '%04d' "$n").tokens
  for i in 1 2 3; do
    gpu-wait list > "$L/q-base-$n-$i.txt" 2>&1
    gpu-wait run --priority 20 --vram 12 -- $E BARO_PROMPT="$p" "$BASE" > "$L/base-$n-$i.log" 2>&1
    echo "base $n run$i exit $? $(rcpt "$L/base-$n-$i.log")" >> "$S"
  done
  gpu-wait list > "$L/q-prof-$n.txt" 2>&1
  gpu-wait run --priority 20 --vram 12 -- $E BARO_PROFILE=1 BARO_PROMPT="$p" "$PROF" > "$L/prof-$n.log" 2>&1
  echo "prof $n exit $? $(rcpt "$L/prof-$n.log") | $(grep -h 'prefill split' "$L/prof-$n.log")" >> "$S"
done
gpu-wait list > "$L/q-llama.txt" 2>&1
pl=$(echo $LENS | tr ' ' ',')
gpu-wait run --priority 20 --vram 14 -- ~/llama.cpp/build/bin/llama-bench -m "$GGUF" -p "$pl" -n 0 -r 3 -ngl 99 -fa on -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -o md > "$L/llama-bench.md" 2> "$L/llama-bench.err"
echo "llama-bench exit $?" >> "$S"; cat "$L/llama-bench.md" >> "$S"
echo "STAGE0 DONE $(date -Is)" >> "$S"
