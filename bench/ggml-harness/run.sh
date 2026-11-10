#!/usr/bin/env bash
# Time every target in a targets file on llama.cpp's own kernels, under rocprofv3, cache-cold.
# usage: bench/ggml-harness/run.sh [TARGETS] [OUT_DIR]
#   TARGETS: lines "label | op_bench args" (default bench/ggml-harness/targets.txt; # comments)
#   OUT_DIR: default .work/ggml-harness/<date>; per target: <label>/tg/trace_kernel_trace.csv + stdout.txt
# Then ggml-harvest-catalog summarises device time per compiled variant into OUT_DIR/catalog.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
T=${1:-bench/ggml-harness/targets.txt}
OUT=${2:-.work/ggml-harness/$(date +%F)}
[ -x bench/ggml-harness/op_bench ] || bench/ggml-harness/build.sh
mkdir -p "$OUT"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${HARNESS_IN_JOB:-}" != 1 ]; then
  # gpu-wait drops the environment: re-enter with everything as arguments
  exec gpu-wait run --vram 8 --timeout 3600 -- env HARNESS_IN_JOB=1 "$PWD/bench/ggml-harness/run.sh" "$PWD/$T" "$PWD/$OUT"
fi
grep -vE '^\s*(#|$)' "$T" | while IFS='|' read -r label args; do
  label=$(echo "$label" | xargs); d="$OUT/$label/tg"; mkdir -p "$d"
  echo "== $label: $args"
  # shellcheck disable=SC2086
  rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- bench/ggml-harness/op_bench $args > "$OUT/$label/stdout.txt" 2> "$d/stderr.log" \
    || { echo "FAIL $label (see $d/stderr.log)"; continue; }
  grep -E '^(arm|wall)' "$OUT/$label/stdout.txt"
done
~/iTools/llm/ggml-harvest/harvest.sh catalog "$OUT" "${LLAMA_ROOT:-$HOME/llama.cpp-master}/ggml/src/ggml-cuda"
