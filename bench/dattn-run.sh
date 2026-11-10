#!/usr/bin/env bash
# Both arms of bench/dattn-protocol.md in ONE gpu-wait stint, every target under rocprofv3 --kernel-trace:
#   R: llama.cpp flash_attn_ext_vec via bench/ggml-harness/op_bench   (bench/dattn-targets-R.txt, run.sh format)
#   O: .work/bench_dattn                                               (bench/dattn-targets-O.txt, "label | SHAPE T ITERS NS NLD PATH")
# usage: bench/dattn-run.sh [OUT_DIR] [R_TARGETS] [O_TARGETS]   (default OUT .work/dattn-run/<date>-<time>)
# Writes OUT/{R,O}/<label>/{stdout.txt,tg/trace_kernel_trace.csv}, OUT/gpu-state.txt, then
# bench/dattn-summarize.sh OUT -> OUT/summary.md (device us/iter per target = the arm's own kernels, warmup included).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-.work/dattn-run/$(date +%F-%H%M%S)}")
RT=$(realpath "${2:-bench/dattn-targets-R.txt}")
OT=$(realpath "${3:-bench/dattn-targets-O.txt}")
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 10 --timeout 3600 -- "$PWD/bench/dattn-run.sh" "$OUT" "$RT" "$OT"
fi
mkdir -p "$OUT"
[ -x .work/bench_dattn ] || ./.venv/bin/mojo build bench/bench_dattn.mojo -o .work/bench_dattn -I kernels
[ -x bench/ggml-harness/op_bench ] || bench/ggml-harness/build.sh
{ date -Is; git rev-parse --short HEAD; cat bench/ggml-harness/op_bench.llama-commit 2>/dev/null
  rocm-smi -d 0 --showclocks --showpower --showtemp --showperflevel 2>/dev/null | grep -E 'sclk|mclk|Power|junction|Perf'
  awk '{printf "power cap %d W\n", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1
} > "$OUT/gpu-state.txt" 2>&1 || true
( while :; do echo "$(date +%s.%N) $(rocm-smi -d 0 --showclocks --showpower 2>/dev/null | grep -E "sclk|mclk|Power" | sed -E "s/^GPU\[0\][[:space:]]*: //" | tr "\n" "|")"; sleep 0.5; done ) > "$OUT/clocks.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "== R: $RT"
HARNESS_IN_JOB=1 bench/ggml-harness/run.sh "$RT" "$OUT/R"
echo "== O: $OT"
grep -vE '^\s*(#|$)' "$OT" | while IFS='|' read -r label args; do
  label=$(echo "$label" | xargs); d="$OUT/O/$label/tg"; mkdir -p "$d"
  echo "== $label: $args"
  # shellcheck disable=SC2086
  rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- .work/bench_dattn $args > "$OUT/O/$label/stdout.txt" 2> "$d/stderr.log" \
    || { echo "FAIL $label (see $d/stderr.log)"; continue; }
  grep -E '^(arm|wall)' "$OUT/O/$label/stdout.txt"
done
bench/dattn-summarize.sh "$OUT"
