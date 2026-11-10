#!/usr/bin/env bash
# Headroom per target: device time per iteration (all kernels the op launched, from the
# rocprofv3 trace via catalog.csv) against the 7900 XTX roofline.
# usage: bench/ggml-harness/summarize.sh [TARGETS] [OUT_DIR]   -> OUT_DIR/headroom.md
# Peaks: 960 GB/s HBM, 122.8 TFLOPS dense fp16/bf16 WMMA (AMD spec sheet, RX 7900 XTX).
# Each trace holds 20 warmup + ITERS timed iterations of the same graph.
set -euo pipefail
cd "$(dirname "$0")/../.."
T=${1:-bench/ggml-harness/targets.txt}
OUT=${2:-.work/ggml-harness/$(date +%F)}
ITERS=${ITERS:-200}
{
echo "| target | device us/iter | GB/s | % of 960 GB/s | TFLOPS | % of 122.8 TFLOPS | roofline % | cache-cold | top kernel |"
echo "|---|---|---|---|---|---|---|---|---|"
grep -vE '^\s*(#|$)' "$T" | while IFS='|' read -r label args; do
  label=$(echo "$label" | xargs); set -- $args; op=$1
  s="$OUT/$label/stdout.txt"; [ -s "$s" ] || { echo "| $label | FAIL | | | | | | | |"; continue; }
  mb=$(grep -oE 'moved [0-9.]+ MB' "$s" | awk '{print $2}')
  cold=$(grep -q 'exceeds' "$s" && echo yes || echo NO)
  case $op in
    mul_mat) flop=$(awk -v n=$3 -v k=$4 -v m=$5 'BEGIN{print 2*n*k*m}') ;;
    flash_attn) flop=$(awk -v nh=$2 -v hd=$4 -v kv=$5 -v nq=$6 'BEGIN{print 4*nh*hd*kv*nq}') ;;
    *) flop=0 ;;
  esac
  awk -F, -v L="$label" -v it=$((ITERS + 20)) -v mb="$mb" -v fl="$flop" -v cold="$cold" '
    NR > 1 && $1 == L { t = $(NF - 7); us += t; if (t > best) { best = t; top = $3 } }  # variant is quoted and holds commas: count from the end
    END {
      if (us == 0) { printf "| %s | NO TRACE | | | | | | %s | |\n", L, cold; exit }
      d = us / it; gbs = mb / d * 1e3; tf = fl / d / 1e6
      bw = 100 * gbs / 960; cp = 100 * tf / 122.8; rf = bw > cp ? bw : cp
      printf "| %s | %.2f | %.0f | %.1f | %.2f | %.1f | %.1f | %s | %s |\n", L, d, gbs, bw, tf, cp, rf, cold, top
    }' "$OUT/catalog.csv"
done
} > "$OUT/headroom.md"
cat "$OUT/headroom.md"
