#!/usr/bin/env bash
# Confirmation stint of bench/dattn-protocol.md, ONE gpu-wait job, fail-closed:
#   1. gates: rebuild + run kernels/test_attn_block (gate 3 bit-identity) and tools/mega-gate.sh
#      (run-tests, mega == launch identity, model-ref 64/64); any failure aborts before timing
#   2. REPS repeats (default 10); odd reps run arm R then O, even reps O then R (order bias)
#      every target under rocprofv3 --kernel-trace; a background sampler logs sclk/mclk/power
#   3. per rep: device us/iter per target = the arm's own kernels from the 21st main-kernel dispatch on
#      (warmup excluded), divided by the main-kernel dispatch count; both read from the trace
#      -> OUT/reps.tsv; OUT/confirm.md = mean, min, max, spread % per arm and target
# usage: bench/dattn-confirm.sh [OUT_DIR] [TARGETS] [REPS]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-.work/dattn-confirm/$(date +%F-%H%M%S)}")
TG=$(realpath "${2:-bench/dattn-confirm-targets.txt}")
REPS=${3:-10}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 12 --timeout 5400 -- "$PWD/bench/dattn-confirm.sh" "$OUT" "$TG" "$REPS"
fi
mkdir -p "$OUT"
{ date -Is; git rev-parse --short HEAD; git status --short | head; cat bench/ggml-harness/op_bench.llama-commit
  awk '{printf "power cap %d W\n", $1/1e6}' /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1
  rocm-smi -d 0 --showperflevel 2>/dev/null | grep -E 'Perf'; } > "$OUT/gpu-state.txt" 2>&1 || true
echo "== gates"
./.venv/bin/mojo build kernels/test_attn_block.mojo -o .work/test_attn_block -I kernels
./.work/test_attn_block | tee "$OUT/gate3.txt"
grep -q "mismatched words 0 of" "$OUT/gate3.txt" || { echo "GATE 3 FAIL"; exit 1; }
tools/mega-gate.sh "$OUT/mega-gate" > "$OUT/mega-gate.log" 2>&1 || true
cat "$OUT/mega-gate/SUMMARY.txt"
if grep -q "^FAIL" "$OUT/mega-gate/SUMMARY.txt" || ! grep -q "^PASS" "$OUT/mega-gate/SUMMARY.txt"; then echo "ENGINE GATE FAIL"; exit 1; fi
./.venv/bin/mojo build bench/bench_dattn.mojo -o .work/bench_dattn -I kernels
( while :; do echo "$(date +%s.%N) $(rocm-smi -d 0 --showclocks --showpower 2>/dev/null | grep -E "sclk|mclk|Power" | sed -E "s/^GPU\[0\][[:space:]]*: //" | tr "\n" "|")"; sleep 0.5; done ) > "$OUT/clocks.log" 2>&1 &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null || true' EXIT
one() {  # arm label args rep
  local d="$OUT/rep$4/$1/$2"; mkdir -p "$d"
  local bin=.work/bench_dattn pat=amar_dattn main='amar_dattn_(split|exact)'
  if [ "$1" = R ]; then bin=bench/ggml-harness/op_bench; pat=flash_attn; main='flash_attn_ext_vec'; fi
  # shellcheck disable=SC2086
  rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- $bin $3 > "$d/stdout.txt" 2> "$d/stderr.log" || { echo "FAIL $1 $2 rep$4"; return; }
  grep -q "exceeds" "$d/stdout.txt" || echo "VOID-ROTATION $1 $2 rep$4"
  local us; us=$(gawk -v pat="$pat" -v main="$main" 'BEGIN{FPAT="([^,]*)|(\"[^\"]*\")"}
    NR>1 && $8 ~ pat { n++; st[n] = $10 + 0; du[n] = $11 - $10; mn[n] = ($8 ~ main) }
    END { PROCINFO["sorted_in"] = "@val_num_asc"; for (i in st) ord[++k] = i
          delete PROCINFO["sorted_in"]; for (i = 1; i <= k; i++) idx[i] = ord[i]
          PROCINFO["sorted_in"] = "@ind_num_asc"
          for (j = 1; j <= k; j++) { i = idx[j]; if (mn[i]) m++; if (m > 20) { t += du[i]; if (mn[i]) c++ } }
          if (c == 0) { printf "NaN"; exit } printf "%.3f", t / c / 1e3 }' "$d/trace_kernel_trace.csv")
  printf "%s\t%s\t%s\t%s\n" "$4" "$1" "$2" "$us" >> "$OUT/reps.tsv"
}
: > "$OUT/reps.tsv"
for rep in $(seq 1 "$REPS"); do
  order="R O"; [ $((rep % 2)) -eq 0 ] && order="O R"
  for arm in $order; do
    grep -vE '^\s*(#|$)' "$TG" | while IFS='|' read -r a label args; do
      a=$(echo "$a" | xargs); label=$(echo "$label" | xargs)
      if [ "$a" = "$arm" ]; then one "$arm" "$label" "$args" "$rep"; fi
    done
  done
  echo "rep $rep done ($order)"
done
{
echo "| target | R mean us | R min-max | R spread % | O mean us | O min-max | O spread % | R/O |"
echo "|---|---|---|---|---|---|---|---|"
gawk -F'\t' '{k=$3; a=$2; v=$4; n[a,k]++; s[a,k]+=v; if(!((a,k) in mn)||v<mn[a,k])mn[a,k]=v; if(!((a,k) in mx)||v>mx[a,k])mx[a,k]=v; keys[k]=1}
  END{for(k in keys){
    rm = n["R",k] ? s["R",k]/n["R",k] : 0; om = n["O",k] ? s["O",k]/n["O",k] : 0
    rs = rm ? sprintf("%.2f-%.2f", mn["R",k], mx["R",k]) : "-"; os = om ? sprintf("%.2f-%.2f", mn["O",k], mx["O",k]) : "-"
    rsp = rm ? sprintf("%.1f", 100*(mx["R",k]-mn["R",k])/rm) : "-"; osp = om ? sprintf("%.1f", 100*(mx["O",k]-mn["O",k])/om) : "-"
    printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", k, rm?sprintf("%.2f",rm):"-", rs, rsp, om?sprintf("%.2f",om):"-", os, osp, (rm&&om)?sprintf("%.2f",rm/om):"-"}}' "$OUT/reps.tsv" | sort
echo
echo "Arm lines (rep 1):"
for d in "$OUT"/rep1/*/*/; do echo "- $(basename "$(dirname "$d")")/$(basename "$d"): $(grep -E '^arm' "$d/stdout.txt" | head -1)"; done
echo
echo "Per-kernel resources (rep 1, from the trace):"
for d in "$OUT"/rep1/*/*/; do
  echo "- $(basename "$(dirname "$d")")/$(basename "$d"):"
  ~/iTools/bin/rocprof-kernels "$d/trace_kernel_trace.csv" --iters 5000 --warmup 20 2>/dev/null | grep -E '^\| [a-z_]' | grep -v copyBuffer | sed 's/^/    /'
done
} > "$OUT/confirm.md"
cat "$OUT/confirm.md"
