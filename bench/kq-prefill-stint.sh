#!/usr/bin/env bash
# usage: bench/kq-prefill-stint.sh ENGINE PACK GGUF OUTDIR [LEN ...]   (default lengths 1024 16384)
# Same-stint prefill bar against llama.cpp (bench/llama-stint-protocol.md, run 8), one gpu-wait job: per length, our
# engine on bench/prefill-prompts/pNNNN.tokens (one process per run, BARO_SPEC=0, BARO_TMAX = max(16384, LEN + 1024)),
# then llama-bench pp LEN on the same gguf (-ngl 99 -fa 1, 2 reps), then ours again; ours = mean of the 2 runs.
# Read-back from every ours log: prompt tokens, pack kq trunk, TMAX; prefill_s is the engine's own number.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 pack=$2 gguf=$3 out=$4; shift 4
lens=${*:-1024 16384}
bench=${LLAMA_BENCH:-$HOME/Projects/imports/llama.cpp/build/bin/llama-bench}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 3000 -- env LLAMA_BENCH="$bench" "$0" "$eng" "$pack" "$gguf" "$out" $lens
fi
mkdir -p "$out"
echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) pack=$pack gguf=$gguf bench=$bench sha=$(sha256sum "$bench" | cut -c1-16) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
ours() { # len run
  local n=$1 r=$2 p log tmax s
  p=bench/prefill-prompts/p$(printf '%04d' "$n").tokens; log=$out/ours-$n-$r.log; tmax=$(( n + 1024 > 16384 ? n + 1024 : 16384 ))
  [ -f "$p" ] || { echo "FAIL ours: no prompt file $p"; exit 1; }
  env BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=$tmax BARO_PROMPT="$p" "$eng" > "$log" 2>&1 || { echo "FAIL ours $n: engine exit"; tail -3 "$log"; exit 1; }
  grep -q "TMAX: $tmax " "$log" || { echo "FAIL ours $n: TMAX read-back"; exit 1; }
  grep -q 'pack kq trunk: True' "$log" || { echo "FAIL ours $n: not a kq trunk"; exit 1; }
  grep -qE "prompt tokens: $n\$" "$log" || { echo "FAIL ours $n: prompt length read-back"; exit 1; }
  s=$(grep -oE 'prefill_s: [0-9.]+' "$log" | head -1 | awk '{print $2}' || true)
  [ -n "$s" ] || { echo "FAIL ours $n: no prefill_s"; exit 1; }
  echo "$n $r $s" >> "$out/ours.txt"
}
: > "$out/ours.txt"; : > "$out/llama.txt"
for n in $lens; do
  ours "$n" 1
  "$bench" -m "$gguf" -p "$n" -n 0 -ngl 99 -fa 1 -r 2 -o csv > "$out/llama-$n.csv" 2> "$out/llama-$n.err" || { echo "FAIL llama-bench $n"; tail -3 "$out/llama-$n.err"; exit 1; }
  ts=$(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /avg_ts/) c = i; next } c { gsub(/"/, "", $c); print $c }' "$out/llama-$n.csv" | tail -1)
  [ -n "$ts" ] || { echo "FAIL llama-bench $n: no avg_ts"; exit 1; }
  echo "$n $ts" >> "$out/llama.txt"
  ours "$n" 2
done
for n in $lens; do
  o=$(awk -v n="$n" '$1 == n { s += $3; k++ } END { printf "%.3f", s / k }' "$out/ours.txt")
  l=$(awk -v n="$n" '$1 == n { printf "%.3f", n / $2 }' "$out/llama.txt")
  awk -v n="$n" -v o="$o" -v l="$l" 'BEGIN { printf "len %6d  ours %8.3f s  llama %7.3f s  ours/llama time %.2fx\n", n, o, l, o / l }'
done | tee "$out/summary.txt"
