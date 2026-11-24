#!/usr/bin/env bash
# speed.sh OUT [ROUNDS]: decode tok/s on the Xiaomi, three arms, order rotated every round, one process per run.
#   bar  llama-bench (956591f45, OpenCL, ngl 99) -p 0 -n 128 -r 1: tg128 from an empty context
#   gpu  engine --gpu 1, prompt qwen3-p1 (41 ids) then 128 tokens: decode over positions 41..168 (more attention
#        work than the bar's 0..127, so the comparison leans against us)
#   cpu  engine CPU path, same prompt and length, --threads 4 --arm kai (the gate's CPU arm)
# Files expected in /data/local/tmp/baro (ph.sh): llama-bench, libomp.so, engine, baro_gpu.cl, qwen3-p1.ids, the GGUF.
# Receipt OUT/speed.tsv (round arm tok/s), OUT/summary.txt (median, min, max, spread % per arm).
set -euo pipefail
out=${1:?usage: speed.sh OUT [ROUNDS]}; rounds=${2:-5}
here=$(cd "$(dirname "$0")" && pwd); m=Qwen3-8B-Q4_0-pure.gguf
mkdir -p "$out"; : > "$out/speed.tsv"
"$here/ph.sh" run 20 "cat /sys/class/thermal/thermal_zone0/temp; ls -la engine baro_gpu.cl llama-bench qwen3-p1.ids | wc -l" | tee "$out/pre.txt"
arms=(bar gpu cpu)
for r in $(seq 1 "$rounds"); do
  for j in 0 1 2; do
    a=${arms[$(( (j + r) % 3 ))]}
    case $a in
      bar) o=$("$here/ph.sh" run 400 "env LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:. timeout 380 ./llama-bench -m $m -ngl 99 -p 0 -n 128 -r 1 -o csv 2>/dev/null | tail -1")
           v=$(echo "$o" | head -1 | awk -F, '{gsub(/"/, "", $0); print $(NF-1)}') ;;
      gpu) o=$("$here/ph.sh" run 400 "timeout 380 ./engine --model $m --ids qwen3-p1.ids --n 128 --layout 48 --gpu 1 2>&1 | grep RESULT")
           v=$(echo "$o" | sed -n 's/.*decode=\([0-9.]*\) tok\/s.*/\1/p') ;;
      cpu) o=$("$here/ph.sh" run 400 "timeout 380 ./engine --model $m --ids qwen3-p1.ids --n 128 --layout 48 --threads 4 --arm kai 2>&1 | grep RESULT")
           v=$(echo "$o" | sed -n 's/.*decode=\([0-9.]*\) tok\/s.*/\1/p') ;;
    esac
    [ -n "$v" ] || { echo "FAIL run: $a round $r: $(echo "$o" | tr '\n' ' ' | cut -c1-300)" >&2; exit 1; }
    printf '%s\t%s\t%s\n' "$r" "$a" "$v" | tee -a "$out/speed.tsv"
  done
done
"$here/ph.sh" run 20 "cat /sys/class/thermal/thermal_zone0/temp" | tee -a "$out/pre.txt"
sort -k2,2 -k3,3g "$out/speed.tsv" | awk -F'\t' '{v[$2][++n[$2]]=$3} END {for (a in v) {k=n[a]; md=(k%2)?v[a][(k+1)/2]:(v[a][k/2]+v[a][k/2+1])/2;
  printf "%s\tmedian %.2f\tmin %.2f\tmax %.2f\tspread %.1f %%\truns %d\n", a, md, v[a][1], v[a][k], 100*(v[a][k]-v[a][1])/md, k}}' | sort | tee "$out/summary.txt"
