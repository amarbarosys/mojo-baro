#!/usr/bin/env bash
# usage: bench/turbo-ktrace.sh ENGINE PACK OUT [K=V ...]   (queues its own gpu-wait job)
# Kernel trace of one resident spec k = 2 decode (p01, n = 200) with rocprofv3 --kernel-trace csv, then wavewait rank and a
# per-instance table (count, avg us, grid X, N = grid / 32 for the row-per-wave GEMVs) in OUT/inst.txt. Knobs after OUT go
# into the engine env (BARO_MEGA=0 BARO_SPEC=1 BARO_TMAX=16384 are fixed and read back). Writes OUT/{arm.txt,engine.log,rank.txt,inst.txt}.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 3 ] || { echo "FAIL usage: bench/turbo-ktrace.sh ENGINE PACK OUT [K=V ...]"; exit 1; }
eng=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); shift 3
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ -z "${KTRACE_DRY:-}" ]; then
  exec "$HOME/.local/bin/gpu-wait" run --vram 20 --timeout 900 --label turbo-ktrace -- "$0" "$eng" "$pack" "$out" "$@"
fi
mkdir -p "$out"; [ -z "$(ls -A "$out/run" 2>/dev/null)" ] || { echo "FAIL ktrace: $out/run not empty"; exit 1; }
lib=$(ls -d "$PWD"/.venv/lib/python*/site-packages/modular/lib)
echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $pack knobs BARO_MEGA=0 BARO_SPEC=1 BARO_TMAX=16384 $* commit $(git rev-parse --short HEAD) dirty='$(git status --porcelain -- kernels serve bench | tr '\n' ' ')' load_start=$(cut -d' ' -f1 /proc/loadavg)" > "$out/arm.txt"
printf '{"id":1,"prompt":[%s],"n":200,"spec":true}\n' "$(tr -s ' \n' ',,' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')" > "$out/req.jsonl"
timeout -s KILL 800 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$out/run" -- \
  bash -c "env LD_LIBRARY_PATH='$lib' BARO_MEGA=0 BARO_SPEC=1 BARO_TMAX=16384 BARO_SERVE=1 BARO_PACK='$pack' $* '$eng' < '$out/req.jsonl'" > "$out/engine.log" 2>&1 \
  || { echo "FAIL ktrace: rocprofv3 exit, see $out/engine.log"; tail -5 "$out/engine.log"; exit 1; }
for w in 'BARO_MEGA: False' 'BARO_SPEC: True' 'TMAX: 16384 ' 'BARO_TURBO:'; do
  grep -q "$w" "$out/engine.log" || { echo "FAIL readback: '$w' missing in $out/engine.log"; exit 1; }
done
grep -q '"done": *true' "$out/engine.log" || { echo "FAIL ktrace: no done line"; exit 1; }
grep -E 'BARO_TURBO:|"done"' "$out/engine.log" | cut -c1-220 | tee -a "$out/arm.txt"
tools/wavewait rank "$out/run" > "$out/rank.txt" || { echo "FAIL rank"; exit 1; }
csv=$(ls "$out"/run/*/*kernel_trace.csv | head -1)
awk -F'","|^"|"$' '/^"KERNEL_DISPATCH"/ { split($0, f, ","); n = f[8]; gsub(/"/, "", n); us = (f[11] - f[10]) / 1000; c[n]++; t[n] += us; g[n] = f[20]; v[n] = f[14] }
  END { for (k in c) printf "%8d %9.1f %10d %6d %5d  %s\n", c[k], t[k] / c[k], g[k], g[k] / 32, v[k], k }' "$csv" | sort -k1,1nr -k2,2nr > "$out/inst.txt"
sed -i '1i   count    avg_us     grid_x   N=g/32 vgpr  kernel' "$out/inst.txt"
head -3 "$out/rank.txt"; tail -2 "$out/rank.txt"; echo "PASS ktrace: $out"
