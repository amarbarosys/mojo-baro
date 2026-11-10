#!/usr/bin/env bash
# usage: gpu-wait run --timeout 1800 -- bench/ailab/burst.sh OUTDIR
# Stability gate (briefs/2026-11-06-multi-session-serve.md): ROUNDS lockstep bursts of N chat requests
# (bench/ailab/load.sh, default 32 x 32 = 1,024) against one baro-serve --parallel N, then a capacity probe:
# OVER (100) concurrent game requests (bench/ailab/states-tp) on --parallel 32 --kv-pool 1 --max-inflight 64:
# the KV pool holds about 8 of them at once, so admissions wait for pages inside the engine, and requests past
# the in-flight bound must get HTTP 429, never a 5xx or a dead engine.
# Pass: every burst request 200, VRAM flat over the second half of the rounds (<= 16 MiB; the first rounds fill
# the runtime's allocation pool), no engine
# fault line, the probe sees >= 1 x 429 and >= 1 x 200 and nothing else, the server answers /health after it.
# Env: ENGINE, PACK, N (32), ROUNDS (32), OVER (8), BARO_ARGS (default "--parallel N --kv-pool 3").
set -euo pipefail
[ $# -eq 1 ] || { echo "FAIL usage: burst.sh OUTDIR"; exit 1; }
cd "$(dirname "$0")/../.."
out=$1; mkdir -p "$out"
engine=${ENGINE:-.work/ailab/engine-seqs32s2}; pack=${PACK:-.work/ailab/pack-kq}
n=${N:-32}; rounds=${ROUNDS:-32}; over=${OVER:-100}
server=serve/target/release/baro-serve
port=18433
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1; }
pid=""
stop() { [ -z "$pid" ] && return 0; kill -- -"$pid" 2>/dev/null || :
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 -- -"$pid" 2>/dev/null || :; pid=""; }
trap stop EXIT
start() {  # log args...
  local log=$1; shift
  # shellcheck disable=SC2068
  BARO_SPEC=0 setsid "$server" --engine "$engine" --pack "$pack" --port "$port" --engine-env BARO_TMAX=20480 \
    --engine-env BARO_SPEC=0 $@ > "$log" 2>&1 & pid=$!
  for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 "$pid" 2>/dev/null || { echo "FAIL start: server died ($log)"; tail -5 "$log"; exit 1; }
    sleep 0.5
  done
  echo "FAIL start: no /health in 300 s"; exit 1
}
for f in "$engine" "$pack/identity.json" "$server"; do [ -e "$f" ] || { echo "FAIL arm: missing $f"; exit 1; }; done
echo "date=$(date -Is) head=$(git rev-parse --short HEAD) engine=$engine sha=$(sha256sum "$engine" | cut -c1-16) n=$n rounds=$rounds" > "$out/arm.txt"
echo "vram_idle=$(vram)" >> "$out/arm.txt"
# shellcheck disable=SC2086
start "$out/burst-serve.log" ${BARO_ARGS:---parallel $n --kv-pool 3}
echo "health=$(curl -s "http://127.0.0.1:$port/health" | jq -c .limits)" >> "$out/arm.txt"
bench/ailab/load.sh "http://127.0.0.1:$port" "$n" 1 "$out/warm" > /dev/null
v0=$(vram); echo "vram_warm=$v0" >> "$out/arm.txt"
: > "$out/vram.tsv"
for k in $(seq 1 "$rounds"); do
  bench/ailab/load.sh "http://127.0.0.1:$port" "$n" 1 "$out/burst-$k" > /dev/null || { echo "FAIL burst: round $k"; exit 1; }
  printf '%d\t%d\n' "$k" "$(vram)" >> "$out/vram.tsv"
done
cat "$out"/burst-*/lat.tsv > "$out/burst-lat.tsv"
v1=$(vram); vh=$(awk -v h=$(( rounds / 2 )) '$1 == h {print $2}' "$out/vram.tsv")
echo "vram_after=$v1 delta_mib=$(( (v1 - v0) / 1048576 )) second_half_mib=$(( (v1 - vh) / 1048576 ))" >> "$out/arm.txt"
echo "burst: $(wc -l < "$out/burst-lat.tsv") requests, VRAM +$(( (v1 - v0) / 1048576 )) MiB total, +$(( (v1 - vh) / 1048576 )) MiB over the second half"
! grep -qE "Memory access fault|NOT-RESIDENT|worker stopped|kv pool exhausted" "$out/burst-serve.log" || { echo "FAIL burst: engine fault line in $out/burst-serve.log"; exit 1; }
[ $(( (v1 - vh) / 1048576 )) -le 16 ] || { echo "FAIL burst: VRAM still growing, +$(( (v1 - vh) / 1048576 )) MiB over rounds $(( rounds / 2 ))..$rounds"; exit 1; }
stop
start "$out/probe-serve.log" --parallel 32 --kv-pool 1 --max-inflight 64 --engine-env BARO_SEQS_TIME=1
: > "$out/probe.tsv"
pids=()
for i in $(seq 1 "$over"); do
  ( f=$(printf 'bench/ailab/states-tp/s%02d.json' "$i")
    code=$(curl -s -o "$out/probe-$i.json" -w '%{http_code}' -H 'content-type: application/json' --data-binary @"$f" \
      --max-time 900 "http://127.0.0.1:$port/v1/chat/completions" || echo 000)
    printf '%d\t%s\t%s\n' "$i" "$code" "$(jq -c '.error.message // .usage' "$out/probe-$i.json" 2>/dev/null | cut -c1-120)" >> "$out/probe.tsv" ) &
  pids+=($!)
done
wait "${pids[@]}"
c429=$(awk -F'\t' '$2 == 429' "$out/probe.tsv" | wc -l); c200=$(awk -F'\t' '$2 == 200' "$out/probe.tsv" | wc -l)
alive=$(curl -sf "http://127.0.0.1:$port/health" > /dev/null && echo yes || echo no)
dfr=$(grep -o "admissions deferred for pages [0-9]*" "$out/probe-serve.log" | tail -1 | awk '{print $NF}')
echo "probe: 200 x $c200, 429 x $c429, other x $((over - c200 - c429)), server alive $alive, admissions deferred ${dfr:-0}" | tee -a "$out/arm.txt"
[ "$c429" -ge 1 ] && [ "$c200" -ge 1 ] && [ $((c200 + c429)) -eq "$over" ] && [ "$alive" = yes ] || { echo "FAIL probe: see $out/probe.tsv"; exit 1; }
stop
echo "vram_end=$(vram)" >> "$out/arm.txt"
echo "PASS burst: $((n * rounds)) requests, VRAM +$(( (v1 - v0) / 1048576 )) MiB total / +$(( (v1 - vh) / 1048576 )) MiB second half, probe 200 x $c200 / 429 x $c429"
