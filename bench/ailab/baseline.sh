#!/usr/bin/env bash
# usage: gpu-wait run --timeout 1800 -- bench/ailab/baseline.sh OUTDIR
# Throughput arms for the street-sim AI lab (briefs/2026-11-06-multi-session-serve.md): the 20 states of
# bench/ailab/states/ in lockstep bursts of N (bench/ailab/load.sh) against
#   baro  = baro-serve + ENGINE + PACK (BARO_TMAX 20480: prompt ~1.8K + max_tokens 16000), one server for every N
#   llama = llama-server --parallel N on the same GGUF, restarted per N
# Env: ARMS ("baro llama"), NS ("1 4 16 32"), ENGINE, PACK, GGUF, LLAMA, BARO_ENV (extra K=V for the engine),
#      BARO_ARGS (extra baro-serve flags, e.g. "--parallel 16 --kv-pool 2"), ROUNDS1 (rounds at N=1, default 20);
#      load.sh reads STATES and SAMPLE from the environment.
# Writes OUTDIR/arm.txt (identity + read-back), OUTDIR/<arm>-n<N>/ (load.sh output), OUTDIR/table.tsv.
# Exit 0 all arms ran clean, 1 any failure (FAIL <step>: <reason>).
set -euo pipefail
[ $# -eq 1 ] || { echo "FAIL usage: baseline.sh OUTDIR"; exit 1; }
cd "$(dirname "$0")/../.."
out=$1; mkdir -p "$out"
arms=${ARMS:-baro llama}; ns=${NS:-1 4 16 32}
engine=${ENGINE:-.work/ailab/engine-main}; pack=${PACK:-.work/ailab/pack-kq}
gguf=${GGUF:-$HOME/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf}
llama=${LLAMA:-$HOME/llama.cpp/build/bin/llama-server}
server=serve/target/release/baro-serve
port=18431
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1; }
rounds() { case $1 in 1) echo "${ROUNDS1:-20}" ;; 4) echo 5 ;; *) echo $(( (32 + $1 - 1) / $1 )) ;; esac; }
pid=""
stop() { [ -z "$pid" ] && return 0; kill -- -"$pid" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 -- -"$pid" 2>/dev/null || true; pid=""; }
trap stop EXIT
up() {  # log
  for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 "$pid" 2>/dev/null || { echo "FAIL start: server died, tail of $1:"; tail -20 "$1"; exit 1; }
    sleep 0.5
  done
  echo "FAIL start: no /health in 300 s ($1)"; exit 1
}
for f in "$engine" "$pack/identity.json" "$gguf" "$server"; do [ -e "$f" ] || { echo "FAIL arm: missing $f"; exit 1; }; done
{ echo "date=$(date -Is) head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve bench/ailab | wc -l)"
  echo "engine=$engine sha=$(sha256sum "$engine" | cut -c1-16) server_sha=$(sha256sum "$server" | cut -c1-16)"
  echo "pack=$pack $(jq -c '{source, pack_sha256}' "$pack/identity.json" 2>/dev/null || head -c 200 "$pack/identity.json")"
  echo "gguf=$gguf states_md5=$(cat bench/ailab/states/*.json | md5sum | cut -c1-12)"
  echo "vram_start=$(vram)"; } > "$out/arm.txt"
printf 'arm\tn\trequests\twall_s\tcompletion_tokens\tagg_tok_s\tp50_s\tp95_s\terrors\n' > "$out/table.tsv"
row() {  # arm n dir
  sed -n 's/.*requests=\([0-9]*\) wall_s=\([0-9.]*\).*/\1 \2/p;s/.*completion_tokens=\([0-9]*\) agg_tok_s=\([0-9.]*\) p50_s=\([0-9.]*\) p95_s=\([0-9.]*\) errors=\([0-9]*\).*/\1 \2 \3 \4 \5/p' "$3/summary.txt" \
    | tr '\n' ' ' | awk -v a="$1" -v n="$2" '{printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", a, n, $1, $2, $3, $4, $5, $6, $7}' >> "$out/table.tsv"
}
for arm in $arms; do
  case $arm in
  baro)
    log=$out/baro-serve.log
    envs=(--engine-env BARO_TMAX=20480 --engine-env BARO_SPEC=0); for kv in ${BARO_ENV:-}; do envs+=(--engine-env "$kv"); done
    # shellcheck disable=SC2086
    BARO_SPEC=0 setsid "$server" --engine "$engine" --pack "$pack" --port "$port" "${envs[@]}" ${BARO_ARGS:-} > "$log" 2>&1 & pid=$!
    echo "baro args=${BARO_ARGS:-none} env=${BARO_ENV:-none}" >> "$out/arm.txt"
    up "$log"
    h=$(curl -s "http://127.0.0.1:$port/health")
    echo "baro health=$(jq -c .limits <<<"$h")" >> "$out/arm.txt"
    grep -E "(BARO_TMAX|BARO_MEGA|BARO_KVQ|BARO_SPEC|BARO_SEQS|BARO_KV_SHARE|seqs: N)|kv dtype|mega:" "$log" | head -16 | sed 's/^/baro log: /' >> "$out/arm.txt" || true
    [ "$(jq -r .limits.tmax <<<"$h")" -ge 20480 ] || { echo "FAIL arm: baro tmax $(jq -r .limits.tmax <<<"$h") < 20480 (prompt + max_tokens 16000)"; exit 1; }
    STATE_DIR="$PWD/bench/ailab/states" STATES=1 SAMPLE= bench/ailab/load.sh "http://127.0.0.1:$port" 1 1 "$out/baro-warm" > /dev/null
    for n in $ns; do bench/ailab/load.sh "http://127.0.0.1:$port" "$n" "$(rounds "$n")" "$out/baro-n$n"; row baro "$n" "$out/baro-n$n"; done
    ! grep -q "NOT-RESIDENT" "$log" || { echo "FAIL fail-word: NOT-RESIDENT in $log"; exit 1; }
    echo "baro vram_loaded=$(vram)" >> "$out/arm.txt"
    stop ;;
  llama)
    [ -x "$llama" ] || { echo "FAIL arm: no $llama"; exit 1; }
    echo "llama version=$("$llama" --version 2>&1 | grep -m1 -i version)" >> "$out/arm.txt"
    kvu=""; "$llama" --help 2>&1 | grep -q -- "--kv-unified" && kvu=--kv-unified
    for n in $ns; do
      log=$out/llama-n$n.log
      LD_LIBRARY_PATH="$(dirname "$llama"):${LD_LIBRARY_PATH:-}" setsid "$llama" -m "$gguf" --jinja -ngl 99 -fa on \
        -c 65536 $kvu --parallel "$n" --host 127.0.0.1 --port "$port" > "$log" 2>&1 & pid=$!
      up "$log"
      echo "llama n=$n $kvu read-back: $(grep -m3 -E "n_slots|n_ctx_seq|kv_unified" "$log" | tr '\n' ' ' | cut -c1-300)" >> "$out/arm.txt"
      STATE_DIR="$PWD/bench/ailab/states" STATES=1 SAMPLE= bench/ailab/load.sh "http://127.0.0.1:$port" 1 1 "$out/llama-warm-n$n" > /dev/null
      bench/ailab/load.sh "http://127.0.0.1:$port" "$n" "$(rounds "$n")" "$out/llama-n$n"; row llama "$n" "$out/llama-n$n"
      echo "llama n=$n vram_loaded=$(vram)" >> "$out/arm.txt"
      stop
    done ;;
  *) echo "FAIL usage: unknown arm $arm"; exit 1 ;;
  esac
done
echo "vram_end=$(vram)" >> "$out/arm.txt"
column -t "$out/table.tsv"
