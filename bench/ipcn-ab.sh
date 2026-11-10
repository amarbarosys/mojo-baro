#!/usr/bin/env bash
# usage: K=4 bench/ipcn-ab.sh OUTDIR [ENGINE]   (run under gpu-wait; K engines, default 2)
# K resident engines on one GPU; engines 2..K map the first one's weights over HIP IPC (BARO_WEIGHTS_EXPORT/IMPORT).
# Phase 1: N requests to engine A alone. Phase 2: the same N requests to A and N to B at the same moment.
# Aggregate tok/s = generated tokens / wall between the first request sent and the last "done" line.
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
out=$1; eng=${2:-.work/bake27/engine-lean}; ieng=${IMPORT_ENGINE:-$eng}; ienv=${IMPORT_ENV:-}; ispec=${IMPORT_SPEC:-true}; mkdir -p "$out"; rm -f "$out"/*.log "$out"/*.fifo
pack=.work/bake27/verify3/pack; shim=$(realpath .work/shim-build/libamarbaro_shim.so); rec=$out/weights.rec
n=${N:-6}; gen=64
mapfile -t prompts < <(ls bench/mtp-prompts/p0[1245]-*.tokens bench/mtp-prompts/p1[04]-*.tokens | head -n "$n")
: > "$out/req.jsonl"; i=0
for tf in "${prompts[@]}"; do i=$((i+1)); printf '{"id":%d,"prompt":[%s],"n":%d,"spec":true}\n' "$i" "$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')" "$gen" >> "$out/req.jsonl"; done
sed "s/\"spec\":true/\"spec\":$ispec/" "$out/req.jsonl" > "$out/req-imp.jsonl"
printf '{"id":99,"prompt":[%s],"n":4,"spec":true}\n' "$(tr -s ' \n' ',,' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')" > "$out/warm.jsonl"
K=${K:-2}; names=(a b c d e f g h); names=("${names[@]:0:$K}")
for x in "${names[@]}"; do mkfifo "$out/$x.fifo"; done
pids=(); trap 'for fd in 3 4 5 6 7 8 9; do eval "exec $fd>&-" 2>/dev/null; done; kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null; true' EXIT
stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
launch() {  # name, extra env
  local x=$1; shift
  ( env BARO_VRAM_LOG=1 BARO_PACK=$pack BARO_SERVE=1 BARO_TMAX=1088 BARO_MEGA=1 BARO_SPEC=1 BARO_EMBD_HOST=1 BARO_SHIM_LIB=$shim "$@" "${ENG_FOR:-$eng}" < "$out/$x.fifo" 2>&1 | stamp > "$out/$x.log" ) &
  pids+=($!)
}
launch a BARO_WEIGHTS_EXPORT=$rec; exec 3> "$out/a.fifo"
for _ in $(seq 120); do [ -s "$rec" ] && break; sleep 1; done; [ -s "$rec" ] || { echo "FAIL a: no weight record"; exit 1; }
fdn=4; for x in "${names[@]:1}"; do ENG_FOR=$ieng launch "$x" BARO_WEIGHTS_IMPORT=$rec $ienv; eval "exec $fdn> \"$out/$x.fifo\""; fdn=$((fdn+1)); done
fd_of() { local i; for i in "${!names[@]}"; do [ "${names[$i]}" = "$1" ] && echo $((i+3)); done; }
for x in "${names[@]}"; do eval "cat \"$out/warm.jsonl\" >&$(fd_of $x)"; done
wait_done() {  # file, count
  for _ in $(seq 240); do [ "$(grep -c '"done": *true' "$out/$1.log" || true)" -ge "$2" ] && return 0; sleep 0.5; done; echo "FAIL $1: $2 done lines not seen"; exit 1
}
for x in "${names[@]}"; do wait_done $x 1; done
grep -q 'weights exported over IPC' "$out/a.log" || { echo "FAIL a: no export read-back"; exit 1; }
for x in "${names[@]:1}"; do grep -q 'weights imported over IPC' "$out/$x.log" || { echo "FAIL $x: no import read-back"; exit 1; }; done
echo "engines $K vram_mib $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 ))" > "$out/arm.txt"
t0=$(date +%s.%N); cat "$out/req.jsonl" >&3; wait_done a $((n + 1)); t1=$(date +%s.%N)
sleep 1
t2=$(date +%s.%N)
for x in "${names[@]}"; do r=req.jsonl; [ "$x" = a ] || r=req-imp.jsonl; eval "cat \"$out/$r\" >&$(fd_of $x)"; done
for x in "${names[@]}"; do if [ "$x" = a ]; then wait_done a $((2 * n + 1)); else wait_done $x $((n + 1)); fi; done; t3=$(date +%s.%N)
python3 - "$t0" "$t1" "$t2" "$t3" "$n" "$gen" "$K" >> "$out/arm.txt" <<'PY'
import sys
t0,t1,t2,t3=map(float,sys.argv[1:5]); n=int(sys.argv[5]); g=int(sys.argv[6]); k=int(sys.argv[7])
single=n*g/(t1-t0); two=k*n*g/(t3-t2)
print(f"single engine A: {n*g} tokens in {t1-t0:.2f} s = {single:.1f} tok/s")
print(f"{k} engines: {k*n*g} tokens in {t3-t2:.2f} s = {two:.1f} tok/s  ratio {two/single:.3f}")
PY
cat "$out/arm.txt"
