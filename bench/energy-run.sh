#!/usr/bin/env bash
# Energy per token of one resident engine run: board power (hwmon power1_average, 100 ms samples) and CPU package
# energy (RAPL intel-rapl:0, also AMD) over the decode, beside tok/s from the engine's own timer.
# usage: bench/energy-run.sh OUT ENGINE N "ENV" [label]   (queues itself under gpu-wait; ENV like "BARO_SPEC=1 BARO_MEGA=1")
# Prints: label tok_s gpu_W_avg gpu_J_per_tok cpu_W_avg cpu_J_per_tok idle_W ... cpu_idle_W cpu_extra_J/tok (package
# energy above the 2 s idle baseline taken just before the request, so the desktop is not billed to the engine); writes OUT/<label>.{log,pw}.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage}; ENG=${2:?usage}; N=${3:?usage}; ENVS=${4:-}; LAB=${5:-run}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 12 --timeout ${ER_TIMEOUT:-900} -- env PATH="$PATH" REQ_SPEC="${REQ_SPEC:-true}" BARO_PACK="${BARO_PACK:-.work/engine-pack-q4}" BARO_TMAX="${BARO_TMAX:-16384}" "$0" "$@"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
mkdir -p "$OUT"
HW=$(ls /sys/class/drm/card1/device/hwmon/*/power1_average | head -1)
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
[ -r "$HW" ] && [ -r "$RAPL" ] || fail sensors "power1_average or RAPL not readable"
idle=0; for _ in 1 2 3 4 5; do idle=$((idle + $(cat "$HW"))); sleep 0.2; done; idle=$((idle / 5))
P=$(ls bench/mtp-prompts/p01-*.tokens)
req="{\"id\":1,\"prompt\":[$(tr -s ' \n' ',,' < "$P" | sed 's/,$//')],\"n\":$N,\"spec\":${REQ_SPEC:-true}}"
mkfifo "$OUT/$LAB.in" 2>/dev/null || true
env BARO_PACK="${BARO_PACK:-.work/engine-pack-q4}" BARO_SERVE=1 BARO_TMAX="${BARO_TMAX:-16384}" $ENVS "$ENG" < "$OUT/$LAB.in" > "$OUT/$LAB.log" 2>&1 &
epid=$!
exec 3> "$OUT/$LAB.in"
for _ in $(seq 600); do grep -q '"ready":true' "$OUT/$LAB.log" && break; sleep 0.1; done
grep -q '"ready":true' "$OUT/$LAB.log" || { kill $epid; fail ready "$OUT/$LAB.log"; }
c0=$(cat "$RAPL"); sleep 2; cidle=$(( ($(cat "$RAPL") - c0) / 2 ))  # CPU package µW with the engine loaded and idle
: > "$OUT/$LAB.pw"
( while :; do echo "$(date +%s%N) $(cat "$HW")" >> "$OUT/$LAB.pw"; sleep 0.1; done ) & spid=$!
e0=$(cat "$RAPL"); t0=$(date +%s%N)
echo "$req" >&3
for _ in $(seq 6000); do grep -q '"done":true' "$OUT/$LAB.log" && break; sleep 0.05; done
t1=$(date +%s%N); e1=$(cat "$RAPL")
kill $spid; exec 3>&-; wait $epid || true
grep -q '"done":true' "$OUT/$LAB.log" || fail done "$OUT/$LAB.log"
python3 - "$OUT/$LAB.log" "$OUT/$LAB.pw" "$t0" "$t1" "$e0" "$e1" "$idle" "$LAB" "$cidle" <<'E'
import json,sys
log,pw,t0,t1,e0,e1,idle,lab,cidle=sys.argv[1:]
d=[json.loads(l) for l in open(log) if l.startswith('{') and '"done":true' in l][-1]
n=d['n']; tok=d['tok_s']; secs=(int(t1)-int(t0))/1e9
s=[int(l.split()[1]) for l in open(pw) if l.strip() and int(t0)<=int(l.split()[0])<=int(t1)]
gw=sum(s)/len(s)/1e6 if s else float('nan')
de=int(e1)-int(e0)
if de < 0: de += int(open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read())
cj=de/1e6
if d.get('fail',0): print('FAIL fail word',d['fail']); sys.exit(1)
print(f"{lab} tok_s {tok:.2f} n {n} wall_s {secs:.2f} gpu_W {gw:.1f} gpu_J/tok {gw*secs/n:.3f} cpu_W {cj/secs:.1f} cpu_J/tok {cj/n:.3f} idle_W {int(idle)/1e6:.1f} samples {len(s)} cpu_idle_W {int(cidle)/1e6:.1f} cpu_extra_J/tok {(cj-int(cidle)/1e6*secs)/n:.3f}")
E
