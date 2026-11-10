#!/usr/bin/env bash
# Same-job A/B for the lean 27B arm: control engine twice (noise floor) vs engine-lean with BARO_EMBD_HOST=1.
# usage: bench/orca27b-lean-ab.sh OUTDIR   (run under gpu-wait; env goes inside this script)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?outdir}; mkdir -p "$OUT"
PACK=.work/bake27/verify3/pack
CTRL=.work/bake27/engine-seq1
LEAN=.work/bake27/engine-lean
used=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
[ "$used" -lt 1500000000 ] || { echo "FAIL gpu-clean: $used bytes in use"; exit 1; }
run() { # arm engine log extra-env...
  local arm=$1 eng=$2 log=$3; shift 3
  for _ in $(seq 60); do
    [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 1500000000 ] && break
    sleep 1
  done
  [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 1500000000 ] || { echo "FAIL $arm: VRAM not free before launch"; exit 1; }
  env BARO_PACK=$PACK BARO_PROMPT=$P BARO_MEGA=1 BARO_SPEC=1 BARO_VRAM_LOG=1 "$@" "$eng" > "$log" 2>&1
  grep -q 'mega fail word: 0' "$log" || { echo "FAIL $arm: fail word not 0 in $log"; exit 1; }
}
for P in bench/mtp-prompts/*.tokens; do
  n=$(basename "$P" .tokens)
  run c1 $CTRL "$OUT/${n}_c1.log"
  run lean $LEAN "$OUT/${n}_lean.log" BARO_EMBD_HOST=1
  run c2 $CTRL "$OUT/${n}_c2.log"
done
python3 - "$OUT" <<'PY'
import sys,glob,os,statistics as st
out=sys.argv[1]
def get(f,k):
    for l in open(f):
        if k in l: return l
def gen(f): return get(f,"GENERATED:").strip()
def spd(f): return float(get(f,"tok/s_gen:").split("tok/s_gen:")[1])
def vram(f):
    v=[l for l in open(f) if "after_first_token" in l]
    return int(v[0].split("bytes:")[1]) if v else 0
rows=[]
for c1 in sorted(glob.glob(out+"/*_c1.log")):
    n=os.path.basename(c1)[:-7]; l=f"{out}/{n}_lean.log"; c2=f"{out}/{n}_c2.log"
    rows.append((n,gen(c1)==gen(c2),gen(c1)==gen(l),spd(c1),spd(c2),spd(l),vram(c1),vram(l)))
    print("%-18s c1==c2 %-5s c1==lean %-5s %6.2f %6.2f %6.2f  vram %.3f -> %.3f GB"%(rows[-1][:1]+rows[-1][1:6]+(rows[-1][6]/1e9,rows[-1][7]/1e9)))
print("median tok/s_gen c1 %.2f c2 %.2f lean %.2f  lean/c1 %.3f"%(st.median(r[3] for r in rows),st.median(r[4] for r in rows),st.median(r[5] for r in rows),st.median(r[5] for r in rows)/st.median(r[3] for r in rows)))
print("identity lean==c1: %d/20, c2==c1: %d/20"%(sum(r[2] for r in rows),sum(r[1] for r in rows)))
PY
