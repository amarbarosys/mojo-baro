#!/usr/bin/env bash
# usage: bench/mega-g-ab.sh OUTDIR PACK ENGINE_CONTROL ENGINE_ARM... (run under gpu-wait; env knobs go inside)
# Megakernel grid size A/B at m=1 (BARO_SPEC=0): arms interleaved per prompt, 3 rounds, clean GPU before every run,
# fail word read on every run, tok/s_gen and generated ids per arm; identity of each arm against the control.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1 pack=$2; shift 2; engs=("$@")
mkdir -p "$out"
prompts=${PROMPTS:-"p01-water p02-python-fib p05-math p19-numbers"}
for round in 1 2 3; do
  for n in $prompts; do
    for i in "${!engs[@]}"; do
      for _ in $(seq 60); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 1500000000 ] && break; sleep 1; done
      log="$out/${n}_a${i}_r${round}.log"
      env BARO_PACK=$pack BARO_PROMPT=bench/mtp-prompts/$n.tokens BARO_MEGA=1 BARO_SPEC=${SPEC:-0} BARO_EMBD_HOST=1 "${engs[$i]}" > "$log" 2>&1
      grep -q 'mega fail word: 0' "$log" || { echo "FAIL a$i $n: fail word"; exit 1; }
    done
  done
done
python3 - "$out" "${#engs[@]}" "$prompts" <<'PY'
import sys,glob,statistics as st
out,na,prompts=sys.argv[1],int(sys.argv[2]),sys.argv[3].split()
def f(p,k):
    return [l for l in open(p) if k in l][0]
for i in range(na):
    sp=[float(f(p,"tok/s_gen:").split("tok/s_gen:")[1]) for p in glob.glob(f"{out}/*_a{i}_r*.log")]
    same=sum(f(p,"GENERATED:")==f(p.replace(f"_a{i}_",f"_a0_"),"GENERATED:") for p in glob.glob(f"{out}/*_a{i}_r*.log"))
    print(f"arm {i}: median tok/s_gen {st.median(sp):.2f} min {min(sp):.2f} max {max(sp):.2f} n={len(sp)}  ids==control {same}/{len(sp)}")
PY
