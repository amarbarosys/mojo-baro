#!/usr/bin/env bash
# One resident engine, all 20 prompts. Replaces 20 process starts + 20 pack loads.
set -uo pipefail
L=$HOME/Projects/mojo/mojo-baro-lanes/MOE
cd "$L"
REF=.work/moe-w3/gate2-force-m1prefill
ENG=${1:-$HOME/Projects/mojo/mojo-baro/.work/engine-res}
OUT=${2:-.work/resgate}
mkdir -p "$OUT"
: > "$OUT/req.jsonl"
id=0
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); id=$((id+1))
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
  fids=$(tr -s ' \n' ',,' < "$REF/$p.ref.ids" | sed 's/,$//')
  echo "{\"id\":$id,\"prompt\":[$ids],\"n\":64,\"spec\":false,\"force\":[$fids]}" >> "$OUT/req.jsonl"
done
echo "engine $(sha256sum "$ENG" | cut -c1-16), 20 requests, ONE process"
t0=$(date +%s.%N)
env BARO_SERVE=1 BARO_MEGA=0 BARO_PREFILL=1 BARO_PACK=.work/moe-w1/pack "$ENG" \
    < "$OUT/req.jsonl" > "$OUT/out.log" 2>&1
t1=$(date +%s.%N)
grep -oE 'forced agreement: [0-9]+ / 64' "$OUT/out.log" | grep -oE '^forced agreement: [0-9]+' | grep -oE '[0-9]+$' > "$OUT/scores.txt"
n=$(grep -c . "$OUT/scores.txt")
python3 -c "
v=[int(x) for x in open('$OUT/scores.txt')]
print('prompts',len(v),'mean',round(sum(v)/len(v),2) if v else 0)
print(v)
"
echo "wall $(python3 -c "print(f'{$t1-$t0:.1f}s')")  (pack loaded once)"
