#!/usr/bin/env bash
# usage: bench/kq-decode-check.sh MAIN_ENGINE NEW_ENGINE KQ_PACK OUTDIR
# Run 9 / run 10 decode check (bench/llama-stint-protocol.md): the kq batched prefill leaves decode untouched. One gpu-wait
# job, the 20 mtp prompts, three arms interleaved per prompt: main with BARO_PREFILL=0 (replays the prompt; since run 9 main
# prefills kq packs too, so the knob is explicit), new with BARO_PREFILL=0 (same path, ids must equal main 20/20, tok/s_gen
# median within 1%), new with prefill on (reported: ids equal count and tok/s_gen median; prompts under 17 tokens replay
# anyway). Read-back per log: TMAX, pack kq trunk, prefill rows.
set -euo pipefail
cd "$(dirname "$0")/.."
main=$1 new=$2 pack=$3 out=$4
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 1500 -- "$0" "$@"
fi
mkdir -p "$out"
echo "main=$main sha=$(sha256sum "$main" | cut -c1-16) new=$new sha=$(sha256sum "$new" | cut -c1-16) pack=$pack load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
run() { # engine prompt log pfrows [env...]
  local e=$1 tf=$2 log=$3 want=$4; shift 4
  env "$@" BARO_PACK="$pack" BARO_SPEC=0 BARO_TMAX=16384 BARO_PROMPT="$tf" "$e" > "$log" 2>&1 || { echo "FAIL run $log: engine exit"; tail -3 "$log"; exit 1; }
  grep -q 'TMAX: 16384 ' "$log" || { echo "FAIL run $log: TMAX read-back"; exit 1; }
  grep -q 'pack kq trunk: True' "$log" || { echo "FAIL run $log: not a kq trunk"; exit 1; }
  grep -q '^GENERATED' "$log" || { echo "FAIL run $log: no GENERATED line"; exit 1; }
  local rows; rows=$(grep -oP ' prefill rows: \K[0-9]+' "$log" | head -1)
  [ "$want" = any ] || [ "$rows" = "$want" ] || { echo "FAIL run $log: prefill rows $rows, want $want"; exit 1; }
}
: > "$out/ids.txt"; : > "$out/ts.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens); n=$(wc -w < "$tf")
  run "$main" "$tf" "$out/$p.main.log" 0 BARO_PREFILL=0
  run "$new" "$tf" "$out/$p.replay.log" 0 BARO_PREFILL=0
  want=any; [ "$((n - 1))" -lt 16 ] && want=0
  run "$new" "$tf" "$out/$p.prefill.log" "$want" BARO_NOOP=1
  for arm in main replay prefill; do
    echo "$p $arm $(grep -oE 'tok/s_gen: [0-9.]+' "$out/$p.$arm.log" | cut -d' ' -f2)" >> "$out/ts.txt"
    echo "$p $arm $(grep '^GENERATED' "$out/$p.$arm.log" | md5sum | cut -c1-12)" >> "$out/ids.txt"
  done
done
python3 - "$out" <<'PY'
import sys, statistics as st
out = sys.argv[1]
ts, ids = {}, {}
for l in open(f"{out}/ts.txt"): p, a, v = l.split(); ts.setdefault(a, {})[p] = float(v)
for l in open(f"{out}/ids.txt"): p, a, h = l.split(); ids.setdefault(a, {})[p] = h
n = len(ts["main"])
same_replay = sum(ids["replay"][p] == ids["main"][p] for p in ids["main"])
same_pf = sum(ids["prefill"][p] == ids["main"][p] for p in ids["main"])
med = {a: st.median(v.values()) for a, v in ts.items()}
ratio = med["replay"] / med["main"]
print(f"main    median {med['main']:.2f} tok/s_gen n={n}")
print(f"replay  median {med['replay']:.2f} tok/s_gen  ratio {ratio:.4f}  ids equal main {same_replay}/{n}")
print(f"prefill median {med['prefill']:.2f} tok/s_gen  ratio {med['prefill'] / med['main']:.4f}  ids equal main {same_pf}/{n} (reported, not gated)")
ok = same_replay == n and abs(ratio - 1) <= 0.01
print(("PASS" if ok else "FAIL") + f": decode unchanged (bar: replay ids 20/20 equal main, median within 1%)")
sys.exit(0 if ok else 1)
PY
