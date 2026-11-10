#!/usr/bin/env bash
# Same-stint decode bar against llama.cpp (mojo-baro bench/llama-stint.sh method), one gpu-wait job:
# wavegroup on every tests/prompts/*.ids (one process each), then llama-server on the same ids, then wavegroup again
# on the first 5 as the drift control. 64 greedy tokens per prompt both sides; llama.cpp in its fast config (-fa on,
# KV f16), ignore_eos, one untimed request first; tok/s = timings.predicted_per_second. Ours = decode_tok_s.
# Also compares the 64 generated ids per prompt with llama.cpp (matching prefix length).
# ALT=ENGINE adds a second engine arm with decode's CLI (e.g. tools/baro_arm.sh), run right after ours.
# usage: tools/stint.sh DECODE_BIN OUTDIR [LLAMA_SERVER]
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 out=$2 srv=${3:-$HOME/Projects/imports/llama.cpp/build/bin/llama-server}
model=${GGUF:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --timeout 1800 -- "$0" "$@"
fi
mkdir -p "$out"
echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) server=$srv sha=$(sha256sum "$srv" | cut -c1-16) model=$model load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"

ours() { # label files...
  local l=$1 eng=${ENG:-$eng}; shift
  for f in "$@"; do
    p=$(basename "$f" .ids)
    "$eng" "$model" "$(cat "$f")" 64 > "$out/$p.$l.log" 2>&1 || { echo "FAIL $l $p: exit $?, $out/$p.$l.log"; exit 1; }
    t=$(grep -oE 'decode_tok_s [0-9.]+' "$out/$p.$l.log" | cut -d' ' -f2 || true)
    [ -n "$t" ] || { echo "FAIL $l $p: no decode_tok_s"; exit 1; }
    echo "$p $t" >> "$out/$l.txt"
  done
}

: > "$out/ours.txt"; : > "$out/drift.txt"; : > "$out/alt.txt"
ours ours tests/prompts/*.ids
[ -z "${ALT:-}" ] || ENG=$ALT ours alt tests/prompts/*.ids
port=$((20000 + RANDOM % 20000))
"$srv" -m "$model" -c 16384 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -np 1 --no-webui --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for _ in $(seq 240); do curl -sf "localhost:$port/health" > /dev/null && break; sleep 0.5; done
curl -sf "localhost:$port/health" > /dev/null || { echo "FAIL llama: server not healthy, $out/server.log"; exit 1; }
req() { curl -sf "localhost:$port/completion" -d "{\"prompt\": [$1], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"ignore_eos\": true, \"return_tokens\": true}"; }
req "$(cat tests/prompts/p01.ids)" > /dev/null
: > "$out/llama.txt"
for f in tests/prompts/*.ids; do
  r=$(req "$(cat "$f")") || { echo "FAIL llama $(basename "$f")"; exit 1; }
  [ "$(echo "$r" | jq .timings.predicted_n)" = 64 ] || { echo "FAIL llama $(basename "$f"): not 64 tokens"; exit 1; }
  echo "$(basename "$f" .ids) $(echo "$r" | jq .timings.predicted_per_second)" >> "$out/llama.txt"
  echo "$r" | jq -r '.tokens | map(tostring) | join(",")' > "$out/$(basename "$f" .ids).llama.ids"
done
kill $pid; wait $pid 2>/dev/null || true
ours drift $(ls tests/prompts/*.ids | head -5)
python3 - "$out" <<'PY'
import sys, statistics as st
d = sys.argv[1]
rd = lambda n: {l.split()[0]: float(l.split()[1]) for l in open(f"{d}/{n}.txt")}
o, l, dr = rd("ours"), rd("llama"), rd("drift")
r = [o[k] / l[k] for k in o]
print(f"ours median {st.median(o.values()):.1f} tok/s, llama.cpp median {st.median(l.values()):.1f}, ratio median {st.median(r):.3f} "
      f"(min {min(r):.3f}, max {max(r):.3f}), faster {sum(x > 1 for x in r)}/{len(r)}, drift {st.median(dr[k] / o[k] for k in dr):.4f}")
same = []
for k in sorted(o):
    a = open(f"{d}/{k}.ours.log").readline().strip().split(",")
    b = open(f"{d}/{k}.llama.ids").read().strip().split(",")
    same.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
if open(f"{d}/alt.txt").read().strip():
    a = rd("alt")
    print(f"alt median {st.median(a.values()):.1f} tok/s, ours/alt ratio median {st.median(o[k] / a[k] for k in a):.3f}, alt/llama.cpp {st.median(a[k] / l[k] for k in a):.3f}")
print(f"ids equal to llama.cpp: {sum(n == 64 for n in same)}/{len(same)} prompts all 64; matching prefix per prompt {same}")
if open(f"{d}/alt.txt").read().strip():
    pre = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    ids = lambda k, l: open(f"{d}/{k}.{l}.log").readline().strip().split(",")
    al = [pre(ids(k, "alt"), open(f"{d}/{k}.llama.ids").read().strip().split(",")) for k in sorted(o)]
    ao = [pre(ids(k, "alt"), ids(k, "ours")) for k in sorted(o)]
    print(f"alt ids equal to llama.cpp: {sum(n == 64 for n in al)}/{len(al)} all 64; alt ids equal to ours: {sum(n == 64 for n in ao)}/{len(ao)} all 64, prefix {ao}")
PY
