#!/usr/bin/env bash
# usage: bench/a2-bar-llama.sh OUTDIR [KV types to force, default "q8_0 f16"]
#   QUICK=N  first N prompts of the 32k set only (smoke); default: 20 prompts x 3 sets
# A2 step 2 bar (bench/PROTOCOL-RULES.md P14): what a known-good lossy KV reaches in llama.cpp
# against its own f32 KV on bench/a2-prompts.sh's sets. tools/llama-force.cpp generates f32 greedy
# ids, then force-feeds them to f32 again (P11 self-check: must be 100%, proves the seq copy and
# the logit indexing) and to each lossy type. Model: the Q4_0-pure GGUF quant-matched to our pack.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1; types=${2:-"q8_0 f16"}; quick=${QUICK:-0}
model=${LLAMA_MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
bin=.work/a2s2/llama-force
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 12 --timeout 2400 -- env QUICK="$quick" LLAMA_MODEL="$model" "$0" "$@"
fi
mkdir -p "$out" .work/a2s2
[ -d .work/a2-prompts/L32768 ] || bench/a2-prompts.sh
if [ ! -x "$bin" ] || [ tools/llama-force.cpp -nt "$bin" ]; then
  g++ -O2 -std=c++17 tools/llama-force.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include \
    -L ~/llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,"$HOME/llama.cpp/build/bin" -o "$bin" \
    || { echo "FAIL build: tools/llama-force.cpp"; exit 1; }
fi
{
  echo "model=$model modelsha=$(sha256sum "$model" | cut -c1-16) llama=$(git -C ~/llama.cpp rev-parse --short HEAD) tool=$(sha256sum "$bin" | cut -c1-16) quick=$quick types='$types'"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"
q=(); [ "$quick" -gt 0 ] && q=("$quick")
run() {  # $1 kv  $2 mode  $3 log
  "$bin" "$model" .work/a2-prompts "$1" "$2" "$out/ref-f32.jsonl" "${q[@]}" > "$out/$3" 2> "$out/$3.err" \
    || { echo "FAIL $1 $2: see $out/$3.err"; tail -3 "$out/$3.err"; exit 1; }
  grep -m1 LLAMA_FORCE "$out/$3"
}
run f32 gen gen-f32.log
[ "$(wc -l < "$out/ref-f32.jsonl")" -gt 0 ] || { echo "FAIL gen: no ids in $out/ref-f32.jsonl"; exit 1; }
run f32 force force-f32.log
grep -q "TOTAL forced agreement: .* = 100.00%" "$out/force-f32.log" \
  || { echo "FAIL self-check: f32 forced against its own greedy ids is not 100%"; grep TOTAL "$out/force-f32.log"; exit 1; }
echo "self-check f32 vs f32: $(grep TOTAL "$out/force-f32.log")"
for t in $types; do
  run "$t" force "force-$t.log"
  python3 - "$out/force-$t.log" "$t" <<'PY'
import re, sys, statistics as st, collections
rows = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r"(L\d+)/(\S+) forced agreement: (\d+) / (\d+)", l)
    if m: rows[m[1]].append((m[2], int(m[3]), int(m[4])))
for s, r in rows.items():
    p = [100.0 * a / n for _, a, n in r]
    low = [f"{nm}={a}" for nm, a, n in r if a < n]
    print(f"{sys.argv[2]} {s}: {len(r)} prompts, min {min(p):.1f}% mean {st.mean(p):.2f}% at 100%: {sum(1 for x in p if x == 100)}/{len(r)} below: {' '.join(low) or '-'}")
PY
done
