#!/usr/bin/env bash
# Ornith-1.5-9B end-to-end per bench/ornith-protocol.md: G3 kernel
# self-consistency, step 3a teacher-forced agreement vs llama.cpp, 3b
# 20-prompt tok/s median, 3c MTP identity, 3d chat smoke through the Rust
# front. GPU-exclusive arms alternate inside this script (P1); the caller
# wraps the whole thing in `gpu-wait run -- bench/ornith-run.sh`.
set -uo pipefail
cd "$(dirname "$0")/.."
GGUF=${ORNITH_GGUF:-$HOME/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf}
PACK=.work/engine-pack-ornith-q8
OUT=${1:-.work/ornith-run}
TOKENIZER=.work/engine-pack-q4/tokenizer.json
mkdir -p "$OUT"
ok() { echo "PASS $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }

[ -d "$PACK" ] || die setup "no $PACK -- run tools/engine-pack.py ... --q8 first"

echo "== build engine =="
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine --target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}" > "$OUT/build-engine.log" 2>&1 \
  || die build "$(grep -m1 error: "$OUT/build-engine.log" | cut -c1-200)"
ok build "engine"

echo "== G3: kernel self-consistency, ours vs ours/numpy, p01-water =="
BARO_PACK=$PACK BARO_SPEC=0 BARO_PROMPT=bench/mtp-prompts/p01-water.tokens ./.work/engine \
  > "$OUT/g3-ours.log" 2>&1 || die g3 "engine exit $?"
cp bench/mtp-prompts/p01-water.tokens "$PACK/prompt-tokens.txt"
BARO_PACK=$PACK ./.venv/bin/python tools/model-ref.py decode 64 > "$OUT/g3-numpy.log" 2>&1 \
  || die g3 "model-ref.py exit $?"
g3_ours=$(grep -oE '^GENERATED:.*' "$OUT/g3-ours.log")
g3_numpy_ids=$(grep -oE '^GENERATED:.*' "$OUT/g3-numpy.log" | tr -d '[],')
echo "ours:  $g3_ours" > "$OUT/g3-compare.txt"
echo "numpy: $g3_numpy_ids" >> "$OUT/g3-compare.txt"
ours_ids=$(echo "$g3_ours" | sed 's/GENERATED://')
python3 -c "
a='''$ours_ids'''.split()
b='''$g3_numpy_ids'''.split()[1:]  # drop the leading 'GENERATED:' token
n=min(len(a),len(b))
agree=sum(1 for i in range(n) if a[i]==b[i])
first_bad=next((i for i in range(n) if a[i]!=b[i]), n)
print(f'G3: {agree}/{n} agree, first divergence at position {first_bad}')
" | tee -a "$OUT/g3-compare.txt"
ok g3 "$(tail -1 "$OUT/g3-compare.txt")"

echo "== llama.cpp: start server on Ornith GGUF (native K-quant) =="
~/llama.cpp/build/bin/llama-server -m "$GGUF" -c 8192 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
  -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8098 > "$OUT/llama-server.log" 2>&1 &
LPID=$!
cleanup_llama() { kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; }
trap cleanup_llama EXIT
up=0
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8098/health >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$LPID" 2>/dev/null || die llama "server exited early: $(tail -5 "$OUT/llama-server.log")"
  sleep 1
done
[ "$up" = 1 ] || die llama "server never came up"
curl -s http://127.0.0.1:8098/props > "$OUT/llama-props.json"
ok llama "up, props $(cat "$OUT/llama-props.json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("default_generation_settings",{}).get("n_ctx"))' 2>/dev/null)"

echo "== step 3a + 3b: per-prompt llama greedy, ours no-spec (tok/s), ours forced (agreement) =="
echo "prompt n_prompt tok_s_gen forced_agreement" > "$OUT/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens)
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
  resp=$(curl -s http://127.0.0.1:8098/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\": [$ids], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}")
  echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(str(t) for t in d["tokens"]))' \
    > "$OUT/$p.llama-tokens.txt" || die 3a "$p: bad llama /completion response: $resp"

  BARO_PACK=$PACK BARO_SPEC=0 BARO_PROMPT="$tf" ./.work/engine > "$OUT/$p.ours-nospec.log" 2>&1 \
    || die 3b "$p: engine exit $?"
  ts=$(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/$p.ours-nospec.log" | cut -d' ' -f2)

  BARO_PACK=$PACK BARO_SPEC=0 BARO_PROMPT="$tf" BARO_FORCE="$OUT/$p.llama-tokens.txt" ./.work/engine \
    > "$OUT/$p.ours-forced.log" 2>&1 || die 3a "$p: forced engine exit $?"
  agree=$(grep '^forced agreement:' "$OUT/$p.ours-forced.log" | sed 's/forced agreement: //; s/ \/ /\//')

  echo "$p $(wc -w < "$tf") $ts $agree" >> "$OUT/results.txt"
done
cleanup_llama; trap - EXIT
ok 3ab "$(cat "$OUT/results.txt")"

echo "== step 3c: MTP identical to no-spec, 20 prompts, k=2 =="
BARO_PACK=$PACK BARO_SPEC=0 bench/mtp-prompts.sh .work/engine "$OUT/mtp" 2 > "$OUT/mtp-run.log" 2>&1 \
  || die 3c "mtp-prompts.sh exit $?"
n_pass=$(grep -c ' PASS$' "$OUT/mtp/results.txt" || true)
n_total=$(ls bench/mtp-prompts/p*.tokens | wc -l)
ok 3c "$n_pass/$n_total identity PASS ($OUT/mtp/results.txt)"

echo "== step 3d: chat smoke through the Rust front =="
(cd serve && cargo build --release) > "$OUT/cargo-build.log" 2>&1 || die 3d "cargo build: $(grep -m1 error "$OUT/cargo-build.log")"
BARO_PACK=$PACK ./serve/target/release/baro-serve --engine .work/engine --pack "$PACK" \
  --tokenizer "$TOKENIZER" --port 0 > "$OUT/rust-server.stdout" 2> "$OUT/rust-server.stderr" &
RPID=$!
cleanup_rust() { kill -9 "$RPID" 2>/dev/null; pkill -9 -P "$RPID" 2>/dev/null; }
trap cleanup_rust EXIT
for _ in $(seq 1 120); do grep -q '^listening on' "$OUT/rust-server.stdout" && break; kill -0 "$RPID" 2>/dev/null || die 3d "server exited: $(tail -5 "$OUT/rust-server.stderr")"; sleep 0.5; done
url=$(grep -m1 -oE 'http://[0-9.:]+' "$OUT/rust-server.stdout") || die 3d "no listening line"
curl -s "$url/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"ornith","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"max_tokens":64}' \
  > "$OUT/chat-smoke.json"
cleanup_rust; trap - EXIT
python3 -c "
import json
d = json.load(open('$OUT/chat-smoke.json'))
content = d['choices'][0]['message']['content']
assert content.strip(), 'empty content'
print('3d content:', content)
" | tee -a "$OUT/SUMMARY.txt" || die 3d "$(cat "$OUT/chat-smoke.json")"
ok 3d "response received, see $OUT/chat-smoke.json"

echo "== done, see $OUT/SUMMARY.txt =="
cat "$OUT/SUMMARY.txt"
