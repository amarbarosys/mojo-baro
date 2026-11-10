#!/usr/bin/env bash
# bench/e14-llama-run.sh: E14's llama.cpp arm, same shape as arm 2 (Text):
# each follower posts its own full document+question prompt from scratch,
# cache_prompt:false, no state shared between followers.
#
# Adapted from tools/llama-ref-run.sh, which posts ONE fixed prompt file N
# times (repeats for a median); E14 needs N DIFFERENT prompts (one per
# follower), so this loops bench/e14-llama-prep.py's per-follower prompt
# files instead of repeating one. Same llama-server flags as
# tools/llama-ref-run.sh / docs/prefill-long-ctx-2026-09-11.md's llama.cpp
# arm (-ngl 99 -fa on -b 2048 -ub 512 -ctk q8_0 -ctv q8_0), context raised to
# fit a 32k document plus a short question and the answer budget.
#
# usage: bench/e14-llama-run.sh MODEL.gguf PROMPT_DIR OUT_DIR [PORT] [N_PREDICT]
set -euo pipefail
model=$1; prompt_dir=$2; out=$3; port=${4:-8098}; n_predict=${5:-64}
mkdir -p "$out"

~/llama.cpp/build/bin/llama-server -m "$model" -c 33280 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
  --host 127.0.0.1 --port "$port" -ctk q8_0 -ctv q8_0 > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
curl -s "http://127.0.0.1:$port/props" > "$out/props.json"

: > "$out/timings.jsonl"
for f in "$prompt_dir"/prompt-*.txt; do
  id=$(basename "$f" .txt)
  # A 32k-token prompt as an inline curl -d argument overflows the shell's
  # arg-length limit (measured: "Argument list too long" at this size, fine
  # at tools/llama-ref-run.sh's 8k). Build the JSON body to a file instead.
  payload="$out/req-$id.json"
  python3 -c "
import json, sys
ids = [int(x) for x in open(sys.argv[1]).read().split()]
json.dump({'prompt': ids, 'n_predict': int(sys.argv[2]), 'temperature': 0, 'top_k': 1,
           'cache_prompt': False, 'return_tokens': True}, open(sys.argv[3], 'w'))
" "$f" "$n_predict" "$payload"
  t0=$(date +%s.%N)
  resp=$(curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' --data "@$payload")
  t1=$(date +%s.%N)
  wall_s=$(python3 -c "print($t1 - $t0)")
  echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
print(json.dumps({'id': '$id', 'wall_s': $wall_s, 'predicted_n': t.get('predicted_n'),
                   'predicted_per_second': t.get('predicted_per_second'),
                   'prompt_n': t.get('prompt_n'), 'prompt_ms': t.get('prompt_ms')}))
" >> "$out/timings.jsonl"
  echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(' '.join(str(t) for t in d.get('tokens', [])))
" > "$out/ans-$id.txt"
done
cat "$out/timings.jsonl"
