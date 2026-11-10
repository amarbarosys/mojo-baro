#!/usr/bin/env bash
# Gate for `tools/baro serve` (briefs/2026-09-16-baro-serve-cache-lane.md):
# per model, build (CPU, outside the GPU queue), then inside one gpu-wait job
# start `baro serve` twice (first start = whatever the build left, second =
# warm) and send one real /v1/completions request by token ids at T=0 each
# time. PASS = the generated ids equal the file's own baro.run.ref.tokens and
# the second start's receipt says engine=hit pack=hit.
#
#   tools/baro-serve-gate.sh OUT_DIR MODEL.gguf [MODEL.gguf ...]
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(readlink -f "$1"); shift
mkdir -p "$out"
py=$here/.venv/bin/python3
port=${GATE_PORT:-8097}
min_free_gb=${GATE_MIN_FREE_GB:-45}
fails=0

for model in "$@"; do
  name=$(basename "${model%.gguf}")
  d=$out/$name; rm -rf "$d"; mkdir -p "$d"
  "$py" "$here/tools/gguf-extract.py" "$model" --meta > "$d/meta.json" || { echo "FAIL $name: meta"; fails=$((fails+1)); continue; }
  jq -r '.["baro.run.prompt.tokens"] // empty' "$d/meta.json" > "$d/prompt.txt"
  jq -r '.["baro.run.ref.tokens"] // empty' "$d/meta.json" | tr ' ' '\n' | grep . > "$d/ref.txt"
  [ -s "$d/prompt.txt" ] && [ -s "$d/ref.txt" ] || { echo "FAIL $name: file carries no baro.run.prompt/ref.tokens"; fails=$((fails+1)); continue; }

  free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge "$min_free_gb" ] || { echo "STOP $name: /home free ${free_gb}G < ${min_free_gb}G"; exit 2; }
  b0=$(date +%s)
  "$here/tools/baro" serve "$model" --no-serve > "$d/build.log" 2>&1 || { echo "FAIL $name: build, see $d/build.log"; fails=$((fails+1)); continue; }
  echo "$name build $(( $(date +%s) - b0 ))s: $(grep '^receipt:' "$d/build.log")"

  nref=$(wc -l < "$d/ref.txt")
  prompt_json="[$(tr ' ' ',' < "$d/prompt.txt")]"
  gpu-wait run --vram 22 --priority 20 --timeout 1800 -- bash -c '
    here=$1; model=$2; d=$3; port=$4; prompt_json=$5; nref=$6
    for run in 1 2; do
      "$here/tools/baro" serve "$model" --port "$port" > "$d/serve$run.log" 2>&1 &
      pid=$!
      for _ in $(seq 600); do
        curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
        kill -0 $pid 2>/dev/null || break
        sleep 1
      done
      curl -sf "http://127.0.0.1:$port/v1/completions" -H "content-type: application/json" \
        -d "{\"prompt\": $prompt_json, \"max_tokens\": $nref, \"temperature\": 0, \"spec\": false}" > "$d/resp$run.json"
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; wait $pid 2>/dev/null
      sleep 2
    done
  ' _ "$here" "$model" "$d" "$port" "$prompt_json" "$nref" > "$d/gpu.log" 2>&1

  verdict=PASS; detail=""
  for run in 1 2; do
    jq -r '.choices[0].tokens[]' "$d/resp$run.json" > "$d/gen$run.txt" 2>/dev/null
    if ! cmp -s "$d/gen$run.txt" "$d/ref.txt"; then
      verdict=FAIL
      detail+=" run$run: $(diff <(nl -ba "$d/ref.txt") <(nl -ba "$d/gen$run.txt") | grep -m1 '^[<>]' || echo 'no response')"
    fi
  done
  grep -q 'engine=hit pack=hit' "$d/serve2.log" || { verdict=FAIL; detail+=" warm: $(grep '^receipt:' "$d/serve2.log" || echo 'no receipt')"; }
  [ "$verdict" = PASS ] || fails=$((fails+1))
  echo "$verdict $name: $(wc -l < "$d/gen1.txt")/$nref and $(wc -l < "$d/gen2.txt")/$nref tokens; warm $(grep -o 'engine=[a-z]* pack=[a-z]* elapsed_s=[0-9]*' "$d/serve2.log")$detail"
done
[ "$fails" = 0 ]
