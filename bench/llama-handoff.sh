#!/usr/bin/env bash
# usage: bench/llama-handoff.sh ENGINE PACK P_TOKENS PQ_TOKENS OUTDIR
#
# LatentOS use 1 gate (bench/llama-handoff-protocol.md): llama.cpp prefills P
# and saves its slot, tools/llama-slot-to-state converts it, our engine loads it
# and decodes P+Q. Compared against our own state at the same position
# (tools/state_diff.mojo) and our own cold continuation (BARO_FORCE agreement).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1 pack=$2 p=$3 pq=$4 out=$5
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 24 --timeout 3600 -- "$0" "$@"
fi
mkdir -p "$out/slots"
model=${LH_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
server=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
tmax=8448
np=$(wc -w < "$p")

# 1. llama.cpp prefill + slot save
port=$((20000 + RANDOM % 20000))
"$server" -m "$model" -ngl 99 -fa on -np 1 -c 9216 --port "$port" --no-webui \
  --slot-save-path "$out/slots" > "$out/server.log" 2>&1 &
srv=$!
trap 'kill "$srv" 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 0.5; done
ids=$(tr -s ' \n' ',' < "$p" | sed 's/^,//; s/,$//')
printf '{"prompt":[%s],"n_predict":1,"temperature":0,"cache_prompt":true}' "$ids" > "$out/req.json"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req.json" > "$out/completion.json"
curl -sf -X POST "localhost:$port/slots/0?action=save" -H 'content-type: application/json' \
  -d '{"filename":"p.slot"}' > "$out/save.json"
kill "$srv"; wait "$srv" 2>/dev/null || true; trap - EXIT

# 2. convert
t0=$(date +%s.%N)
.work/llama-slot-to-state "$out/slots/p.slot" "$pack" "$out/llama.state" > "$out/convert.log" 2>&1
t1=$(date +%s.%N)

# 3. our own state at the same position: prompt P + first token of Q
head -c 0 /dev/null
p1="$out/p1.tokens"
{ cat "$p"; printf ' %s' "$(awk '{print $('"$np"'+1)}' "$pq")"; } > "$p1"
env BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_PROMPT="$p1" BARO_STATE_SAVE="$out/ours.state" "$eng" > "$out/ours-p1.log" 2>&1

# 4. our cold continuation of P+Q
env BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_PROMPT="$pq" "$eng" > "$out/cold-pq.log" 2>&1
grep '^GENERATED' "$out/cold-pq.log" | sed 's/^GENERATED: *//' > "$out/cold-pq.gen"

# 5. load the converted state: greedy, then teacher-forced on step 4's tokens
env BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_PROMPT="$pq" BARO_STATE_LOAD="$out/llama.state" "$eng" > "$out/load-pq.log" 2>&1
env BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_PROMPT="$pq" BARO_STATE_LOAD="$out/llama.state" \
  BARO_SPEC=0 BARO_FORCE="$out/cold-pq.gen" "$eng" > "$out/force-pq.log" 2>&1

grep -q "cached: $np " "$out/load-pq.log" || { echo "VOID: the loaded state was not reused (prefix lookup missed)"; grep -hE 'cached: [0-9]+' "$out/load-pq.log"; exit 3; }

# 6. layout check
echo "./.venv/bin/mojo build tools/state_diff.mojo -o .work/state_diff" | tools/build-many.sh state-diff > "$out/state-diff-build.log" 2>&1 || { echo "FAIL build state_diff: see $out/state-diff-build.log"; exit 1; }
.work/state_diff "$out/ours.state" "$out/llama.state" > "$out/state-diff.txt" 2>&1 || true

echo "== receipts"
python3 -c "import json;t=json.load(open('$out/completion.json'))['timings'];print('llama prompt_n',t.get('prompt_n'),'prompt_ms',round(t.get('prompt_ms',0),1))"
python3 -c "import json;print('slot save',json.load(open('$out/save.json')))"
echo "convert_s $(python3 -c "print(round($t1 - $t0, 3))")"; cat "$out/convert.log"
grep -hE 'state loaded|cached: [0-9]+' "$out/load-pq.log" | head -3
grep -hE 'prefill_s|tok/s_gen' "$out/cold-pq.log" "$out/load-pq.log" | head -4
[ "$(grep '^GENERATED' "$out/cold-pq.log")" = "$(grep '^GENERATED' "$out/load-pq.log")" ] && echo "greedy identical to cold" || echo "greedy differs from cold"
grep -hiE 'agree' "$out/force-pq.log" | tail -2
cat "$out/state-diff.txt"
