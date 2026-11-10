#!/usr/bin/env bash
# Teacher-forced agreement of an engine build against llama.cpp (ornith-run.sh step
# 3a, any engine/pack/GGUF): llama.cpp greedy N tokens per prompt, then our engine
# forced along llama's tokens (BARO_FORCE) reports how many of its own argmaxes agree.
# Run inside one gpu-wait job: llama-server and the engine alternate on the GPU.
# usage: bench/llama-agree.sh ENGINE PACK GGUF OUTDIR [N_PROMPTS] [N_TOKENS]
set -euo pipefail
cd "$(dirname "$0")/.."
usage="usage: bench/llama-agree.sh ENGINE PACK GGUF OUTDIR [N_PROMPTS] [N_TOKENS]"
eng=$(realpath "${1:?$usage}"); pack=$(realpath "${2:?$usage}"); gguf=$(realpath "${3:?$usage}")
out=$(realpath -m "${4:?$usage}"); np=${5:-5}; nt=${6:-64}; port=${LLAMA_PORT:-8099}
. bench/min16k.sh; min16k "${BARO_TMAX:-16384}" 16000
llama=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
mkdir -p "$out"
fail() { echo "FAIL $1: $2"; exit 1; }
for f in "$eng" "$llama"; do [ -x "$f" ] || fail setup "not executable: $f"; done
echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) pack=$pack gguf=$gguf prompts=$np tokens=$nt job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"

"$llama" -m "$gguf" -c 16384 -ngl 99 -fa on -t 8 --host 127.0.0.1 --port "$port" > "$out/llama-server.log" 2>&1 &
lpid=$!
trap 'kill "$lpid" 2> /dev/null || true; wait "$lpid" 2> /dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$port/health" > /dev/null || fail llama "never healthy: $out/llama-server.log"

for tf in $(ls bench/mtp-prompts/p*.tokens | head -n "$np"); do
  p=$(basename "$tf" .tokens)
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
  curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\": [$ids], \"n_predict\": $nt, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}" \
    | python3 -c 'import sys,json; print(" ".join(str(t) for t in json.load(sys.stdin)["tokens"]))' > "$out/$p.llama.txt" \
    || fail llama "$p: bad /completion response"
done
kill "$lpid" 2> /dev/null; wait "$lpid" 2> /dev/null || true
for _ in $(seq 60); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 1500000000 ] && break; sleep 1; done

echo "prompt agree" > "$out/results.txt"
for tf in $(ls bench/mtp-prompts/p*.tokens | head -n "$np"); do
  p=$(basename "$tf" .tokens)
  BARO_TMAX=${BARO_TMAX:-16384} BARO_PACK=$pack BARO_SPEC=${AGREE_SPEC:-0} BARO_PROMPT=$tf BARO_FORCE=$out/$p.llama.txt "$eng" > "$out/$p.forced.log" 2>&1 \
    || fail engine "$p: exit $? ($out/$p.forced.log)"
  ! grep -q 'NOT-RESIDENT' "$out/$p.forced.log" || fail engine "$p: NOT-RESIDENT, tokens invalid ($out/$p.forced.log)"
  a=$(grep -m1 '^forced agreement:' "$out/$p.forced.log" | sed 's/forced agreement: //; s/ \/ /\//') \
    || fail engine "$p: no agreement line ($out/$p.forced.log)"
  echo "$p $a" | tee -a "$out/results.txt"
done
python3 - "$out/results.txt" <<'PY'
import statistics, sys
rows = [l.split()[1] for l in open(sys.argv[1]).read().split("\n")[1:] if l.strip()]
fr = [int(a) / int(b) for a, b in (r.split("/") for r in rows)]
print(f"median {statistics.median(fr):.4f} min {min(fr):.4f} over {len(fr)} prompts")
PY
