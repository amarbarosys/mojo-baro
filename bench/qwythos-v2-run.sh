#!/usr/bin/env bash
# Qwythos-9B-v2-MTP-Q6_K vs llama.cpp per bench/qwythos-v2-protocol.md: step 3a
# teacher-forced agreement, step 3b 20-prompt tok/s median. Scope trimmed from
# bench/ornith-run.sh to 3a+3b only (G1-G3 and 3c/3d already proven on this
# code path and out of scope per the brief). GPU-exclusive arms alternate
# inside this script (P1); the caller wraps the whole thing in
# `gpu-wait run -- bench/qwythos-v2-run.sh`.
set -uo pipefail
cd "$(dirname "$0")/.."
GGUF=${QV2_GGUF:-$HOME/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf}
PACK=.work/qv2/engine-pack-q8
OUT=${1:-.work/qv2/run}
mkdir -p "$OUT"
ok() { echo "PASS $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }

echo "== pack (--q8, K-quant source) =="
if [ ! -d "$PACK" ]; then
  ./.venv/bin/python tools/engine-pack.py "$GGUF" "$PACK" --q8 > "$OUT/pack.log" 2>&1 \
    || die pack "$(tail -5 "$OUT/pack.log")"
fi
pack_tensors=$(grep -oE '^packed [0-9]+ tensors' "$OUT/pack.log" | grep -oE '[0-9]+')
ok pack "packed $pack_tensors tensors ($OUT/pack.log)"

echo "== build engine =="
./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine --target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}" > "$OUT/build-engine.log" 2>&1 \
  || die build "$(grep -m1 error: "$OUT/build-engine.log" | cut -c1-200)"
ok build "engine"

echo "== llama.cpp: start server on Qwythos-v2 GGUF (native K-quant) =="
~/llama.cpp/build/bin/llama-server -m "$GGUF" -c 8192 -ngl 99 -fa on -b 2048 -ub 512 -t 8 \
  -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8099 > "$OUT/llama-server.log" 2>&1 &
LPID=$!
cleanup_llama() { kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; }
trap cleanup_llama EXIT
up=0
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$LPID" 2>/dev/null || die llama "server exited early: $(tail -5 "$OUT/llama-server.log")"
  sleep 1
done
[ "$up" = 1 ] || die llama "server never came up"
curl -s http://127.0.0.1:8099/props > "$OUT/llama-props.json"
ok llama "up, props $(cat "$OUT/llama-props.json" | python3 -c 'import sys,json; d=json.load(sys.stdin); g=d.get("default_generation_settings",{}); print("n_ctx", g.get("n_ctx"))' 2>/dev/null)"
kv_read=$(python3 -c '
import sys, json
d = json.load(open("'"$OUT"'/llama-props.json"))
print(d.get("default_generation_settings", {}))
' 2>/dev/null)
echo "llama props (full): $kv_read" >> "$OUT/SUMMARY.txt"

echo "== step 3a + 3b: per-prompt llama greedy, ours no-spec (tok/s), ours forced (agreement) =="
echo "prompt n_prompt tok_s_gen forced_agreement" > "$OUT/results.txt"
for tf in bench/mtp-prompts/p*.tokens; do
  p=$(basename "$tf" .tokens)
  ids=$(tr -s ' \n' ',,' < "$tf" | sed 's/,$//')
  resp=$(curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\": [$ids], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}")
  echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(str(t) for t in d["tokens"]))' \
    > "$OUT/$p.llama-tokens.txt" || die 3a "$p: bad llama /completion response: $resp"
  prompt_n=$(echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("tokens_evaluated", d.get("prompt_n", "?")))' 2>/dev/null)
  n_expect=$(wc -w < "$tf")
  [ "$prompt_n" = "$n_expect" ] || echo "WARN $p: prompt_n=$prompt_n expected $n_expect" | tee -a "$OUT/SUMMARY.txt"

  BARO_PACK=$PACK BARO_SPEC=0 BARO_PROMPT="$tf" ./.work/engine > "$OUT/$p.ours-nospec.log" 2>&1 \
    || die 3b "$p: engine exit $?"
  ts=$(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/$p.ours-nospec.log" | cut -d' ' -f2)
  pt_n=$(grep -oE 'prompt tokens: [0-9]+' "$OUT/$p.ours-nospec.log" | head -1 | grep -oE '[0-9]+')

  BARO_PACK=$PACK BARO_SPEC=0 BARO_PROMPT="$tf" BARO_FORCE="$OUT/$p.llama-tokens.txt" ./.work/engine \
    > "$OUT/$p.ours-forced.log" 2>&1 || die 3a "$p: forced engine exit $?"
  agree=$(grep '^forced agreement:' "$OUT/$p.ours-forced.log" | sed 's/forced agreement: //; s/ \/ /\//')
  force_n=$(grep -oE 'BARO_FORCE:.*\( [0-9]+ ids' "$OUT/$p.ours-forced.log" | grep -oE '\( [0-9]+' | grep -oE '[0-9]+')

  echo "$p $(wc -w < "$tf") $ts $agree pack_tensors=$pack_tensors prompt_tokens=$pt_n force_ids=$force_n" >> "$OUT/results.txt"
done
cleanup_llama; trap - EXIT
ok 3ab "$(cat "$OUT/results.txt")"

echo "== done, see $OUT/SUMMARY.txt =="
cat "$OUT/SUMMARY.txt"
