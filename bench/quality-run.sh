#!/usr/bin/env bash
# One model's quality row (bench/quality-protocol.md, PROTOCOL-RULES P15-P19).
#   bench/quality-run.sh KEY            full row
#   QUICK=N bench/quality-run.sh KEY    first N task items, 2 PPL chunks (iteration only, never cited)
# Runs OUTSIDE gpu-wait: CPU steps (tokenizer server, prep, builds, scoring) hold no GPU; each GPU step is
# its own gpu-wait job. llama.cpp outputs go through $HOME/iTools/dev/refcache/refcache.sh (model sha + llama.cpp
# commit + inputs, P17; NOT the $HOME/iTools/bin/refcache shim, which now points to an incompatible rewrite with
# a different CLI); the ours arm always runs. Any failure prints FAIL <step> and exits non-zero (P16).
set -euo pipefail
export BARO_NEURONS=0
cd "$(dirname "$0")/.."
KEY=$1; QUICK=${QUICK:-0}
bench/preflight.sh --check
PY=$PWD/.venv/bin/python3; MOJO=$PWD/.venv/bin/mojo; LB=$HOME/llama.cpp/build/bin
ACCEL=(--target-accelerator "${BARO_TARGET_ACCELERATOR:-gfx1100}")
WIKI=$HOME/Models/quant-lab/wikitext-2-raw/wiki.test.raw
OUT=.work/quality/$KEY; rm -rf "$OUT"; mkdir -p "$OUT"
ok() { echo "OK $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }
trap 'rc=$?; [ $rc = 0 ] || echo "FAIL $KEY: exit $rc at line $LINENO, see $OUT" | tee -a "$OUT/SUMMARY.txt"' EXIT
field() { "$PY" -c "import json,sys; print(json.load(open('bench/quality-models.json'))['$KEY']['$1'])"; }
gpu() { local t=$1; shift; gpu-wait run --vram "$((vram + 2))" --priority 20 --timeout "$t" -- "$@"; }
baro_gguf=$(field baro_gguf); llama_gguf=$(field llama_gguf); engine=$(field engine); vram=$(field vram_gb)
llama_sha=$(git -C $HOME/llama.cpp rev-parse --short HEAD)
echo "KEY=$KEY engine=$engine quick=$QUICK llama_gguf=$llama_gguf llama.cpp=$llama_sha head=$(git rev-parse --short HEAD)" | tee "$OUT/arm.txt"

tools/baro serve "$baro_gguf" --no-serve > "$OUT/cache.log" 2>&1 || die cache "$OUT/cache.log"
cache_engine=$(awk '/^engine: (hit|miss)/{print $3}' "$OUT/cache.log")
cache_pack=$(awk '/^pack: (hit|miss)/{print $3}' "$OUT/cache.log")
[ -x "$cache_engine" ] && [ -s "$cache_pack/index.txt" ] || die cache "no engine/pack in $OUT/cache.log"
meta=$("$PY" tools/gguf-extract.py "$baro_gguf" --meta)
renv=$(jq -r '.["baro.run.env"] // empty' <<<"$meta")

TOKP=8198
"$LB/llama-server" -m "$llama_gguf" -ngl 0 -c 8192 -t 8 --host 127.0.0.1 --port $TOKP > "$OUT/tok-server.log" 2>&1 &
TOKPID=$!
trap 'rc=$?; kill $TOKPID 2>/dev/null || true; [ $rc = 0 ] || echo "FAIL $KEY: exit $rc, see $OUT" | tee -a "$OUT/SUMMARY.txt"' EXIT
for _ in $(seq 600); do curl -sf localhost:$TOKP/health >/dev/null 2>&1 && break; kill -0 $TOKPID || die tok "cpu server exited"; sleep 1; done
TOK=http://127.0.0.1:$TOKP

# Perplexity: every family now (spark.mojo has logprobs since 7c3a2ce). Engine built from HEAD on the cached pack.
chunks=$([ "$QUICK" -gt 0 ] && echo 2 || echo 8)
if [ "$engine" = spark ]; then
  mkdir -p "$OUT/prof"; .work/gen-profile "$baro_gguf" "$OUT/prof/profile.mojo" > "$OUT/build.log" 2>&1 || die build "$OUT/build.log"
  "$MOJO" build "${ACCEL[@]}" serve/spark.mojo -I . -I kernels -I serve -I "$OUT/prof" -o "$OUT/engine-head" >> "$OUT/build.log" 2>&1 || die build "$OUT/build.log"
else
  dflag=(); [ "$engine" = moe ] && dflag=(-D BARO_MODEL=qwen35moe)
  "$MOJO" build "${ACCEL[@]}" serve/engine.mojo -I . -I kernels "${dflag[@]}" -o "$OUT/engine-head" > "$OUT/build.log" 2>&1 || die build "$OUT/build.log"
fi
gpu 1800 env $renv "$PY" bench/quality-ppl-run.py --engine "$OUT/engine-head" --pack "$cache_pack" --text "$WIKI" --tok-url $TOK \
  --ref-prompt "$(jq -r '.["baro.run.prompt.tokens"]' <<<"$meta")" --ref-tokens "$(jq -r '.["baro.run.ref.tokens"]' <<<"$meta")" \
  --chunks "$chunks" --out "$OUT/ppl-ours" > "$OUT/ppl-ours.log" 2>&1 || die ppl-ours "$(tail -2 "$OUT/ppl-ours.log")"
ok ppl-ours "$(tail -1 "$OUT/ppl-ours.log")"
REFCACHE_DIR=$PWD/.work/refcache $HOME/iTools/dev/refcache/refcache.sh --key-file "$llama_gguf" --key-file "$WIKI" --key "llama.cpp=$llama_sha ppl c512 chunks$chunks" \
  --out "$OUT/ppl-llama.log" -- gpu-wait run --vram "$((vram + 2))" --priority 20 --timeout 900 -- bash -c '"$1" -m "$2" -f "$3" -c 512 --chunks "$4" -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 > "$5" 2>&1' \
  _ "$LB/llama-perplexity" "$llama_gguf" "$WIKI" "$chunks" "$OUT/ppl-llama.log" 2>&1 | tee -a "$OUT/SUMMARY.txt"
ok ppl-llama "$(grep 'Final estimate' "$OUT/ppl-llama.log")"

# Task eval: identical ids to both arms; prep and scoring on CPU.
"$PY" bench/quality-task-ids.py prep --tok-url $TOK --limit "$QUICK" --out "$OUT/task" > "$OUT/task.log" 2>&1 || die prep "$OUT/task.log"
ok prep "$(tail -1 "$OUT/task.log")"
spec=$([ "$engine" = dense ] && echo 1 || echo 0)
gpu 2400 "$PY" bench/quality-task-ids.py ours --engine "$cache_engine" --pack "$cache_pack" --env "$renv" --spec "$spec" --out "$OUT/task" >> "$OUT/task.log" 2>&1 || die ours "$OUT/task.log"
ok ours "$(tail -1 "$OUT/task.log")"
REFCACHE_DIR=$PWD/.work/refcache $HOME/iTools/dev/refcache/refcache.sh --key-file "$llama_gguf" --key-file "$OUT/task/prompts.json" --key "llama.cpp=$llama_sha task n300 np8 topk1" \
  --out "$OUT/task/llama-ids.json" -- gpu-wait run --vram "$((vram + 2))" --priority 20 --timeout 2400 -- bash -c '"$1" -m "$2" -c 16384 -np 8 -ngl 99 -fa on -ctk f16 -ctv f16 -t 8 --host 127.0.0.1 --port 8199 > "$3/llama-server.log" 2>&1 & p=$!
    trap "kill $p" EXIT
    for _ in $(seq 600); do curl -sf localhost:8199/health >/dev/null 2>&1 && break; kill -0 $p || exit 1; sleep 1; done
    "$4" bench/quality-task-ids.py llama --url http://127.0.0.1:8199 --out "$3"' _ "$LB/llama-server" "$llama_gguf" "$OUT/task" "$PY" 2>&1 | tee -a "$OUT/SUMMARY.txt"
"$PY" bench/quality-task-ids.py score --tok-url $TOK --out "$OUT/task" >> "$OUT/task.log" 2>&1 || die score-task "$OUT/task.log"
ok task "$(tail -1 "$OUT/task.log")"

"$PY" bench/quality-score.py --key "$KEY" --ppl-ours "$OUT/ppl-ours/ppl-result.json" --ppl-llama-log "$OUT/ppl-llama.log" \
  --task-dir "$OUT/task" --out "$OUT/result.json" > "$OUT/score.log" || die score "$OUT/score.log"
ok result "$(jq -c '{ppl_ratio,delta_pp,verdict}' "$OUT/result.json")"
rm -rf "$OUT/ppl-ours/dump" "$OUT/engine-head"
