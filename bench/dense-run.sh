#!/usr/bin/env bash
# Dense-family per-target verify: BARO_FORCE teacher-forced agreement vs llama.cpp
# on the 20-prompt set, plus no-spec tok/s_gen. bench/dense-protocol.md, PROFILE step 3.
# Usage: bench/dense-run.sh GGUF PACKDIR PROFILEDIR OUTDIR [PORT]
# PROMPTS=glob limits the set (quick runs, P18: a verdict needs the full 20 and at least 15 exercised).
# LLAMA_CACHE=dir caches the llama.cpp reference tokens per (gguf sha, ids sha) (P17); ours is never cached.
# DEFS="-D K=V ..." extra mojo build defines for the engine (an arm knob: print it in the arm file).
# Run inside gpu-wait; engine and llama-server never overlap. LLAMA_NGL: reference layers on the GPU, default 0 (CPU); a GPU reference differs from CPU on 6 of 8 prompts, so LLAMA_NGL=99 is exploratory only.
set -uo pipefail
cd "$(dirname "$0")/.."
GGUF=$1; PACK=$2; PROFILE=$3; OUT=${4:-.work/dense/run}; PORT=${5:-8099}
. bench/min16k.sh; min16k "$(grep -oP "^comptime TMAX = \K[0-9]+" "$PROFILE/profile.mojo")" 16000
mkdir -p "$OUT"
BIN="$OUT/spark-engine"
ok() { echo "PASS $1: $2" | tee -a "$OUT/SUMMARY.txt"; }
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }

echo "== build =="
# profile dir first: tools/profile.mojo (the TOML profile tool) would shadow the generated profile module; tools/ for
# gguf_reader (2026-10-12, the old line no longer built)
./.venv/bin/mojo build serve/spark.mojo ${DEFS:-} -I "$PROFILE" -I . -I kernels -I serve -I tools -o "$BIN" > "$OUT/build.log" 2>&1 \
  || die build "$(grep -m1 error: "$OUT/build.log" | cut -c1-200)"
ok build "$BIN"

echo "== tokenize 20 prompts with our own tokenizer =="
for tf in bench/mtp-prompts/${PROMPTS:-p*}.txt; do
  p=$(basename "$tf" .txt)
  ./.work/baro-tokenize encode "$tf" "$GGUF" > "$OUT/$p.ids" 2>"$OUT/$p.tok.log" \
    || die tok "$p: $(cat "$OUT/$p.tok.log")"
done
ok tok "20/20 tokenized"

echo "== llama.cpp: start server (CPU reference, f32 KV: GPU f16+FA was 0.7 relL2 off f64, open3 item 2) =="
ngl=${LLAMA_NGL:-0}; echo "reference ngl $ngl"
~/llama.cpp/build/bin/llama-server -m "$GGUF" -c 16384 -ngl "$ngl" -fa off -ctk f32 -ctv f32 \
  -b 2048 -ub 512 -t 16 --host 127.0.0.1 --port "$PORT" > "$OUT/llama-server.log" 2>&1 &
LPID=$!
cleanup_llama() { kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; }
trap cleanup_llama EXIT
up=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$LPID" 2>/dev/null || die llama "server exited early: $(tail -5 "$OUT/llama-server.log")"
  sleep 1
done
[ "$up" = 1 ] || die llama "server never came up"
curl -s "http://127.0.0.1:$PORT/props" > "$OUT/llama-props.json"
ok llama "up"

echo "== per-prompt: llama greedy (reference), ours no-spec (tok/s), ours forced (agreement) =="
echo "prompt n_ids tok_s_gen forced_agreement" > "$OUT/results.txt"
for tf in bench/mtp-prompts/${PROMPTS:-p*}.txt; do
  p=$(basename "$tf" .txt)
  ids=$(tr -s '\n' ',' < "$OUT/$p.ids" | sed 's/,$//')
  # P17: the reference arm is a pure function of (gguf, ids); LLAMA_CACHE=dir keeps it across arms of one job
  ck="${LLAMA_CACHE:-}/$(sha256sum "$GGUF" | cut -c1-16)-$(sha256sum "$OUT/$p.ids" | cut -c1-16).llama-tokens.txt"
  if [ -n "${LLAMA_CACHE:-}" ] && [ -s "$ck" ]; then
    cp "$ck" "$OUT/$p.llama-tokens.txt"; echo "refcache hit $p key=$(basename "$ck")"
  else
    resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\": [$ids], \"n_predict\": 64, \"temperature\": 0, \"top_k\": 1, \"cache_prompt\": false, \"return_tokens\": true}")
    echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" ".join(str(t) for t in d["tokens"]))' \
      > "$OUT/$p.llama-tokens.txt" || die run "$p: bad /completion response: $resp"
    if [ -n "${LLAMA_CACHE:-}" ]; then mkdir -p "$LLAMA_CACHE"; cp "$OUT/$p.llama-tokens.txt" "$ck"; echo "refcache miss $p key=$(basename "$ck")"; fi
  fi

  env BARO_PROMPT="$OUT/$p.ids" BARO_PACK="$PACK" BARO_GEN=64 BARO_SPEC=0 "$BIN" > "$OUT/$p.ours-nospec.log" 2>&1 \
    || die run "$p: engine exit $?"
  ts=$(grep -oE 'tok/s_gen: [0-9.]+' "$OUT/$p.ours-nospec.log" | cut -d' ' -f2)

  env BARO_PROMPT="$OUT/$p.ids" BARO_PACK="$PACK" BARO_GEN=64 BARO_SPEC=0 BARO_FORCE="$OUT/$p.llama-tokens.txt" "$BIN" \
    > "$OUT/$p.ours-forced.log" 2>&1 || die run "$p: forced engine exit $?"
  agree=$(grep '^forced agreement:' "$OUT/$p.ours-forced.log" | sed 's/forced agreement: //; s/ \/ /\//')

  echo "$p $(wc -l < "$OUT/$p.ids") $ts $agree" >> "$OUT/results.txt"
done
cleanup_llama; trap - EXIT
# the gate: forced agreement per prompt against MIN_PCT (default 90; the known-good targets sit at 95 to 98).
# A results table with 0/64 everywhere used to print PASS here (2026-09-17, m.ledger/mojo-baro.md).
MIN_PCT=${MIN_PCT:-90}; MIN_REF=${MIN_REF:-16}
# a reference that ends after < MIN_REF tokens scores a near-tie, not the engine: NOT EXERCISED, and at least 15 must remain
verdict=$(python3 - "$OUT/results.txt" "$MIN_PCT" "$MIN_REF" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]).read().splitlines()[1:]]
short = [r[0] for r in rows if int(r[3].split("/")[1]) < int(sys.argv[3])]
rows = [r for r in rows if r[0] not in short]
pct = [100.0 * int(r[3].split("/")[0]) / int(r[3].split("/")[1]) for r in rows]
bad = [r[0] for r, x in zip(rows, pct) if x < float(sys.argv[2])]
print(f"min {min(pct):.1f}% mean {sum(pct)/len(pct):.1f}% over {len(pct)} prompts, bar {sys.argv[2]}%; below bar: {len(bad)} {' '.join(bad)}; NOT EXERCISED (ref < {sys.argv[3]}): {len(short)} {' '.join(short)}")
sys.exit(1 if bad or len(rows) < 15 else 0)
PY
) && ok agreement "$verdict" || die agreement "$verdict"
ok run "$(cat "$OUT/results.txt")"
echo "== done, see $OUT/SUMMARY.txt =="
