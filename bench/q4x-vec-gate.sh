#!/usr/bin/env bash
# Identity gate for q4x Q4X_VEC (vector-input step, latent-flash E8 retry): feeding the token_embd row of g0 as a
# vector must equal step(g0) byte for byte. Per prompt: g0 = the first generated token (flash rows next_ids.txt),
# row = latent-wa's CPU dequant of token_embd[g0]. Arm A: prefill, Q4X_FORCE=g0, gen 3. Arm B: prefill,
# Q4X_VEC=row Q4X_VEC_IDS=g0, gen 2. PASS = every hidden row (f32) byte-identical and A's ids after g0 = B's ids.
# Falsifier on the first prompt: row of g0+1 with id g0 must change the hidden bytes (a gate that cannot fail is not one).
# usage: bench/q4x-vec-gate.sh OUT_DIR [N_PROMPTS]   (queues itself under gpu-wait; Q4X_BIN overrides the binary)
set -euo pipefail
OUT=${1:?out dir}; NP=${2:-3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
BIN=${Q4X_BIN:-.work/q4x-vec/q4x}
NEXT=${Q4X_NEXT_IDS:-$HOME/.cache/baro/qwen4exp/latent/flash-rows-2026-10-12/next_ids.txt}
LWA=${LATENT_WA:-$HOME/iTools/bin/latent-wa}
D=2560
cd "$REPO"
if [ -z "${Q4X_VEC_INNER:-}" ]; then
  mkdir -p "$OUT"
  { echo "date $(date -Is)"; echo "commit $(git rev-parse HEAD) dirty $(git status --porcelain serve/q4x.mojo | wc -l)"
    echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -c1-16)"; echo "gguf $GGUF size $(stat -c %s "$GGUF")"
    echo "env Q4X_PREFILL=1 Q4X_TMAX=256 Q4X_HIDDEN_F32=1"; } > "$OUT/receipt.txt"
  i=0
  for p in $(ls bench/q4x-prompts/*.prompt | head -n "$NP"); do
    i=$((i + 1)); g=$(sed -n "${i}p" "$NEXT")
    "$LWA" --rows "$GGUF" "$g" 1 "$OUT/rows.tmp" && head -c $((D * 4)) "$OUT/rows.tmp" > "$OUT/$(basename "$p" .prompt).vec"
    [ "$i" -eq 1 ] && "$LWA" --rows "$GGUF" $((g + 1)) 1 "$OUT/rows.tmp" && head -c $((D * 4)) "$OUT/rows.tmp" > "$OUT/falsifier.vec"
  done
  rm -f "$OUT/rows.tmp"
  rc=0; gpu-wait run --vram 20 --timeout 1800 -- env Q4X_VEC_INNER=1 Q4X_GGUF="$GGUF" Q4X_BIN="$BIN" Q4X_NEXT_IDS="$NEXT" HOME="$HOME" bash "$REPO/bench/q4x-vec-gate.sh" "$OUT" "$NP" || rc=$?
  gpu-wait stats --days 1 | tail -3 >> "$OUT/receipt.txt"
  exit $rc
fi
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
run() { local tag=$1; shift; rm -f "$OUT/$tag.bin" "$OUT/$tag.bin.n"
  env Q4X_HIDDEN="$OUT/$tag.bin" Q4X_HIDDEN_F32=1 Q4X_PREFILL=1 Q4X_TMAX=256 "$@" > "$OUT/$tag.log" 2>&1; }
fail=0; names=""; i=0
for p in $(ls bench/q4x-prompts/*.prompt | head -n "$NP"); do
  i=$((i + 1)); b=$(basename "$p" .prompt); g=$(sed -n "${i}p" "$NEXT"); ids=$(jq -r 'join(" ")' "$p")
  why=""
  run "$b.A" env Q4X_FORCE="$g" "$BIN" "$GGUF" 3 $ids || why="arm A exit"
  [ -z "$why" ] && { run "$b.B" env Q4X_VEC="$OUT/$b.vec" Q4X_VEC_IDS="$g" "$BIN" "$GGUF" 2 $ids || why="arm B exit"; }
  if [ -z "$why" ]; then
    ia=$(grep '^ids ' "$OUT/$b.A.log" | cut -d' ' -f2-); ib=$(grep '^ids ' "$OUT/$b.B.log" | cut -d' ' -f2-)
    if [ "${ia%% *}" != "$g" ]; then why="stale g0: arm A predicted ${ia%% *}, next_ids $g"
    elif [ "${ia#* }" != "$ib" ]; then why="ids A '${ia#* }' vs B '$ib'"
    elif [ "$(cat "$OUT/$b.A.bin.n")" != "$(cat "$OUT/$b.B.bin.n")" ]; then why="row count $(cat "$OUT/$b.A.bin.n") vs $(cat "$OUT/$b.B.bin.n")"
    elif ! cmp -s "$OUT/$b.A.bin" "$OUT/$b.B.bin"; then why="hidden bytes differ: $(cmp "$OUT/$b.A.bin" "$OUT/$b.B.bin" || :)"
    fi
  fi
  if [ "$i" -eq 1 ] && [ -z "$why" ]; then
    run "$b.F" env Q4X_VEC="$OUT/falsifier.vec" Q4X_VEC_IDS="$g" "$BIN" "$GGUF" 2 $ids || why="falsifier exit"
    [ -z "$why" ] && cmp -s "$OUT/$b.A.bin" "$OUT/$b.F.bin" && why="falsifier: row of $((g + 1)) gave identical bytes, gate cannot fail"
    [ -z "$why" ] && echo "falsifier ok: row of $((g + 1)) changed the hidden bytes" | tee -a "$OUT/receipt.txt"
  fi
  if [ -n "$why" ]; then fail=$((fail + 1)); names="$names $b"; echo "FAIL $b: $why (logs $OUT/$b.*.log)" | tee -a "$OUT/receipt.txt"
  else echo "ok $b g0 $g rows $(cat "$OUT/$b.A.bin.n") ids ${ia#* }" | tee -a "$OUT/receipt.txt"; fi
done
if [ "$fail" -gt 0 ]; then echo "FAIL $fail/$NP$names" | tee -a "$OUT/receipt.txt"; exit 1; fi
echo "PASS $NP/$NP byte-identical hidden rows, step_vec(token_embd row) = step(token)" | tee -a "$OUT/receipt.txt"
