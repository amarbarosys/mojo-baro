#!/usr/bin/env bash
# Check for the q4x soft step (Q4X_SOFT, latent-flash E8 retry O2): after prefill, one soft step with no rescale
# (top-K softmax . token_embd rows through the engine's own dequant) must point the same way as latent-wa's
# full-vocabulary soft embedding of the same prompt-end hidden: cos >= 0.999 per prompt, and the step's PLE id must
# be the recorded greedy token. Needs SOFT_REF (latent-wa OUT.soft.f32 of the flash rows, same prompt order).
# usage: bench/q4x-soft-check.sh OUT_DIR [N_PROMPTS]   (queues itself under gpu-wait; Q4X_BIN overrides the binary)
set -euo pipefail
OUT=${1:?out dir}; NP=${2:-20}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
BIN=${Q4X_BIN:-.work/q4x-soft/q4x}
NEXT=${Q4X_NEXT_IDS:-$HOME/.cache/baro/qwen4exp/latent/flash-rows-2026-10-12/next_ids.txt}
REF=${SOFT_REF:-.work/flash-probe.soft.f32}
K=${Q4X_SOFT_K:-1024}
cd "$REPO"
if [ -z "${Q4X_SOFT_INNER:-}" ]; then
  mkdir -p "$OUT"
  { echo "date $(date -Is)"; echo "commit $(git rev-parse HEAD) dirty $(git status --porcelain serve/q4x.mojo | wc -l)"
    echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -c1-16)"; echo "ref $REF sha256 $(sha256sum "$REF" | cut -c1-16)"
    echo "env Q4X_PREFILL=1 Q4X_TMAX=256 Q4X_SOFT=1 Q4X_SOFT_K=$K Q4X_SOFT_NORM=0"; } > "$OUT/receipt.txt"
  rc=0; gpu-wait run --vram 20 --timeout 1800 -- env Q4X_SOFT_INNER=1 Q4X_GGUF="$GGUF" Q4X_BIN="$BIN" Q4X_NEXT_IDS="$NEXT" SOFT_REF="$REF" Q4X_SOFT_K="$K" HOME="$HOME" bash "$REPO/bench/q4x-soft-check.sh" "$OUT" "$NP" || rc=$?
  gpu-wait stats --days 1 | tail -3 >> "$OUT/receipt.txt"
  exit $rc
fi
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
mkdir -p "$OUT"
fail=0; names=""; i=0
for p in $(ls bench/q4x-prompts/*.prompt | head -n "$NP"); do
  i=$((i + 1)); b=$(basename "$p" .prompt); g=$(sed -n "${i}p" "$NEXT"); ids=$(jq -r 'join(" ")' "$p")
  rm -f "$OUT/$b.soft" "$OUT/$b.soft.ids"
  if ! env Q4X_PREFILL=1 Q4X_TMAX=256 Q4X_SOFT=1 Q4X_SOFT_K="$K" Q4X_SOFT_NORM=0 Q4X_SOFT_OUT="$OUT/$b.soft" "$BIN" "$GGUF" 1 $ids > "$OUT/$b.log" 2>&1; then
    fail=$((fail + 1)); names="$names $b"; echo "FAIL $b: q4x exit (log $OUT/$b.log)" | tee -a "$OUT/receipt.txt"; continue
  fi
  line=$("$REPO/.venv/bin/python" -I -c '
import sys, numpy as np
a = np.fromfile(sys.argv[1], np.float32).astype(np.float64)
r = np.fromfile(sys.argv[2], np.float32).reshape(-1, 2560)[int(sys.argv[3])].astype(np.float64)
c = a @ r / np.linalg.norm(a) / np.linalg.norm(r)
print(f"{c:.6f} {np.linalg.norm(a) / np.linalg.norm(r):.6f}")' "$OUT/$b.soft" "$REF" $((i - 1)))
  cos=${line% *}; ratio=${line#* }; sid=$(cat "$OUT/$b.soft.ids")
  why=""
  [ "$sid" != "$g" ] && why="PLE id $sid vs greedy $g"
  awk -v c="$cos" 'BEGIN { exit !(c >= 0.999) }' || why="${why:+$why; }cos $cos < 0.999"
  if [ -n "$why" ]; then fail=$((fail + 1)); names="$names $b"; echo "FAIL $b: $why (norm ratio $ratio)" | tee -a "$OUT/receipt.txt"
  else echo "ok $b cos $cos norm ratio $ratio id $sid" | tee -a "$OUT/receipt.txt"; fi
done
if [ "$fail" -gt 0 ]; then echo "FAIL $fail/$NP$names" | tee -a "$OUT/receipt.txt"; exit 1; fi
echo "PASS $NP/$NP q4x top-$K soft step matches latent-wa's full soft (cos >= 0.999), PLE id = greedy" | tee -a "$OUT/receipt.txt"
