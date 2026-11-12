#!/usr/bin/env bash
# q4x soft-step host top-K (sort instead of k x |cand| selection): rows byte-identical old vs new, ms/step before/after.
# usage: bench/latent-bridge-soft-ab.sh OUT_DIR   (runs itself under gpu-wait; OLD=.work/lb/q4x-base NEW=.work/lb/q4x-soft)
# Old binary prints no soft timing: its ms/step = (wall with Q4X_SOFT=8 - wall with Q4X_SOFT=0) / 8, same process shape.
set -euo pipefail
OUT=${1:?out dir}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
ITEMS=$HOME/.cache/baro/latent-bridge/data/items.tsv
OLD=${OLD:-.work/lb/q4x-base}; NEW=${NEW:-.work/lb/q4x-soft}
if [ -z "${AB_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --mem 44 --timeout 1800 -- env AB_INNER=1 OLD="$OLD" NEW="$NEW" HOME="$HOME" bash "$REPO/bench/latent-bridge-soft-ab.sh" "$OUT"
fi
cd "$REPO"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
{ echo "date $(date -Is) commit $(git rev-parse HEAD)"; echo "old $OLD $(sha256sum "$OLD" | cut -c1-16) new $NEW $(sha256sum "$NEW" | cut -c1-16)"
  echo "env Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_SOFT=8 Q4X_SOFT_K=1024 Q4X_SOFT_NORM=0, prompts = pre ids of gt3 mh-train-0 ge5"; } > "$OUT/receipt.txt"
fail=0
for id in gt3 mh-train-0 ge5; do
  ids=$(awk -F'\t' -v id=$id '$1==id { print $6 }' "$ITEMS")
  for arm in old new; do
    bin=$([ $arm = old ] && echo "$OLD" || echo "$NEW")
    s0=$(date +%s%N); timeout 300 env Q4X_PREFILL=1 Q4X_TMAX=512 "$bin" "$GGUF" 1 $ids > "$OUT/$id.$arm.s0.log" 2>&1; s1=$(date +%s%N)
    timeout 300 env Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_SOFT=8 Q4X_SOFT_OUT="$OUT/$id.$arm.rows" "$bin" "$GGUF" 1 $ids > "$OUT/$id.$arm.log" 2>&1; s2=$(date +%s%N)
    echo "$id $arm wall_soft0_ms $(( (s1 - s0) / 1000000 )) wall_soft8_ms $(( (s2 - s1) / 1000000 )) delta_ms_per_step $(( (s2 - s1 - s1 + s0) / 8000000 )) $(grep -o 'ms/step [0-9.]*' "$OUT/$id.$arm.log" || true)" | tee -a "$OUT/receipt.txt"
  done
  if cmp -s "$OUT/$id.old.rows" "$OUT/$id.new.rows" && cmp -s "$OUT/$id.old.rows.ids" "$OUT/$id.new.rows.ids"; then echo "PASS $id rows + ids byte-identical" | tee -a "$OUT/receipt.txt"
  else echo "FAIL $id rows differ" | tee -a "$OUT/receipt.txt"; fail=$((fail + 1)); fi
done
[ $fail = 0 ] && echo "PASS 3/3" | tee -a "$OUT/receipt.txt" || { echo "FAIL $fail/3" | tee -a "$OUT/receipt.txt"; exit 1; }
