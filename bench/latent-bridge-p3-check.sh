#!/usr/bin/env bash
# latent-bridge P3 parity smoke:
#  1. argv mode, new q4x vs base q4x (main): ids and every f32 hidden row byte-identical on 2 items
#  2. resident mode (Q4X_STDIN=1), requests A B A: both A rows byte-identical to the fresh-process A
#  3. Q4X_DUMP_ATT rows: count = tail per request, file size = rows x (6144 + 2560) x 4 B, all finite
#  4. stdin @VEC: the rows + ids Q4X_SOFT_OUT wrote, replayed in a request, give the soft run's ids
# usage: bench/latent-bridge-p3-check.sh OUT_DIR      (runs itself under gpu-wait; Q4X_BIN, Q4X_BASE override)
# gpu-wait does not forward the caller's env: every knob goes into the queued command by name.
set -euo pipefail
OUT=${1:?out dir}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
ITEMS=${ITEMS:-$HOME/.cache/baro/latent-bridge/data/items.tsv}
BIN=${Q4X_BIN:-.work/lb/q4x}; BASE=${Q4X_BASE:-.work/lb/q4x-base}
if [ -z "${P3_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 20 --mem ${P3_MEM:-44} --timeout 1500 -- env P3_INNER=1 Q4X_ZC=${Q4X_ZC:-1} Q4X_GGUF="$GGUF" ITEMS="$ITEMS" Q4X_BIN="$BIN" Q4X_BASE="$BASE" HOME="$HOME" bash "$REPO/bench/latent-bridge-p3-check.sh" "$OUT"
fi
cd "$REPO"
echo "page cache before warm: $(fincore -nb "$GGUF" | awk '{print $1}') of $(stat -c %s "$GGUF") B"
[ "${Q4X_ZC:-1}" = 0 ] || cat "$GGUF" > /dev/null
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 Q4X_ZC=${Q4X_ZC:-1}
seq_of() { awk -F'\t' -v id="$1" '$1==id { print $6" "$8" "$9 }' "$ITEMS"; }
A=$(seq_of gt3); B=$(seq_of mh-train-0)
[ -n "$A" ] && [ -n "$B" ] || { echo "FAIL items: gt3 or mh-train-0 missing in $ITEMS"; exit 1; }
TA=$(echo $A | wc -w); TB=$(echo $B | wc -w)
{
  echo "date $(date -Is)"; echo "commit $(git rev-parse HEAD)"
  echo "binary $BIN sha256 $(sha256sum "$BIN" | cut -c1-16) base $BASE sha256 $(sha256sum "$BASE" | cut -c1-16)"
  echo "gguf sha256 $(sha256sum "$GGUF" | cut -c1-16); items $ITEMS sha256 $(sha256sum "$ITEMS" | cut -c1-16)"
  echo "env Q4X_ZC=$Q4X_ZC Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_HIDDEN_F32=1; A=gt3 T=$TA, B=mh-train-0 T=$TB"
} > "$OUT/receipt.txt"
fail=0; names=""
check() { if [ "$2" = 1 ]; then echo "PASS $1" | tee -a "$OUT/receipt.txt"; else echo "FAIL $1" | tee -a "$OUT/receipt.txt"; fail=$((fail + 1)); names="$names $1"; fi; }
run1() { timeout 300 env Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_HIDDEN_F32=1 Q4X_HIDDEN="$OUT/$2.bin" "$1" "$GGUF" 4 $3 > "$OUT/$2.log" 2>&1; }
for x in A B; do
  ids=${!x}; rm -f "$OUT"/{base,new}-$x.bin*
  run1 "$BASE" base-$x "$ids" || echo "FAIL base-$x: exit $? (124 = hung past 300 s)" | tee -a "$OUT/receipt.txt"
  run1 "$BIN" new-$x "$ids" || echo "FAIL new-$x: exit $? (124 = hung past 300 s)" | tee -a "$OUT/receipt.txt"
  ok=1; [ -s "$OUT/new-$x.bin" ] || ok=0; cmp -s "$OUT/base-$x.bin" "$OUT/new-$x.bin" || ok=0
  [ -n "$(grep '^ids' "$OUT/new-$x.log")" ] && [ "$(grep '^ids' "$OUT/base-$x.log")" = "$(grep '^ids' "$OUT/new-$x.log")" ] || ok=0
  check "argv-$x (hidden bytes + ids, new == base)" $ok
done
rm -f "$OUT"/stdin.*
printf 'a1 4 %s\nb1 4 %s\na2 4 %s\n' "$A" "$B" "$A" | timeout 300 env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_HIDDEN_F32=1 \
  Q4X_HIDDEN="$OUT/stdin.bin" Q4X_DUMP_ATT="$OUT/stdin.att" Q4X_INDEX="$OUT/stdin.idx" "$BIN" "$GGUF" > "$OUT/stdin.log" 2>&1 || echo "FAIL stdin run: exit $? (124 = hung past 300 s)" | tee -a "$OUT/receipt.txt"
RA=$(((TA + 3) * 2560 * 4)); RB=$(((TB + 3) * 2560 * 4))   # N_GEN 4: prompt rows + 3 decode rows
head -c $RA "$OUT/stdin.bin" > "$OUT/stdin-a1.bin"; tail -c $RA "$OUT/stdin.bin" > "$OUT/stdin-a2.bin"
ok=1; cmp -s "$OUT/stdin-a1.bin" "$OUT/new-A.bin" || ok=0; cmp -s "$OUT/stdin-a2.bin" "$OUT/new-A.bin" || ok=0
[ "$(stat -c %s "$OUT/stdin.bin")" = "$((2 * RA + RB))" ] || ok=0
for t in a1 a2; do [ "$(grep "^out $t " "$OUT/stdin.log" | cut -d' ' -f3-)" = "$(grep '^ids' "$OUT/new-A.log")" ] || ok=0; done
check "stdin-reset (A after B byte-identical to fresh A, ids equal)" $ok
ok=1; [ "$(cat "$OUT/stdin.idx")" = "$(printf 'a1 %d 15\nb1 %d 15\na2 %d 15' $((TA + 3)) $((TB + 3)) $((TA + 3)))" ] || ok=0
[ "$(stat -c %s "$OUT/stdin.att")" = "$((3 * 15 * (6144 + 2560) * 4))" ] || ok=0
./.venv/bin/python -I -c "import numpy as np,sys; a=np.fromfile(sys.argv[1],np.float32); sys.exit(0 if np.isfinite(a).all() and np.abs(a).max()>0 else 1)" "$OUT/stdin.att" || ok=0
check "dump-att (index, 3 x (12 tail + 3 decode) rows of q 6144 + attn_out 2560 f32, finite)" $ok
timeout 300 env Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_SOFT=4 Q4X_SOFT_OUT="$OUT/s4" "$BIN" "$GGUF" 4 $A > "$OUT/soft4.log" 2>&1 || echo "FAIL soft4 run: exit $?" | tee -a "$OUT/receipt.txt"
printf 'v1 4 @%s %s\n' "$OUT/s4" "$A" | timeout 300 env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX=512 "$BIN" "$GGUF" > "$OUT/vec.log" 2>&1 || echo "FAIL vec run: exit $?" | tee -a "$OUT/receipt.txt"
ok=1; [ "$(grep '^out v1 ' "$OUT/vec.log" | cut -d' ' -f3-)" = "$(grep '^ids' "$OUT/soft4.log")" ] && [ -n "$(grep '^ids' "$OUT/soft4.log")" ] || ok=0
check "stdin-vec (@rows of Q4X_SOFT_OUT replayed per request: ids equal the soft run)" $ok
if [ $fail = 0 ]; then echo "PASS 5/5" | tee -a "$OUT/receipt.txt"; else echo "FAIL $fail/5$names" | tee -a "$OUT/receipt.txt"; exit 1; fi
