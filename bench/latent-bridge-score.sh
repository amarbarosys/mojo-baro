#!/usr/bin/env bash
# latent-bridge score harness. One arm = one resident q4x
# process (Q4X_STDIN) under one gpu-wait job; eval stream = pre [payload] post -> answer (Q4X_FORCE = post ids,
# predictions at forced positions skipped, Q4X_STOP = im_end).
# usage: bench/latent-bridge-score.sh ARM SEL OUT_ROOT [N]     SEL = test | val; N = first N items per task (half)
#   cot   A-turn greedy CoT (up to 300 ids, stop at im_end), feeds h32 / t     nocot  no payload
#   h32   first 32 CoT ids as text payload      t   full CoT as text payload (ceiling, precondition arm)
#   s32   32 soft steps (Q4X_SOFT, K 1024, rows rescaled to the median token_embd row norm 0.469174), rows saved
#   n32   Gaussian rows at each s32 row's norm, s32 ids      x32   another item's s32 rows + ids (derangement)
# out: OUT_ROOT/ARM-SEL[-N]/{requests.txt,q4x.log,gen.tsv,score.txt,receipt.txt}
# Exit 3 + UNVERIFIED while exchange/latent-bridge-prereg-*.md is not FROZEN (CLAUDE.md exploratory rule).
set -euo pipefail
ARM=${1:?arm}; SEL=${2:?sel}; ROOT=${3:?out root}; N=${4:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
C=$HOME/.cache/baro/latent-bridge; ITEMS=$C/data/items.tsv
GGUF=$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q4X=${Q4X_BIN:-$REPO/.work/lb/q4x-next}; LD=$REPO/.work/latent-data
TAGN=$([ "$N" = 0 ] && echo "" || echo "-$N"); O=$ROOT/$ARM-$SEL$TAGN; mkdir -p "$O"
SOFT_NORM=0.469174; IM_END=248046
POST=$(awk -F'\t' 'NR == 1 { print $8 }' "$ITEMS"); LQ=$(echo $POST | wc -w)
awk -F'\t' '{ print $8 }' "$ITEMS" | sort -u | wc -l | grep -qx 1 || { echo "FAIL items: post ids differ across items"; exit 1; }
sel_ids() { awk -v n="$N" '{ t = ($0 ~ /^mh-/) ? "m" : "g"; if (n == 0 || c[t]++ < n) print }' "$C/data/sel-$SEL.txt"; }
cot_of() { awk -v id="$1" '$1 == "out" && $2 == id { s = ""; for (i = 4; i <= NF && $i != '"$IM_END"'; i++) if (max == 0 || i - 3 <= max) s = s " " $i; print s }' max="$2" "$ROOT/cot-$SEL$TAGN/q4x.log"; }
env_arm="Q4X_FORCE=$(echo $POST | tr ' ' ,) Q4X_STOP=$IM_END"; ngen=$((LQ + 16)); skip=$LQ
: > "$O/requests.txt"
for id in $(sel_ids); do
  pre=$(awk -F'\t' -v id="$id" '$1 == id { print $6 }' "$ITEMS")
  case $ARM in
    cot) echo "$id 300 $pre" ;;
    nocot|s32) echo "$id $ngen $pre" ;;
    h32) echo "$id $ngen $pre$(cot_of "$id" 32)" ;;
    t) echo "$id $ngen $pre$(cot_of "$id" 0)" ;;
    n32|x32) echo "$id $ngen @$ROOT/$ARM-$SEL$TAGN/rows/$id $pre" ;;
    *) echo "FAIL args: unknown ARM $ARM" >&2; exit 2 ;;
  esac >> "$O/requests.txt"
done
case $ARM in
  cot) env_arm="Q4X_STOP=$IM_END"; skip=0 ;;
  s32) env_arm="$env_arm Q4X_SOFT=32 Q4X_SOFT_K=1024 Q4X_SOFT_NORM=$SOFT_NORM Q4X_SOFT_OUT=$O/rows"; mkdir -p "$O/rows" ;;
  n32|x32) mkdir -p "$O/rows"; "$LD" $([ $ARM = n32 ] && echo noise || echo cross) <(sel_ids) "$ROOT/s32-$SEL$TAGN/rows" "$O/rows" 2560 ;;
esac
{
  echo "date $(date -Is) arm $ARM sel $SEL n_per_task $N requests $(wc -l < "$O/requests.txt") commit $(git -C "$REPO" rev-parse HEAD)"
  echo "q4x $Q4X sha256 $(sha256sum "$Q4X" | cut -c1-16) gguf sha256 $(sha256sum "$GGUF" | cut -c1-16) items sha256 $(sha256sum "$ITEMS" | cut -c1-16)"
  echo "env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX=1024 $env_arm; ngen $ngen skip $skip"
} > "$O/receipt.txt"
t0=$(date +%s)
gpu-wait run --vram 20 --mem 44 --timeout 7200 -- bash -c "exec timeout 7000 env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX=1024 $env_arm \
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 '$Q4X' '$GGUF' < '$O/requests.txt' > '$O/q4x.log' 2>&1" \
  || echo "FAIL q4x: exit $? (log $O/q4x.log)" | tee -a "$O/receipt.txt"
got=$(grep -c '^out ' "$O/q4x.log" || true); want=$(wc -l < "$O/requests.txt")
echo "requests answered $got of $want in $(( $(date +%s) - t0 )) s" | tee -a "$O/receipt.txt"
[ "$got" = "$want" ] || { echo "FAIL arm $ARM: $got of $want answered" | tee -a "$O/receipt.txt"; exit 1; }
[ $ARM = cot ] && { echo "PASS cot generated"; exit 0; }
"$LD" gen "$O/q4x.log" "$skip" "$GGUF" "$O/gen.tsv" > /dev/null
"$LD" score "$ITEMS" "$O/gen.tsv" > "$O/score.txt"
grep '^score' "$O/score.txt" | tee -a "$O/receipt.txt"
if ! grep -qs '^Status: FROZEN' "$REPO"/exchange/latent-bridge-prereg-*.md; then echo "UNVERIFIED: prereg not frozen, exploratory run" | tee -a "$O/receipt.txt"; exit 3; fi
