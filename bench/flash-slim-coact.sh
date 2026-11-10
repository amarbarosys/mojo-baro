#!/usr/bin/env bash
# Expert co-activation structure of Flash (exploratory, CPU only): per layer, the co-pick graph of a Q4X_EXPERTS
# trace (tools/flash-slim coact) split into K balanced parts by graph-shard (iTools, newera shard), then scored by
# tools/flash-slim groups against 200 random relabelings with the same part sizes: intra-part share of the
# co-activation weight, and part/category mutual information. Logs per layer in OUT/, summary in OUT/summary.tsv.
# usage: bench/flash-slim-coact.sh OUT_DIR TRACE SEQLEN CATS [K...]   (default K: 4 8)
set -euo pipefail
OUT=${1:?out dir}; TRACE=${2:?trace}; SEQ=${3:?seqlen}; CATS=${4:?cats}; shift 4
KS=${*:-4 8}
REPO=$(cd "$(dirname "$0")/.." && pwd)
TOOL=${FLASH_SLIM_TOOL:-$REPO/.work/flash-slim/flash-slim}
command -v graph-shard > /dev/null || { echo "FAIL preflight: graph-shard not on PATH (~/iTools/bin)"; exit 1; }
mkdir -p "$OUT"
{ echo "date $(date -Is)"; echo "commit $(git -C "$REPO" rev-parse HEAD)"; echo "trace $TRACE sha256 $(sha256sum "$TRACE" | cut -c1-16) seq $SEQ cats $CATS"
  echo "tool $TOOL sha256 $(sha256sum "$TOOL" | cut -c1-16)"; echo "graph-shard source $(readlink -f ~/iTools/science/graph-shard/shard.mojo) $(git -C ~/Projects/newera rev-parse --short HEAD)"; } > "$OUT/receipt.txt"
printf 'layer\tk\tedges\tcut\tintra\trandom\tz_intra\tmi_bits\tmi_random\tz_mi\n' > "$OUT/summary.tsv"
fail=0; names=""
for l in $(seq 0 47); do
  "$TOOL" coact "$TRACE" $l "$OUT/l$l.edges" > "$OUT/l$l.coact.log" || { fail=$((fail + 1)); names="$names l$l"; continue; }
  for k in $KS; do
    if ! graph-shard "$OUT/l$l.edges" $k --parts "$OUT/l$l.k$k.parts" > "$OUT/l$l.k$k.shard.log" 2>&1 \
       || ! "$TOOL" groups "$TRACE" $SEQ $CATS $l "$OUT/l$l.k$k.parts" > "$OUT/l$l.k$k.groups.log"; then
      fail=$((fail + 1)); names="$names l$l.k$k"; continue
    fi
    g=$(head -1 "$OUT/l$l.k$k.groups.log"); s=$(tail -1 "$OUT/l$l.k$k.shard.log")
    awk -v l=$l -v k=$k -v s="$s" '{ for (i = 1; i <= NF; i++) v[$i] = $(i + 1); split(s, t, "\t")
      printf "%d\t%d\t%s\t%s\t%.4f\t%.4f\t%.1f\t%.4f\t%.4f\t%.1f\n", l, k, t[2], t[4], v["intra"], $11, $15, v["mi_bits"], $19, $23 }' <<< "$g" >> "$OUT/summary.tsv"
  done
done
for k in $KS; do
  awk -v k=$k -F'\t' 'NR > 1 && $2 == k { n++; i += $5; r += $6; m += $8; mr += $9; if ($8 > bm) { bm = $8; bl = $1 } }
    END { printf "k %d layers %d mean intra %.4f (random %.4f) mean MI %.4f bits (random %.4f) max MI %.4f at layer %d\n", k, n, i / n, r / n, m / n, mr / n, bm, bl }' "$OUT/summary.tsv" | tee -a "$OUT/receipt.txt"
done
if [ "$fail" -gt 0 ]; then echo "FAIL $fail steps:$names" | tee -a "$OUT/receipt.txt"; exit 1; fi
echo "PASS 48 layers x K {$KS}, summary $OUT/summary.tsv" | tee -a "$OUT/receipt.txt"
