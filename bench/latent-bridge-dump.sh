#!/usr/bin/env bash
# latent-bridge P3 dumps. One source x one selection, in
# chunks of CHUNK requests, one gpu-wait job per chunk, resumable (a chunk whose .idx is complete is skipped).
# usage: bench/latent-bridge-dump.sh SRC SEL [CHUNK]
#   SRC = flash-student | flash-teacher | qwythos9b | qwen35-4b      SEL = train | val (data/sel-SEL.txt order)
# out: ~/.cache/baro/latent-bridge/dumps/SRC-SEL.{hid,idx[,att]} + SRC-SEL.d/ (chunks, logs, receipt.txt)
#   flash-*: q4x resident mode (Q4X_STDIN), f32 final hidden (student every position, teacher last 12), layer-47
#            q + attn_output of the last 12 positions; sharers: llama-server --embeddings --pooling none, same ids.
# gpu-wait carries neither env nor stdin: knobs and the redirect live inside the queued bash -c.
set -euo pipefail
SRC=${1:?src}; SEL=${2:?sel}; CHUNK=${3:-500}
REPO=$(cd "$(dirname "$0")/.." && pwd)
C=$HOME/.cache/baro/latent-bridge; D=$C/dumps/$SRC-$SEL.d; ITEMS=$C/data/items.tsv; mkdir -p "$D"
GGUF=$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q4X=${Q4X_BIN:-$REPO/.work/lb/q4x}; LD=$REPO/.work/latent-data; LS=$HOME/Projects/imports/llama.cpp/build/bin/llama-server
case $SRC in
  flash-student) cols='$6" "$8" "$9'; W=2560; TAIL=0 ;;
  flash-teacher) cols='$6" "$7" "$8" "$9'; W=2560; TAIL=12 ;;
  qwythos9b) cols='$6" "$8" "$9'; W=4096; M=$HOME/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-04867e2.gguf ;;
  qwen35-4b) cols='$6" "$8" "$9'; W=2560; M=$HOME/Models/qwen3.5-4b-mtp/Qwen3.5-4B-MTP-Q4_0.gguf ;;
  *) echo "FAIL args: unknown SRC $SRC"; exit 2 ;;
esac
awk -F'\t' 'NR == FNR { want[$1] = NR; next } ($1 in want) { print want[$1] "\t" $1 " 1 " '"$cols"' }' \
  "$C/data/sel-$SEL.txt" "$ITEMS" | sort -n | cut -f2 > "$D/requests.txt"
[ "$(wc -l < "$D/requests.txt")" = "$(wc -l < "$C/data/sel-$SEL.txt")" ] || { echo "FAIL requests: ids of sel-$SEL missing in items.tsv"; exit 1; }
rm -f "$D"/req-*; split -l "$CHUNK" -d -a 3 "$D/requests.txt" "$D/req-"
fail=0; names=""
for r in "$D"/req-*; do
  k=${r##*req-}; want=$(wc -l < "$r")
  if [ -f "$D/$k.idx" ] && [ "$(wc -l < "$D/$k.idx")" = "$want" ]; then echo "skip chunk $k ($want done)"; continue; fi
  rm -f "$D/$k".{hid,att,idx,log}
  t0=$(date +%s)
  if [[ $SRC == flash-* ]]; then
    gpu-wait run --vram 20 --mem 42 --timeout 2400 -- bash -c "exec env Q4X_STDIN=1 Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_HIDDEN_F32=1 \
      Q4X_HIDDEN_TAIL=$TAIL Q4X_HIDDEN='$D/$k.hid' Q4X_DUMP_ATT='$D/$k.att' Q4X_DUMP_TAIL=12 Q4X_INDEX='$D/$k.idx' \
      MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 '$Q4X' '$GGUF' < '$r' > '$D/$k.log' 2>&1" || true
  else
    gpu-wait run --vram 8 --timeout 2400 -- bash -c "
      '$LS' -m '$M' -ngl 99 --embeddings --pooling none --port 18731 -c 1024 -ub 1024 -b 1024 > '$D/$k.server.log' 2>&1 & sp=\$!
      for i in \$(seq 1 120); do curl -s localhost:18731/health | grep -q ok && break; sleep 1; done
      while read -r tag n ids; do
        printf '{\"content\": [%s], \"embd_normalize\": -1}' \"\$(echo \$ids | tr ' ' ',')\" > '$D/$k.req.json'
        curl -s --fail localhost:18731/embedding -d @'$D/$k.req.json' > '$D/$k.rep.json' || { echo \"FAIL curl \$tag\"; break; }
        rows=\$('$LD' emb '$D/$k.rep.json' '$D/$k.hid' $W) || { echo \"FAIL emb \$tag\"; break; }
        echo \"\$tag \$rows 0\" >> '$D/$k.idx'
      done < '$r' > '$D/$k.log' 2>&1
      kill \$sp; wait \$sp 2>/dev/null || true" || true
  fi
  got=$(wc -l < "$D/$k.idx" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then echo "PASS chunk $k: $want requests in $(( $(date +%s) - t0 )) s"
  else echo "FAIL chunk $k: $got of $want requests (log $D/$k.log)"; fail=$((fail + 1)); names="$names $k"; fi
done
[ $fail = 0 ] || { echo "FAIL $fail chunks:$names"; exit 1; }
cat "$D"/[0-9][0-9][0-9].idx > "$C/dumps/$SRC-$SEL.idx"; cat "$D"/[0-9][0-9][0-9].hid > "$C/dumps/$SRC-$SEL.hid"
rows=$(awk '{ s += $2 } END { print s }' "$C/dumps/$SRC-$SEL.idx"); arows=$(awk '{ s += $3 } END { print s }' "$C/dumps/$SRC-$SEL.idx")
ok=1; [ "$(stat -c %s "$C/dumps/$SRC-$SEL.hid")" = "$((rows * W * 4))" ] || ok=0
if [[ $SRC == flash-* ]]; then
  cat "$D"/[0-9][0-9][0-9].att > "$C/dumps/$SRC-$SEL.att"
  [ "$(stat -c %s "$C/dumps/$SRC-$SEL.att")" = "$((arows * (6144 + 2560) * 4))" ] || ok=0
fi
{
  echo "date $(date -Is) src $SRC sel $SEL requests $(wc -l < "$D/requests.txt") hidden rows $rows (width $W) att rows $arows"
  echo "commit $(git -C "$REPO" rev-parse HEAD)"
  if [[ $SRC == flash-* ]]; then echo "q4x $Q4X sha256 $(sha256sum "$Q4X" | cut -c1-16) env Q4X_PREFILL=1 Q4X_TMAX=512 Q4X_HIDDEN_F32=1 Q4X_HIDDEN_TAIL=$TAIL Q4X_DUMP_TAIL=12"
    grep -h '^prompt ' "$D"/[0-9][0-9][0-9].log | awk '{ for (i = 1; i <= NF; i++) if ($i == "ms/token") { s += $(i + 1); n++ } } END { printf "prefill ms/token mean over requests %.2f (n %d)\n", s / n, n }'
  else echo "llama-server $LS model $M sha256 $(sha256sum "$M" | cut -c1-16) --embeddings --pooling none embd_normalize -1"; fi
  echo "hid sha256 $(sha256sum "$C/dumps/$SRC-$SEL.hid" | cut -c1-16)"
} | tee "$D/receipt.txt"
[ $ok = 1 ] && echo "PASS dump $SRC-$SEL" || { echo "FAIL dump $SRC-$SEL: file sizes do not match the index"; exit 1; }
