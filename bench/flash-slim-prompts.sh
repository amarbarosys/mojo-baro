#!/usr/bin/env bash
# Flash-slim round 1 inputs (bench/flash-slim-protocol.md): 8 categories, disjoint trace and eval documents,
# tokenized with our tokenizer from the Flash GGUF, cut to 1024 trace and 256 eval tokens per category.
# Writes OUT/trace.ids (8 x 1024 ids, one per line, category order below), OUT/eval.ids (8 x 256) and
# OUT/manifest.txt (documents used, token counts, sha256 of every id file). CPU only.
# usage: bench/flash-slim-prompts.sh OUT_DIR [trace2|decode]
# decode (bench/flash-slim-decode-protocol.md): 10 skills + 10 papers documents in no trace or eval batch, from token
# 256 on: OUT/decode/NN.prompt (64 ids, spaces) and NN.force (next 64 ids, commas), OUT/manifest-decode.txt.
# trace2 (round 2, bench/flash-slim-r2-protocol.md): 16 skills + 16 papers sequences of 256 tokens, one document run
# each, spread evenly over the corpus, never an eval document; OUT/trace2-b1..b4.ids (4 skills + 4 papers each) and
# OUT/manifest2.txt. Round 1 outputs are not touched.
set -euo pipefail
export LC_ALL=C
OUT=${1:?out dir}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
TOK=${BARO_TOKENIZE:-$REPO/.work/baro-tokenize}
PAPERS=$HOME/Projects/AiTrainer/experiments/mojo-baro/work/research-paper-corpus/chunks-capped.jsonl
DOCS=$HOME/Projects/AiTrainer/datasets/external/python-3.14-docs-text
CATS="skills papers notes mojo python math prose docs"
NT=1024; NE=256
# mojo and python documents come from a pinned tree (default: the round-1 tree in the pre-squash archive,
# which reproduces the round-1 manifest sha256), so edits and deletions on main cannot move the corpus
SRC=${FLASH_SLIM_SRC:-$HOME/Projects/mojo/mojo-baro-archive}; REV=${FLASH_SLIM_REV:-b78da09}; SNAP=$OUT/repo-$REV
mkdir -p "$OUT/text"
[ -d "$SNAP" ] || { mkdir -p "$SNAP" && git -C "$SRC" archive "$REV" serve tools | tar -x -C "$SNAP"; } || { echo "FAIL rev: cannot archive $REV"; exit 1; }
[ -x "$TOK" ] || { echo "FAIL tokenizer: $TOK missing (build tools/baro-tokenize.mojo)"; exit 1; }

# docs CAT: one document path or record per line, sorted, before the index filter
files() {
  case $1 in
    skills) find -L "$HOME/Brain/Skills" -name SKILL.md -not -path '*/.git/*' -print0 | xargs -0 realpath | sort -u ;;
    notes) ls "$HOME"/Brain/mojo/mojo-baro/2026-*.md | sort ;;
    mojo) git -C "$SRC" ls-tree -r --name-only "$REV" | grep -E '^serve/.*\.mojo$' | sort | sed "s|^|$SNAP/|" ;;
    python) git -C "$SRC" ls-tree -r --name-only "$REV" | grep -E '^tools/.*\.py$' | sort | sed "s|^|$SNAP/|" ;;
    docs) find "$DOCS" -type f -name '*.txt' | sort ;;
  esac
}
# text CAT MOD REM BYTES OUT: selected documents (index % MOD == REM) concatenated until BYTES are reached, cut there
text() {
  local cat=$1 mod=$2 rem=$3 want=$4 out=$5
  case $cat in
    papers) awk -v m="$mod" -v r="$rem" 'match($0, /"page_id": "[^:"]*/) { id = substr($0, RSTART + 12, RLENGTH - 12)
              if (!(id in ix)) ix[id] = n++; if (ix[id] % m == r) { print; if (++c >= 400) exit } }' "$PAPERS" \
              | jq -r '.text + "\n"' > "$out.full" ;;
    math) awk -v m="$mod" -v r="$rem" '(NR - 1) % m == r { print; if (++c >= 400) exit }' "$HOME/Models/datasets/gsm8k/main/test.jsonl" \
            | jq -r '"Question: " + .question + "\nAnswer: " + .answer + "\n"' > "$out.full" ;;
    prose) awk -v m="$mod" -v r="$rem" '/^ = [^=].* = $/ { n++ } n > 0 && (n - 1) % m == r' "$HOME/Models/quant-lab/wikitext-2-raw/wiki.test.raw" > "$out.full" ;;
    *) local i=0 have=0
       : > "$out.full"; : > "$out.docs"
       while IFS= read -r f && [ "$have" -lt "$want" ]; do
         if [ $((i % mod)) -eq "$rem" ]; then
           cat "$f" >> "$out.full"; printf '\n\n' >> "$out.full"; echo "$f" >> "$out.docs"
           have=$(stat -c %s "$out.full")
         fi
         i=$((i + 1))
       done < <(files "$cat") ;;
  esac
  head -c "$want" "$out.full" > "$out"
  rm -f "$out.full"
}
# seq2 CAT J FILE...: sequence J of trace2 from the listed documents (concatenated until 3072 bytes), 256 ids
seq2() {
  local cat=$1 j=$2; shift 2
  local t=$OUT/trace2/$cat-$j
  : > "$t.txt"; : > "$t.docs"
  for f in "$@"; do
    [ "$(stat -c %s "$t.txt")" -ge 3072 ] && break
    cat "$f" >> "$t.txt"; printf '\n\n' >> "$t.txt"; echo "$f" >> "$t.docs"
  done
  "$TOK" encode "$t.txt" "$GGUF" > "$t.all" 2> "$t.log" || { echo "FAIL tokenize trace2 $cat $j: see $t.log"; exit 1; }
  [ "$(wc -l < "$t.all")" -ge 256 ] || { echo "FAIL trace2 $cat $j: $(wc -l < "$t.all") tokens < 256"; exit 1; }
  head -n 256 "$t.all" > "$t.ids"
  echo "$cat $j docs $(wc -l < "$t.docs") first $(head -1 "$t.docs") sha256 $(sha256sum "$t.ids" | cut -c1-16)" >> "$OUT/manifest2.txt"
}
# dec1 NAME TEXTFILE: prompt/force pair from tokens 257..320 / 321..384, or return 1 if the text is shorter
dec1() {
  local d=$OUT/decode
  "$TOK" encode "$2" "$GGUF" > "$d/$1.all" 2> "$d/$1.log" || { echo "FAIL tokenize decode $1: see $d/$1.log"; exit 1; }
  [ "$(wc -l < "$d/$1.all")" -ge 384 ] || return 1
  sed -n 257,320p "$d/$1.all" | tr '\n' ' ' | sed 's/ $//' > "$d/$1.prompt"
  sed -n 321,384p "$d/$1.all" | paste -sd, > "$d/$1.force"
}
if [ "${2:-}" = decode ]; then
  d=$OUT/decode; mkdir -p "$d"; rm -f "$d"/*.prompt "$d"/*.force; : > "$OUT/manifest-decode.txt"
  n=0
  while IFS= read -r f; do
    [ $n -ge 10 ] && break
    printf -v nm 's%02d' $n
    if dec1 "$nm" "$f"; then echo "$nm $f" >> "$OUT/manifest-decode.txt"; n=$((n + 1)); fi
  done < <(files skills | awk '(NR - 1) % 7 == 3' | tail -n +2)
  [ $n = 10 ] || { echo "FAIL decode: $n of 10 skills documents reach 384 tokens"; exit 1; }
  mapfile -t PID < <(awk 'match($0, /"page_id": "[^:"]*/) { id = substr($0, RSTART + 12, RLENGTH - 12); if (!(id in s)) { s[id] = 1; if ((n++) % 50 == 25) print id } }' "$PAPERS" | tail -n +2 | head -n 20)
  awk -v pick="${PID[*]}" -v dir="$d" 'BEGIN { n = split(pick, p, " "); for (k = 1; k <= n; k++) want[p[k]] = k - 1 }
    match($0, /"page_id": "[^:"]*/) { id = substr($0, RSTART + 12, RLENGTH - 12); if ((id in want) && c[id]++ < 10) print > (dir "/pchunks-" want[id] ".jsonl") }' "$PAPERS"
  n=0
  for k in $(seq 0 19); do
    [ $n -ge 10 ] && break
    jq -r '.text + "\n"' "$d/pchunks-$k.jsonl" > "$d/paper-$k.src"
    printf -v nm 'p%02d' $n
    if dec1 "$nm" "$d/paper-$k.src"; then echo "$nm paper ${PID[$k]}" >> "$OUT/manifest-decode.txt"; n=$((n + 1)); fi
  done
  [ $n = 10 ] || { echo "FAIL decode: $n of 10 papers reach 384 tokens"; exit 1; }
  for f in "$d"/*.prompt "$d"/*.force; do echo "$(basename "$f") sha256 $(sha256sum "$f" | cut -c1-16)" >> "$OUT/manifest-decode.txt"; done
  cat "$OUT/manifest-decode.txt" | head -20
  echo "PASS decode 20/20 prompt/force pairs in $d"
  exit 0
fi
if [ "${2:-}" = trace2 ]; then
  mkdir -p "$OUT/trace2"; : > "$OUT/manifest2.txt"
  mapfile -t SK < <(files skills | awk '(NR - 1) % 7 != 3')
  for j in $(seq 0 15); do i=$((j * ${#SK[@]} / 16)); seq2 skills $j "${SK[@]:$i:20}"; done
  mapfile -t PID < <(awk 'match($0, /"page_id": "[^:"]*/) { id = substr($0, RSTART + 12, RLENGTH - 12); if (!(id in s)) { s[id] = 1; print id } }' "$PAPERS")
  pick=""
  for j in $(seq 0 15); do i=$((j * ${#PID[@]} / 16)); [ $((i % 50)) = 25 ] && i=$((i + 1)); pick="$pick ${PID[$i]}"; done
  awk -v pick="$pick" -v dir="$OUT/trace2" 'BEGIN { n = split(pick, p, " "); for (k = 1; k <= n; k++) want[p[k]] = k - 1 }
    match($0, /"page_id": "[^:"]*/) { id = substr($0, RSTART + 12, RLENGTH - 12); if ((id in want) && c[id]++ < 6) print > (dir "/paperchunks-" want[id] ".jsonl") }' "$PAPERS"
  for j in $(seq 0 15); do jq -r '.text + "\n"' "$OUT/trace2/paperchunks-$j.jsonl" > "$OUT/trace2/paper-$j.src"; seq2 papers $j "$OUT/trace2/paper-$j.src"; done
  for b in 1 2 3 4; do
    : > "$OUT/trace2-b$b.ids"
    for cat in skills papers; do for j in $(seq $((4 * b - 4)) $((4 * b - 1))); do cat "$OUT/trace2/$cat-$j.ids" >> "$OUT/trace2-b$b.ids"; done; done
    echo "trace2-b$b.ids lines $(wc -l < "$OUT/trace2-b$b.ids") sha256 $(sha256sum "$OUT/trace2-b$b.ids" | cut -c1-16)" >> "$OUT/manifest2.txt"
  done
  echo "sequence order per batch: skills x4, papers x4" >> "$OUT/manifest2.txt"
  cat "$OUT/manifest2.txt"
  echo "PASS trace2 32/32 sequences in $OUT"
  exit 0
fi
mod() { case $1 in mojo|math|prose) echo 2 ;; papers) echo 50 ;; *) echo 7 ;; esac; }
rem() { case $1 in papers) echo "$2" | sed 's/^0$/0/;s/^1$/25/' ;; mojo|math|prose) echo "$2" ;; *) [ "$2" = 0 ] && echo 0 || echo 3 ;; esac; }

: > "$OUT/trace.ids"; : > "$OUT/eval.ids"; : > "$OUT/manifest.txt"
fail=0; names=""
for cat in $CATS; do
  for part in trace eval; do
    if [ $part = trace ]; then n=$NT; side=0; else n=$NE; side=1; fi
    m=$(mod $cat); r=$(rem $cat $side)
    text $cat "$m" "$r" $((n * 12)) "$OUT/text/$cat.$part.txt"
    "$TOK" encode "$OUT/text/$cat.$part.txt" "$GGUF" > "$OUT/text/$cat.$part.all" 2> "$OUT/text/$cat.$part.log" \
      || { echo "FAIL tokenize $cat $part: see $OUT/text/$cat.$part.log"; exit 1; }
    got=$(wc -l < "$OUT/text/$cat.$part.all")
    if [ "$got" -lt "$n" ]; then fail=$((fail + 1)); names="$names $cat.$part($got)"; continue; fi
    head -n "$n" "$OUT/text/$cat.$part.all" > "$OUT/text/$cat.$part.ids"
    cat "$OUT/text/$cat.$part.ids" >> "$OUT/$part.ids"
    echo "$cat $part mod $m rem $r bytes $(stat -c %s "$OUT/text/$cat.$part.txt") tokens_available $got kept $n sha256 $(sha256sum "$OUT/text/$cat.$part.ids" | cut -c1-16)" >> "$OUT/manifest.txt"
  done
done
if [ "$fail" -gt 0 ]; then echo "FAIL $fail/16 parts short of tokens:$names"; exit 1; fi
for f in trace eval; do echo "$f.ids lines $(wc -l < "$OUT/$f.ids") sha256 $(sha256sum "$OUT/$f.ids" | cut -c1-16)" >> "$OUT/manifest.txt"; done
echo "categories (sequence order) $CATS" >> "$OUT/manifest.txt"
cat "$OUT/manifest.txt"
echo "PASS 16/16 parts in $OUT"
