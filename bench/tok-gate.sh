#!/usr/bin/env bash
# Tokenizer correctness gate: serve/tokenizer.mojo against llama.cpp and HF tokenizers, on every GGUF on disk.
#
#   bench/tok-gate.sh              full gate: every model, four arms, then the mutant arm
#   bench/tok-gate.sh --quick      three vocab files only (qwen35, llama-bpe, llama-spm), no mutants
#   bench/tok-gate.sh --mutants    only the mutant arm: each broken tokenizer must FAIL the quick gate
#   bench/tok-gate.sh --models F   model list from file F (one GGUF path per line)
#
# Arms, per distinct tokenizer (models with equal tokenizer.* keys share one run, on the smallest file):
#   llama   ids with specials == llama-tokenize -f CASE --no-escape --ids            (every case)
#   hf      ids without BOS   == HF tokenizers on a tokenizer.json found on disk whose vocab equals the GGUF's
#           (NOT-EXERCISED when no such file is on disk: the gate never builds the json with our own code)
#   round   decode(encode(case), keep_special) == the case bytes
#   pieces  decode([id]) bytes of EVERY id == llama.cpp token_to_piece rules, computed by the oracle from the
#           GGUF (covers tokens that are partial, invalid UTF-8 on their own)
# Every model file must also load (info). Known, explained deviations: bench/tok-gate-known.tsv.
# Cases: bench/tok-corpus/whole/* as they are, bench/tok-corpus/lines/* whole and one case per line.
# Env: TOK_BIN (prebuilt baro-tokenize), LLAMA_TOKENIZE, TOK_GATE_WORK, TOK_GATE_OUT (receipt tsv), JOBS.
# Exit 0 only when every model has verdict PASS and (full mode) every mutant was killed.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${TOK_GATE_WORK:-.work/tok-gate}
LT=${LLAMA_TOKENIZE:-$HOME/Models/llama.cpp/build/bin/llama-tokenize}
PY=.venv/bin/python; ORA=tools/tok-gate-oracle.py; KNOWN=bench/tok-gate-known.tsv
JOBS=${JOBS:-8}; QUICK=0; MUT=1; ONLY_MUT=0; MODELS=""
while [ $# -gt 0 ]; do case "$1" in
  --quick) QUICK=1; MUT=0;; --mutants) ONLY_MUT=1;; --models) MODELS=$2; shift;;
  *) echo "FAIL args: unknown $1"; exit 2;; esac; shift; done
[ -x "$LT" ] || { echo "FAIL oracle: no llama-tokenize at $LT"; exit 2; }
$PY -c 'import tokenizers' 2>/dev/null || { echo "FAIL oracle: $PY has no tokenizers module"; exit 2; }
mkdir -p "$W"

build() {  # build SRC_DIR OUT: baro-tokenize against the tokenizer.mojo in SRC_DIR
  rm -f "$2"
  ./.venv/bin/mojo build tools/baro-tokenize.mojo -I . -I "$1" -o "$2" > "$2.log" 2>&1 || { echo "FAIL build: see $2.log"; exit 2; }
}

mutants() {
  local killed=0 n=0 m d
  for m in drop-merge wrong-pre no-specials decode-drop-byte spm-score; do
    n=$((n + 1)); d=$W/mut/$m; mkdir -p "$d"
    case $m in
      drop-merge)       sed 's/^\( *\)self.merge_of\[(a << 32) | b\] = (i << 32) | ab$/\1if i != 300:\n\1    self.merge_of[(a << 32) | b] = (i << 32) | ab/' serve/tokenizer.mojo > "$d/tokenizer.mojo";;
      wrong-pre)        sed 's/\[\\p{L}\\p{M}\]+|\\p{N}| ?\[^\\s\\p{L}\\p{M}\\p{N}\]/\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]/' serve/tokenizer.mojo > "$d/tokenizer.mojo";;  # qwen35 gets the qwen2 regex (no \p{M}); a digit-run mutant is equivalent here, Qwen vocabs have no multi-digit token
      no-specials)      sed 's/^\( *\)if not self.special_first\[Int(tb\[at\])\]:$/\1if True:/' serve/tokenizer.mojo > "$d/tokenizer.mojo";;
      decode-drop-byte) sed 's/^\( *\)if b >= 0:$/\1if b >= 0 and b != 195:/' serve/tokenizer.mojo > "$d/tokenizer.mojo";;
      spm-score)        sed 's/return self.score\[a\] > self.score\[b\] or/return self.score[a] < self.score[b] or/' serve/tokenizer.mojo > "$d/tokenizer.mojo";;
    esac
    cmp -s serve/tokenizer.mojo "$d/tokenizer.mojo" && { echo "FAIL mutant $m: the edit changed nothing"; exit 2; }
    build "$d" "$d/baro-tokenize"
    if TOK_BIN=$d/baro-tokenize TOK_GATE_WORK=$d/run TOK_GATE_OUT=$d/gate.tsv bash "$0" --quick > "$d/gate.log" 2>&1; then
      echo "MUTANT $m SURVIVED (see $d/gate.log)"
    else
      grep -q '^FAIL tok-gate:' "$d/gate.log" || { echo "FAIL mutant $m: gate died without a verdict, see $d/gate.log"; exit 2; }
      killed=$((killed + 1)); echo "MUTANT $m killed: $(grep -c 'FAIL$' "$d/gate.tsv" || true) model rows FAIL"
    fi
  done
  [ "$killed" = "$n" ] && echo "PASS mutants: $killed/$n killed" || { echo "FAIL mutants: $killed/$n killed"; return 1; }
}
if [ $ONLY_MUT = 1 ]; then mutants; exit $?; fi

BIN=${TOK_BIN:-}
if [ -z "$BIN" ]; then BIN=$W/baro-tokenize; build serve "$BIN"; fi

# cases
C=$W/cases; rm -rf "$C"; mkdir -p "$C"
for f in bench/tok-corpus/whole/* bench/tok-corpus/lines/*; do cp "$f" "$C/w-$(basename "$f")"; done
for f in bench/tok-corpus/lines/*; do split -l 1 -a 3 -d "$f" "$C/l-$(basename "$f" .txt)-"; done
ls "$C"/* | sort > "$W/cases.txt"; NC=$(wc -l < "$W/cases.txt")
[ "$NC" -ge 50 ] || { echo "FAIL cases: only $NC"; exit 2; }

# models
if [ -n "$MODELS" ]; then cp "$MODELS" "$W/models.txt"
else  # mmproj and tts files carry no text tokenizer of ours
  find "$HOME/Models" "$HOME/.cache/baro" -name '*.gguf' -size +100k 2>/dev/null \
    | grep -v -e mmproj -e '/tts/' -e '-0000[2-9]-of-' | sort > "$W/models.txt"
fi
[ $QUICK = 1 ] && { grep -E 'ggml-vocab-(qwen35|llama-bpe|llama-spm)\.gguf$' "$W/models.txt" > "$W/models.q" || true; mv "$W/models.q" "$W/models.txt"; }
NM=$(wc -l < "$W/models.txt"); [ "$NM" -ge 1 ] || { echo "FAIL models: none found"; exit 2; }

# fingerprints (cached by path, size, mtime)
mkdir -p "$W/fp"; : > "$W/index.tsv"
while read -r g; do
  key=$(printf '%s' "$g $(stat -c '%s %Y' "$g")" | sha1sum | cut -c1-16)
  [ -s "$W/fp/$key" ] || $PY $ORA fp "$g" > "$W/fp/$key" 2> "$W/fp/$key.err" || printf 'ERR\t-\t-\t0\t-\n' > "$W/fp/$key"
  printf '%s\t%s\t%s\n' "$(cat "$W/fp/$key")" "$(stat -c %s "$g")" "$g" >> "$W/index.tsv"
done < "$W/models.txt"

known() {  # known FP PRE ARM -> allowed mismatch count (0 when no line)
  [ -f "$KNOWN" ] || { echo 0; return; }
  awk -F'\t' -v fp="$1" -v pre="$2" -v arm="$3" '$1 !~ /^#/ && ($1 == fp || $1 == pre) && $2 == arm {n = $3} END {print n + 0}' "$KNOWN"
}
miss() {  # miss OURS REF -> number of differing lines (line count mismatch counts every line)
  if [ "$(wc -l < "$1")" != "$(wc -l < "$2")" ]; then wc -l < "$2"; return; fi
  awk 'NR == FNR {a[FNR] = $0; next} a[FNR] != $0 {n++} END {print n + 0}' "$1" "$2"
}

mapfile -t JSONS < <(find "$HOME/Models" "$HOME/.cache/baro" -name tokenizer.json -not -path '*/tts/*' 2>/dev/null | sort)
jkey=$(stat -c '%n %s %Y' "${JSONS[@]}" | sha1sum | cut -c1-16)
[ -s "$W/jsonfp-$jkey.tsv" ] || $PY $ORA jsonfp "${JSONS[@]}" > "$W/jsonfp-$jkey.tsv"
for fp in $(cut -f1 "$W/index.tsv" | grep -v -e '^-$' -e '^ERR$' | sort -u); do
  D=$W/g/$fp; rm -rf "$D"; mkdir -p "$D/llama"
  rep=$(awk -F'\t' -v fp="$fp" '$1 == fp' "$W/index.tsv" | sort -t$'\t' -k6,6n | head -1 | cut -f7)
  vfp=$(awk -F'\t' -v fp="$fp" '$1 == fp {print $5; exit}' "$W/index.tsv")
  pre=$(awk -F'\t' -v fp="$fp" '$1 == fp {print $3; exit}' "$W/index.tsv")
  echo "$rep" > "$D/rep"
  if ! { "$BIN" ids "$W/cases.txt" "$rep" > "$D/ours.ids" && "$BIN" ids-raw "$W/cases.txt" "$rep" > "$D/ours.raw" \
      && "$BIN" roundtrip "$W/cases.txt" "$rep" > "$D/ours.rt" && "$BIN" pieces x "$rep" > "$D/ours.pieces"; } 2> "$D/ours.err"; then
    echo "ours-error: $(tail -1 "$D/ours.err" | cut -c1-160)" > "$D/result"; continue
  fi
  # llama arm; a file llama-tokenize cannot open (a draft head with no context of its own) has no llama oracle
  loads=1; "$LT" -m "$rep" -p hi --ids --log-disable 2>/dev/null | tail -1 | grep -q '^\[' || loads=0
  [ $loads = 0 ] || nl -ba -w4 -nrz "$W/cases.txt" | xargs -P "$JOBS" -L 1 sh -c '"$0" -m "$1" -f "$4" --no-escape --ids --log-disable 2>/dev/null | tail -1 | tr -d "[]," > "$2/llama/$3"' "$LT" "$rep" "$D" || true
  : > "$D/llama.ids"; for i in $(seq -f '%04g' 1 "$NC"); do { cat "$D/llama/$i" 2>/dev/null || true; } | head -1 | sed 's/ *$//' > "$D/one"; [ -s "$D/one" ] || echo > "$D/one"; cat "$D/one" >> "$D/llama.ids"; done
  ml=$(miss "$D/ours.ids" "$D/llama.ids"); kl=$(known "$fp" "$pre" llama); [ $loads = 1 ] || ml=-
  # hf arm
  hfj=$(awk -F'\t' -v v="$vfp" '$1 == v {print $2; exit}' "$W/jsonfp-$jkey.tsv"); mh=-; kh=0
  if [ -z "$hfj" ]; then  # a json shorter than the GGUF vocab matches when the GGUF only adds UNUSED padding after it
    ntok=$(awk -F'\t' -v fp="$fp" '$1 == fp {print $4; exit}' "$W/index.tsv")
    for n in $(awk -F'\t' -v m="$ntok" '$3 > 0 && $3 < m {print $3}' "$W/jsonfp-$jkey.tsv" | sort -un); do
      p=$($PY $ORA prefixfp "$rep" "$n" 2>/dev/null || true)
      [ "${p#*$'\t'}" = unused-tail ] && hfj=$(awk -F'\t' -v v="${p%%$'\t'*}" -v n="$n" '$1 == v && $3 == n {print $2; exit}' "$W/jsonfp-$jkey.tsv")
      [ -z "$hfj" ] || break
    done
  fi
  echo "$hfj" > "$D/hf.json.path"
  if [ -n "$hfj" ]; then
    if $PY $ORA hf "$hfj" "$W/cases.txt" > "$D/hf.ids" 2> "$D/hf.err"; then mh=$(miss "$D/ours.raw" "$D/hf.ids"); else mh=$NC; fi
    kh=$(known "$fp" "$pre" hf)
  fi
  # round trip and pieces
  mr=$(grep -vc '^OK$' "$D/ours.rt" || true); [ "$(wc -l < "$D/ours.rt")" = "$NC" ] || mr=$NC; kr=$(known "$fp" "$pre" round)
  if $PY $ORA pieces "$rep" > "$D/ref.pieces" 2> "$D/pieces.err"; then mp=$(miss "$D/ours.pieces" "$D/ref.pieces"); else mp=999999; fi
  kp=$(known "$fp" "$pre" pieces)
  v=PASS
  [ "$mr" -le "$kr" ] || v=FAIL; [ "$mp" -le "$kp" ] || v=FAIL
  if [ "$ml" != - ]; then [ "$ml" -le "$kl" ] || v=FAIL; fi
  if [ "$mh" != - ]; then [ "$mh" -le "$kh" ] || v=FAIL; fi
  [ "$ml" != - ] || [ "$mh" != - ] || v=FAIL  # no id oracle at all is not a pass
  hfs="NOT-EXERCISED"; [ "$mh" = - ] || hfs="$((NC - mh))/$NC"
  lls="NOT-EXERCISED"; [ "$ml" = - ] || lls="$((NC - ml))/$NC"
  dev=""; [ "$kl$kh$kr$kp" = 0000 ] || dev=" (known: llama $kl hf $kh round $kr pieces $kp)"
  printf 'llama %s\thf %s\tround %s/%s\tpieces-diff %s%s\t%s' "$lls" "$hfs" "$((NC - mr))" "$NC" "$mp" "$dev" "$v" > "$D/result"
done

# per-model table
OUT=${TOK_GATE_OUT:-$W/gate.tsv}; : > "$OUT"; pass=0; fail=0; skip=0
while IFS=$'\t' read -r fp model pre ntok vfp size g; do
  if [ "$fp" = - ]; then printf '%s\t-\t-\tno tokenizer in file\tSKIP\n' "$g" >> "$OUT"; skip=$((skip + 1)); continue; fi
  if [ "$fp" = ERR ]; then printf '%s\t-\t-\toracle cannot read the header\tFAIL\n' "$g" >> "$OUT"; fail=$((fail + 1)); continue; fi
  if [ "$(known "$fp" "$model" unsupported)" != 0 ]; then
    if "$BIN" info x "$g" > "$W/info.out" 2>&1; then printf '%s\t%s\t%s/%s\tlisted unsupported but loads: delete the line in %s\tFAIL\n' "$g" "$fp" "$model" "$pre" "$KNOWN" >> "$OUT"; fail=$((fail + 1))
    else printf '%s\t%s\t%s/%s\tUNSUPPORTED family, named load error (%s)\tSKIP\n' "$g" "$fp" "$model" "$pre" "$KNOWN" >> "$OUT"; skip=$((skip + 1)); fi
    continue
  fi
  if ! "$BIN" info x "$g" > "$W/info.out" 2>&1; then
    printf '%s\t%s\t%s/%s\tLOAD: %s\tFAIL\n' "$g" "$fp" "$model" "$pre" "$(tail -1 "$W/info.out" | sed 's/.*execution: //' | cut -c1-120)" >> "$OUT"; fail=$((fail + 1)); continue
  fi
  r=$(cat "$W/g/$fp/result" 2>/dev/null || echo "no verdict")
  case "$r" in *PASS) pass=$((pass + 1));; *FAIL) fail=$((fail + 1));; *) r="$r"$'\t'FAIL; fail=$((fail + 1));; esac
  printf '%s\t%s\t%s/%s\t%s\n' "$g" "$fp" "$model" "$pre" "$r" >> "$OUT"
done < "$W/index.tsv"
sed "s|$HOME/||" "$OUT" | cut -f1,3- | column -t -s$'\t' | cut -c1-230
echo "cases $NC  models $NM  tokenizers $(cut -f1 "$W/index.tsv" | grep -v -e '^-$' -e '^ERR$' | sort -u | wc -l)  pass $pass  fail $fail  skip(no tokenizer or unsupported family) $skip  receipt $OUT"
[ $((pass + fail + skip)) = "$NM" ] || { echo "FAIL tok-gate: verdict count $((pass + fail + skip)) != models $NM"; exit 1; }
[ "$pass" -ge 1 ] || { echo "FAIL tok-gate: nothing passed"; exit 1; }
[ "$fail" = 0 ] || { echo "FAIL tok-gate: $fail model(s)"; exit 1; }
if [ $MUT = 1 ]; then mutants || { echo "FAIL tok-gate: mutant arm"; exit 1; }; fi
echo "PASS tok-gate: $pass/$((pass + fail)) models, $NC cases"
