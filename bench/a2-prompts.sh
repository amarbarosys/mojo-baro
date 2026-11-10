#!/usr/bin/env bash
# usage: bench/a2-prompts.sh [OUTDIR]        default .work/a2-prompts (deterministic, regenerated on demand)
# Shared-document prompt sets for the A2 gate (bench/a2-gate.sh). For L in 8192 16384 32768 the
# document is the first L-128 ids of bench/prefill-prompts/p32768.tokens and each of the 20
# bench/mtp-prompts questions is appended as ids (no re-tokenization, so the doc end is an exact
# prefix boundary). Every prompt of a set shares the document, and the three documents are prefixes
# of one another, so one 32k prefill serves all 60 prompts through prefix checkpoints (the gate
# sends a "ckpt" hint at the doc end). The 128-id margin keeps prompt + 64 generated under
# BARO_TMAX=L. Writes L<L>/pNN-name.tokens, L<L>/doclen, manifest.txt (sha256 of the inputs).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/a2-prompts}
doc=bench/prefill-prompts/p32768.tokens
[ -f "$doc" ] || { echo "FAIL a2-prompts: $doc missing"; exit 1; }
mkdir -p "$out"
sha256sum "$doc" bench/mtp-prompts/p*.tokens > "$out/manifest.txt"
for L in 8192 16384 32768; do
  d=$((L - 128)); mkdir -p "$out/L$L"; echo "$d" > "$out/L$L/doclen"
  awk -v n="$d" '{for (i = 1; i <= NF; i++) {c++; if (c <= n) printf "%s ", $i; if (c >= n) exit}}' "$doc" > "$out/L$L/doc.ids"
  [ "$(wc -w < "$out/L$L/doc.ids")" = "$d" ] || { echo "FAIL a2-prompts: doc prefix for L=$L is short"; exit 1; }
  for tf in bench/mtp-prompts/p*.tokens; do
    { cat "$out/L$L/doc.ids"; tr -s ' \n' ' ' < "$tf"; echo; } > "$out/L$L/$(basename "$tf")"
  done
  rm "$out/L$L/doc.ids"
done
echo "a2-prompts: 3 sets x $(ls bench/mtp-prompts/p*.tokens | wc -l) prompts in $out (doc ids $((8192-128))/$((16384-128))/$((32768-128)))"
