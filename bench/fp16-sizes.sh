#!/usr/bin/env bash
# usage: [REPEATS=n] [SPREAD_MAX=f] bench/fp16-sizes.sh <bench.mojo> <tag> [sizes...]
# Patches comptime M/N/K in <bench.mojo>, builds once and runs REPEATS times per
# size, appends the median sample's JSON to .work/fp16-<tag>.jsonl, restores the
# source. REPEATS defaults to 3: one sample cannot detect its own
# contamination, and the noise on this card is one-sided (see bench/fp16-median.py).
set -eu; cd "$(dirname "$0")/.."
src=$1; tag=$2; shift 2; sizes=${@:-512 2048 4096}
repeats=${REPEATS:-3}; spread_max=${SPREAD_MAX:-0.02}
S=$PWD/.work/shim-build
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10
orig=$(mktemp -p .work); cp "$src" "$orig"; trap 'cp "$orig" "$src"; rm -f "$orig"' EXIT
for s in $sizes; do
  sed -i -E "s/^comptime (M|N|K) = [0-9]+$/comptime \1 = $s/" "$src"
  ./.venv/bin/mojo build "$src" -o .work/fp16_${tag}_$s -I kernels \
    -Xlinker -L"$S" -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker "$S" 2>&1 | grep -E "error" -A3 || true
  smp=.work/fp16-$tag-$s.samples; : > "$smp"
  for ((i=1; i<=repeats; i++)); do ./.work/fp16_${tag}_$s >> "$smp"; done
  row=$(bench/fp16-median.py "$smp" "$spread_max")   # non-zero exit here aborts the sweep (set -e)
  printf '%s\n' "$row" | tee -a .work/fp16-$tag.jsonl
done
