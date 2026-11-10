#!/usr/bin/env bash
# usage: bench/family-gate.sh GGUF PACK OUTDIR
# One command from a dense GGUF to a gated, tuned spark engine (bench/family-gen-protocol.md):
#   1 llama-eval-callback on 1 token, CPU (the op graph llama.cpp builds for this arch)
#   2 tools/gen-profile.mojo --recipe: activation from the ops, coverage check (exit 1 names every missing template)
#   3 tools/tune-family.sh: split-K knobs raced on the card, OUTDIR/tune/flags
#   4 bench/dense-run.sh with DEFS = the flags: forced agreement vs llama.cpp on the 20 prompts + tok/s_gen
# EXPLORE=1 caps the verdict at UNVERIFIED (exit 3). PACK must exist (tools/engine-pack.py GGUF PACK --dense).
set -euo pipefail
cd "$(dirname "$0")/.."
gguf=$1 pack=$2 out=$3
fail() { echo "FAIL $1: $2" | tee -a "$out/SUMMARY.txt"; exit 1; }
mkdir -p "$out/prof"; : > "$out/SUMMARY.txt"
[ -f "$pack/index.txt" ] || fail pack "no $pack/index.txt"
printf 'Hi' > "$out/one.txt"
HIP_VISIBLE_DEVICES= ~/Projects/imports/llama.cpp/build/bin/llama-eval-callback -m "$gguf" -ngl 0 -c 512 -n 1 --temp 0 \
  -f "$out/one.txt" > "$out/evalcb.log" 2>&1 || fail evalcb "$(tail -1 "$out/evalcb.log")"
[ -x .work/gen-profile ] || ./.venv/bin/mojo build tools/gen-profile.mojo -I tools -o .work/gen-profile > "$out/genprofile-build.log" 2>&1 \
  || fail genprofile-build "see $out/genprofile-build.log"
.work/gen-profile "$gguf" "$out/prof/profile.mojo" --tmax 16384 --recipe "$out/evalcb.log" > "$out/recipe.log" 2>&1 \
  || { grep FAIL "$out/recipe.log" | tee -a "$out/SUMMARY.txt"; fail recipe "$(tail -1 "$out/recipe.log")"; }
grep -q "WARN" "$out/recipe.log" && fail recipe "$(grep WARN "$out/recipe.log")"
echo "PASS recipe: $(grep '^recipe:' "$out/recipe.log")" | tee -a "$out/SUMMARY.txt"
# a bake that passed this gate carries baro.tune.* (tools/gguf-embed.py --gen-only): same card and same spark/kernels
# commit -> its flags, no search. The flags are data: only BARO_SP_* defines are accepted, and every gate below re-runs.
kv() { gguf-kv "$gguf" 2>/dev/null | sed -n "s/^$1 *'\(.*\)'\$/\1/p"; }
sc=$(git log -1 --format=%h -- serve/spark.mojo kernels)
mkdir -p "$out/tune"
if [ "$(kv baro.tune.arch)" = gfx1100 ] && [ "$(kv baro.tune.commit)" = "$sc" ] && kv baro.tune.flags | grep -qxE '( ?-DBARO_SP_[A-Z]+_S=[0-9]+)*'; then
  kv baro.tune.flags > "$out/tune/flags"; echo "from bake: baro.tune.* (spark commit $sc)" > "$out/tune/summary.txt"
  echo "PASS tune: flags [$(cat "$out/tune/flags")] from the bake, search skipped" | tee -a "$out/SUMMARY.txt"
else
  tools/tune-family.sh "$gguf" "$pack" "$out/prof" "$out/tune" > "$out/tune.log" 2>&1 || fail tune "$(grep -m1 FAIL "$out/tune.log")"
  echo "PASS tune: flags [$(cat "$out/tune/flags")]" | tee -a "$out/SUMMARY.txt"
fi
DEFS="$(cat "$out/tune/flags")" gpu-wait run --vram 22 --timeout 3600 -- env DEFS="$(cat "$out/tune/flags")" LLAMA_CACHE="$PWD/.work/refcache-dense" bench/dense-run.sh "$gguf" "$pack" "$out/prof" "$out/dense" \
  > "$out/dense.log" 2>&1 || fail dense "$(grep -m1 FAIL "$out/dense/SUMMARY.txt" 2>/dev/null || tail -1 "$out/dense.log")"
cat "$out/dense/SUMMARY.txt" >> "$out/SUMMARY.txt"
lb=$HOME/Projects/imports/llama.cpp/build/bin/llama-bench
gpu-wait run --vram 22 --timeout 600 -- "$lb" -m "$gguf" -p 0 -n 64 -ngl 99 -fa 1 -r 3 -o csv > "$out/llama-bench.csv" 2> "$out/llama-bench.err" \
  || fail llama-bench "$(tail -1 "$out/llama-bench.err")"
ours=$(awk '$3 ~ /^[0-9.]+$/ { print $3 }' "$out/dense/results.txt" | sort -g | awk '{ v[NR] = $1 } END { print v[int((NR + 1) / 2)] }')
theirs=$(awk -F, '!c && /avg_ts/ { for (i = 1; i <= NF; i++) if ($i == "avg_ts") c = i; next } c && NF > 20 { gsub(/"/, "", $c); print $c }' "$out/llama-bench.csv" | tail -1)
[ -n "$theirs" ] && [ -n "$ours" ] || fail speed "no number (ours '$ours', llama-bench '$theirs')"
echo "SPEED ours median $ours tok/s_gen (20 prompts) vs llama-bench tg64 $theirs = $(awk -v a="$ours" -v b="$theirs" 'BEGIN { printf "%.3fx", a / b }')" | tee -a "$out/SUMMARY.txt"
if [ -n "${EXPLORE:-}" ]; then echo "UNVERIFIED: EXPLORE=1" | tee -a "$out/SUMMARY.txt"; exit 3; fi
echo "PASS family-gate: $gguf" | tee -a "$out/SUMMARY.txt"
# BAKE=DST.gguf: write the recipe, profile and tune result back (a new file, the source is untouched)
[ -n "${BAKE:-}" ] && ./.venv/bin/python3 tools/gguf-embed.py "$gguf" "$BAKE" --gen-only="$out" | tee -a "$out/SUMMARY.txt"
exit 0
