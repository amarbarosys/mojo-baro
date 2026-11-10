#!/usr/bin/env bash
# tools/spark-pack.mojo against its oracle tools/spark-pack.py: byte-identical pack.bin and index.txt on
#   1. a synthetic spark-shaped GGUF (tools/spark-pack-fixture.py) that hits Q8_0, Q4_K, Q5_K, Q6_K and F32, with a
#      K-quant and a Q8_0 token_embd (the f32 lookup and the q8 tied head),
#   2. the real model when given: tools/spark-pack-check.sh [MODEL.gguf]   (CPU only)
set -euo pipefail
cd "$(dirname "$0")/.."
w=.work/spark-pack-check; rm -rf "$w"; mkdir -p "$w"
./.venv/bin/mojo build --fp-mode contract=off tools/spark-pack.mojo -I serve -I tools -o .work/spark-pack > "$w/build.log" 2>&1 \
  || { echo "FAIL build: $(grep -m1 error: "$w/build.log")"; exit 1; }
fails=0; names=""
check() { # label gguf
  mkdir -p "$w/$1/py" "$w/$1/mojo"
  ./.venv/bin/python3 tools/spark-pack.py "$2" "$w/$1/py" > "$w/$1/py.log" 2>&1 || { echo "FAIL $1: oracle failed, $w/$1/py.log"; return 1; }
  .work/spark-pack "$2" "$w/$1/mojo" > "$w/$1/mojo.log" 2>&1 || { echo "FAIL $1: mojo failed, $w/$1/mojo.log"; return 1; }
  for f in pack.bin index.txt; do cmp -s "$w/$1/py/$f" "$w/$1/mojo/$f" || { echo "FAIL $1: $f differs"; return 1; }; done
  diff -q "$w/$1/py.log" "$w/$1/mojo.log" > /dev/null || { echo "FAIL $1: summary line differs"; return 1; }
  echo "PASS $1: $(cat "$w/$1/mojo.log")"
}
for seed in 7 11; do
  ./.venv/bin/python3 tools/spark-pack-fixture.py "$w/fix$seed.gguf" "$seed" > /dev/null
  check "fixture-$seed" "$w/fix$seed.gguf" || { fails=$((fails + 1)); names="$names fixture-$seed"; }
done
# Q8_0 token_embd: the fixture's first matrix type is fixed, so a second fixture swaps it in
sed 's/mat("token_embd.weight", 8, Q.Q5_K)/mat("token_embd.weight", 8, Q.Q8_0)/' tools/spark-pack-fixture.py > "$w/fixture-q8embd.py"
./.venv/bin/python3 "$w/fixture-q8embd.py" "$w/fixq8.gguf" 3 > /dev/null
check fixture-q8-embd "$w/fixq8.gguf" || { fails=$((fails + 1)); names="$names fixture-q8-embd"; }
if [ -n "${1:-}" ]; then check model "$1" || { fails=$((fails + 1)); names="$names model"; }; fi
[ $fails = 0 ] || { echo "FAIL $fails:$names"; exit 1; }
echo "PASS spark-pack-check"
