#!/usr/bin/env bash
# tools/engine-pack.mojo against its oracle tools/engine-pack.py: byte-identical pack.bin and index.txt, the same
# identity.json apart from pack_mtime_ns, the same stdout summary lines (numpy's divide-by-zero warnings go to
# stderr), all built strict IEEE (--fp-mode contract=off: a fused x * id + 8.5 moved q4 nibbles), on
#   1. tiny GGUFs from tools/engine-pack-fixture.py: the qwen35 trunk with --q8, --q4, --q2b3, --tq1, --tq2,
#      --q8 --q4-draft and no flag (bf16 transposed), three --dense shapes, --arch qwen35moe; then --identity after a tokenizer.json lands,
#      fresh and cached,
#   2. real models when given: tools/engine-pack-check.sh [MODEL.gguf "FLAGS"]...   (CPU only; wall s and peak RSS
#      of both tools go to $w/<label>/time.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
w=.work/engine-pack-check; rm -rf "$w"; mkdir -p "$w"
./.venv/bin/mojo build --fp-mode contract=off tools/engine-pack.mojo -I serve -I tools -o .work/engine-pack > "$w/build.log" 2>&1 \
  || { echo "FAIL build: $(grep -m1 error: "$w/build.log")"; exit 1; }
fails=0; names=""
same_identity() { # dirA dirB
  cmp -s <(grep -v pack_mtime_ns "$1/identity.json") <(grep -v pack_mtime_ns "$2/identity.json")
}
check() { # label gguf flags...
  local l=$1 g=$2; shift 2
  mkdir -p "$w/$l/py" "$w/$l/mojo"
  /usr/bin/time -f "py %e s %M KB" -o "$w/$l/time.txt" ./.venv/bin/python3 tools/engine-pack.py "$g" "$w/$l/py" "$@" \
    > "$w/$l/py.log" 2> "$w/$l/py.err" || { echo "FAIL $l: oracle failed, $w/$l/py.err"; return 1; }
  /usr/bin/time -a -f "mojo %e s %M KB" -o "$w/$l/time.txt" .work/engine-pack "$g" "$w/$l/mojo" "$@" \
    > "$w/$l/mojo.log" 2> "$w/$l/mojo.err" || { echo "FAIL $l: mojo failed, $w/$l/mojo.err"; return 1; }
  for f in pack.bin index.txt; do cmp -s "$w/$l/py/$f" "$w/$l/mojo/$f" || { echo "FAIL $l: $f differs"; return 1; }; done
  same_identity "$w/$l/py" "$w/$l/mojo" || { echo "FAIL $l: identity.json differs"; return 1; }
  diff -q "$w/$l/py.log" "$w/$l/mojo.log" > /dev/null || { echo "FAIL $l: summary lines differ"; return 1; }
  echo "PASS $l: $(head -1 "$w/$l/mojo.log") | $(tr '\n' ' ' < "$w/$l/time.txt")"
}
run() { check "$@" || { fails=$((fails + 1)); names="$names $1"; }; }
for seed in 7 11 12; do
  ./.venv/bin/python3 tools/engine-pack-fixture.py "$w/trunk$seed.gguf" trunk "$seed" > /dev/null
  run "trunk-$seed-q8" "$w/trunk$seed.gguf" --q8
  run "trunk-$seed-q4" "$w/trunk$seed.gguf" --q4
  run "trunk-$seed-q2b3" "$w/trunk$seed.gguf" --q2b3
  run "trunk-$seed-tq1" "$w/trunk$seed.gguf" --tq1
  run "trunk-$seed-tq2" "$w/trunk$seed.gguf" --tq2
  run "trunk-$seed-q8-draft" "$w/trunk$seed.gguf" --q8 --q4-draft
  run "trunk-$seed-bf16" "$w/trunk$seed.gguf"
done
for kind in dense-llama dense-qwen2 dense-bf16; do
  for seed in 7 8; do
    ./.venv/bin/python3 tools/engine-pack-fixture.py "$w/$kind$seed.gguf" "$kind" "$seed" > /dev/null
    run "$kind-$seed" "$w/$kind$seed.gguf" --dense
  done
done
for seed in 7 8; do
  ./.venv/bin/python3 tools/engine-pack-fixture.py "$w/moe$seed.gguf" moe "$seed" > /dev/null
  run "moe-$seed" "$w/moe$seed.gguf" --arch qwen35moe
done
# --identity: a tokenizer.json lands after packing; the first call hashes it, the second is cached
idcheck() {
  local d=$w/trunk-7-q8
  for s in py mojo; do printf '{"model": "fixture"}\n' > "$d/$s/tokenizer.json"; done
  for pass in fresh cached; do
    ./.venv/bin/python3 tools/engine-pack.py --identity "$d/py" > "$d/py-id-$pass.log" 2>&1
    .work/engine-pack --identity "$d/mojo" > "$d/mojo-id-$pass.log" 2>&1
    same_identity "$d/py" "$d/mojo" || { echo "FAIL identity-$pass: identity.json differs"; return 1; }
    diff -q "$d/py-id-$pass.log" "$d/mojo-id-$pass.log" > /dev/null || { echo "FAIL identity-$pass: output differs"; return 1; }
    echo "PASS identity-$pass: $(cat "$d/mojo-id-$pass.log")"
  done
}
idcheck || { fails=$((fails + 1)); names="$names identity"; }
while [ $# -ge 2 ]; do
  cat "$1" > /dev/null  # page cache warm for both tools, so the oracle's first read is not the only cold one
  # shellcheck disable=SC2086
  run "model-$(basename "$1" .gguf)" "$1" $2
  rm -f "$w/model-$(basename "$1" .gguf)"/{py,mojo}/pack.bin
  shift 2
done
[ $fails = 0 ] || { echo "FAIL $fails:$names"; exit 1; }
echo "PASS engine-pack-check"
