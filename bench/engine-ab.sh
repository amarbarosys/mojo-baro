#!/usr/bin/env bash
# Token identity between two engine binaries on one pack: one-shot T=0 run each,
# GENERATED lines must be equal. For toolchain upgrades the old binary needs its
# own runtime: OLD_LIB=<old venv>/lib/python3.X/site-packages/modular/lib.
# usage: bench/engine-ab.sh OLD_ENGINE NEW_ENGINE PACK PROMPT_TOKENS_FILE
# Run inside a GPU job: gpu-wait run --vram 10 --timeout 600 -- bench/engine-ab.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."
old=$1; new=$2; pack=$3; prompt=$4
out=.work/engine-ab; mkdir -p "$out"
# With MAX's pool uncapped (unset or 0, which the GPU job environment may pass) the
# engine segfaults in a device-to-device copy; tools/baro uses 2 as well.
p=${MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:-0}
[ "$p" = 0 ] && p=2
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=$p
LD_LIBRARY_PATH=${OLD_LIB:-} BARO_PROMPT=$prompt BARO_PACK=$pack "$old" > "$out/old.log" 2>&1 || { echo "FAIL old engine: see $out/old.log"; exit 1; }
BARO_PROMPT=$prompt BARO_PACK=$pack "$new" > "$out/new.log" 2>&1 || { echo "FAIL new engine: see $out/new.log"; exit 1; }
a=$(grep -m1 '^GENERATED:' "$out/old.log" || true); b=$(grep -m1 '^GENERATED:' "$out/new.log" || true)
[ -n "$a" ] || { echo "FAIL: no GENERATED line from the old engine"; exit 1; }
for arm in old new; do echo "$arm $(grep -oE 'tok/s_gen: [0-9.]+' "$out/$arm.log")"; done
[ "$a" = "$b" ] || { echo "FAIL: GENERATED differs"; diff <(tr ' ' '\n' <<<"$a") <(tr ' ' '\n' <<<"$b") | head; exit 1; }
echo "PASS engine-ab: $(($(wc -w <<<"$a") - 1)) tokens identical"
