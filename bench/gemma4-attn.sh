#!/usr/bin/env bash
# gemma4 decode attention (kernels/gemma4_attn.mojo): build, ISA receipt, then one GPU job.
# usage: bench/gemma4-attn.sh test                     correctness gate (f64 oracle, mutants)
#        bench/gemma4-attn.sh bench [HD WIN T NS]      microseconds per launch, cold and hot
#        bench/gemma4-attn.sh ktrace HD WIN T NS       device time per kernel (rocprofv3 --kernel-trace)
# env: G4_GB, G4_NLD, G4_BT (0 = the tuned picks g4_gb / g4_nld / g4_bt), OUT (.work/g4), TIMEOUT (600 s)
# The build and the receipt run on the CPU; the script then queues itself under gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?usage: gemma4-attn.sh test|bench|ktrace [HD WIN T NS]}
shift
GB=${G4_GB:-0} NLD=${G4_NLD:-0} BT=${G4_BT:-0} OUT=${OUT:-.work/g4}
BIN=$OUT/test_gemma4_attn-gb$GB-nld$NLD-bt$BT
if [ -z "${G4_INJOB:-}" ]; then
  mkdir -p "$OUT"
  rm -f "$BIN"
  ./.venv/bin/mojo build kernels/test_gemma4_attn.mojo -I kernels -I serve -D G4_GB="$GB" -D G4_NLD="$NLD" -D G4_BT="$BT" \
    -o "$BIN" --target-accelerator gfx1100 > "$OUT/build.log" 2>&1 || { echo "FAIL build: see $OUT/build.log"; exit 1; }
  [ -x "$BIN" ] || { echo "FAIL build: no binary, see $OUT/build.log"; exit 1; }
  ./.venv/bin/python tools/isa-receipt.py --summary attn_ "$BIN" > "$BIN.isa" || { cat "$BIN.isa"; echo "FAIL isa: a kernel spills"; exit 1; }
  grep amar_attn_decode_ge "$BIN.isa" | sort -u -k3,3n
  exec gpu-wait run --timeout "${TIMEOUT:-600}" -- env G4_INJOB=1 G4_GB="$GB" G4_NLD="$NLD" G4_BT="$BT" OUT="$OUT" "$0" "$mode" "$@"
fi
log=$OUT/$mode-gb$GB-nld$NLD-bt$BT-$(date +%H%M%S).log
{ uptime; sha256sum "$BIN"; echo "args: $mode $*"; } > "$log"
case $mode in
  test)
    "$BIN" >> "$log" 2>&1 || { grep -E 'FAIL|rror' "$log" | head -20; echo "FAIL test: $log"; exit 1; }
    echo "pass lines $(grep -c ' PASS$' "$log"), mutants caught $(grep -c ' CAUGHT$' "$log")"
    tail -1 "$log" ;;
  bench)
    "$BIN" bench "$@" >> "$log" 2>&1 || { tail -5 "$log"; echo "FAIL bench: $log"; exit 1; }
    grep -c '^BENCH' "$log" > /dev/null || { echo "FAIL bench: no BENCH line in $log"; exit 1; }
    grep '^BENCH' "$log" ;;
  ktrace)
    kt=$PWD/$OUT/kt-$(date +%H%M%S)
    /opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d "$kt" -- "$PWD/$BIN" bench "$@" >> "$log" 2>&1 \
      || { tail -5 "$log"; echo "FAIL ktrace: $log"; exit 1; }
    grep '^BENCH' "$log" || true
    echo "ktrace dir $kt"
    find "$kt" -name '*.csv' | head ;;
  *) echo "FAIL mode: $mode"; exit 1 ;;
esac
echo "log $log"
