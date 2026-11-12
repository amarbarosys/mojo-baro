#!/usr/bin/env bash
# usage: bench/rdna4-selftest.sh [--pack DIR] [--out FILE]
#
# Architecture self-test for cards we have never run on (written for RDNA4, gfx1200/gfx1201; works on any AMD
# GPU). Needs no model weights unless --pack is given. Writes one local text file and prints a markdown block to
# paste into a GitHub issue. Nothing is sent anywhere. No prompts, paths under $HOME or host names are recorded.
#
# Sections: card (gpu-info.py: gfx, CUs, Infinity Cache, ROCm, Mojo, commit), intrinsics (tools/rdna4-probe for
# this card's gfx), WMMA numeric (kernels/test_wmma_rdna34.mojo built for this gfx and run), engine smoke (only
# with --pack: engine start, compute-unit line, BARO_MEGA line, fail word), kernel log (amdgpu lines only).
# A section that cannot run says SKIP and why; a failing section is the useful part of the report.
# SELFTEST_RUN="gpu-wait run --shared --vram 1 --timeout 300 --" wraps only the GPU steps (our box; users leave it unset).
set -uo pipefail
cd "$(dirname "$0")/.."
PACK=""; OUT=""
while [ $# -gt 0 ]; do
  case $1 in
    --pack) PACK=$2; shift ;;
    --out) OUT=$2; shift ;;
    -h|--help) sed -n '2,11p' "$0" | cut -c3-; exit 0 ;;
    *) echo "rdna4-selftest: unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done
MOJO=./.venv/bin/mojo
RUN=${SELFTEST_RUN:-}
[ -x "$MOJO" ] || { echo "FAIL rdna4-selftest: ./.venv/bin/mojo missing, run 'uv sync' (README)" >&2; exit 1; }
D=.work/selftest; mkdir -p "$D" results
GPUJSON=$(./bench/gpu-info.py) || { echo "FAIL rdna4-selftest: bench/gpu-info.py found no AMD GPU" >&2; exit 1; }
j() { printf '%s' "$GPUJSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('$1'))"; }
GFX=$(j gfx); COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
OUT=${OUT:-results/selftest-$GFX-$COMMIT.txt}
fails=0
{
echo "## mojo-baro architecture self-test"
echo
echo "card:     $(j gpu) ($GFX), compute units $(j compute_units), wavefront $(j wavefront), infinity cache KB $(j infinity_cache_kb)"
echo "software: rocm $(j rocm), $("$MOJO" --version 2>&1 | head -1), commit $COMMIT$( [ -z "$(git status --porcelain 2>/dev/null)" ] || echo ' (dirty tree)')"
echo
echo "### intrinsics (compile only)"
echo '```'
if tools/rdna4-probe/rdna4-probe.sh "$GFX" > "$D/probe.log" 2>&1; then sed -n '/^probe /,$p' "$D/probe.log"
else echo "probe script failed:"; tail -5 "$D/probe.log"; fails=$((fails+1)); fi
echo '```'
echo
echo "### WMMA numeric (16x16x16 f16 -> f32, K=64, bar 2^-19 sum|ab|)"
echo '```'
if timeout 600 "$MOJO" build kernels/test_wmma_rdna34.mojo --target-accelerator "$GFX" -o "$D/wmma" > "$D/wmma-build.log" 2>&1; then
  $RUN bash -c "timeout 120 $D/wmma > $D/wmma.log 2>&1"; rc=$?
  grep -E "PASS|FAIL|C\[" "$D/wmma.log" | head -5; echo "exit $rc"; [ $rc = 0 ] || fails=$((fails+1))
else echo "BUILD FAILED:"; grep -m5 -E "error|LLVM ERROR" "$D/wmma-build.log"; fails=$((fails+1)); fi
echo '```'
echo
echo "### engine smoke"
echo '```'
if [ -z "$PACK" ]; then echo "SKIP: no --pack given (needs a model pack, see README 'tools/baro')"
elif timeout 900 "$MOJO" build serve/engine.mojo -I . -I kernels --target-accelerator "$GFX" -o "$D/engine" > "$D/engine-build.log" 2>&1; then
  $RUN bash -c "env BARO_PACK=$PACK BARO_TMAX=16384 timeout 300 $D/engine < /dev/null > $D/engine.log 2>&1"; rc=$?
  grep -E "compute units|BARO_MEGA:|NOT-RESIDENT|fail|Error|ready" "$D/engine.log" | grep -v "^BARO_MEGA_WIN" | head -8; echo "exit $rc"; [ $rc = 0 ] || fails=$((fails+1))
else echo "ENGINE BUILD FAILED:"; grep -m5 -E "error|LLVM ERROR" "$D/engine-build.log"; fails=$((fails+1)); fi
echo '```'
echo
echo "### kernel log (amdgpu lines, last 15)"
echo '```'
{ journalctl -k -b --no-pager -o cat 2>/dev/null || dmesg 2>/dev/null; } | grep -i amdgpu | grep -viE "serial|uuid" | tail -15 || echo "SKIP: no access to the kernel log"
echo '```'
echo
echo "sections failed: $fails"
} | tee "$OUT"
echo "wrote $OUT (paste it into a 'hardware report' issue; logs in $D/ if we ask for more)" >&2
[ "$fails" = 0 ]
