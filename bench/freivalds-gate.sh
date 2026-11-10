#!/usr/bin/env bash
# Freivalds integer-GEMM check gate (exchange/freivalds-prereg-2026-10-12.md, frozen).
#   bench/freivalds-gate.sh [OUT]      builds kernels/test_freivalds.mojo AOT into .work/, queues itself
#                                      under gpu-wait, records the arm (VRAM free, clock caps, binary
#                                      hash, HEAD) and runs the gate; output tee'd to OUT
#                                      (default .work/freivalds-gate.txt).
# EXPLORE=1: run without a passing bench/preflight.sh --check; verdict capped at UNVERIFIED, exit 3.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/freivalds-gate.txt}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --priority "${PRIO:-50}" --vram 2 --timeout 1800 -- env EXPLORE="${EXPLORE:-0}" "$0" "$out"
fi
mkdir -p .work
explore=${EXPLORE:-0}
if ! bench/preflight.sh --check >/dev/null 2>&1; then
  if [ "$explore" = 1 ]; then echo "preflight --check not passing: EXPLORE=1, verdict capped at UNVERIFIED"
  else echo "FAIL preflight: bench/preflight.sh --check failed (run bench/preflight.sh, or EXPLORE=1)"; exit 1; fi
fi
./.venv/bin/mojo build kernels/test_freivalds.mojo -I kernels -o .work/test_freivalds > .work/freivalds-build.log 2>&1 \
  || { echo "FAIL build: .work/freivalds-build.log"; exit 1; }
{
  echo "== arm: HEAD $(git rev-parse --short HEAD) binary $(sha256sum .work/test_freivalds | cut -c1-12) $(date -u +%FT%TZ)"
  gw=$(command -v gpu-wait || echo "$HOME/.local/bin/gpu-wait")
  { "$gw" list 2>/dev/null | head -2 | sed "s/^/queue: /"; } || true
  { "$gw" gpu 2>/dev/null | tr ",{}" "\n\n\n" | grep -iE "vram|mem.*(free|used|total)" | head -6 | sed "s/^/gpu: /"; } || true
  { /opt/rocm/bin/rocm-smi --showmeminfo vram --showclocks --json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); c=d.get("card0",{}); print("rocm-smi:", {k:v for k,v in c.items() if "VRAM" in k or "clock" in k.lower() or "sclk" in k or "mclk" in k})' 2>/dev/null; } || true
  { /opt/rocm/bin/rocm-smi --showmaxpower 2>/dev/null | grep -i "power cap"; } || true
  ./.work/test_freivalds
} 2>&1 | tee "$out"
rc=${PIPESTATUS[0]}
if [ "$explore" = 1 ]; then echo "UNVERIFIED (EXPLORE=1): see $out"; exit 3; fi
grep -q "^PASS" "$out" || { echo "FAIL gate: $out"; exit 1; }
exit "$rc"
