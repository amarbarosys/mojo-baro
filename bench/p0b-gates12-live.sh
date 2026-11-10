#!/usr/bin/env bash
# Runs P0b gates 1 and 2 back to back as one gpu-wait job (coordinator
# 2026-09-17): each gate script boots and tears down its own pair of
# engines, so this is just sequencing, not shared state between them.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/p0b-gates12}
bench/p0b-gate1-placement.sh "$out/gate1"
bench/p0b-gate2-failover.sh "$out/gate2"
echo "PASS p0b-gates12-live"
