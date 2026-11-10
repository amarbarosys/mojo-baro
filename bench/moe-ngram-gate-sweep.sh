#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fail=()
for w in 1 3 5; do
  BARO_TMAX=1024 BARO_NGRAM_MIN=$w python3 bench/moe-ngram.py .work/moe-ngram/control .work/moe-ngram/engine-gate .work/moe-ngram/pack .work/moe-ngram/gate-min$w || fail+=(min$w)
done
[ ${#fail[@]} -eq 0 ] || { echo "FAIL ${#fail[@]}/3 ${fail[*]}"; exit 1; }
