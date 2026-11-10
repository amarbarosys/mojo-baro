#!/usr/bin/env bash
# Every model in bench/quality-models.json (or the keys given), one quality row each (bench/quality-run.sh).
# Not wrapped in gpu-wait: each GPU step inside quality-run.sh is its own job (P19). Ends with
# "PASS N/N" or "FAIL k/N: <keys>" and a non-zero exit on any failure (P16).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .work/quality
keys=("$@"); [ ${#keys[@]} -gt 0 ] || mapfile -t keys < <(jq -r 'keys[] | select(startswith("_") | not)' bench/quality-models.json)
failed=()
for k in "${keys[@]}"; do
  echo "=== $k $(date -Is) ===" | tee -a .work/quality/sweep.log
  bench/quality-run.sh "$k" 2>&1 | tee -a .work/quality/sweep.log || failed+=("$k")
done
gpu-wait stats --days 1 | head -1 | tee -a .work/quality/sweep.log
if [ ${#failed[@]} -gt 0 ]; then echo "FAIL ${#failed[@]}/${#keys[@]}: ${failed[*]}" | tee -a .work/quality/sweep.log; exit 1; fi
echo "PASS ${#keys[@]}/${#keys[@]}" | tee -a .work/quality/sweep.log
gpu-wait stats --days 1 | head -1 | tee -a .work/quality/sweep.log
