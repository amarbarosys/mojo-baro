#!/usr/bin/env bash
# mojo-baro's own Llama engine (serve/spark.mojo harness, dense bf16 pack) behind decode's CLI, for tools/stint.sh ALT=.
# usage: tools/baro_arm.sh MODEL IDS_CSV N   (MODEL ignored: the pack is BARO_PACK; N must be 64, the harness default)
# prints the generated ids comma separated, then decode_tok_s. Needs BARO_ENGINE and BARO_PACK set.
set -euo pipefail
[ "$3" = 64 ] || { echo "FAIL baro_arm: N must be 64"; exit 1; }
mkdir -p "$(dirname "$0")/../.work/baro-arm"
d=$(mktemp -d "$(dirname "$0")/../.work/baro-arm/p.XXXX")
trap 'rm -rf "$d"' EXIT
echo "$2" | tr ',' ' ' > "$d/prompt.tokens"
log=$(BARO_PROMPT="$d/prompt.tokens" BARO_MEGA=0 BARO_SPEC=0 "$BARO_ENGINE" 2>&1) || { echo "FAIL baro_arm: engine"; echo "$log" | tail -3; exit 1; }
ids=$(echo "$log" | grep -oP '^generated: \K.*' || true)
ts=$(echo "$log" | grep -oP 'tok/s_gen: \K[0-9.]+' || true)
[ -n "$ids" ] && [ -n "$ts" ] || { echo "FAIL baro_arm: no generated ids or tok/s_gen"; echo "$log" | tail -3; exit 1; }
echo "$ids" | xargs | tr ' ' ','
echo "decode_tok_s $ts"
