#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOJO=${MOJO:-"$ROOT/.venv/bin/mojo"}
OUT=${1:-"$ROOT/.work/semantic-latent-build"}
mkdir -p "$OUT"
# granite-4.2-3b profile dropped 2026-11-15 (model deleted): the granite arch compile is no longer covered here
profiles=(
    "$HOME/Models/library/profiles/llama-3.2-1b-instruct-q4_K_M__Llama-3.2-1B-Instruct-Q4_K_M"
)
for profile in "${profiles[@]}"; do
    if [ ! -d "$profile" ]; then
        echo "SKIP semantic Spark profile unavailable: $profile" >&2
        exit 3
    fi
    name=$(basename "$profile" | cut -d_ -f1)
    "$MOJO" build --target-accelerator gfx1100 "$ROOT/serve/spark.mojo" \
        -I "$ROOT" -I "$ROOT/kernels" -I "$ROOT/serve" -I "$profile" \
        -o "$OUT/$name" >"$OUT/$name.build.log" 2>&1
    printf '%s: COMPILE_PASS\n' "$name"
done
printf 'SEMANTIC_SPARK_COMPILE_PASS\n'
