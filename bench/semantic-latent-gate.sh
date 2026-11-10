#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOJO=${MOJO:-"$ROOT/.venv/bin/mojo"}
OUT=${OUT:-"$ROOT/.work/semantic-latent-gate"}
GPU_WAIT=${GPU_WAIT:-"$HOME/.local/bin/gpu-wait"}
mkdir -p "$OUT"

SENDER_PROF="$HOME/Models/library/profiles/granite-4.2-3b-bf16__granite-4.2-3b-BF16"
RECEIVER_PROF="$HOME/Models/library/profiles/llama-3.2-1b-instruct-q4_K_M__Llama-3.2-1B-Instruct-Q4_K_M"

SENDER_PACK="$HOME/.cache/baro/granite-d8ea7d76a98aa5af/packs/2faa21c21375fe7c1de05b25343e8d565fae13ca33a87bcaa4645bfcac0fafb3"
RECEIVER_PACK="$HOME/.cache/baro/llama-531b581265e3c68f/packs/40044b9be32ca6a5bf29f2a130f30cd630d5a2d30ceba610be11ef4d3d2ec960"

if [ ! -d "$SENDER_PROF" ] || [ ! -d "$RECEIVER_PROF" ] || [ ! -d "$SENDER_PACK" ] || [ ! -d "$RECEIVER_PACK" ]; then
    echo "SKIP semantic latent hardware gate: profiles or weight packs missing" >&2
    exit 3
fi

"$MOJO" build --target-accelerator gfx1100 "$ROOT/serve/spark.mojo" \
    -I "$SENDER_PROF" -I "$ROOT" -I "$ROOT/kernels" -I "$ROOT/serve" -I "$ROOT/tools" \
    -o "$OUT/sender_engine" >"$OUT/sender.build.log" 2>&1
"$MOJO" build --target-accelerator gfx1100 "$ROOT/serve/spark.mojo" \
    -I "$RECEIVER_PROF" -I "$ROOT" -I "$ROOT/kernels" -I "$ROOT/serve" -I "$ROOT/tools" \
    -o "$OUT/receiver_engine" >"$OUT/receiver.build.log" 2>&1



RAW_DUMP_PATH="$OUT/sender_hidden_raw.f32"
PROMPT_PATH="$OUT/prompt.txt"
echo "1 2 3 4 5 6 7 8" > "$PROMPT_PATH"

ENV_PATH="$ROOT/.work/regescore-latent/fixed-identity.env"
ADAPTER_DIR="$ROOT/.work/regescore-latent/adapter-mlp-256"

# Run Sender hidden extraction
"$GPU_WAIT" run --priority 90 --timeout 120 -- env BARO_PACK="$SENDER_PACK" BARO_PROMPT="$PROMPT_PATH" BARO_SEMANTIC_DUMP="$RAW_DUMP_PATH" BARO_GEN=1 "$OUT/sender_engine" >"$OUT/sender.log" 2>&1
if [ ! -f "$RAW_DUMP_PATH" ]; then
    echo "FAIL: Sender dump failed to produce $RAW_DUMP_PATH" >&2
    cat "$OUT/sender.log" >&2
    exit 1
fi

# Use existing lowrank-payload.f32 and manifest from .work/regescore-latent for strict payload digest verification
PAYLOAD_PATH="$ROOT/.work/regescore-latent/lowrank-payload.f32"
MANIFEST_PATH="$ROOT/.work/regescore-latent/lowrank-payload.manifest"

# Run Receiver control
"$GPU_WAIT" run --priority 90 --timeout 120 -- env BARO_PACK="$RECEIVER_PACK" BARO_PROMPT="$PROMPT_PATH" BARO_GEN=16 "$OUT/receiver_engine" >"$OUT/control.log" 2>&1

# Run Receiver semantic latent handoff with adapter
"$GPU_WAIT" run --priority 90 --timeout 120 -- env $(cat "$ENV_PATH" | xargs) BARO_PACK="$RECEIVER_PACK" BARO_PROMPT="$PROMPT_PATH" BARO_SEMANTIC_MANIFEST="$MANIFEST_PATH" BARO_SEMANTIC_INPUT="$PAYLOAD_PATH" BARO_SEMANTIC_ADAPTER_DIR="$ADAPTER_DIR" BARO_GEN=16 "$OUT/receiver_engine" >"$OUT/semantic.log" 2>&1

if grep -q "semantic adapter: applied non-linear MLP" "$OUT/semantic.log" && grep -q "prefill_s:" "$OUT/semantic.log"; then
    echo "PASS: Multi-Model Semantic Latent Handoff Gate (Sender -> Adapter -> Receiver)"
    exit 0
else
    echo "FAIL: Receiver semantic execution failed" >&2
    cat "$OUT/semantic.log" >&2
    exit 1
fi
