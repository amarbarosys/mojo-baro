#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOJO=${MOJO:-"$ROOT/.venv/bin/mojo"}
OUT=${OUT:-"$ROOT/.work/vision-latent-gate"}
GPU_WAIT=${GPU_WAIT:-"$HOME/.local/bin/gpu-wait"}
mkdir -p "$OUT"

PACK=${PACK:-"$ROOT/.work/moe-ngram/pack"}
ENG=${ENGINE:-"$ROOT/.work/vision/engine-moe"}
ADAPTER_DIR="$ROOT/.work/regescore-latent/adapter-mlp-256"

if [ ! -d "$PACK" ] || [ ! -d "$ADAPTER_DIR" ]; then
    echo "SKIP vision latent hardware gate: required packs/adapters missing" >&2
    exit 3
fi

if [ ! -x "$ENG" ]; then
    mkdir -p "$ROOT/.work/vision"
    "$MOJO" build --target-accelerator gfx1100 "$ROOT/serve/engine.mojo" \
        -I "$ROOT" -I "$ROOT/kernels" -I "$ROOT/serve" -D BARO_MODEL=qwen35moe \
        -o "$ENG" >"$OUT/engine_build.log" 2>&1
fi

BEMB_PATH="$OUT/test_image.bemb"
python3 - "$BEMB_PATH" <<'PY'
import struct, sys
path = sys.argv[1]
hdr = struct.pack("<4sIIII", b"BEMB", 1, 4, 2048, 4)
pos_data = b"".join(struct.pack("<III", i, 0, i) for i in range(4))
float_data = struct.pack("<%sf" % (4 * 2048), *([0.5] * (4 * 2048)))
with open(path, "wb") as f:
    f.write(hdr + pos_data + float_data)
PY

PROMPT_JSON="$OUT/req.jsonl"
cat <<EOJSON > "$PROMPT_JSON"
{"id":1,"prompt":[151644,248056,248056,248056,248056,9920],"embeds":["$BEMB_PATH"],"n":16,"temperature":0}
EOJSON

"$GPU_WAIT" run --priority 90 --timeout 120 -- bash -c 'env BARO_PACK="'"$PACK"'" BARO_TMAX=4096 BARO_SPEC=0 BARO_MEGA=0 BARO_PREFILL=1 BARO_SERVE=1 BARO_SEMANTIC_ADAPTER_DIR="'"$ADAPTER_DIR"'" MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 "'"$ENG"'" < "'"$PROMPT_JSON"'"' > "$OUT/run.log" 2>&1

if grep -q "vision: applied semantic latent MLP adapter" "$OUT/run.log" && grep -q '"done":true' "$OUT/run.log"; then
    echo "PASS: Vision Latent Integration Gate"
    exit 0
else
    echo "FAIL: Vision Latent Integration Gate failed" >&2
    cat "$OUT/run.log" >&2
    exit 1
fi
