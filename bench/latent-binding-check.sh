#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOJO=${MOJO:-"$ROOT/.venv/bin/mojo"}
OUT=${1:-"$ROOT/.work/latent-binding-gate"}
SENDER_DIM=${SENDER_DIM:-2560}
RECEIVER_DIM=${RECEIVER_DIM:-2048}
POSITIONS=${POSITIONS:-8}

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="" python3 "$ROOT/tools/latent-binding-oracle.py" \
    --out "$OUT" --sender-dim "$SENDER_DIM" --receiver-dim "$RECEIVER_DIM" --positions "$POSITIONS" > "$OUT/generator.json"
export LATENT_BINDING_NOW_S=$(date +%s)
"$MOJO" build "$ROOT/bench/latent_binding_check.mojo" -I "$ROOT" -o "$OUT/checker"
CUDA_VISIBLE_DEVICES="" "$OUT/checker" \
    "$OUT/binding.manifest" "$OUT/affine.f32" "$OUT/input.f32" "$OUT/expected.f32" \
    "$SENDER_DIM" "$RECEIVER_DIM" "$POSITIONS" | tee "$OUT/parity.txt"

python3 - "$OUT" "$RECEIVER_DIM" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
receiver_dim = int(sys.argv[2])
manifest = bytearray((root / "binding.manifest").read_bytes())
manifest[16:20] = (receiver_dim + 1).to_bytes(4, "little")
(root / "wrong-dimension.manifest").write_bytes(manifest)
weights = (root / "affine.f32").read_bytes()
(root / "truncated.f32").write_bytes(weights[:-4])
PY
if CUDA_VISIBLE_DEVICES="" "$OUT/checker" "$OUT/wrong-dimension.manifest" "$OUT/affine.f32" "$OUT/input.f32" "$OUT/expected.f32" "$SENDER_DIM" "$RECEIVER_DIM" "$POSITIONS" > "$OUT/wrong-dimension.log" 2>&1; then
    echo "FAIL wrong manifest dimension was accepted" >&2
    exit 1
fi
if CUDA_VISIBLE_DEVICES="" "$OUT/checker" "$OUT/binding.manifest" "$OUT/truncated.f32" "$OUT/input.f32" "$OUT/expected.f32" "$SENDER_DIM" "$RECEIVER_DIM" "$POSITIONS" > "$OUT/truncated.log" 2>&1; then
    echo "FAIL truncated adapter was accepted" >&2
    exit 1
fi
printf 'dimensions=%s->%s positions=%s parity=PASS negatives=PASS\n' "$SENDER_DIM" "$RECEIVER_DIM" "$POSITIONS" | tee "$OUT/SUMMARY.txt"
