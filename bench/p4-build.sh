#!/usr/bin/env bash
# Builds the three P4 gate binaries from THIS tree into .work/p4/bin (CPU only, no queue):
# baro-serve (release), engine-dense (serve/engine.mojo, gfx1100), engine-qwen
# (serve/spark.mojo against the Qwen2.5-7B profile, built under igpu-env so the
# kernels target gfx1030). Prints the commit and sha256 of each artifact.
# P4_QWEN_NATIVE=1 builds engine-qwen for the default device instead (gfx1100, no override), the
# control arm that runs the same spark kernels on the XTX; P4_QWEN_ONLY=1 skips the other two.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${P4_BIN:-$ROOT/.work/p4/bin}
MOJO=${MOJO:-$ROOT/.venv/bin/mojo}
PROFILE=${P4_QWEN_PROFILE:-$HOME/Models/qwen2.5-7b-instruct-import/profile}
IGPU_ENV=${IGPU_ENV:-$HOME/iTools/bin/igpu-env}
fail() { echo "FAIL $1: $2 (log $3)"; exit 1; }

mkdir -p "$OUT"
[ -x "$MOJO" ] || fail toolchain "no mojo at $MOJO" none
[ -f "$PROFILE/profile.mojo" ] || fail profile "no profile.mojo in $PROFILE" none
cd "$ROOT"

if [ "${P4_QWEN_ONLY:-0}" != 1 ]; then
(cd serve && CARGO_TARGET_DIR="$ROOT/.work/p4/target" cargo build --release) > "$OUT/cargo.log" 2>&1 \
  || fail baro-serve "cargo build" "$OUT/cargo.log"
cp "$ROOT/.work/p4/target/release/baro-serve" "$OUT/baro-serve"

"$MOJO" build serve/engine.mojo -I . -I kernels -o "$OUT/engine-dense" > "$OUT/engine-dense.log" 2>&1 \
  || fail engine-dense "mojo build" "$OUT/engine-dense.log"
fi

WRAP=(env -u HSA_OVERRIDE_GFX_VERSION -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES "$IGPU_ENV" --run)
[ "${P4_QWEN_NATIVE:-0}" = 1 ] && WRAP=(env -u HSA_OVERRIDE_GFX_VERSION)
"${WRAP[@]}" "$MOJO" build serve/spark.mojo -I . -I kernels -I serve -I "$PROFILE" ${P4_QWEN_DEFINES:-} \
  -o "$OUT/engine-qwen" > "$OUT/engine-qwen.log" 2>&1 \
  || fail engine-qwen "mojo build (native=${P4_QWEN_NATIVE:-0})" "$OUT/engine-qwen.log"

{
  echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain -- serve kernels | wc -l)"
  echo "qwen_defines=${P4_QWEN_DEFINES:-none} qwen_native=${P4_QWEN_NATIVE:-0}"
  sha256sum "$OUT"/baro-serve "$OUT"/engine-dense "$OUT/engine-qwen" "$PROFILE/profile.mojo" 2>/dev/null || true
} | tee "$OUT/BUILD.txt"
echo "PASS p4-build: binaries in $OUT"
