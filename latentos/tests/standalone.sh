#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MOJO=${MOJO:-"$ROOT/.venv/bin/mojo"}
CC=${CC:-cc}
OUT=${LATENTOS_STANDALONE_OUT:-}
BACKEND=reference

while [ "$#" -gt 0 ]; do
    case "$1" in
        --backend)
            [ "$#" -ge 2 ] || { echo "--backend needs a value" >&2; exit 2; }
            BACKEND=$2
            shift 2
            ;;
        --backend=*)
            BACKEND=${1#--backend=}
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
[ "$BACKEND" = reference ] || { echo "only --backend reference is implemented" >&2; exit 2; }
[ -x "$MOJO" ] || { echo "missing Mojo compiler: $MOJO" >&2; exit 1; }

mkdir -p "$ROOT/.work"; TMP=$(mktemp -d "$ROOT/.work/latentos-standalone.XXXXXX")
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT
cp -a "$ROOT/latentos" "$TMP/latentos"

# Optional regenerated fixtures let the gate validate format migrations without
# mutating checked-in binary fixtures. The normal path remains self-contained.
if [ -n "${LATENTOS_FIXTURE_DIR:-}" ]; then
    cp "$LATENTOS_FIXTURE_DIR"/{affine.f32,input.f32,expected.f32,binding.manifest,fixture.json} "$TMP/latentos/tests/fixtures/"
fi

python3 - "$TMP/latentos" <<'PY'
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
for path in root.rglob("*.mojo"):
    text = path.read_text()
    if re.search(r"^\s*(?:from|import)\s+[^\n]*(?:baro|yahai)\b", text, re.M):
        raise SystemExit(f"forbidden product import: {path}")
PY

"$MOJO" build "$TMP/latentos/tests/reference_lifecycle.mojo" -I "$TMP" -o "$TMP/reference-lifecycle"
"$TMP/reference-lifecycle"
"$MOJO" build "$TMP/latentos/tests/router_fixture.mojo" -I "$TMP" -o "$TMP/router-fixture"
"$TMP/router-fixture"
"$MOJO" build "$TMP/latentos/tests/ipc_small.mojo" -I "$TMP" -o "$TMP/ipc-small"
"$TMP/ipc-small"

"$CC" -shared -fPIC "$TMP/latentos/crypto_shim.c" -lsodium -o "$TMP/libLatentCrypto.so"
"$MOJO" build "$TMP/latentos/tests/crypto_smoke.mojo" -I "$TMP" -o "$TMP/crypto-smoke" \
    -Xlinker -L"$TMP" -Xlinker -lLatentCrypto -Xlinker -lsodium -Xlinker -rpath -Xlinker "$TMP"
"$TMP/crypto-smoke"
"$MOJO" build "$TMP/latentos/tests/golden_binding.mojo" -I "$TMP" -o "$TMP/golden-binding"
LATENT_BINDING_NOW_S=$(date +%s) "$TMP/golden-binding" "$TMP/latentos/tests/fixtures/binding.manifest" "$TMP/latentos/tests/fixtures/affine.f32" "$TMP/latentos/tests/fixtures/input.f32" "$TMP/latentos/tests/fixtures/expected.f32"
"$MOJO" build "$TMP/latentos/tests/transport_smoke.mojo" -I "$TMP" -o "$TMP/transport-smoke" \
    -Xlinker -L"$TMP" -Xlinker -lLatentCrypto -Xlinker -lsodium -Xlinker -rpath -Xlinker "$TMP"
"$TMP/transport-smoke"

if [ -n "$OUT" ]; then
    mkdir -p "$OUT"
    {
        printf 'backend=reference\n'
        printf 'package=%s\n' "$TMP/latentos"
        ldd "$TMP/reference-lifecycle" || true
        ldd "$TMP/transport-smoke" || true
    } > "$OUT/dependencies.txt"
fi
printf 'STANDALONE_REFERENCE_PASS\n'
