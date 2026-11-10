#!/usr/bin/env bash
# Build and run latentos/tests/fsm_fixes.mojo (+ fsm_fixes_recv.mojo when present) against the checkout in ROOT
# (default: this one; pass another worktree to run the same checks on it).
set -euo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
MOJO=${MOJO:-$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/mojo}
OUT=$(cd "$(dirname "$0")/../.." && pwd)/.work/fix-front-test/latentos-$(basename "$ROOT"); mkdir -p "$OUT"
cc -shared -fPIC "$ROOT/latentos/crypto_shim.c" -lsodium -o "$OUT/libLatentCrypto.so" || { echo "FAIL shim: cc" >&2; exit 1; }
for t in ${TESTS:-fsm_fixes fsm_fixes_recv boot_result}; do
  [ -f "$ROOT/latentos/tests/$t.mojo" ] || continue
  rm -f "$OUT/$t"
  "$MOJO" build "$ROOT/latentos/tests/$t.mojo" -I "$ROOT" -o "$OUT/$t" -Xlinker -L"$OUT" -Xlinker -lLatentCrypto -Xlinker -lsodium -Xlinker -rpath -Xlinker "$OUT" > "$OUT/$t.build.log" 2>&1 \
    || { echo "FAIL build $t: see $OUT/$t.build.log" >&2; grep -m5 error "$OUT/$t.build.log" >&2; exit 1; }
  "$OUT/$t"
done
