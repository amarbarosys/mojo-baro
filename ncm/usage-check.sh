#!/usr/bin/env bash
# Audit row C17: `ncm MODEL 64 64` (the DIR forgotten) and a bare `ncm` fail with the usage line instead of running a
# self-check that ignores both numbers (or indexing a[1] out of range). The guard runs before the first GPU call.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/.work/fix-front-test/ncm; mkdir -p "$(dirname "$out")"
rm -f "$out"
"$here/.venv/bin/mojo" build -I "$here/tools" --target-accelerator gfx1100 "$here/ncm/ncm.mojo" -o "$out" > "$out.build.log" 2>&1 \
  || { echo "FAIL build: see $out.build.log" >&2; grep -m3 error "$out.build.log" >&2; exit 1; }
for args in "" "model.gguf 64 64" "model.gguf 64 64 dir extra"; do
  # shellcheck disable=SC2086
  if msg=$("$out" $args 2>&1); then echo "FAIL C17: 'ncm $args' was accepted" >&2; exit 1; fi
  case $msg in *"FAIL usage: ncm MODEL [NB B DIR]"*) ;; *) echo "FAIL C17: 'ncm $args' gave: $(printf %s "$msg" | tail -2)" >&2; exit 1;; esac
done
echo "ncm usage: PASS (C17)"
