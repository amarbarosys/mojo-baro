#!/usr/bin/env bash
# Re-embed the COMMITTED repo sources into a new gguf with lineage; never overwrites.
# usage: tools/loop-embed-winner.sh SRC-BARO.gguf ITER
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1; iter=$2
dst="${src%.gguf}-loop-$iter.gguf"
[ ! -e "$dst" ] || { echo "refusing to overwrite $dst"; exit 1; }
parent=$(./.venv/bin/python3 tools/gguf-extract.py "$src" --meta | jq -r '.["baro.kernel.commit"]')
# Split layout (2026-09-08): the file list is the import closure of
# serve/window.mojo + serve/registry.mojo (tools/embed-files.py); serve/engine.mojo
# -- the stopwatch -- is deliberately NOT embedded. The source gguf's own
# baro.kernel.files is legacy (it carried engine.mojo and the shim) and is ignored.
files=$(./.venv/bin/python3 tools/embed-files.py)
for p in $files; do [ -f "$p" ] || { echo "missing source: $p" >&2; exit 1; }; done
echo "embedding: $files"
BARO_KERNEL_PARENT="$parent" ./.venv/bin/python3 tools/gguf-embed.py "$src" "$dst" $files
echo "$dst" > .work/loop/CHAMPION
echo "wrote $dst (parent $parent); CHAMPION updated"
