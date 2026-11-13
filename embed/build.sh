#!/usr/bin/env bash
# Build the standard embedder (embed/serve_embed.mojo) into $BARO_EMBED_DIR (default ~/.cache/baro/embed), with the
# WordPiece normaliser table beside it: the paths baro-serve reads by default (BARO_EMBED_BIN, BARO_EMBED_NORM).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
OUT=${BARO_EMBED_DIR:-$HOME/.cache/baro/embed}
MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo
mkdir -p "$OUT"
"$MOJO" build embed/serve_embed.mojo -I embed -o "$OUT/serve_embed.new" > "$OUT/build.log" 2>&1 || { echo "FAIL build: see $OUT/build.log"; exit 1; }
mv "$OUT/serve_embed.new" "$OUT/serve_embed"
cp embed/norm.tsv "$OUT/norm.tsv"
echo "built $OUT/serve_embed"
