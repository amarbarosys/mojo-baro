#!/usr/bin/env bash
# Run a materialised pack tool: pack-run.sh TOOL MODEL OUTDIR [FLAGS...]. TOOL sits in a directory that also holds
# what it needs (written there from the bake's baro.run.src.* by tools/baro or tools/gguf-closure.sh):
#   *.py    the repo venv's python3 (gguf-extract.py beside it), as before
#   *.mojo  built once, strict IEEE, with -I <its directory> (its baro.run.pack.deps closure: shard_index.mojo, gguf_reader.mojo),
#           the binary cached beside it under a key of every .mojo source there, then run
# Shared by tools/baro and tools/gguf-closure.sh so a bake builds its pack the same way in serve and in verify.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
tool=${1:?usage: pack-run.sh TOOL MODEL OUTDIR [FLAGS...]}; shift
[ -f "$tool" ] || { echo "FAIL pack-run: no pack tool $tool"; exit 1; }
case $tool in
  *.py) exec "$here/.venv/bin/python3" "$tool" "$@" ;;
  *.mojo)
    # strict IEEE: the default contract=fast fuses x * id + 8.5 into an FMA and moves q4 nibbles off the oracle's
    fp=(--fp-mode contract=off)
    d=$(cd "$(dirname "$tool")" && pwd); key=$( (cat "$d"/*.mojo; echo "${fp[*]}") | sha256sum | cut -c1-16)
    bin=$d/.$(basename "$tool" .mojo)-$key
    if [ ! -x "$bin" ]; then
      "$here/.venv/bin/mojo" build "${fp[@]}" "$tool" -I "$d" -o "$bin.new" > "$d/pack-tool-build.log" 2>&1 \
        || { echo "FAIL pack-run: build of $tool, see $d/pack-tool-build.log"; exit 1; }
      mv "$bin.new" "$bin"
    fi
    exec "$bin" "$@" ;;
  *) echo "FAIL pack-run: $tool is neither .py nor .mojo"; exit 1 ;;
esac
