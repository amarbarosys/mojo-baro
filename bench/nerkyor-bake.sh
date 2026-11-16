#!/usr/bin/env bash
# usage: bench/nerkyor-bake.sh OUTDIR : self-describing bake of nerkyor 27B (tools/bake.sh) + strict closure gate
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(realpath -m "$1"); mkdir -p "$out"
src=$HOME/Models/qwen38-27b-nerkyor-coder-q4lynn/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf
pack=$(realpath ../nerkyor-vs-forge/.work/nerkyor/pack-q4); prompt=bench/mtp-prompts/p01-water.tokens
sha=$(git rev-parse --short HEAD); dst=$HOME/Models/qwen38-27b-nerkyor-coder-q4lynn/nerkyor-27B-BARO-$sha.gguf
env BARO_PACK="$pack" BARO_TMAX=16384 BARO_MEGA=0 BARO_SPEC=0 BARO_PROMPT="$PWD/$prompt" .work/v8/engine-base > "$out/ref.log" 2>&1 || { echo "FAIL ref: see $out/ref.log"; exit 1; }
grep '^GENERATED:' "$out/ref.log" | sed 's/^GENERATED: *//' | tr -s ' ' '\n' | sed '/^$/d' > "$out/ref.ids"
[ -s "$out/ref.ids" ] || { echo "FAIL ref: no GENERATED ids"; exit 1; }
echo "ref ids $(wc -l < "$out/ref.ids")"
tools/bake.sh "$src" "$dst" qwythos "$out/ref.ids" "$prompt" '' '--kv=baro.run.build.flags= -D BARO_SEQ_CAP=1 -D BARO_SLOTS=9 -D BARO_DOT_UNROLL=2 -D BARO_TURBO=1' > "$out/bake.log" 2>&1 || { echo "FAIL bake: see $out/bake.log"; exit 1; }
BARO_PACK="$pack" BARO_TMAX=16384 BARO_MEGA=0 BARO_SPEC=0 tools/gguf-closure.sh "$dst" "$out/ref.ids" "$out/closure" > "$out/closure.log" 2>&1 || { echo "FAIL closure: see $out/closure.log"; exit 1; }
tail -5 "$out/closure.log"; echo "PASS nerkyor-bake $dst"
