#!/usr/bin/env bash
# Build serve/engine.mojo (this worktree's CURRENT source, not a gguf's
# embedded historical commit) + a pack, for the quality-protocol harness
# (bench/quality-protocol.md). Dense qwen35/qwen35moe only: the forced-token
# logprob dump (BARO_DUMP_LOGITS_DIR, window.mojo) is engine.mojo-only,
# serve/spark.mojo has no top_logprobs wiring at all (confirmed live on
# Llama-3.2-1B 2026-09-16: request accepted, field silently ignored, 0 rows
# dumped -- QUESTION sent to a lane pane, spark-family models out of this script).
#
# Deliberately NOT tools/gguf-closure.sh's approach (rebuild from the gguf's
# baro.kernel.commit): every current bake predates today's sampling-lane
# logprob landing, so that path silently builds an engine without the dump
# feature at all (same failure, root cause was the commit, not the family,
# but the family gap is separately real -- see above).
# usage: bench/quality-build.sh MODEL.gguf OUTDIR [--moe|--spark]
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; out=$2; mode=${3:-}
MOJO=$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo
PY=$HOME/Projects/mojo/mojo-baro/.venv/bin/python3
mkdir -p "$out"

if [ "$mode" = "--spark" ]; then
  echo "== spark profile (tools/gen-profile.mojo, task-eval only, no logprob dump on this path) =="
  [ -x .work/gen-profile ] || "$MOJO" build tools/gen-profile.mojo -I tools -o .work/gen-profile > "$out/genprofile-build.log" 2>&1 \
    || { cat "$out/genprofile-build.log"; exit 1; }
  mkdir -p "$out/profiledir"
  .work/gen-profile "$model" "$out/profiledir/profile.mojo" > "$out/genprofile.log" 2>&1 \
    || { cat "$out/genprofile.log"; exit 1; }
  echo "== engine (serve/spark.mojo, current worktree HEAD) =="
  "$MOJO" build serve/spark.mojo -I . -I kernels -I serve -I "$out/profiledir" -o "$out/engine" > "$out/build.log" 2>&1 \
    || { grep -m1 error: "$out/build.log"; exit 1; }
  [ -x "$out/engine" ] || { echo "engine build FAILED"; cat "$out/build.log"; exit 1; }
  echo "== pack (tools/engine-pack.py --dense) =="
  "$PY" tools/engine-pack.py "$model" "$out/pack" --dense > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; exit 1; }
  echo "engine: $out/engine  pack: $out/pack"
  exit 0
fi

echo "== engine (current worktree HEAD, not the gguf's embedded commit) =="
if [ "$mode" = "--moe" ]; then
  "$MOJO" build serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -o "$out/engine" > "$out/build.log" 2>&1 \
    || { grep -m1 error: "$out/build.log"; exit 1; }
else
  "$MOJO" build serve/engine.mojo -I . -I kernels -o "$out/engine" > "$out/build.log" 2>&1 \
    || { grep -m1 error: "$out/build.log"; exit 1; }
fi
[ -x "$out/engine" ] || { echo "engine build FAILED"; cat "$out/build.log"; exit 1; }

echo "== pack (tools/engine-pack.py, current worktree) =="
if [ "$mode" = "--moe" ]; then
  "$PY" tools/engine-pack.py "$model" "$out/pack" --arch qwen35moe > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; exit 1; }
else
  "$PY" tools/engine-pack.py "$model" "$out/pack" --q8 > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; exit 1; }
fi
echo "engine: $out/engine  pack: $out/pack"
