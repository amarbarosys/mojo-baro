#!/usr/bin/env bash
# Assemble the build harness of a loop iteration and print `split=<0|1> entry=<path rel. to DIR>` for eval.
# Split layout (2026-09-08, P-A): the embedded sources carry window.mojo and no main(); the harness -- pack load,
# stopwatch, prints -- is serve/engine.mojo at the iteration's own commit, taken from git, never from the
# candidate. harness/ also carries every local module the harness imports that the sources do not (serve/harness,
# serve/prefix since 5a82337/7b4b9d0), walked transitively (latent_mlp -> semantic_latent) through serve/ then
# kernels/ (state.mojo; iteration 008 preflight). The entry is not named harness.mojo, which would shadow the
# imported module of that name. Legacy ggufs embed an engine.mojo with main() and build from src/ as before.
# Shared by tools/loop-gate.sh and tools/loop-propose.mojo (compile repair), so both build the same way.
# usage: tools/loop-harness.sh .work/loop/ITER
set -euo pipefail
cd "$(dirname "$0")/.."
dir=$1
if [ -f "$dir/src/window.mojo" ] && ! grep -q '^def main' "$dir/src/engine.mojo" 2>/dev/null; then
  kcommit=$(python3 -c "import json;print(json.load(open('$dir/meta.json'))['baro.kernel.commit'])")
  rm -rf "$dir/harness"; mkdir -p "$dir/harness"
  git show "$kcommit:serve/engine.mojo" > "$dir/harness/closure_main.mojo" || { echo "FAIL loop-harness: no serve/engine.mojo at commit $kcommit" >&2; exit 1; }
  new=1; while [ $new = 1 ]; do new=0
    for m in $(cat "$dir"/harness/*.mojo | sed -n 's/^from \([a-z_]*\) import.*/\1/p' | sort -u); do
      { [ -f "$dir/src/$m.mojo" ] || [ -f "$dir/harness/$m.mojo" ]; } && continue
      for sd in serve kernels; do git cat-file -e "$kcommit:$sd/$m.mojo" 2>/dev/null && { git show "$kcommit:$sd/$m.mojo" > "$dir/harness/$m.mojo"; new=1; break; }; done
    done
  done
  echo "split=1 entry=harness/closure_main.mojo kcommit=$kcommit"
else
  echo "split=0 entry=src/engine.mojo"
fi
