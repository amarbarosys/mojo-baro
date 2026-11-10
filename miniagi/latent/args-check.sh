#!/usr/bin/env bash
# Audit row C16: the stage selector's illegal combinations fail by name (miniagi/latent/stagecheck.mojo, the guard run.mojo
# calls before training). CPU only: builds and runs test_stagecheck.mojo. (run.mojo itself does not build at the moment:
# Model.sort_ids is called and defined nowhere, so the guard is tested as the module it lives in.)
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
out=$here/.work/fix-front-test/test_stagecheck; mkdir -p "$(dirname "$out")"
rm -f "$out"
"$here/.venv/bin/mojo" build "$here/miniagi/latent/test_stagecheck.mojo" -I "$here/miniagi/latent" -o "$out" > "$out.build.log" 2>&1 \
  || { echo "FAIL build: see $out.build.log" >&2; grep -m3 error "$out.build.log" >&2; exit 1; }
"$out"
