#!/usr/bin/env bash
# dense-tmax-check.sh: BARO_TMAX reaches a dense (spark.mojo) engine through `tools/baro serve` (CPU only, no GPU).
# The dense harness compiles TMAX in; before this check existed BARO_TMAX was silently ignored there and /health
# kept the baked 16384. Steps: serve --no-serve with BARO_TMAX=N must build or hit the engine slot -tN and leave
# TMAX N in the extracted profile.mojo; without BARO_TMAX the baked slot is used (no -t suffix).
# usage: tools/dense-tmax-check.sh [MODEL.gguf]     BARO=path/to/other/tools/baro runs the check against another copy
# Needs the cached pack of the model (the first run of the unit made it); about 40 s with the engine slot cached.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-tools/baro}
M=${1:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-38e5113.gguf}
N=${TMAX:-98304}
[ -f "$M" ] || { echo "FAIL model: $M missing"; exit 1; }
out=$(BARO_TMAX=$N BARO_NEURONS=0 "$BARO" serve "$M" --no-serve --port 8123 2>&1) || { echo "FAIL serve: $(echo "$out" | tail -2)"; exit 1; }
echo "$out" | grep -q "engine_path=[^ ]*-t$N" || { echo "FAIL slot: no engine slot -t$N in the receipt"; exit 1; }
id=$(echo "$out" | sed -n 's/^id: //p' | head -1)
grep -q "^comptime TMAX = $N$" "$HOME/.cache/baro/$id/src/profile.mojo" || { echo "FAIL tmax: profile.mojo does not hold TMAX $N"; exit 1; }
out0=$(BARO_NEURONS=0 "$BARO" serve "$M" --no-serve --port 8123 2>&1) || { echo "FAIL baked: $(echo "$out0" | tail -2)"; exit 1; }
if echo "$out0" | grep -q "engine_path=[^ ]*-t[0-9]"; then echo "FAIL baked: BARO_TMAX unset still picked a -t slot"; exit 1; fi
echo "PASS dense-tmax-check: TMAX $N slot and profile, baked slot untouched"
