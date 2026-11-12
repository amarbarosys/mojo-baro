#!/usr/bin/env bash
# usage (as the engine binary of bench/ab-resident.sh arm B): FACE_ENGINE=ENGINE FACE_FPS=60 FACE_OUT=DIR engine-with-window.sh
# Starts the face (offscreen SDL3 window, no desktop window) at FACE_FPS frames a second on the same card, runs the
# engine with this script's stdin and stdout, stops the face. The face's own fps lines go to FACE_OUT/face-fps.log
# (read back, never assumed: window-cost.sh refuses an arm whose window ran below 90% of the requested rate).
set -euo pipefail
: "${FACE_ENGINE:?}" "${FACE_FPS:?}" "${FACE_OUT:?}"
SDL_VIDEODRIVER=offscreen "${FACE_BIN:-.work/face}" --front 127.0.0.1:1 --fps "$FACE_FPS" --stats >> "$FACE_OUT/face-fps.log" 2>> "$FACE_OUT/face.err" &
fp=$!
trap 'kill $fp 2>/dev/null || true; wait $fp 2>/dev/null || true' EXIT
echo "window start fps=$FACE_FPS" >> "$FACE_OUT/face-fps.log"
for _ in $(seq 100); do [ "$(grep -c "^fps" "$FACE_OUT/face-fps.log")" -gt 0 ] && break; kill -0 $fp 2>/dev/null || { echo "FAIL window: face exited, see $FACE_OUT/face.err" >&2; exit 1; }; sleep 0.1; done
"$FACE_ENGINE"
