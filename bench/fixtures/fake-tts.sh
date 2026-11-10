#!/usr/bin/env bash
set -euo pipefail
out=${3:?output path required}
python3 - "$out" <<'PY'
import struct, sys, wave
path = sys.argv[1]
rate = 16000
frames = b"\x00\x00" * (rate // 10)
with wave.open(path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(frames)
PY
