#!/usr/bin/env bash
# usage: face/gate.sh [OUTDIR]     (default .work/face-gate)
# F0 gate (docs/lanes/2026-11-12-face-plan.md): the net test, then the face renders 3 frames headless (no window,
# no display) on the GPU, converted to PNG; tesseract (img-read) must read the chat hint, the engine panel title and the status line
# back from EVERY frame. Exit non-zero on any miss. The front is pointed at a dead port so the status line is
# deterministic: "front: unreachable". FACE_EXPECT_EXTRA="text" adds one required string (falsifier: a wrong one must fail).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/face-gate}; mkdir -p "$out" .work
fail() { echo "FAIL $1: $2"; exit 1; }
mojo=./.venv/bin/mojo
cap() { systemd-run --user --scope -q -p MemoryMax=8G "$@"; }
cap $mojo build -I . -I face face/net_test.mojo -o .work/net_test > "$out/net_test.build.log" 2>&1 || fail build "net_test, see $out/net_test.build.log"
timeout 30 .work/net_test > "$out/net_test.log" 2>&1 || fail net_test "see $out/net_test.log"
cap $mojo build --target-accelerator amdgpu:gfx1100 -I . -I face face/app.mojo -o .work/face > "$out/face.build.log" 2>&1 || fail build "face, see $out/face.build.log"
rm -f "$out"/face-*.ppm "$out"/face-*.png
cap gpu-wait run --shared --vram 1 --timeout 300 -- .work/face --shots "$out" --front 127.0.0.1:1 > "$out/run.log" 2>&1 || fail run "see $out/run.log"
need=("Type a message below" "Model card" "front: unreachable")
[ -z "${FACE_EXPECT_EXTRA:-}" ] || need+=("$FACE_EXPECT_EXTRA")
for f in 1 2 3; do
  [ -s "$out/face-$f.ppm" ] || fail frame "face-$f.ppm missing"
  magick "$out/face-$f.ppm" "$out/face-$f.png" || fail convert "frame $f"
  img-read --min-chars 5 "$out/face-$f.png" > "$out/face-$f.ocr.txt" 2>&1 || fail ocr "img-read frame $f"
  for s in "${need[@]}"; do grep -qiF "$s" "$out/face-$f.ocr.txt" || fail ocr "frame $f: '$s' not read back, see $out/face-$f.ocr.txt"; done
  grep -qE "frame $f\b" "$out/face-$f.ocr.txt" || fail ocr "frame $f: its own frame number not read back"
done
echo "PASS face gate: net test + 3 headless frames, OCR read ${#need[@]} strings and the frame number in each ($out)"
