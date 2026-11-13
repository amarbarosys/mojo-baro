#!/usr/bin/env bash
# usage: face/settings_gate.sh [OUTDIR]     (default .work/settings-gate)
# F4 gate (docs/lanes/2026-11-12-face-settings-plan.md): settings_test (TOML load, staging) on the CPU, then one headless frame
# per settings tab (no window, no display; one gpu-wait --shared --vram 1 job, the layer renderer needs a device), read back
# with tesseract (img-read). Required in the frames: the 6 tab names on every frame; per tab the labels of its settings;
# on Engine+Advanced the staged mark, the selected row's help text and its Default / Restart lines; Sampling basic must NOT
# show the advanced-only label. SETTINGS_GATE_FALSIFY=1 asks for a label that is not drawn: the gate must then exit non-zero.
# Exit non-zero on any miss: FAIL <step>: <reason>.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/settings-gate}; mkdir -p "$out" .work
fail() { echo "FAIL $1: $2"; exit 1; }
mojo=./.venv/bin/mojo
cap() { systemd-run --user --scope -q -p MemoryMax=8G "$@"; }
gpu_cap() { local label=$1; shift; cap gpu-wait run --timeout 300 --vram 1 --label "$label" -- "$@"; }
gpu_cap F5-settings-test "$mojo" build -I . -I face face/settings_test.mojo -o .work/settings_test > "$out/test.build.log" 2>&1 || fail build "settings_test, see $out/test.build.log"
timeout 30 .work/settings_test > "$out/settings_test.log" 2>&1 || fail settings_test "see $out/settings_test.log"
gpu_cap F5-presets-test "$mojo" build -I . -I face face/presets_test.mojo -o .work/presets_test > "$out/presets.build.log" 2>&1 || fail build "presets_test, see $out/presets.build.log"
timeout 30 .work/presets_test > "$out/presets_test.log" 2>&1 || fail presets_test "see $out/presets_test.log"
gpu_cap F5-face-app "$mojo" build --target-accelerator amdgpu:gfx1100 -I . -I face face/app.mojo -o .work/face > "$out/app.build.log" 2>&1 || fail build "face/app, see $out/app.build.log"
gpu_cap F5-settings-gate "$mojo" build --target-accelerator amdgpu:gfx1100 -I . -I face face/settings_gate.mojo -o .work/settings_gate > "$out/gate.build.log" 2>&1 || fail build "settings_gate, see $out/gate.build.log"
rm -f "$out"/tab-*.ppm "$out"/tab-*.png "$out"/*.ocr.txt
cap gpu-wait run --shared --vram 1 --timeout 300 -- env SDL_VIDEODRIVER=offscreen .work/settings_gate "$out" > "$out/render.log" 2>&1 || fail render "see $out/render.log"
tabs=("Chat" "Sampling" "Engine" "Model" "Front" "Presets")
for f in 1b 2a 1 2 3 4 5; do
  [ -s "$out/tab-$f.ppm" ] || fail frame "tab-$f.ppm missing"
  magick "$out/tab-$f.ppm" "$out/tab-$f.png" || fail convert "tab-$f"
  magick "$out/tab-$f.png" -resize 200% "$out/tab-$f.ocr.png" || fail convert "tab-$f ocr"  # enlarged: small light text reads badly at 1x
  { tesseract "$out/tab-$f.ocr.png" - --psm 11
    for r in 750x160+30+120 470x400+780+120; do magick "$out/tab-$f.png" -crop $r -resize 300% "$out/tab-$f.reg.png"; tesseract "$out/tab-$f.reg.png" - --psm 6; done; } > "$out/tab-$f.ocr.txt" 2>/dev/null || fail ocr "tesseract tab-$f"  # the engine img-read wraps, sparse-text mode
  magick "$out/tab-$f.png" -crop 560x40+10+8 -resize 300% "$out/tab-$f.bar.png" || fail convert "tab-$f bar"  # the tab row, enlarged: small pill text reads badly in the full frame
  tesseract "$out/tab-$f.bar.png" - --psm 7 > "$out/tab-$f.bar.ocr.txt" 2>/dev/null || fail ocr "tesseract tab-$f bar"
  case $f in 1|1b) act=Sampling;; 2|2a) act=Engine;; 3) act=Model;; 4) act=Front;; 5) act=Presets;; esac
  grep -qiF "$act" "$out/tab-$f.ocr.txt" || fail ocr "tab-$f: panel title '$act' not read, see $out/tab-$f.ocr.txt"  # the active pill (filled) is read as the panel title
  for t in "${tabs[@]}"; do [ "$t" = "$act" ] || grep -qiF "$t" "$out/tab-$f.bar.ocr.txt" || fail ocr "tab-$f: tab name '$t' not read, see $out/tab-$f.bar.ocr.txt"; done
done
need() { grep -qiF "$2" "$out/tab-$1.ocr.txt" || fail ocr "tab-$1: '$2' not read, see $out/tab-$1.ocr.txt"; }
absent() { ! grep -qiF "$2" "$out/tab-$1.ocr.txt" || fail ocr "tab-$1: '$2' read but must be hidden"; }
need 1b "Temperature"; need 1b "How random the next word"; need 1b "0 staged"; absent 1b "Stream tokens"
need 2a "Context window"; need 2a "KV cache type"; need 2a "staged, restart"; need 2a "Needs an engine restart"; need 2a "Default"; need 2a "16384"; need 2a "Status"
need 3 "GGUF path"; need 3 "Find pack"; need 3 "Model card"; need 3 "tmax 32768"
need 4 "No settings in this view"
need 5 "Presets"; need 5 "Engine restarts"; need 5 "BARO_TMAX"; need 5 "FITS: gate fixture"
need 2 "Context window"; absent 2 "KV cache type"
need 1b "${SETTINGS_GATE_FALSIFY:+NO SUCH LABEL }Temperature"
echo "PASS settings gate: settings_test + presets_test + 7 headless frames (including Apply plan OCR) ($out)"
