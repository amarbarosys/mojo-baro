#!/usr/bin/env bash
# usage: face/chat-gate.sh [OUTDIR]     (default .work/F1-gate-live)
# F1 gate (docs/lanes/2026-11-12-face-plan.md, item F1). Needs a real engine, so the body runs inside ONE gpu-wait job:
#   baro-serve (llama-3.2-1b BARO pack, TMAX 16384) up -> chat_gate sends 2 prompts through the pane (typed text, Return)
#   and renders frames; tesseract (img-read) must read the first words of each streamed answer back from its frame.
#   Cancel: 1) Stop clicked at 200 bytes, 2) the Chat dropped at 200 bytes (window close); the front log must gain one
#   "engine: client dropped request N; cancelling" line for each, and none before (the 2 normal prompts add none).
# Falsifier (FACE_FALSIFY=1): the OCR expectation is replaced by words nobody streamed; the gate must exit non-zero.
# Env: FACE_PACKDIR (default: main checkout .work/baro-llama-3.2-1b-...), FACE_PORT (default 18433).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/F1-gate-live}; mkdir -p "$out" .work
here=$PWD
if [ -z "${FACE_INNER:-}" ]; then
  fail() { echo "FAIL $1: $2"; exit 1; }
  mojo=./.venv/bin/mojo
  cap() { systemd-run --user --scope -q -p MemoryMax=8G "$@"; }
  cap $mojo build -I . -I face face/chat_test.mojo -o .work/chat_test > "$out/chat_test.build.log" 2>&1 || fail build "chat_test, see $out/chat_test.build.log"
  timeout 60 .work/chat_test > "$out/chat_test.log" 2>&1 || fail chat_test "see $out/chat_test.log"
  cap $mojo build --target-accelerator amdgpu:gfx1100 -I . -I face face/chat_gate.mojo -o .work/chat_gate > "$out/chat_gate.build.log" 2>&1 || fail build "chat_gate, see $out/chat_gate.build.log"
  cap gpu-wait run --shared --vram 4 --timeout 900 -- env FACE_INNER=1 PATH="$PATH" FACE_FALSIFY="${FACE_FALSIFY:-}" FACE_PACKDIR="${FACE_PACKDIR:-}" FACE_PORT="${FACE_PORT:-}" bash "$here/face/chat-gate.sh" "$out" > "$out/inner.log" 2>&1 \
    || { tail -5 "$out/inner.log"; fail inner "see $out/inner.log"; }
  tail -1 "$out/inner.log"
  exit 0
fi
# ---- inside the gpu-wait job ----
fail() { echo "FAIL $1: $2"; exit 1; }
port=${FACE_PORT:-18433}
pack=${FACE_PACKDIR:-$HOME/Projects/mojo/mojo-baro/.work/baro-llama-3.2-1b-instruct-q4_k_m-q4_K_M-BARO-93a8b08}
srv="$out/server.log"; : > "$srv"
BARO_TMAX=16384 serve/target/release/baro-serve --engine "$pack/engine" --pack "$pack/pack" --port "$port" > "$srv" 2>&1 &
sp=$!
trap 'kill $sp 2>/dev/null || true; wait $sp 2>/dev/null || true' EXIT
for _ in $(seq 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || true)" = 200 ] && break
  kill -0 $sp 2>/dev/null || fail front "baro-serve exited, see $srv"
  sleep 1
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ] || fail front "no 200 on /health, see $srv"
dropped() { grep -c "client dropped request" "$srv" || true; }
rm -f "$out"/chat-*.ppm "$out"/chat-*.png
# 1. two prompts through the pane; frames; OCR of the streamed text
timeout 240 .work/chat_gate --front 127.0.0.1:$port --out "$out" \
  --prompt "Write one short sentence about a lighthouse keeper." \
  --prompt "Now write one short sentence about a ship." > "$out/two-prompts.log" 2>&1 || fail run "see $out/two-prompts.log"
[ "$(grep -c '^ANSWER' "$out/two-prompts.log")" = 2 ] || fail run "not 2 answers, see $out/two-prompts.log"
[ "$(dropped)" = 0 ] || fail cancel "a normal answer produced a 'client dropped' line (baseline must be 0)"
norm() { tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' ' ' | tr -s ' \n' ' '; }
for n in 1 2; do
  magick "$out/chat-$n.ppm" "$out/chat-$n.png" || fail convert "answer $n"
  img-read --min-chars 5 "$out/chat-$n.png" > "$out/chat-$n.ocr.txt" 2>&1 || fail ocr "img-read answer $n"
  ans=$(grep "^ANSWER $n " "$out/two-prompts.log" | sed 's/^[^|]*| //')
  want=$(printf '%s' "$ans" | norm | awk '{print $1" "$2" "$3" "$4}')
  [ -z "${FACE_FALSIFY:-}" ] || want="zebra quartz nobody streamed"
  got=$(norm < "$out/chat-$n.ocr.txt")
  case "$got" in *"$want"*) ;; *) fail ocr "answer $n: '$want' not read back from chat-$n.png, see $out/chat-$n.ocr.txt";; esac
  echo "OCR answer $n: read back '$want'"
done
# the user's first prompt is in frame 2 as well (the list holds both turns)
grep -qi "lighthouse" "$out/chat-2.ocr.txt" || fail ocr "frame 2 lost the first turn"
# 2. cancel: Stop clicked mid-answer
long="Count from 1 to 3000, one number per line, nothing else."
timeout 120 .work/chat_gate --front 127.0.0.1:$port --out "$out" --prompt "$long" --stop-after 200 > "$out/stop.log" 2>&1 || fail stop "see $out/stop.log"
grep -q "^STOP stopped | bytes [0-9]* busy False" "$out/stop.log" || fail stop "pane did not report stopped and idle, see $out/stop.log"
for _ in $(seq 100); do [ "$(dropped)" -ge 1 ] && break; sleep 0.1; done
[ "$(dropped)" = 1 ] || fail cancel "Stop: front log has $(dropped) 'client dropped' lines, want 1, see $srv"
# 3. cancel: window closed (Chat dropped) mid-answer
timeout 120 .work/chat_gate --front 127.0.0.1:$port --out "$out" --prompt "$long" --close-after 200 > "$out/close.log" 2>&1 || fail close "see $out/close.log"
for _ in $(seq 100); do [ "$(dropped)" -ge 2 ] && break; sleep 0.1; done
[ "$(dropped)" = 2 ] || fail cancel "close: front log has $(dropped) 'client dropped' lines, want 2, see $srv"
grep "client dropped request" "$srv" > "$out/dropped-lines.txt"
# 4. the front still answers after both cancels (engine not wedged)
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ] || fail front "health not 200 after the cancels"
echo "PASS chat gate: 2 prompts streamed and OCR-read back, Stop and close each cancelled at the front ($(wc -l < "$out/dropped-lines.txt") 'client dropped' lines, $out)"
