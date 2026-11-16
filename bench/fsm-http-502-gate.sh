#!/usr/bin/env bash
# fsm-http-502-gate: B-M1 live. A real engine whose fail word is non-zero must answer HTTP 502 `engine_invalid_tokens`
# through the real baro-serve front; the same engine without the knob answers 200 (control, so the probe can fail).
#   usage: bench/fsm-http-502-gate.sh ENGINE OUT      env: BARO_PACK (default .work/engine-pack-q4)  FRONT (default serve/target/release/baro-serve)
# The knob is BARO_TEST_FAILWORD=1 (serve/engine.mojo): it presets the megakernel fail word, so the live barrier path ends `fail`:1.
# Arm read-back: OUT/arm.txt (binary shas, env) and the engine's own "test knob:" log line per arm. BARO_TMAX=16384.
# Sibling: bench/fsm-wire-gate.sh (same gpu-wait re-exec with env inside the queued command; this gate has no preflight
# check, so its best verdict on an unstamped tree is the same as that gate's EXPLORE=1: read the commit line in arm.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=${1:?ENGINE}; out=${2:?OUT}; pack=${BARO_PACK:-.work/engine-pack-q4}; front=${FRONT:-serve/target/release/baro-serve}
fail() { echo "FAIL $1: $2" >&2; exit 1; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  gpu-wait run --timeout 300 --vram 24 -- env BARO_PACK="$pack" FRONT="$front" "$0" "$@"; exit $?
fi
mkdir -p "$out"; [ -x "$eng" ] || fail setup "no engine $eng"; [ -x "$front" ] || fail setup "no front $front"
[ -f "$pack/prompt-tokens.txt" ] || fail setup "no $pack/prompt-tokens.txt"
toks=$(tr -s ' \n' ',' < "$pack/prompt-tokens.txt" | sed 's/^,*//; s/,*$//')
{ echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ' ')'"
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16)"; echo "front=$front sha=$(sha256sum "$front" | cut -c1-16)"
  echo "pack=$pack  env BARO_TMAX=16384 (both arms), BARO_TEST_FAILWORD=1 (knob arm only)"; } | tee "$out/arm.txt"
arm() {  # NAME [KEY=VAL engine env]...  -> OUT/NAME.{log,status,body}
  local name=$1; shift; local port=$((8700 + RANDOM % 200)) envs=(--engine-env BARO_TMAX=16384)
  for kv in "$@"; do envs+=(--engine-env "$kv"); done
  "$front" --engine "$eng" --pack "$pack" --port "$port" "${envs[@]}" > "$out/$name.log" 2>&1 &
  local pid=$!
  for _ in $(seq 300); do curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  curl -s -o "$out/$name.body" -w '%{http_code}' "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' \
    -d "{\"prompt\": [$toks], \"max_tokens\": 8, \"temperature\": 0, \"spec\": false}" > "$out/$name.status" || true
  kill $pid 2>/dev/null; wait $pid 2>/dev/null || true; sleep 2
}
arm ctl
arm knob BARO_TEST_FAILWORD=1
cs=$(cat "$out/ctl.status"); ks=$(cat "$out/knob.status")
echo "control status=$cs  knob status=$ks  knob body: $(head -c 300 "$out/knob.body")"
grep -q "BARO_TEST_FAILWORD=1" "$out/knob.log" || fail readback "the engine never logged the knob: the arm is not the arm"
! grep -q "BARO_TEST_FAILWORD=1" "$out/ctl.log" || fail readback "control arm logged the knob"
[ "$cs" = 200 ] || fail control "control answered $cs, not 200: the probe cannot tell"
[ "$ks" = 502 ] || fail knob "knob arm answered $ks, not 502"
grep -q engine_invalid_tokens "$out/knob.body" || fail code "502 body lacks engine_invalid_tokens"
echo "PASS fsm-http-502-gate: control 200, fail word 1 -> 502 engine_invalid_tokens (receipts in $out)"
