#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
out=${1:-.work/team-B/codex/p3a}
mkdir -p "$out"

model=${BARO_WHISPER_MODEL:-$HOME/Models/whisper/ggml-large-v3-turbo-q5_0.bin}
baro=${BARO_SERVE_BIN:-serve/target/release/baro-serve}
whisper_cli=${WHISPER_CLI:-$HOME/Models/whisper.cpp/build/bin/whisper-cli}
port=${BARO_PORT:-18183}
whisper_port=${BARO_WHISPER_PORT:-18184}
idle=${BARO_WHISPER_IDLE_SECS:-2}
gpu_wait=${GPU_WAIT_BIN:-$HOME/.local/bin/gpu-wait}
manifest=bench/fixtures/p3a/MANIFEST.tsv
arm="$out/arm.txt"
receipt="$out/receipt.log"

fail() {
  echo "FAIL $*" | tee -a "$receipt" >&2
  exit 1
}

cat > "$arm" <<EOF
model=$model
language=en
beam=5
threads=8
port=$port
whisper_port=$whisper_port
baro=$baro
gpu_wait=$gpu_wait
EOF

if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  echo "FAIL GPU step: dry-run stopped before baro-serve and whisper startup" | tee "$receipt" >&2
  exit 1
fi

if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  bench/preflight.sh --check
  exec "$gpu_wait" run --vram 4 --timeout 1200 -- env P3A_JOB=1 "$0" "$@"
fi

[ -x "$baro" ] || fail "missing baro-serve: $baro"
[ -x "$whisper_cli" ] || fail "missing whisper-cli: $whisper_cli"
[ -f "$model" ] || fail "missing whisper model: $model"
[ -f "$manifest" ] || fail "missing manifest: $manifest"

: > "$receipt"
echo "P3a gate" | tee -a "$receipt"
echo "model=$model" | tee -a "$receipt"
echo "language=en beam=5 threads=8 device=gate GPU" | tee -a "$receipt"
echo "audio fixture manifest=$manifest" | tee -a "$receipt"
echo "model_sha256=$(sha256sum "$model" | cut -d' ' -f1)" | tee -a "$receipt"
"$gpu_wait" gpu > "$out/gpu-before.txt" 2>&1 || fail "gpu-wait gpu before snapshot failed"
echo "gpu-before=$out/gpu-before.txt" | tee -a "$receipt"

server_out="$out/baro.stdout"
server_err="$out/baro.stderr"
rm -f "$server_out" "$server_err"
BARO_WHISPER_MODEL="$model" \
BARO_WHISPER_LANGUAGE=en \
BARO_WHISPER_BEAM=5 \
BARO_WHISPER_THREADS=8 \
BARO_WHISPER_PORT="$whisper_port" \
BARO_WHISPER_IDLE_SECS="$idle" \
  "$baro" --audio-only --host 127.0.0.1 --port "$port" >"$server_out" 2>"$server_err" &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$port/health" > "$out/health.json" 2>/dev/null; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 1
done
[ "$ready" = 1 ] || { cat "$server_err" >&2; fail "baro-serve did not become ready"; }

no_engine_child() {
  ! ps --ppid "$server_pid" -o args= | grep -Eq 'BARO_SERVE=1|(^|/)engine([[:space:]]|$)'
}
no_engine_child || fail "audio-only baro-serve spawned an LLM engine"

completion_status=$(curl -sS -o "$out/completions.json" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$port/v1/completions" \
  -H 'content-type: application/json' -d '{"prompt":"audio-only must reject this","max_tokens":1}')
[ "$completion_status" = 503 ] || fail "audio-only completion status=$completion_status, expected 503"
echo "PASS audio-only completion returns 503" | tee -a "$receipt"

while IFS=$'\t' read -r file _; do
  [ "$file" = file ] && continue
  wav="bench/fixtures/p3a/$file"
  base=${file%.wav}
  ref="$out/ref-$base.txt"
  ref_err="$out/ref-$base.stderr"
  response="$out/response-$base.json"
  [ -f "$wav" ] || fail "manifest fixture missing: $wav"

  "$whisper_cli" -m "$model" -f "$wav" -t 8 -bs 5 -l en -np -nt >"$ref" 2>"$ref_err" \
    || fail "whisper-cli reference failed for $file"
  curl -fsS -X POST "http://127.0.0.1:$port/v1/audio/transcriptions" \
    -F "file=@$wav" -F "model=$model" -F 'language=en' -F 'response_format=json' \
    >"$response" || fail "endpoint request failed for $file"

  python3 - "$ref" "$response" "$file" <<'PY'
import json
import pathlib
import sys

ref_path, response_path, name = map(pathlib.Path, sys.argv[1:])
reference = ref_path.read_text().strip()
payload = json.loads(response_path.read_text())
actual = payload.get("text")
if not isinstance(actual, str):
    raise SystemExit(f"{name}: response has no string text field")
if actual.strip() != reference:
    raise SystemExit(f"{name}: transcript mismatch\nref={reference!r}\nbaro={actual!r}")
print(f"PASS {name}: transcript identical after edge-whitespace trim")
PY
done < "$manifest" | tee -a "$receipt"

sleep 3
no_engine_child || fail "audio-only baro-serve spawned an LLM engine after transcription"
"$gpu_wait" gpu > "$out/gpu-after.txt" 2>&1 || fail "gpu-wait gpu after snapshot failed"
echo "gpu-after=$out/gpu-after.txt" | tee -a "$receipt"
grep -Ei 'whisper|model|language|beam|thread|device|listening' "$server_err" >> "$receipt" || true
echo "health=$(cat "$out/health.json")" >> "$receipt"
echo "PASS P3a 20/20" | tee -a "$receipt"
