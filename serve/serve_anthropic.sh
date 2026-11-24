#!/usr/bin/env bash
# Long-running Anthropic endpoint on Qwythos-9B-v2 (the model and settings pilot_anthropic.sh proved: YaRN x4, int8 KV,
# TMAX 131072 default = max that fits 24 GB (262144 OOMs: 8.12 GB alloc), env TMAX overrides) so a Claude Code session can work on it: ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}.
# Direct invocation self-submits via gpu-wait (holds the GPU for --timeout seconds, default 4 h). Stop: gpu-wait cancel <id>.
# Gotcha: needs .work/baro-front built (pilot_anthropic.sh builds it); the window is 65,536 tokens, Claude Code's own
# prompt is ~47k, so long sessions compact often.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"; cd "${REPO_DIR}"
if [ "${BARO_ANTHROPIC_SERVE_ADMITTED:-0}" != "1" ]; then
  exec gpu-wait run --vram 16 --mem 12 --timeout "${SERVE_TIMEOUT:-14400}" --cwd "${REPO_DIR}" --label anthropic-serve-qwythos -- \
    env BARO_ANTHROPIC_SERVE_ADMITTED=1 TMAX="${TMAX:-131072}" SERVE_PORT="${SERVE_PORT:-18765}" "${SCRIPT_DIR}/serve_anthropic.sh" "$@"
fi
PORT="${SERVE_PORT:-18765}"
MODEL_PACK="${MODEL_PACK:-$HOME/.cache/baro/qwen35-c19706672fb25b4a/packs/5657eb0f4ce2e32245090603d4e45e6eb18098f353c902bd61a22a9a56adece9}"
MODEL_ENGINE="${MODEL_ENGINE:-$HOME/.cache/baro/qwen35-c19706672fb25b4a/engine-df9114edc69b}"
FRONT="${REPO_DIR}/.work/baro-front"
[ -x "${FRONT}" ] || { echo "FAIL front: ${FRONT} missing, run serve/pilot_anthropic.sh once"; exit 1; }
[ -x "${MODEL_ENGINE}" ] && [ -d "${MODEL_PACK}" ] || { echo "FAIL model: engine or pack missing"; exit 1; }
echo "SERVE Qwythos-9B-v2 Q4_0 YaRN x4 int8 KV TMAX ${TMAX:-131072} on http://127.0.0.1:${PORT}"
exec env BARO_NEURONS=0 BARO_SERVE=1 BARO_SPEC=1 BARO_TMAX="${TMAX:-131072}" BARO_LOG_PROMPTS=1 BARO_DEFAULT_TOP_P=0.95 \
  BARO_DEFAULT_TOP_K=20 BARO_PACK="${MODEL_PACK}" "${FRONT}" "${MODEL_ENGINE}" "${MODEL_PACK}" --host 127.0.0.1 --port "${PORT}" --workers 4
