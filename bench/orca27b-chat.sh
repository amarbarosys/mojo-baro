#!/usr/bin/env bash
# Chat with OrcaSAQ-2-27B in omp: starts the server if port 8330 is closed, then runs omp with serve/prompts/orca27b-omp-system.md as the system prompt, plus orca27b-memory.md and the index of ~/Brain/orca27b/.
# usage: bench/orca27b-chat.sh [omp args]      window: W=163840 (default) or W=131072 next to a heavy desktop
set -euo pipefail
cd "$(dirname "$0")/.."
if ! curl -sf localhost:8330/health >/dev/null; then
  mkdir -p .work/chat; W=${W:-163840} setsid nohup bench/orca27b-serve.sh > .work/chat/serve.log 2>&1 &
  for _ in $(seq 240); do curl -sf localhost:8330/health >/dev/null && break; sleep 1; done
  curl -sf localhost:8330/health >/dev/null || { echo "FAIL health: see .work/chat/serve.log"; exit 1; }
fi
exec omp --model orca27b/pack --system-prompt "$(cat serve/prompts/orca27b-omp-system.md serve/prompts/orca27b-memory.md ~/Brain/orca27b/MEMORY.md)" "$@"
