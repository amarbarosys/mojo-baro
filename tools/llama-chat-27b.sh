#!/usr/bin/env bash
# Reference chat arm: llama-server on Qwen3.8-27B-OBLITERATED Q4_K_M with the
# model card's settings (greedy, repeat_penalty 1.15, thinking off via the
# bundled template, no system prompt) and a 100k context. DeerFlow points at
# it (`~/Projects/deer-flow/config.yaml`, model `qwen3.8-27b-obliterated`)
# until baro-serve carries the same contract. Run through gpu-wait.
# usage: tools/llama-chat-27b.sh [port]   (default 8084)
set -euo pipefail
port="${1:-8084}"
model="${LLAMA_CHAT_GGUF:-$HOME/Models/qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf}"
exec "$HOME/llama.cpp/build/bin/llama-server" -m "$model" \
  -c 102400 -np 1 -ngl 99 -fa on -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-format deepseek \
  --temp 0 --top-k 0 --top-p 1.0 --min-p 0 --repeat-penalty 1.15 \
  --alias qwen3.8-27b-obliterated --host 127.0.0.1 --port "$port" --metrics
