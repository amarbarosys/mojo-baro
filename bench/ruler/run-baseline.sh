#!/usr/bin/env bash
# bench/ruler/run-baseline.sh: llama.cpp baseline arms for the RULER gate
# (bench/ruler-protocol.md item 4). One server launch per (arm, size);
# -c = size + max_tokens + margin (must cover the full decode budget, not
# just RULER's small per-task tokens_to_generate -- this model needs
# --max-tokens 1024 to get past its own reasoning, see the protocol's
# "Reasoning-model finding"); server killed between sizes; /props + server
# log saved next to the run as the P1 receipt.
#
# Usage: bench/ruler/run-baseline.sh ARM SIZE [SIZE ...]
#   ARM  = bf16 | k8v4
#   SIZE = 4096 | 8192 | 16384 | 32768 | 65536 | 131072
# N = 25 at 4096-32768, N = 10 at 65536/131072 (brief's own scope gate);
# LIMIT=N overrides (arm B ran at N=10, amendment 2026-09-08).
# Parallel slots: -np NP (4 through 32k, 2 at 64k, 1 at 128k) with -c scaled
# per slot and run.py --workers NP; -ub 2048. Accuracy gate is unaffected;
# wall_ms per prompt is no longer comparable with the -np 1 runs.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
PORT=${PORT:-8199}
OUT=${OUT:-.work/ruler-baseline}
MAX_TOKENS=${MAX_TOKENS:-1024}

ARM=$1; shift
ctk_flags=()
if [ "$ARM" = "k8v4" ]; then
  ctk_flags=(-ctk q8_0 -ctv q4_0)
elif [ "$ARM" != "bf16" ]; then
  echo "ARM must be bf16 or k8v4, got $ARM" >&2; exit 1
fi

srv_wrapper=""
srv_pid=""
cleanup() {
  [ -n "$srv_pid" ] && kill -9 "$srv_pid" 2>/dev/null
  [ -n "$srv_wrapper" ] && kill -9 "$srv_wrapper" 2>/dev/null
}
trap cleanup EXIT

for size in "$@"; do
  limit=25; np=4
  case $size in
    65536) limit=10; np=2 ;;
    131072) limit=10; np=1 ;;
  esac
  limit=${LIMIT:-$limit}
  ctx=$(( (size + MAX_TOKENS + 256) * np ))
  dst="$OUT/$ARM/$size"
  mkdir -p "$dst"

  "$HOME/.local/bin/gpu-wait" run --priority 50 --vram 8 --preemptible -- \
    "$HOME/llama.cpp/build/bin/llama-server" -m "$MODEL" -c "$ctx" -ngl 99 -fa on \
    -b 4096 -ub 2048 -t 8 -np "$np" "${ctk_flags[@]}" --host 127.0.0.1 --port "$PORT" \
    > "$dst/server.log" 2>&1 &
  srv_wrapper=$!

  up=0
  for _ in $(seq 1 240); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    sleep 2
  done
  if [ "$up" != "1" ]; then
    echo "server did not come up for size=$size arm=$ARM" >&2
    cleanup; srv_wrapper=""; continue
  fi
  curl -s "http://127.0.0.1:$PORT/props" > "$dst/props.json"
  # anchored at the start of the command line -- excludes the gpu-wait
  # wrapper process, whose own argv also contains this substring
  srv_pid=$(pgrep -f "^$HOME/llama.cpp/build/bin/llama-server -m $MODEL -c $ctx " | head -1)

  echo "=== arm=$ARM size=$size limit=$limit ctx=$ctx np=$np ub=2048 pid=$srv_pid ===" | tee -a "$dst/run.log"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["default_generation_settings"]; print("props: n_ctx=%s" % d["n_ctx"])' "$dst/props.json" | tee -a "$dst/run.log"
  t0=$(date +%s)
  ./.venv/bin/python3 bench/ruler/run.py --base-url "http://127.0.0.1:$PORT/v1" --model qwythos \
    --prompts bench/ruler/prompts --out "$dst/responses" --sizes "$size" --max-tokens "$MAX_TOKENS" \
    --limit "$limit" --workers "$np" | tee -a "$dst/run.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  echo "wall_s=$((t1 - t0)) rc=$rc" | tee -a "$dst/run.log"

  cleanup
  # wait for the port to actually free before the next size's server binds it
  for _ in $(seq 1 30); do
    curl -sf -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break
    sleep 1
  done
  srv_wrapper=""; srv_pid=""
  sleep 2
done
trap - EXIT
