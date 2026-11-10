#!/usr/bin/env bash
# usage: bench/ailab/llama.sh up [--ngl N] [--parallel P] [--ctx C] [--port PORT] [--gguf PATH] [--extra "FLAGS"]
#        bench/ailab/llama.sh down | status
# One llama-server on the Ornith gguf, detached, killed by PID. State: .work/ailab/llama.env or $LLAMA_ENV (URL= PORT= PID= ARM=).
# --ngl 0 (default) is the CPU server (--device none, no GPU backend touched): P1 template and tokenize checks.
# --ngl > 0 only inside a `gpu-wait run` job (GPU_WAITING_ROOM_JOB set), else FAIL: the card is never taken bare.
# The arm is read back from the running server (/props total_slots, n_ctx) and /proc/PID/cmdline, written to
# .work/ailab/llama.arm and printed; a slot count different from --parallel is FAIL.
# GPU arm flags follow the multi-session report: -fa on, --kv-unified when parallel > 1 (--no-unified: one KV per slot, c/par each; the
# one-token forced control needs it: with --kv-unified every id_slot request re-evaluated its whole prompt), --jinja always.
set -euo pipefail
cd "$(dirname "$0")/../.."
envf=${LLAMA_ENV:-.work/ailab/llama.env}  # LLAMA_ENV: a per-run state file, so two gates never share one (a `down` kills the PID in it)
bin=${LLAMA:-$HOME/Projects/imports/llama.cpp/build/bin/llama-server}
mkdir -p .work/ailab
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
cmd=${1:-}; shift || true
case $cmd in
up)
  kv=""; unified=1; ngl=0; par=1; ctx=""; port=18462; gguf=${GGUF:-$HOME/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf}; extra=""
  while [ $# -gt 0 ]; do
    case $1 in
      --no-unified) unified=0; shift; continue ;;
      --kv) kv=$2 ;;
      --ngl) ngl=$2 ;; --parallel) par=$2 ;; --ctx) ctx=$2 ;; --port) port=$2 ;; --gguf) gguf=$2 ;; --extra) extra=$2 ;;
      *) echo "FAIL args: unknown $1"; exit 1 ;;
    esac
    shift 2 || { echo "FAIL args: $1 needs a value"; exit 1; }
  done
  [ -x "$bin" ] || { echo "FAIL up: no llama-server at $bin"; exit 1; }
  [ -f "$gguf" ] || { echo "FAIL up: no gguf $gguf"; exit 1; }
  if [ "$ngl" != 0 ] && [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
    echo "FAIL up: --ngl $ngl needs a gpu-wait run job (GPU_WAITING_ROOM_JOB unset); the card is never taken bare"; exit 1
  fi
  if [ -f "$envf" ]; then
    old=$(sed -n 's/^PID=//p' "$envf")
    if alive "$old"; then echo "FAIL up: llama-server already running, pid $old (llama.sh down first)"; exit 1; fi
    rm -f "$envf"
  fi
  if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "FAIL up: port $port in use"; exit 1; fi
  [ -n "$ctx" ] || ctx=$(( par * 8192 ))
  flags=(-m "$gguf" --host 127.0.0.1 --port "$port" --parallel "$par" -c "$ctx" --jinja -t 8 --no-webui)
  fa=on; if [ "$kv" = f32 ]; then fa=off; fi  # f32 K/V under flash attention is converted to f16 inside the kernel (ggml-cuda/fattn.cu need_f16_K), so f32 KV needs -fa off
  if [ -n "$kv" ]; then flags+=(-ctk "$kv" -ctv "$kv" -lv 4); fi  # -lv 4: at the default verbosity this build prints no llama_kv_cache line, so the read-back would be blind
  if [ "$ngl" = 0 ]; then flags+=(--device none -ngl 0 --no-warmup); else flags+=(-ngl "$ngl" -fa "$fa"); fi
  if [ "$par" -gt 1 ] && [ "$unified" = 1 ]; then flags+=(--kv-unified); fi
  log=.work/ailab/llama-$port.log
  # shellcheck disable=SC2086
  setsid "$bin" "${flags[@]}" $extra > "$log" 2>&1 < /dev/null &
  pid=$!
  ok=0
  for _ in $(seq 600); do
    if ! alive "$pid"; then echo "FAIL up: llama-server died before /health, see $log"; exit 1; fi
    if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || true)" = 200 ]; then ok=1; break; fi
    sleep 0.5
  done
  if [ "$ok" != 1 ]; then kill "$pid" 2>/dev/null || true; echo "FAIL up: no /health in 300 s, see $log"; exit 1; fi
  if [ -n "$kv" ]; then  # read-back from the server's own log, never from the command line
    kvline=$(grep -m1 -E "K \\(${kv}\\)" "$log" || :); falin=$(grep -m1 -i "flash_attn" "$log" || :)
    if [ -z "$kvline" ]; then kill "$pid" 2>/dev/null || true; echo "FAIL up: --kv $kv asked, the log has no 'K ($kv)' line (see $log)"; exit 1; fi
    if [ "$ngl" != 0 ] && [ "$fa" = off ] && ! grep -qiE "flash_attn *= *disabled" "$log"; then kill "$pid" 2>/dev/null || true; echo "FAIL up: -fa off asked, the log reads: ${falin:-no flash_attn line} (see $log)"; exit 1; fi
    echo "kv read-back: ${kvline:0:160} | ${falin:0:80}" >> "${envf%.env}.kv"
  fi
  props=$(curl -s -m 5 "http://127.0.0.1:$port/props") || { kill "$pid" || true; echo "FAIL up: /props unreadable"; exit 1; }
  slots=$(printf '%s' "$props" | jq -r '.total_slots // "?"'); nctx=$(printf '%s' "$props" | jq -r '.default_generation_settings.n_ctx // "?"')
  if [ "$slots" != "$par" ]; then kill "$pid" 2>/dev/null || true; echo "FAIL up: read back total_slots=$slots, wanted --parallel $par"; exit 1; fi
  arm="llama-server pid=$pid port=$port ngl=$ngl kv=${kv:-default(f16)} fa=$([ "$ngl" = 0 ] && echo "auto(cpu, not passed)" || echo "$fa") parallel=$par total_slots=$slots n_ctx=$nctx gguf=$gguf sha=$(sha256sum "$gguf" | cut -c1-16) cmdline=$(tr '\0' ' ' < /proc/$pid/cmdline)"
  printf '%s\n' "$arm" > "${envf%.env}.arm"
  printf 'URL=http://127.0.0.1:%s\nPORT=%s\nPID=%s\nARM=%s\n' "$port" "$port" "$pid" "${envf%.env}.arm" > "$envf"
  echo "$arm"
  ;;
down)
  [ -f "$envf" ] || { echo "llama.sh down: no llama.env, nothing to stop"; exit 0; }
  pid=$(sed -n 's/^PID=//p' "$envf")
  if alive "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 60); do alive "$pid" || break; sleep 0.5; done
    if alive "$pid"; then kill -KILL "$pid" 2>/dev/null || true; sleep 1; fi
  fi
  if alive "$pid"; then echo "FAIL down: llama-server $pid still alive"; exit 1; fi
  rm -f "$envf"; echo "llama.sh down: pid $pid gone"
  ;;
status)
  [ -f "$envf" ] || { echo "llama.sh status: down"; exit 1; }
  pid=$(sed -n 's/^PID=//p' "$envf")
  if alive "$pid"; then echo "llama.sh status: up pid $pid $(sed -n 's/^URL=//p' "$envf")"; else echo "FAIL status: llama.env names dead pid $pid"; exit 1; fi
  ;;
*) echo "usage: llama.sh up [--ngl N --parallel P --ctx C --port PORT --gguf PATH --extra FLAGS] | down | status"; exit 1 ;;
esac
