#!/usr/bin/env bash
# usage: bench/ailab/render-parity.sh [OUT]            (CPU only, no gpu-wait: no GPU is touched)
# P1: every captured runner request (bench/ailab/states-run, states-run-tp: round 1 and later rounds) is rendered and
# tokenized on both sides and the prompt ids compared.
#   ours  = baro-serve on bench/stub-engine.py (the engine request line holds the ids the engine would prefill)
#   llama = CPU llama-server (--ngl 0 --device none) /apply-template + /tokenize, one server, killed by PID
# Env: PACK (default .work/ailab/pack-kq), SETS (body dirs), PORT (baro-serve, default 18461).
# Writes OUT/arm.txt (both arms read back), OUT/ours|llama/<id>.ids, OUT/verdict.tsv. Exit 0 = every request equal,
# 1 = any difference or failure (FAIL <step>: <reason>; a DIFF prints the first differing token and both strings).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:-.work/ailab/render-parity}; mkdir -p "$out"
out=$(realpath "$out"); export LLAMA_ENV=$out/llama.env  # this gate's own llama state file
pack=${PACK:-.work/ailab/pack-kq}; port=${PORT:-18461}
tmax=${TMAX:-24576}  # prompt + max_tokens 16000 must fit: the runner's prompts reach 4.7K tokens (20480 answers 400 past 4480)
sets=${SETS:-"bench/ailab/states-run bench/ailab/states-run-tp bench/ailab/render-edge"}
serve=serve/target/release/baro-serve
for f in "$serve" "$pack/tokenizer.json" bench/stub-engine.py; do [ -e "$f" ] || { echo "FAIL preflight: missing $f"; exit 1; }; done
for d in $sets; do [ -s "$d/MANIFEST.tsv" ] || { echo "FAIL preflight: $d has no MANIFEST.tsv (run capture.py)"; exit 1; }; done
if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "FAIL preflight: port $port in use"; exit 1; fi
spid=""
cleanup() {
  if [ -n "$spid" ]; then
    pkill -TERM -P "$spid" || :
    kill -TERM "$spid" 2>/dev/null || :
    for _ in $(seq 20); do kill -0 "$spid" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$spid" 2>/dev/null || :
  fi
  bench/ailab/llama.sh down >/dev/null 2>&1 || :
}
trap cleanup EXIT
: > "$out/ours-req.jsonl"
FAKE_OUT="$out/ours-req.jsonl" BARO_PACK="$pack" BARO_SPEC=0 \
  setsid "$serve" --engine bench/stub-engine.py --pack "$pack" --port "$port" --engine-env BARO_TMAX=$tmax \
  > "$out/serve.log" 2>&1 < /dev/null &
spid=$!
ok=0
for _ in $(seq 60); do
  kill -0 "$spid" 2>/dev/null || { echo "FAIL serve: baro-serve died, see $out/serve.log"; exit 1; }
  if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || true)" = 200 ]; then ok=1; break; fi
  sleep 0.5
done
[ "$ok" = 1 ] || { echo "FAIL serve: no /health in 30 s, see $out/serve.log"; exit 1; }
larm=$(bench/ailab/llama.sh up --ngl 0 --parallel 1) || { echo "FAIL llama: up failed"; exit 1; }
{
  echo "ours: baro-serve sha=$(sha256sum "$serve" | cut -c1-16) engine=bench/stub-engine.py pack=$pack tokenizer_sha=$(sha256sum "$pack/tokenizer.json" | cut -c1-16)"
  echo "ours health: $(curl -s -m 5 "http://127.0.0.1:$port/health")"
  echo "llama: $larm"
  for d in $sets; do echo "set $d md5=$(cat "$d/SET.md5")"; done
  echo "commit=$(git rev-parse --short HEAD)"
} | tee "$out/arm.txt"
python3 bench/ailab/render_cmp.py --ours "http://127.0.0.1:$port" --fake-out "$out/ours-req.jsonl" \
  --llama "$(sed -n 's/^URL=//p' "$LLAMA_ENV")" --out "$out" $sets 2>&1 | tee "$out/compare.log"
rc=${PIPESTATUS[0]}
exit "$rc"
