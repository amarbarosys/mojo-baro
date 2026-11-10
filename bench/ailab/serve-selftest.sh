#!/usr/bin/env bash
# usage: bench/ailab/serve-selftest.sh [OUTDIR]     (CPU only: bench/stub-engine.py, no queue, no GPU)
# P3 CPU check: after baro-serve gets SIGTERM or SIGKILL, no engine child is left after 5 s.
#   term       engine honours stdin EOF (the normal stub)
#   kill       SIGKILL of baro-serve, normal stub (EOF reaches the engine through the dead parent's pipe)
#   kill-stuck SIGKILL of baro-serve, engine ignores EOF (STUB_STUCK=1, a mid-decode hang): only PDEATHSIG ends it
#   term-stuck SIGTERM, stuck engine: gone within 40 s (SLOW=1 only; the server waits 30 s for the engine, then kills it)
#   warm-fail  the stub reports cached 0 on ask 2: `up` fails with 'FAIL up/warm' (the runner warm in serve.sh)
# Also checks serve.sh itself: up writes serve.env, down removes it and asserts the engine is gone.
# Exit 0 = all cases pass; any failure prints FAIL <case>: <reason>.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:-.work/ailab/selftest}; mkdir -p "$out"
export STUB_VRAM=1200000000 FAKE_OUT=$PWD/$out/req.jsonl STUB_CACHED=3569   # the runner warm in `up` needs a cached prefix
base=(QUEUE=none ENGINE=bench/stub-engine.py HEALTH_WAIT=30 DIR="$out/run" PORT="${PORT:-18534}" N=4)
: > "$FAKE_OUT"
gone() {  # pid seconds
  local i
  for i in $(seq $(( $2 * 10 ))); do
    if ! kill -0 "$1" 2> /dev/null; then return 0; fi
    sleep 0.1
  done
  return 1
}
fails=0
case_sig() {  # name signal stuck limit
  local name=$1 sig=$2 stuck=$3 lim=$4 pid epid
  rm -rf "$out/run"
  env STUB_STUCK="$stuck" bench/ailab/serve.sh up "${base[@]}" > "$out/$name.up.log" 2>&1 \
    || { echo "FAIL $name: up failed, see $out/$name.up.log"; fails=$((fails + 1)); return 0; }
  pid=$(. "$out/run/serve.env"; echo "$PID"); epid=$(. "$out/run/serve.env"; echo "$ENGINE_PID")
  kill "-$sig" "$pid"
  gone "$pid" 40 || kill -KILL "$pid" 2> /dev/null || :
  if gone "$epid" "$lim"; then
    echo "PASS $name: SIG$sig on baro-serve $pid, engine $epid gone within ${lim} s"
  else
    kill -KILL "$epid" 2> /dev/null || :
    echo "FAIL $name: engine $epid alive ${lim} s after SIG$sig on baro-serve $pid"; fails=$((fails + 1))
  fi
  rm -rf "$out/run"
}
case_sig term TERM "" 5
case_sig kill KILL "" 5
case_sig kill-stuck KILL 1 5
if [ -n "${SLOW:-}" ]; then case_sig term-stuck TERM 1 40; fi
# the runner warm in `up` fails the start when ask 2 restores no cached prefix (the cold-burst 429 guard)
rm -rf "$out/run"
if STUB_CACHED=0 bench/ailab/serve.sh up "${base[@]}" > "$out/warmfail.up.log" 2>&1 || ! grep -q "FAIL up/warm" "$out/warmfail.up.log"; then
  echo "FAIL warm-fail: up with cached 0 did not fail with 'FAIL up/warm'"; fails=$((fails + 1)); else echo "PASS warm-fail: up refused a server whose runner warm restores no prefix"; fi
rm -rf "$out/run"
bench/ailab/serve.sh up "${base[@]}" > "$out/down.up.log" 2>&1 || { echo "FAIL down: up failed, see $out/down.up.log"; exit 1; }
bench/ailab/serve.sh down DIR="$out/run" || { echo "FAIL down: serve.sh down"; fails=$((fails + 1)); }
if [ -e "$out/run/serve.env" ]; then echo "FAIL down: serve.env left behind"; fails=$((fails + 1)); fi
[ "$fails" -eq 0 ] || { echo "FAIL serve-selftest: $fails case(s)"; exit 1; }
echo "PASS serve-selftest"
