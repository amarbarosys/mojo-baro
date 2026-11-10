#!/usr/bin/env bash
# usage: bench/gtt-check.sh ENGINE OUTDIR    runs itself under gpu-wait
# Three one-shot runs of ENGINE on the default pack: A default env, B BARO_GTT_WARN_BYTES=1, C B plus BARO_GTT_REFUSE=1.
# Expect A no WARN exit 0, B WARN exit 0, C WARN + FAIL gtt exit 3 and no GENERATED. Exit 1 with FAIL on any miss.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
. bench/min16k.sh; min16k "${BARO_TMAX:-16384}" 16000
eng=$(realpath "${1:?engine}"); out=$(realpath -m "${2:?outdir}")
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$out"
  exec gpu-wait run --label gtt-check --vram 8 --timeout 600 -- env GPU_WAITING_ROOM_JOB=1 PATH="$PATH" "$0" "$eng" "$out"
fi
export BARO_TMAX=${BARO_TMAX:-16384} BARO_PROMPT=$here/bench/mtp-prompts/p01-water.tokens
fails=0
run() {  # label want_rc env...
  local label=$1 want=$2; shift 2; local rc=0
  env "$@" "$eng" > "$out/$label.log" 2>&1 < /dev/null || rc=$?
  echo "$label: exit $rc (want $want) tmax $(grep -m1 '^TMAX:' "$out/$label.log" || echo '?') | $(grep -m1 '^GTT:' "$out/$label.log" || echo 'no GTT line') | warn=$(grep -c '^WARN gtt' "$out/$label.log" || true) fail=$(grep -c '^FAIL gtt' "$out/$label.log" || true) generated=$(grep -c '^GENERATED' "$out/$label.log" || true) | $(grep -m1 -o 'tok/s_gen: [0-9.]*' "$out/$label.log" || true)"
  [ "$rc" = "$want" ] || { echo "FAIL $label: exit $rc, want $want, see $out/$label.log"; fails=$((fails+1)); }
}
run A 0 BARO_X=1
run B 0 BARO_GTT_WARN_BYTES=1
run C 3 BARO_GTT_WARN_BYTES=1 BARO_GTT_REFUSE=1
[ "$(grep -c '^WARN gtt' "$out/A.log" || true)" = 0 ] || { echo "FAIL A: WARN at default threshold"; fails=$((fails+1)); }
[ "$(grep -c '^WARN gtt' "$out/B.log" || true)" -ge 1 ] || { echo "FAIL B: no WARN at threshold 1"; fails=$((fails+1)); }
grep -q '^FAIL gtt' "$out/C.log" || { echo "FAIL C: no FAIL line"; fails=$((fails+1)); }
! grep -q '^GENERATED' "$out/C.log" || { echo "FAIL C: ran past the refusal"; fails=$((fails+1)); }
[ "$fails" = 0 ] && echo "PASS gtt-check" || exit 1
