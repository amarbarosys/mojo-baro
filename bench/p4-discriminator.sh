#!/usr/bin/env bash
# P4 round 2 discriminator: is the iGPU identity miss state bleed or gfx1030 numerics?
# Drives engine-qwen directly over its stdin protocol (no baro-serve in the loop).
#   fresh arms : one request (P4_DISC_PROMPT) in a fresh process, P4_DISC_FRESH times
#   seq arms   : several requests in one process, the target prompt after other history
# Verdict: fresh arms differ among themselves -> NONDETERMINISTIC-FRESH (numerics or race,
# one request, no history); fresh arms agree but a seq arm's target output differs ->
# HISTORY-DEPENDENT (state bleed); everything equal -> NOT-REPRODUCED.
# Run under gpu-wait; the script strips gpu-wait's injected ROCm defaults and reaches
# the iGPU through igpu-env, exactly as .work/p4/run-two-engines.sh does.
set -euo pipefail
export PATH="/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin:${PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${P4_IGPU_ENGINE:-$ROOT/.work/p4/bin/engine-qwen}
PACK=${P4_IGPU_PACK:-$ROOT/.work/team-B/codex/p4/qwen-pack}
OUT=${P4_OUT:-$ROOT/.work/p4/discriminator}
TARGET=${P4_DISC_PROMPT:-p06-translate}
FRESH=${P4_DISC_FRESH:-3}
SEQS=${P4_DISC_SEQS:-"p05-math,$TARGET,$TARGET,$TARGET p01-water,$TARGET"}
GEN=${P4_DISC_GEN:-64}
fail() { echo "FAIL $1: $2 (log ${3:-none})"; exit 1; }

for p in "$ENGINE" "$PACK/index.txt" "$PACK/pack.bin" "$ROOT/bench/mtp-prompts/$TARGET.tokens"; do
  [ -e "$p" ] || fail preflight "missing $p"
done
[ "$(sha256sum "$PACK/index.txt" | awk '{print $1}')" = \
  e2ef587fa49be96a8d95d2715038398b8a45757428fb0f80c7469e9be2471084 ] || fail preflight "Qwen7 pack index hash mismatch"
mkdir -p "$OUT"
reqline() { # id prompt-name
  local f="$ROOT/bench/mtp-prompts/$2.tokens"
  [ -f "$f" ] || fail preflight "missing prompt $f"
  printf '{"id":%d,"prompt":[%s],"n":%d,"spec":false}\n' "$1" "$(tr -s ' \n' ',' < "$f" | sed 's/^,//; s/,$//')" "$GEN"
}
if [ "${P4_CPU_PREFLIGHT:-0}" = 1 ]; then
  reqline 1 "$TARGET" > "$OUT/preflight-req.jsonl"
  python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read()); assert len(d["prompt"])>1 and d["n"]>0' "$OUT/preflight-req.jsonl" \
    || fail preflight "request line is not valid JSON" "$OUT/preflight-req.jsonl"
  echo "PASS p4-discriminator CPU preflight: paths, pack hash, request line"
  exit 0
fi
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run under gpu-wait"

{
  echo "engine=$ENGINE sha256=$(sha256sum "$ENGINE" | awk '{print $1}')"
  echo "pack=$PACK"
  echo "target=$TARGET fresh=$FRESH gen=$GEN seqs=$SEQS"
  echo "igpu_env:"; igpu-env
} > "$OUT/arm.txt" 2>&1

run_arm() { # arm-name prompt,prompt,...
  local arm=$1 id=0 name
  : > "$OUT/$arm.req.jsonl"
  for name in ${2//,/ }; do id=$((id + 1)); reqline "$id" "$name" >> "$OUT/$arm.req.jsonl"; done
  igpu-env --run env BARO_SERVE=1 BARO_PACK="$PACK" "$ENGINE" < "$OUT/$arm.req.jsonl" > "$OUT/$arm.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do grep -q '"ready":true' "$OUT/$arm.log" 2>/dev/null && break; sleep 1; done
  rocm-smi --showpids > "$OUT/$arm.showpids.log" 2>&1 || true
  wait "$pid" || fail "$arm" "engine exited non-zero" "$OUT/$arm.log"
  grep -q '"error"' "$OUT/$arm.log" && fail "$arm" "engine rejected a request" "$OUT/$arm.log"
  [ "$(grep -c '^generated:' "$OUT/$arm.log")" -eq "$id" ] || fail "$arm" "expected $id generated lines" "$OUT/$arm.log"
  grep -q "engine-qwen" "$OUT/$arm.showpids.log" || fail "$arm" "rocm-smi did not list the engine PID" "$OUT/$arm.showpids.log"
  local i=0
  for name in ${2//,/ }; do
    i=$((i + 1))
    grep '^generated:' "$OUT/$arm.log" | sed -n "${i}p" | sed 's/^generated: *//; s/ *$//' > "$OUT/$arm.r$i.$name.tokens"
  done
  echo "ran $arm: $2 ($(grep -c '^tok/s_gen' "$OUT/$arm.log") requests, $(grep '^tok/s_gen' "$OUT/$arm.log" | tail -1))"
}

for k in $(seq 1 "$FRESH"); do run_arm "fresh$k" "$TARGET"; done
s=0
for seq in $SEQS; do s=$((s + 1)); run_arm "seq$s" "$seq"; done

ref="$OUT/fresh1.r1.$TARGET.tokens"
fresh_diff=0; hist_diff=0
{
  echo "arm,request,prompt,first_mismatch_vs_fresh1"
  for f in "$OUT"/fresh*.r*."$TARGET".tokens "$OUT"/seq*.r*."$TARGET".tokens; do
    b=$(basename "$f" .tokens)
    mm=$(paste -d' ' <(tr ' ' '\n' < "$ref") <(tr ' ' '\n' < "$f") | awk '$1 != $2 {print NR - 1; exit}')
    echo "${b%%.*},$(echo "$b" | cut -d. -f2),$TARGET,${mm:-none}"
    if [ -n "$mm" ]; then case "$b" in fresh*) fresh_diff=1 ;; *) hist_diff=1 ;; esac; fi
  done
} > "$OUT/discriminator.csv"
cat "$OUT/discriminator.csv"
if [ "$fresh_diff" = 1 ]; then v=NONDETERMINISTIC-FRESH
elif [ "$hist_diff" = 1 ]; then v=HISTORY-DEPENDENT
else v=NOT-REPRODUCED; fi
echo "VERDICT p4-discriminator: $v" | tee "$OUT/VERDICT.txt"
