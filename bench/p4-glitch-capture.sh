#!/usr/bin/env bash
# P4 round 2: catch the iGPU glitch with its logits. K fresh engine-qwen processes, ONE
# request each (same prompt, T=0, top_logprobs=1 so the engine dumps every step's logits
# row into BARO_DUMP_LOGITS_DIR), then tools/p4-row-diff.py compares token streams and
# rows bitwise. One request per fresh process means no request history exists, so any
# disagreement between runs is not state bleed.
# Run under gpu-wait. P4_IGPU_ENGINE / P4_IGPU_PACK / P4_OUT / P4_CAP_PROMPT / P4_CAP_RUNS.
set -euo pipefail
export PATH="/opt/rocm/bin:$HOME/iTools/bin:/usr/bin:/bin:${PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${P4_IGPU_ENGINE:-$ROOT/.work/p4/bin/engine-qwen}
PACK=${P4_IGPU_PACK:-$ROOT/.work/team-B/codex/p4/qwen-pack}
OUT=${P4_OUT:-$ROOT/.work/p4/glitch-capture}
TARGET=${P4_CAP_PROMPT:-p08-sql}
RUNS=${P4_CAP_RUNS:-12}
GEN=${P4_CAP_GEN:-64}
fail() { echo "FAIL $1: $2 (log ${3:-none})"; exit 1; }

PROMPT="$ROOT/bench/mtp-prompts/$TARGET.tokens"
for p in "$ENGINE" "$PACK/index.txt" "$PACK/pack.bin" "$PROMPT" "$ROOT/tools/p4-row-diff.py"; do
  [ -e "$p" ] || fail preflight "missing $p"
done
[ "$(sha256sum "$PACK/index.txt" | awk '{print $1}')" = \
  e2ef587fa49be96a8d95d2715038398b8a45757428fb0f80c7469e9be2471084 ] || fail preflight "Qwen7 pack index hash mismatch"
mkdir -p "$OUT"
printf '{"id":1,"prompt":[%s],"n":%d,"spec":false,"temperature":0,"top_logprobs":1}\n' \
  "$(tr -s ' \n' ',' < "$PROMPT" | sed 's/^,//; s/,$//')" "$GEN" > "$OUT/req.jsonl"
python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read()); assert len(d["prompt"])>1' "$OUT/req.jsonl" \
  || fail preflight "request line is not valid JSON" "$OUT/req.jsonl"
if [ "${P4_CPU_PREFLIGHT:-0}" = 1 ]; then
  python3 "$ROOT/tools/p4-row-diff.py" --selftest || fail preflight "p4-row-diff selftest"
  echo "PASS p4-glitch-capture CPU preflight: paths, pack hash, request line, analyzer selftest"
  exit 0
fi
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run under gpu-wait"

{
  echo "engine=$ENGINE sha256=$(sha256sum "$ENGINE" | awk '{print $1}')"
  echo "pack=$PACK"
  echo "target=$TARGET runs=$RUNS gen=$GEN"
  echo "igpu_env:"; igpu-env
} > "$OUT/arm.txt" 2>&1

for k in $(seq 1 "$RUNS"); do
  d="$OUT/run$k"; rm -rf "$d"; mkdir -p "$d"
  igpu-env --run env BARO_SERVE=1 BARO_PACK="$PACK" BARO_DUMP_LOGITS_DIR="$d" "$ENGINE" < "$OUT/req.jsonl" > "$d/engine.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 120); do grep -q '"ready":true' "$d/engine.log" 2>/dev/null && break; sleep 1; done
  rocm-smi --showpids > "$d/showpids.log" 2>&1 || true
  wait "$pid" || fail "run$k" "engine exited non-zero" "$d/engine.log"
  grep -q '"error"' "$d/engine.log" && fail "run$k" "engine rejected the request" "$d/engine.log"
  grep -q "engine-qwen" "$d/showpids.log" || fail "run$k" "rocm-smi did not list the engine PID" "$d/showpids.log"
  grep '^generated:' "$d/engine.log" | sed 's/^generated: *//; s/ *$//' > "$d/tokens"
  [ "$(wc -w < "$d/tokens")" -eq "$GEN" ] || fail "run$k" "expected $GEN tokens" "$d/engine.log"
  [ "$(ls "$d"/row-*.bin | wc -l)" -eq "$GEN" ] || fail "run$k" "expected $GEN logits rows" "$d/engine.log"
  echo "ran run$k: $(grep '^tok/s_gen' "$d/engine.log")"
done
python3 "$ROOT/tools/p4-row-diff.py" "$OUT" | tee "$OUT/REPORT.txt"
