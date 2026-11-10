#!/usr/bin/env bash
# usage: bench/tier-bell-long.sh ENGINE OUTDIR [LEN]
# Tier doorbell on a long prompt (MoE, BARO_TIER=64 pinned zero-copy). Batched prefill enqueues every
# chunk without a host sync, so the first decode doorbell sat behind seconds of queued prefill and the
# 5 s spin watchdog raised "doorbell for layer 0 never rang" (2026-10-09, 32k). Arms: BELL1 = doorbell
# on, BELL0 = BARO_TIER_BELL=0 (prepare() path, the control). PASS = every arm finishes with a done
# line, no watchdog error, generated ids equal across arms. ARMS="BELL1" runs one arm. LEN = the
# a2-prompts set (default 32768, prompt p01).
set -euo pipefail
cd "$(dirname "$0")/.."
eng=${1:?engine}; out=$(realpath -m "${2:?out dir}"); len=${3:-32768}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && command -v gpu-wait >/dev/null; then
  exec gpu-wait run --vram 20 --timeout 1800 -- env ARMS="${ARMS:-BELL1 BELL0}" "$0" "$eng" "$out" "$len"
fi
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] || fail setup "engine missing: $eng"
[ -d .work/a2-prompts/L$len ] || bench/a2-prompts.sh
tf=.work/a2-prompts/L$len/p01-water.tokens
[ -s "$tf" ] || fail setup "prompt missing: $tf"
mkdir -p "$out"
T=${TIER_PACK:-$HOME/.cache/baro/qwen35moe-f2628b18234dbbb4/tier-e5c7fc0d}
echo "{\"id\":1,\"prompt\":[$(tr -s ' \n' ',' < "$tf" | sed 's/,$//')],\"n\":32,\"spec\":false,\"temperature\":0}" > "$out/req.jsonl"
echo "engine $eng sha256 $(sha256sum "$eng" | cut -c1-16) pack $T len $len prompt_ids $(wc -w < "$tf") load $(cut -d' ' -f1-3 /proc/loadavg)" | tee "$out/arm.txt"
for arm in ${ARMS:-BELL1 BELL0}; do
  e="BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_ZC=1"; [ "$arm" = BELL0 ] && e="$e BARO_TIER_BELL=0"
  rc=0; env BARO_PACK="$T" BARO_SERVE=1 BARO_SPEC=0 BARO_TMAX=$((len + 1024)) $e timeout 1500 "$eng" < "$out/req.jsonl" > "$out/$arm.log" 2>&1 || rc=$?
  grep -oE 'mode [a-z+]+( cpu_max [0-9]+)?  bell [A-Za-z]+' "$out/$arm.log" | head -1 | sed "s/^/$arm /" | tee -a "$out/arm.txt"
  if grep -m1 -oE 'never rang[^"]*' "$out/$arm.log"; then fail "$arm" "doorbell watchdog fired, see $out/$arm.log"; fi
  [ $rc -eq 0 ] || fail "$arm" "engine exit $rc, see $out/$arm.log"
  if grep -qE 'mega fail word: [1-9]' "$out/$arm.log"; then fail "$arm" "fail word set"; fi
  grep -q '"done"' "$out/$arm.log" || fail "$arm" "no done line, see $out/$arm.log"
  n=$(grep -cE '"tok":[0-9]+' "$out/$arm.log" || true); [ "$n" -gt 0 ] || fail "$arm" "no tokens served"
  md5=$(grep -oE '"tok":[0-9]+' "$out/$arm.log" | md5sum | cut -c1-12)
  echo "$arm env='$e' tok $n ids_md5 $md5 $(grep -m1 '"done"' "$out/$arm.log" | grep -oE '"(prefill_s|tok_s|cached)":[0-9.]+' | tr '\n' ' ')" | tee -a "$out/arm.txt"
  echo "$md5" > "$out/$arm.md5"
done
if [ -f "$out/BELL1.md5" ] && [ -f "$out/BELL0.md5" ]; then
  cmp -s "$out/BELL1.md5" "$out/BELL0.md5" || fail identity "BELL1 ids $(cat "$out/BELL1.md5") != BELL0 $(cat "$out/BELL0.md5")"
fi
echo "PASS tier-bell-long"
