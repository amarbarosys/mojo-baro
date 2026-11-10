#!/usr/bin/env bash
# GTT base rate and phase timeline for the q4x 16K prefill (lane open3, item 3; exchange/open3-item3-prereg.md).
# usage: bench/q4x-gtt-trace.sh OUT [N_PER_PROMPT=6]    runs itself under gpu-wait
# env: Q4X_BIN (engine), Q4X_GGUF, PROMPTS ("a-capabilities b-q4x-source"), PROMPT_DIR, NT (32), EXTRA ("VAR=val ..." added to every run),
#      EXTRA_B (a second arm: every prompt and iteration runs arm A then arm B, order swapped on odd iterations), PLEN (ids per prompt, default 2048, the standard tier: about 15 s a run; 16384 is the long tier, 75 s healthy and minutes under memory pressure)
# Per run: OUT/run-NN-<prompt>.jsonl (tools/q4x-gtt-trace.py), summary line in OUT/summary.txt; ends FAIL k/N when a run died or read a fail word.
set -euo pipefail
OUT=${1:?out dir}; NPP=${2:-6}
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/bench/min16k.sh"; min16k 16640 16000
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$OUT" "$REPO/.work"
  [ "$REPO/.work/q4x-gtt-report" -nt "$REPO/tools/q4x-gtt-report.mojo" ] || "$REPO/.venv/bin/mojo" build "$REPO/tools/q4x-gtt-report.mojo" -I "$REPO" -I "$HOME/iTools/lib/mojo" -o "$REPO/.work/q4x-gtt-report" \
    || { echo "FAIL setup: tools/q4x-gtt-report.mojo does not build"; exit 1; }
  exec gpu-wait run --label open3-gtt-trace --vram 20 --timeout 7200 -- env GPU_WAITING_ROOM_JOB=1 PATH="$PATH" Q4X_BIN="${Q4X_BIN:-}" Q4X_GGUF="${Q4X_GGUF:-}" \
    PROMPTS="${PROMPTS:-}" PROMPT_DIR="${PROMPT_DIR:-}" NT="${NT:-}" EXTRA="${EXTRA:-}" EXTRA_B="${EXTRA_B-@@}" PLEN="${PLEN:-}" "$REPO/bench/q4x-gtt-trace.sh" "$OUT" "$NPP"
fi
cd "$REPO"; "$REPO/bench/preflight.sh" --check
BIN=$(realpath "${Q4X_BIN:-.work/open3/i3/q4x}"); GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
PROMPTS=${PROMPTS:-"a-capabilities b-q4x-source"}; PD=${PROMPT_DIR:-exchange/receipts/open3/item3}; NT=${NT:-32}; PLEN=${PLEN:-2048}
SUF=""; [ "$PLEN" = 2048 ] && SUF=".2k"; [ "$PLEN" = 16384 ] || [ "$PLEN" = 2048 ] || { echo "FAIL setup: PLEN $PLEN has no prompt files"; exit 1; }
[ "${EXTRA_B-@@}" = "@@" ] && ARMS="A" || ARMS="A B"
for f in "$BIN" "$GGUF"; do [ -e "$f" ] || { echo "FAIL setup: missing $f"; exit 1; }; done
for p in $PROMPTS; do [ "$(jq length "$PD/$p$SUF.prompt")" = "$PLEN" ] || { echo "FAIL setup: $PD/$p$SUF.prompt is not $PLEN ids"; exit 1; }; done
mkdir -p "$OUT"
{ echo "bin=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16) gguf=$GGUF commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain serve kernels tools bench | wc -l)"
  for p in $PROMPTS; do echo "prompt $p$SUF sha256=$(sha256sum "$PD/$p$SUF.prompt" | cut -c1-16) ids=$PLEN"; done
  echo "knobs: Q4X_TMAX=16640 Q4X_CTX=16640 Q4X_PREFILL=1 Q4X_CHUNK=8192 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 NT=$NT N_PER_PROMPT=$NPP arms=[$ARMS] extra_A='${EXTRA:-}' extra_B='${EXTRA_B-}'"
  echo "gtt_start $(cat /sys/class/drm/card1/device/mem_info_gtt_used) vram_free_start $(~/iTools/bin/vram-free 2>/dev/null || echo ?)"
  echo "job=${GPU_WAITING_ROOM_JOB:-} clock=$(date +%FT%T%z)"; } | tee "$OUT/arm.txt"
fail=0; n=0
for i in $(seq 1 "$NPP"); do
  for p in $PROMPTS; do
   ord="$ARMS"; if [ "$ARMS" != A ] && [ $((i % 2)) = 0 ]; then ord="B A"; fi
   for arm in $ord; do
    n=$((n + 1)); r=$(printf '%02d' "$n"); f="$OUT/run-$r-$arm-$p"
    X="${EXTRA:-}"; if [ "$arm" = B ]; then X="${EXTRA_B-}"; fi
    rc=0
    env $X Q4X_TMAX=16640 Q4X_CTX=16640 Q4X_PREFILL=1 Q4X_CHUNK=8192 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 \
      "$REPO/tools/q4x-gtt-trace.py" "$f.jsonl" "$BIN" "$GGUF" "$NT" "$PD/$p$SUF.prompt" || rc=$?
    line=$(python3 - "$f.jsonl" "$r" "$p/$arm" "$rc" <<'PY'
import json, sys
f, r, p, rc = sys.argv[1:]
rec = [json.loads(l) for l in open(f)]
s = [x for x in rec if x["k"] == "s"]; L = [x["line"] for x in rec if x["k"] == "l"]
g0, pk = s[0]["gtt"], max(s, key=lambda x: x["gtt"]); eg = max((x["eng"].get("gtt", 0) for x in s), default=0)
ex = [x for x in s if x["gtt"] - g0 > 2**30]
own = max(pk["cl"], key=lambda c: c[2])[1:] if pk["cl"] else "-"
bad = [l for l in L if "NOT-RESIDENT" in l or "FAIL" in l]
e0, e1 = s[0]["eng"], s[-1]["eng"]; rdmb = (e1.get("rd", 0) - e0.get("rd", 0)) / 2**20; maj = e1.get("maj", 0) - e0.get("maj", 0)
slots = next((l for l in L if "expert cache" in l), "-"); ids = next((l for l in L if l.startswith("ids")), "")
print(f"run {r} {p} rc={rc} samples={len(s)} gtt_first={g0/2**30:.2f} gtt_peak={pk['gtt']/2**30:.2f} t_peak={(pk['t']-s[0]['t'])/1e9:.1f}s over1G_samples={len(ex)} "
      f"eng_gtt_max_gb={eg/2**20:.2f} disk_rd_mb={rdmb:.0f} majflt={maj} top_other={own} nids={len(ids.split())-1} failwords={len(bad)} | {slots[:70]}")
PY
)
    echo "$line" | tee -a "$OUT/summary.txt"
    case "$line" in *" rc=0 "*"failwords=0"*) ;; *) fail=$((fail + 1));; esac
   done
  done
done
"$REPO/.work/q4x-gtt-report" "$OUT" | tee -a "$OUT/summary.txt"
[ "$fail" = 0 ] || { echo "FAIL $fail/$n: runs died or read a fail word, see $OUT/summary.txt"; exit 1; }
