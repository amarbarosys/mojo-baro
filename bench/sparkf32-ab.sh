#!/usr/bin/env bash
# sparkf32 A/B: bf16 GEMV activations (control, -D BARO_ACT_BF16=1) vs f32 (default) on the K-quant packs, one GPU job
# (bench/sparkq4k-protocol.md amendment A2). Usage: bench/sparkf32-ab.sh OUT   (re-execs itself under gpu-wait)
#   MODELS="vibe minicpm5"   arms per model: vibe runs both activation builds, minicpm5 f32 only (its bf16 numbers are
#                            the sparkq4k receipts on the same kernels); PROMPTS=glob subset; EXPLORE=1 as in sparkq4k-ab.sh
# Per arm: bench/dense-run.sh (llama.cpp CPU control row, no-spec tok/s_gen); read-back is the engine's "act:" word.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bench/sparkf32-ab.sh OUT}
MODELS=${MODELS:-"vibe minicpm5"}; PROMPTS=${PROMPTS:-}; EXPLORE=${EXPLORE:-0}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ -x "$HOME/.local/bin/gpu-wait" ]; then
  mkdir -p "$OUT"
  exec "$HOME/.local/bin/gpu-wait" run --timeout 5400 -- env MODELS="$MODELS" PROMPTS="$PROMPTS" EXPLORE="$EXPLORE" \
    bash -c "'$0' '$OUT' > '$OUT/job.log' 2>&1"
fi
mkdir -p "$OUT"
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }
if [ "$EXPLORE" = 1 ]; then echo "EXPLORE=1: preflight check skipped, verdict capped at UNVERIFIED" | tee "$OUT/SUMMARY.txt"
else bench/preflight.sh --check || die preflight "no passing bench/preflight.sh for this tree"; : > "$OUT/SUMMARY.txt"; fi
M=$HOME/Models
gguf() { case $1 in vibe) echo "$M/vibethinker-1.5b-gguf/vibethinker-1.5b-Q4_K_M.gguf";; minicpm5) echo "$M/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf";; esac; }
prof() { case $1 in vibe) echo "$M/vibethinker-1.5b-gguf/profile";; minicpm5) echo "$M/minicpm5-2b/profile";; esac; }
pack() { case $1 in vibe) echo .work/sparkq4k/packs/vibe-kq;; minicpm5) echo .work/sparkq4k/packs/minicpm5-kq;; esac; }
arms() { case $1 in vibe) echo "bf16 f32";; *) echo "f32";; esac; }
defs() { [ "$1" = bf16 ] && echo "-D BARO_ACT_BF16=1" || echo "-D BARO_ACT_BF16=0"; }
{ echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')'"
  echo "models=$MODELS prompts='${PROMPTS:-all}' explore=$EXPLORE min_pct=90 min_ref=16 spec=0 gen=64"
  for m in $MODELS; do echo "model $m tmax=$(grep -ho 'TMAX = [0-9]*' "$(prof "$m")/profile.mojo") pack=$(pack "$m") index_sha=$(sha256sum "$(pack "$m")/index.txt" | cut -c1-16) arms='$(arms "$m")'"; done
  echo "power_cap=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)"
} | tee "$OUT/arm.txt"
export MIN_PCT=90 MIN_REF=16 PROMPTS
fails=0
for m in $MODELS; do
  for a in $(arms "$m"); do
    d="$OUT/$m-$a"; mkdir -p "$d"
    echo "== $m $a $(date +%T)" | tee -a "$OUT/SUMMARY.txt"
    DEFS="$(defs "$a")" LLAMA_CACHE="$OUT/refcache-$m" bench/dense-run.sh "$(gguf "$m")" "$(pack "$m")" "$(prof "$m")" "$d" > "$d/run.log" 2>&1 || true
    v=$(grep -E '^(PASS|FAIL) agreement' "$d/SUMMARY.txt" || echo "FAIL agreement: no verdict line, see $d/run.log")
    echo "$m-$a $v" | tee -a "$OUT/SUMMARY.txt"
    want=$([ "$a" = bf16 ] && echo bfloat16 || echo float32)
    rb=$(grep -h -m1 -o 'act: [a-z0-9]*' "$d"/*.ours-nospec.log 2>/dev/null | sort | uniq -c | tr '\n' ' ')
    echo "$m-$a defs='$(defs "$a")' engine_sha=$(sha256sum "$d/spark-engine" 2>/dev/null | cut -c1-16) readback='$rb'" | tee -a "$OUT/arm.txt"
    case "$rb" in *"act: $want"*) ;; *) echo "FAIL $m-$a: read-back '$rb' is not the $want arm" | tee -a "$OUT/SUMMARY.txt"; fails=$((fails + 1));; esac
    if grep -l -E 'NOT-RESIDENT|Error|error|FAIL' "$d"/*.ours-*.log 2>/dev/null | head -3 | grep -q .; then
      echo "FAIL $m-$a: fail word in an engine log" | tee -a "$OUT/SUMMARY.txt"; fails=$((fails + 1)); fi
    case $v in PASS*) ;; *) [ "$m" = vibe ] || fails=$((fails + 1));; esac
  done
  [ "$m" = vibe ] || continue
  python3 - "$OUT" <<'PY' | tee -a "$OUT/SUMMARY.txt"
import sys, statistics
out = sys.argv[1]
def rows(a):
    return {r.split()[0]: r.split() for r in open(f"{out}/vibe-{a}/results.txt").read().splitlines()[1:]}
b, f = rows("bf16"), rows("f32")
common = sorted(set(b) & set(f))
tb = [float(b[p][2]) for p in common]; tf = [float(f[p][2]) for p in common]
rat = sorted(float(f[p][2]) / float(b[p][2]) for p in common)
print(f"speed vibe: n={len(common)} tok/s_gen median bf16 {statistics.median(tb):.2f} f32 {statistics.median(tf):.2f} "
      f"ratio-of-medians {statistics.median(tf)/statistics.median(tb):.3f} paired median {statistics.median(rat):.3f} "
      f"min {rat[0]:.3f} max {rat[-1]:.3f}; p04 bf16 {b.get('p04-list-planets', ['', '', '', 'n/a'])[3]} f32 {f.get('p04-list-planets', ['', '', '', 'n/a'])[3]}")
PY
done
"$HOME/.local/bin/gpu-wait" stats --days 1 >> "$OUT/SUMMARY.txt" 2>/dev/null || true
if [ "$fails" != 0 ]; then echo "FAIL $fails arms, see $OUT/SUMMARY.txt" | tee -a "$OUT/SUMMARY.txt"; exit 1; fi
if [ "$EXPLORE" = 1 ]; then echo "UNVERIFIED (EXPLORE=1): not a gate" | tee -a "$OUT/SUMMARY.txt"; exit 3; fi
echo "PASS sparkf32-ab: $OUT/SUMMARY.txt" | tee -a "$OUT/SUMMARY.txt"
