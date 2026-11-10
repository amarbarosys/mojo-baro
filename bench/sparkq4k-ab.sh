#!/usr/bin/env bash
# sparkq4k A/B: q8 pack (control) vs native K-quant pack, three dense models, one GPU job (bench/sparkq4k-protocol.md).
# Usage: bench/sparkq4k-ab.sh OUT            (re-execs itself under gpu-wait; env and redirects stay inside the job)
#   MODELS="vibe minicpm5 qwen7b"  subset of arms (default all three)
#   PROMPTS=glob                  bench/mtp-prompts subset for iteration (P18: a verdict needs the full set)
#   EXPLORE=1                     skip the preflight stamp check; verdict capped at UNVERIFIED, exit 3
# Per model: bench/dense-run.sh on the q8 pack, then on the K-quant pack (arm order alternates per model),
# identity + no-spec tok/s_gen per prompt; the engine's own "matrices: q8 N kquant M" line is the read-back.
# PASS needs every arm's gate verdict as the protocol bars say; the speed ratio is reported, never judged.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bench/sparkq4k-ab.sh OUT}
MODELS=${MODELS:-"vibe minicpm5 qwen7b"}; PROMPTS=${PROMPTS:-}; EXPLORE=${EXPLORE:-0}
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ -x "$HOME/.local/bin/gpu-wait" ]; then
  mkdir -p "$OUT"
  exec "$HOME/.local/bin/gpu-wait" run --timeout 7200 -- env MODELS="$MODELS" PROMPTS="$PROMPTS" EXPLORE="$EXPLORE" \
    bash -c "'$0' '$OUT' > '$OUT/job.log' 2>&1"
fi
mkdir -p "$OUT"
die() { echo "FAIL $1: $2" | tee -a "$OUT/SUMMARY.txt"; exit 1; }
if [ "$EXPLORE" = 1 ]; then echo "EXPLORE=1: preflight check skipped, verdict capped at UNVERIFIED" | tee "$OUT/SUMMARY.txt"
else bench/preflight.sh --check || die preflight "no passing bench/preflight.sh for this tree"; : > "$OUT/SUMMARY.txt"; fi
M=$HOME/Models
gguf() { case $1 in vibe) echo "$M/vibethinker-1.5b-gguf/vibethinker-1.5b-Q4_K_M.gguf";; minicpm5) echo "$M/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf";;
  qwen7b) echo "$M/qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-3c6c25c.gguf";; esac; }
prof() { case $1 in vibe) echo "$M/vibethinker-1.5b-gguf/profile";; minicpm5) echo "$M/minicpm5-2b/profile";; qwen7b) echo "$M/qwen2.5-7b-instruct-import/profile";; esac; }
pack() { case $1-$2 in vibe-q8) echo "$M/vibethinker-1.5b-gguf/pack";; minicpm5-q8) echo "$M/minicpm5-2b/pack";; qwen7b-q8) echo .work/sparkq4k/packs/qwen7b-q8;;
  vibe-kq) echo .work/sparkq4k/packs/vibe-kq;; minicpm5-kq) echo .work/sparkq4k/packs/minicpm5-kq;; qwen7b-kq) echo .work/sparkq4k/packs/qwen7b-kq;; esac; }
{ echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')'"
  echo "models=$MODELS prompts='${PROMPTS:-all}' explore=$EXPLORE min_pct=90 min_ref=16 spec=0 gen=64 tmax=$(grep -ho 'TMAX = [0-9]*' "$(prof vibe)/profile.mojo")"
  echo "power_cap=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1) vram_used=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)"
  for m in $MODELS; do for a in q8 kq; do p=$(pack "$m" "$a")
    echo "arm $m-$a pack=$p index_sha=$(sha256sum "$p/index.txt" | cut -c1-16) pack_bytes=$(stat -c %s "$p/pack.bin") gguf_sha=$(sha256sum "$(gguf "$m")" | cut -c1-16)"
  done; done
} | tee "$OUT/arm.txt"
export MIN_PCT=90 MIN_REF=16 PROMPTS
order() { case $1 in minicpm5) echo "kq q8";; *) echo "q8 kq";; esac; }
fails=0
for m in $MODELS; do
  for a in $(order "$m"); do
    d="$OUT/$m-$a"; mkdir -p "$d"
    echo "== $m $a $(date +%T)" | tee -a "$OUT/SUMMARY.txt"
    LLAMA_CACHE="$OUT/refcache-$m" bench/dense-run.sh "$(gguf "$m")" "$(pack "$m" "$a")" "$(prof "$m")" "$d" > "$d/run.log" 2>&1 || true
    v=$(grep -E '^(PASS|FAIL) agreement' "$d/SUMMARY.txt" || echo "FAIL agreement: no verdict line, see $d/run.log")
    echo "$m-$a $v" | tee -a "$OUT/SUMMARY.txt"
    echo "$m-$a engine_sha=$(sha256sum "$d/spark-engine" 2>/dev/null | cut -c1-16) readback='$(grep -h -m1 -o 'matrices: q8 [0-9]* kquant [0-9]*' "$d"/*.ours-nospec.log 2>/dev/null | sort | uniq -c | tr '\n' ' ')'" | tee -a "$OUT/arm.txt"
    if grep -l -E 'NOT-RESIDENT|Error|error|FAIL' "$d"/*.ours-*.log 2>/dev/null | head -3 | grep -q .; then
      echo "FAIL $m-$a: fail word in an engine log: $(grep -l -E 'NOT-RESIDENT|Error|error|FAIL' "$d"/*.ours-*.log | head -3 | tr '\n' ' ')" | tee -a "$OUT/SUMMARY.txt"; fails=$((fails + 1))
    fi
    case $v in PASS*) ;; *) [ "$m-$a" = vibe-q8 ] || fails=$((fails + 1));; esac
    [ "$m-$a" = vibe-q8 ] && case $v in PASS*) echo "FAIL vibe-q8: control passed, the known p04 failure did not reproduce" | tee -a "$OUT/SUMMARY.txt"; fails=$((fails + 1));; esac
  done
  python3 - "$OUT" "$m" <<'PY' | tee -a "$OUT/SUMMARY.txt"
import sys, statistics
out, m = sys.argv[1], sys.argv[2]
def rows(a):
    return {r.split()[0]: r.split() for r in open(f"{out}/{m}-{a}/results.txt").read().splitlines()[1:]}
q8, kq = rows("q8"), rows("kq")
common = sorted(set(q8) & set(kq))
t8 = [float(q8[p][2]) for p in common]; tk = [float(kq[p][2]) for p in common]
rat = sorted(float(kq[p][2]) / float(q8[p][2]) for p in common)
p04 = kq.get("p04-list-planets", ["", "", "", "n/a"])[3]
print(f"speed {m}: n={len(common)} tok/s_gen median q8 {statistics.median(t8):.2f} kq {statistics.median(tk):.2f} "
      f"ratio-of-medians {statistics.median(tk)/statistics.median(t8):.3f} paired median {statistics.median(rat):.3f} "
      f"min {rat[0]:.3f} max {rat[-1]:.3f}; kq p04 {p04}")
PY
done
p04=$(awk '$1=="p04-list-planets"{print $4}' "$OUT/vibe-kq/results.txt" 2>/dev/null || true)
if [ -n "$p04" ] && [ "${p04%/*}" -lt 58 ]; then echo "FAIL vibe-kq: p04 $p04 < 58/64" | tee -a "$OUT/SUMMARY.txt"; fails=$((fails + 1)); fi
"$HOME/.local/bin/gpu-wait" stats --days 1 >> "$OUT/SUMMARY.txt" 2>/dev/null || true
if [ "$fails" != 0 ]; then echo "FAIL $fails arms, see $OUT/SUMMARY.txt" | tee -a "$OUT/SUMMARY.txt"; exit 1; fi
if [ "$EXPLORE" = 1 ]; then echo "UNVERIFIED (EXPLORE=1): all arms as predicted, not a gate" | tee -a "$OUT/SUMMARY.txt"; exit 3; fi
echo "PASS sparkq4k-ab: $OUT/SUMMARY.txt" | tee -a "$OUT/SUMMARY.txt"
