#!/usr/bin/env bash
# amdgpu tuning A/B (bench/amdgpu-tune-protocol.md), one gpu-wait job: 12 resident q4x arms over perf level
# (AUTO/HIGH/COMP in blocks ABCCBA) x HSA_ENABLE_INTERRUPT (base/int, alternating), sclk/mclk/power sampled by a
# separate process, perf level read back per arm, CPU seconds per arm (INT read-back), pcie-bw at start and end,
# ids of every arm compared with arm 1. Perf level restored to auto on every exit path.
# usage: bench/amdgpu-tune-ab.sh OUT_DIR      env NEW=q4x-bin PCIEBW=bin NP=10 DRY=1 (no gpu-wait, for a stub NEW)
set -euo pipefail
OUT=${1:?usage: amdgpu-tune-ab.sh OUT_DIR}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next-gsq/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
NEW=${NEW:-$REPO/.work/q4x-serve/q4x2}; PCIEBW=${PCIEBW:-$REPO/.work/amdgpu-tune/pcie-bw}; NP=${NP:-10}
DEV=/sys/class/drm/card1/device; HW=$(ls -d $DEV/hwmon/hwmon* | head -1)
ARMS="AUTO:base AUTO:int HIGH:int HIGH:base COMP:base COMP:int COMP:int COMP:base HIGH:base HIGH:int AUTO:int AUTO:base"
NG=64
if [ -z "${TUNE_INNER:-}" ] && [ -z "${DRY:-}" ]; then
  mkdir -p "$OUT"
  for f in "$NEW" "$PCIEBW" "$GGUF.warm"; do [ -e "$f" ] || { echo "FAIL preflight: $f missing"; exit 1; }; done
  sudo -n true || { echo "FAIL preflight: sudo -n needed for the perf level"; exit 1; }
  rc=0
  gpu-wait run --vram 20 --timeout 1200 -- env TUNE_INNER=1 Q4X_GGUF="$GGUF" NEW="$NEW" PCIEBW="$PCIEBW" NP="$NP" HOME="$HOME" \
    bash "$REPO/bench/amdgpu-tune-ab.sh" "$OUT" || rc=$?
  gpu-wait stats --days 1 | tail -3 >> "$OUT/receipt.txt"
  [ $rc = 0 ] || { echo "FAIL job rc $rc, see $OUT/receipt.txt"; exit $rc; }
  exit 0
fi
cd "$REPO"; mkdir -p "$OUT"
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
unset Q4X_WARM Q4X_CHUNK Q4X_ECACHE_GB Q4X_PREFILL HSA_ENABLE_INTERRUPT
R=$OUT/receipt.txt
fail() { echo "FAIL $1: $2 (receipt $R)" | tee -a "$R"; exit 1; }
sw() { echo "$2" | sudo -n tee "$DEV/$1" > /dev/null; }
level() { cat $DEV/power_dpm_force_performance_level; }
profile() { grep '\*' $DEV/pp_power_profile_mode | awk '{print $2}' | tr -d '*:'; }
setpower() { # AUTO | HIGH | COMP
  if [ "$(level)" != manual ]; then sw power_dpm_force_performance_level manual; fi
  sw pp_power_profile_mode 0
  case $1 in AUTO) sw power_dpm_force_performance_level auto;; HIGH) sw power_dpm_force_performance_level high;;
    COMP) sw pp_power_profile_mode 5;; esac
}
restore() { setpower AUTO || :; echo "restored perf level $(level) profile $(profile)" >> "$R"; }
trap restore EXIT
link() { echo "link $(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed) x$(cat /sys/bus/pci/devices/0000:00:01.1/current_link_width) iommu $(journalctl -k -b 2>/dev/null | grep -m1 -o 'Default domain type: [A-Za-z]*' || echo unknown) cmdline_iommu $(tr ' ' '\n' < /proc/cmdline | grep -m1 '^iommu=' || echo none)"; }
{
  echo "date $(date -Is) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain -- serve kernels bench/amdgpu-tune-ab.sh | wc -l)"
  echo "engNEW=$NEW sha256 $(sha256sum "$NEW" | cut -c1-16) gguf $GGUF warm sha256 $(sha256sum "$GGUF.warm" | cut -c1-16) NP $NP NG $NG"
  echo "$(link) | perf level $(level) profile $(profile) | cap $(($(cat $HW/power1_cap) / 1000000)) W | kernel $(uname -r)"
} >> "$R"
pcie() { "$PCIEBW" > "$OUT/pcie-$1.log" 2>&1 || fail pcie "$1 run failed"; echo "pcie-$1 $(grep -iE 'h2d|d2h|GB/s' "$OUT/pcie-$1.log" | tr -s ' ' | tr '\n' ';' | cut -c1-400)" | tee -a "$R"; }
[ -n "${DRY:-}" ] || pcie start
P=$(ls bench/q4x-prompts/*.prompt | head -"$NP"); P1=$(echo "$P" | head -1)
{ for pf in $P; do echo "$(basename "$pf" .prompt) $NG $(jq -r 'join(" ")' "$pf")"; done; echo "again-$(basename "$P1" .prompt) $NG $(jq -r 'join(" ")' "$P1")"; } > "$OUT/requests.txt"
i=0
for a in $ARMS; do
  i=$((i + 1)); pw=${a%:*}; it=${a#*:}; name=$(printf 'a%02d-%s-%s' $i $pw $it); log=$OUT/$name.log
  setpower "$pw"
  [ "$pw" != AUTO ] || [ "$(level)" = auto ] || fail "$name" "level $(level), want auto"
  [ "$pw" != HIGH ] || [ "$(level)" = high ] || fail "$name" "level $(level), want high"
  [ "$pw" != COMP ] || { [ "$(level)" = manual ] && [ "$(profile)" = COMPUTE ]; } || fail "$name" "level $(level) profile $(profile), want manual COMPUTE"
  if [ $it = int ]; then envs="HSA_ENABLE_INTERRUPT=0"; else envs=""; fi
  ( while true; do echo "$(cat $HW/freq1_input) $(cat $HW/freq2_input) $(cat $HW/power1_average)"; sleep 0.1; done ) > "$OUT/$name.clk" &
  spid=$!
  env Q4X_STDIN=1 $envs /usr/bin/time -f 'cpu_user %U cpu_sys %S wall %e rss %M' "$NEW" "$GGUF" < "$OUT/requests.txt" > "$log" 2>&1 || { kill $spid || :; fail "$name" "resident run failed: $(grep -m1 -E 'Error|error' "$log" | cut -c1-160)"; }
  kill $spid || :; wait $spid 2>/dev/null || :
  [ "$(grep -c '^out ' "$log")" = $((NP + 1)) ] || fail "$name" "not $((NP + 1)) answers"
  grep '^out ' "$log" > "$OUT/$name.ids"
  [ $i = 1 ] || cmp -s "$OUT/$name.ids" "$OUT/$(ls "$OUT" | grep '^a01-.*\.ids$')" || fail identity "$name ids differ from arm 1"
  clk=$(awk '$1 > 500e6 { n++; s += $1; m += $2; p += $3 } END { if (n) printf "sclk_mhz %.0f mclk_mhz %.0f power_w %.1f busy_samples %d", s / n / 1e6, m / n / 1e6, p / n / 1e6, n; else print "sclk_mhz 0 mclk_mhz 0 power_w 0 busy_samples 0" }' "$OUT/$name.clk")
  st=$(awk '/^gen ms\/token/ { g = $3 } /^out / { t = $2 } /^expert misses/ { if (t !~ /^again/ && n++ > 0) print g, $3 / ($5 / 480) }' "$log" | sort -n | awk '{ g[NR] = $1; m[NR] = $2 } END { asort(m); printf "steady_gen_ms %.3f steady_misses %.2f", (NR % 2) ? g[(NR + 1) / 2] : (g[NR / 2] + g[NR / 2 + 1]) / 2, (NR % 2) ? m[(NR + 1) / 2] : (m[NR / 2] + m[NR / 2 + 1]) / 2 }')
  echo "arm $name level $(level) profile $(profile) env[$envs] | $st | $(grep '^cpu_user' "$log") | $clk" | tee -a "$R"
done
echo "identity PASS: $i arms, ids equal arm 1" | tee -a "$R"
[ -n "${DRY:-}" ] || pcie end
python3 - "$R" <<'EOF' | tee -a "$R"
import re, statistics as S, sys
A = []
for l in open(sys.argv[1]):
    m = re.match(r'arm a(\d+)-(\w+)-(\w+) .*steady_gen_ms ([\d.]+) steady_misses ([\d.]+) \| cpu_user ([\d.]+) cpu_sys ([\d.]+) wall ([\d.]+).*sclk_mhz (\d+)', l)
    if m: A.append(dict(i=int(m[1]), pw=m[2], it=m[3], g=float(m[4]), mi=float(m[5]), cpu=float(m[6]) + float(m[7]), sclk=int(m[9])))
blocks = [A[k:k + 2] for k in range(0, len(A), 2)]
ri = []; rc = []
for b in blocks:
    base = [x for x in b if x['it'] == 'base'][0]; it = [x for x in b if x['it'] == 'int'][0]
    ri.append(it['g'] / base['g']); rc.append(it['cpu'] / max(base['cpu'], 0.01))
    print(f"INT block {b[0]['pw']} arms {b[0]['i']},{b[1]['i']}: gen int/base {ri[-1]:.4f} cpu int/base {rc[-1]:.2f}")
print(f"INT pooled median gen {S.median(ri):.4f} (pairs below 1: {sum(r < 1 for r in ri)} of {len(ri)}) cpu {S.median(rc):.2f}")
def bm(k): return S.median([x['g'] for x in blocks[k]])
def bs(k): return S.median([x['sclk'] for x in blocks[k]])
print(f"HIGH/AUTO pair 1 (blocks 1,2) {bm(1) / bm(0):.4f} pair 2 (blocks 6,5) {bm(4) / bm(5):.4f} pooled {S.median([bm(1), bm(4)]) / S.median([bm(0), bm(5)]):.4f} | sclk HIGH {S.median([bs(1), bs(4)]):.0f} AUTO {S.median([bs(0), bs(5)]):.0f}")
print(f"COMP/AUTO pooled {S.median([bm(2), bm(3)]) / S.median([bm(0), bm(5)]):.4f} | sclk COMP {S.median([bs(2), bs(3)]):.0f}")
first = A[0]['g']; rest_auto = S.median([x['g'] for x in A if x['pw'] == 'AUTO' and x['i'] != 1])
print(f"first arm {first:.3f} vs other AUTO arms {rest_auto:.3f}")
EOF
echo "job PASS" | tee -a "$R"
