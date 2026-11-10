#!/usr/bin/env bash
# bench/cert-hook-protocol.md timed A/B: decode tok/s with the per-token certificate on (B) vs off (A), ABBA per prompt,
# one binary, the 20 mtp-prompts, through bench/decode-arms.sh. Void unless the PCIe link reads 16.0 GT/s at start and end.
# usage: bench/cert-ab.sh OUT        (re-execs itself under gpu-wait)
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(realpath -m "${1:?usage: cert-ab.sh OUT}")
ENG=$(realpath "${ENG:-.work/cert-hook/engine}")
PACK=$(readlink -f "${PACK:-.work/engine-pack-q4}")
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$ENG" ] || fail setup "engine $ENG missing"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 22 --timeout 7200 -- bash "$0" "$out"
fi
mkdir -p "$out"
card=$(grep -l . /sys/class/drm/card*/device/mem_info_vram_total | xargs -n1 dirname | while read -r d; do echo "$(cat "$d/mem_info_vram_total") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
link() { echo "root $(cat /sys/bus/pci/devices/0000:00:01.1/current_link_speed) gpu $(cat "$card/current_link_speed")"; }
l0=$(link); echo "link_start $l0" >"$out/receipt.txt"
[[ $l0 == *"root 16.0 GT/s"*"gpu 16.0 GT/s"* ]] || fail link "start: $l0 (void: needs 16.0 GT/s on both)"
arm="BARO_TMAX=16384"
PACK=$PACK PROMPTS="bench/mtp-prompts/p*.tokens" bench/decode-arms.sh "$out/da" \
  A1="$ENG,$arm" B1="$ENG,$arm,BARO_CERT=1" B2="$ENG,$arm,BARO_CERT=1" A2="$ENG,$arm" | tee "$out/da.txt" || fail arms "decode-arms, $out/da"
l1=$(link); echo "link_end $l1" >>"$out/receipt.txt"
[[ $l1 == *"root 16.0 GT/s"*"gpu 16.0 GT/s"* ]] || fail link "end: $l1 (void)"
echo "engine $ENG sha256 $(sha256sum "$ENG" | cut -c1-16) env common BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=16384 arm B adds BARO_CERT=1" >>"$out/receipt.txt"
for f in "$out"/da/A1.*.log "$out"/da/A2.*.log; do
  grep -q '^cert' "$f" && fail readback "cert output in an A log: $f"
done
nb=0; tokens=0
for f in "$out"/da/B1.*.log "$out"/da/B2.*.log; do
  grep -q '^cert on seed 1 ' "$f" || fail readback "no 'cert on seed 1' line in $f"
  c=$(grep -c '^cert tok' "$f"); ok=$(grep -c '^cert tok .* pass 7/7 ' "$f")
  [ "$c" -gt 0 ] && [ "$c" -eq "$ok" ] || fail cert "$f: $ok of $c cert lines pass 7/7"
  nb=$((nb + 1)); tokens=$((tokens + c))
done
echo "readback B logs $nb all carry 'cert on seed 1', $tokens cert lines all pass 7/7; A logs carry none" >>"$out/receipt.txt"
awk -F'\t' 'NR > 1 {v[$1] = v[$1] " " $4} END {for (k in v) print k, v[k]}' "$out/da/results.tsv" | sort >"$out/toks.txt"
python3 - "$out/da/results.tsv" >>"$out/receipt.txt" <<'PY'
import sys, statistics as st
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
def arm(names): return [float(r[3]) for r in rows if r[0] in names]
def ident(): 
    ref = {r[1]: r[4] for r in rows if r[0] == "A1"}
    return sum(1 for r in rows if ref[r[1]] == r[4]), len(rows)
a, b = arm(("A1", "A2")), arm(("B1", "B2"))
pa = {}
for r in rows: pa.setdefault((r[1], r[0][0]), []).append(float(r[3]))
ratios = sorted(st.median(pa[(p, "A")]) / st.median(pa[(p, "B")]) for p in {k[0] for k in pa})
i, n = ident()
print(f"A median tok/s {st.median(a):.3f} (n {len(a)}, min {min(a):.2f}, max {max(a):.2f})")
print(f"B median tok/s {st.median(b):.3f} (n {len(b)}, min {min(b):.2f}, max {max(b):.2f})")
print(f"B/A {st.median(a) / st.median(b):.4f}  per-prompt median ratios min {ratios[0]:.4f} median {st.median(ratios):.4f} max {ratios[-1]:.4f}")
print(f"A1 vs A2 median {st.median(arm(('A1',))) / st.median(arm(('A2',))):.4f}  B1 vs B2 {st.median(arm(('B1',))) / st.median(arm(('B2',))):.4f}")
print(f"identity (GENERATED md5 equal to A1) {i}/{n}")
PY
cat "$out/receipt.txt"
