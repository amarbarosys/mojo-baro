#!/usr/bin/env bash
# usage: bench/spec-sample-ab.sh ENGINE OUTDIR TEMP TOP_P [ENV]
#   e.g. bench/spec-sample-ab.sh .work/b1/engine-a1 .work/b1/g4 0.7 0.9 "BARO_PACK=.work/engine-pack-q4"
#
# A1 gate 4 (bench/spec-sample-protocol.md): 20-prompt medians for speculative
# SAMPLING on and off, at the same temperature, in one resident process, plus
# the same pair at temperature 0 so the T=0 speculative gain measured in the
# same stint is what the T>0 gain is compared against.
#
# One engine process serves all four arms (serve/PROTOCOL.md, BARO_SERVE=1), so
# the 21 GB pack loads once instead of 80 times and no arm pays a cold cache the
# others do not. The arms differ only by the request's own fields, and each
# arm's parameters are read back from the engine's own done line (P1): tok_s,
# and drafted/accepted/k, which appear only when that request actually
# speculated. An arm whose done line carries no "drafted" is not a spec arm,
# whatever the request said.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; temp=$3; topp=$4; envx=${5:-}
mkdir -p "$out"
sha=$(sha256sum "$eng" | cut -c1-16)
{
  echo "engine=$eng sha=$sha temp=$temp top_p=$topp env='$envx'"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"

# One request set, four arms: (T, spec) in {(temp,1),(temp,0),(0,1),(0,0)}.
# id encodes the arm: arm*100 + prompt index, so one stream of done lines can be
# split afterwards without depending on order.
: > "$out/requests.jsonl"
arm=0
for a in "$temp 1" "$temp 0" "0 1" "0 0"; do
  set -- $a
  t=$1; sp=$2
  i=0
  for tf in bench/mtp-prompts/p*.tokens; do
    i=$((i + 1))
    p=$(tr -s ' \n' ',' < "$tf" | sed 's/,$//')
    tp=$topp; [ "$t" = "0" ] && tp=1.0
    echo "{\"id\":$((arm * 100 + i)),\"prompt\":[$p],\"n\":64,\"spec\":$([ "$sp" = 1 ] && echo true || echo false),\"temperature\":$t,\"top_p\":$tp,\"seed\":$((1000 + i))}"
  done >> "$out/requests.jsonl"
  arm=$((arm + 1))
done

env BARO_SERVE=1 $envx "$eng" < "$out/requests.jsonl" > "$out/engine.out" 2>"$out/engine.err"
grep '^{' "$out/engine.out" > "$out/lines.jsonl" || true
grep -q '"ready":true' "$out/lines.jsonl" || { echo "VOID: engine never printed its ready line, see $out/engine.err" >&2; exit 3; }

python3 - "$out" <<'PY'
import json, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1])
arms = {0: "T>0 spec", 1: "T>0 no-spec", 2: "T=0 spec", 3: "T=0 no-spec"}
rows = {a: [] for a in arms}
for line in (out / "lines.jsonl").read_text().splitlines():
    try:
        d = json.loads(line)
    except Exception:
        continue
    if not d.get("done"):
        continue
    a, i = divmod(int(d["id"]), 100)
    rows.setdefault(a, []).append(d)
bad = False
print(f"{'arm':<14}{'n':>4}{'median tok/s':>14}{'min':>9}{'max':>9}{'drafted':>9}{'accepted':>10}{'acc rate':>10}")
med = {}
for a, name in arms.items():
    rs = rows.get(a, [])
    if len(rs) != 20:
        print(f"{name:<14}{len(rs):>4}   VOID: expected 20 done lines")
        bad = True
        continue
    ts = [r["tok_s"] for r in rs]
    dr = sum(r.get("drafted", 0) or 0 for r in rs)
    ac = sum(r.get("accepted", 0) or 0 for r in rs)
    med[a] = st.median(ts)
    rate = f"{ac / dr:.3f}" if dr else "n/a"
    print(f"{name:<14}{len(rs):>4}{st.median(ts):>14.2f}{min(ts):>9.2f}{max(ts):>9.2f}{dr:>9}{ac:>10}{rate:>10}")
    # P1: a spec arm must have actually speculated, and a no-spec arm must not.
    if ("spec" in name and not name.startswith("T=0 no")) and a in (0, 2) and dr == 0:
        print(f"    VOID: {name} reports no drafted tokens, so it did not speculate")
        bad = True
    if a in (1, 3) and dr:
        print(f"    VOID: {name} reports {dr} drafted tokens, so it did speculate")
        bad = True
if len(med) == 4:
    g_hot = med[0] / med[1]
    g_cold = med[2] / med[3]
    print(f"\nspeculative gain at T>0: {g_hot:.3f}x   at T=0: {g_cold:.3f}x   "
          f"ratio of gains: {g_hot / g_cold:.3f}")
    print("gate 4 (frozen): the T>0 gain is within 5% of the T=0 gain, i.e. the ratio in 0.95 to 1.05")
(out / "summary.txt").write_text("see stdout")
sys.exit(1 if bad else 0)
PY
