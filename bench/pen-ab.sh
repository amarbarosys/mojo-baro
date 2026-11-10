#!/usr/bin/env bash
# usage: bench/pen-ab.sh ENGINE_A ENGINE_B OUTDIR [ENV]
#   e.g. bench/pen-ab.sh .work/a6/engine-a6 .work/a6/engine-a6b .work/a6/pen-ab
# Penalties and top_logprobs at T=0 on two engine builds, same stint (bench/chat-protocol.md A6.3).
# Three arms per engine, one resident process each (BARO_SERVE=1), 20 prompts, n=64, spec off:
#   U: plain greedy   P: presence 0.3 / frequency 0.5   L: top_logprobs 5
# Identity per (arm, prompt): engine B's token stream equals engine A's. Receipts (P1): arm P's
# stream differs from arm U's on the same engine (the penalty acted), arm L's token lines carry a
# "logprob" field (the logprobs path acted); no drafted count on any done line (spec really off).
set -euo pipefail
cd "$(dirname "$0")/.."
ea=$1; eb=$2; out=$3; envx=${4:-}
mkdir -p "$out"
{
  echo "engineA=$ea shaA=$(sha256sum "$ea" | cut -c1-16) engineB=$eb shaB=$(sha256sum "$eb" | cut -c1-16) env='$envx'"
  echo "power_cap_uW=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap) vddgfx=$(grep -A1 OD_VDDGFX_OFFSET /sys/class/drm/card1/device/pp_od_clk_voltage | tail -1)"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
} | tee "$out/arm.txt"
: > "$out/requests.jsonl"
arm=0
for extra in "" ",\"presence_penalty\":0.3,\"frequency_penalty\":0.5" ",\"top_logprobs\":5"; do
  i=0
  for tf in bench/mtp-prompts/p*.tokens; do
    i=$((i + 1))
    p=$(tr -s ' \n' ',' < "$tf" | sed 's/,$//')
    echo "{\"id\":$((arm * 100 + i)),\"prompt\":[$p],\"n\":64,\"spec\":false,\"temperature\":0$extra}"
  done >> "$out/requests.jsonl"
  arm=$((arm + 1))
done
for e in A B; do
  eng=$ea; [ $e = B ] && eng=$eb
  env BARO_SERVE=1 $envx "$eng" < "$out/requests.jsonl" > "$out/engine-$e.out" 2> "$out/engine-$e.err" || true
  grep '^{' "$out/engine-$e.out" > "$out/lines-$e.jsonl" || true
  grep -q '"ready":true' "$out/lines-$e.jsonl" || { echo "FAIL pen-ab: engine $e never printed its ready line, see $out/engine-$e.err"; exit 3; }
  grep -m1 '^BARO_MEGA:' "$out/engine-$e.out" | sed "s/^/engine $e /"
done
python3 - "$out" <<'PY'
import json, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1])
arms = {0: "U plain T=0", 1: "P penalties T=0", 2: "L top_logprobs 5 T=0"}
def load(e):
    toks, done, lp = {}, {}, set()
    for line in (out / f"lines-{e}.jsonl").read_text().splitlines():
        try: d = json.loads(line)
        except Exception: continue
        if "id" not in d: continue
        if d.get("done"): done[d["id"]] = d
        elif "tok" in d:
            toks.setdefault(d["id"], []).append(d["tok"])
            if "logprob" in d: lp.add(d["id"])
    return toks, done, lp
A, B = load("A"), load("B")
bad = False
rows = ["arm engine n median_tok_s min max identical_to_A"]
for a, name in arms.items():
    ids = [a * 100 + i for i in range(1, 21)]
    for e, (toks, done, lp) in (("A", A), ("B", B)):
        ts = [done[i]["tok_s"] for i in ids if i in done]
        if len(ts) != 20: print(f"VOID: {name} engine {e} has {len(ts)}/20 done lines"); bad = True; continue
        if any(done[i].get("drafted") for i in ids): print(f"VOID: {name} engine {e} speculated"); bad = True
        same = sum(1 for i in ids if A[0].get(i) == toks.get(i))
        rows.append(f"{name.replace(' ', '_')} {e} {len(ts)} {st.median(ts):.2f} {min(ts):.2f} {max(ts):.2f} {same}/20")
        if e == "B" and same != 20: print(f"FAIL identity: {name}: engine B == engine A on {same}/20 prompts"); bad = True
for e, (toks, done, lp) in (("A", A), ("B", B)):
    diff = sum(1 for i in range(1, 21) if toks.get(100 + i) != toks.get(i))
    nlp = sum(1 for i in range(1, 21) if 200 + i in lp)
    print(f"receipt engine {e}: penalties changed the stream on {diff}/20 prompts; logprob lines on {nlp}/20 prompts")
    if diff == 0: print(f"VOID: engine {e}: penalties changed nothing"); bad = True
    if nlp != 20: print(f"VOID: engine {e}: top_logprobs produced no logprob field on {20 - nlp} prompts"); bad = True
(out / "results.txt").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
sys.exit(1 if bad else 0)
PY
