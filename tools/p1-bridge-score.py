#!/usr/bin/env python3
"""usage: p1-bridge-score.py OUT MODEL PREFLIGHT FALSIFY PROMPT...

Scores bench/p1-bridge-gate.sh by the rules frozen in bench/p1-bridge-protocol.md (599d8fd and
amendment 1). Nothing here is tunable: the thresholds are the note's, and changing one means
amending the note first. Exit 0 only when the gate passes as the plan wrote it (or, under
PREFLIGHT, when the harness proved itself: no voids and the falsifier failed as it must)."""
import json
import sys
from pathlib import Path

out, model, pre, falsify = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
names = sys.argv[5:]
N = len(names)


def load(p, arm):
    f = out / "ids" / f"{p}.{arm}.json"
    return json.loads(f.read_text()) if f.exists() else None


def first_div(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


rows, voids = [], []
for p in names:
    n_prompt = len((out / "ids" / f"{p}.ids").read_text().strip().split(","))
    cold = load(p, "cold")
    # Amendment 3: a cold run that reaches EOS before 32 tokens is a legitimate answer, not a void.
    # The frozen void list names two things only (reuse not held, cold not cold). Every arm is then
    # compared over everything it generated, and a different LENGTH is a divergence (first_div).
    if cold is None or not cold["tokens"] or (len(cold["tokens"]) < 32 and cold.get("stop_type") != "eos"):
        voids.append(f"{p}: no usable cold reference ({0 if cold is None else len(cold['tokens'])} tokens, stop {None if cold is None else cold.get('stop_type')})"); continue
    if cold["timings"].get("cache_n", 0) != 0:
        voids.append(f"{p}: cold arm was not cold (cache_n {cold['timings'].get('cache_n')})")
    row = {"p": p, "n_prompt": n_prompt, "n_cold": len(cold["tokens"])}
    for arm in ("ctrlL", "primary") + (("falsify",) if falsify else ()):
        d, r = load(p, arm), load(p, f"{arm}.restore")
        if d is None or r is None:
            voids.append(f"{p}: {arm} did not run"); continue
        reused = r.get("n_restored") == n_prompt - 1 and d["timings"].get("cache_n") == n_prompt - 1
        if not reused:
            msg = f"{p}: {arm} n_restored {r.get('n_restored')} cache_n {d['timings'].get('cache_n')} expected {n_prompt - 1}"
            # the frozen void rule names the bridged items and the cold arm; a control that did not
            # reuse is reported and leaves that prompt UNATTRIBUTED, it does not void the gate
            (voids if arm != "ctrlL" else row.setdefault("notes", [])).append(msg)
        row[arm] = first_div(cold["tokens"], d["tokens"]) if reused else "void"
    a, b = load(p, "primary"), load(p, "ctrlL")
    if a is not None and b is not None and row.get("primary") != "void" and row.get("ctrlL") != "void":
        row["vsL"] = first_div(b["tokens"], a["tokens"])
    ours = load(p, "ours")
    if ours is not None:
        # our engine has no EOS stop on this route, so it is compared over the cold run's length
        row["ctrlN"] = first_div(cold["tokens"], ours["choices"][0]["tokens"][: len(cold["tokens"])])
    rows.append(row)

same = lambda arm: sum(1 for r in rows if arm in r and r[arm] is None)  # noqa: E731
have_n = all("ctrlN" in r for r in rows) and rows
print(f"\n# gate 4 score: {model}, {N} prompts{' (PREFLIGHT, stand-in state, not a gate result)' if pre else ''}")
print("prompt            |P|  ctrlL  ctrlN  primary  falsify   (first divergence index, = identical)")
fmt = lambda v: "=" if v is None else str(v)  # noqa: E731
for r in rows:
    print(f"{r['p']:<17} {r['n_prompt']:>3}  {fmt(r.get('ctrlL','-')):>5}  {fmt(r.get('ctrlN','-')):>5}  {fmt(r.get('primary','-')):>7}  {fmt(r.get('falsify','-')):>7}")
    for n in r.get("notes", []):
        print(f"    control L not usable: {n}")
prim, ctl = same("primary"), same("ctrlL")
S = len(rows)
short = [f"{r['p']} ({r['n_cold']})" for r in rows if r["n_cold"] < 32]
if short:
    print(f"cold run ended at EOS before 32 tokens, compared over its full length: {', '.join(short)}")
print(f"identical of {S} scored ({N} prompts): control L {ctl}, control N {same('ctrlN') if have_n else 'n/a'}, primary {prim}" + (f", falsifier {same('falsify')}" if falsify else ""))
# Reported, never gated (amendment 2): does OUR state make llama.cpp produce what ITS OWN state
# at the same position makes it produce? Both arms restore at |P|-1 and evaluate one token, so
# this comparison has no batch-shape difference in it, unlike anything measured against cold.
vs = [r for r in rows if "vsL" in r]
print(f"reported only: primary identical to control L on {sum(1 for r in vs if r['vsL'] is None)} of {len(vs)}" + "".join(f"; {r['p']} differs at {r['vsL']}" for r in vs if r["vsL"] is not None))

if voids:
    print(f"RESULT {model}: VOID, {len(voids)} void(s); a void is a failure (P10)")
    for v in voids:
        print(f"  void {v}")
    sys.exit(1)
if falsify and same("falsify") > 2:
    print(f"RESULT {model}: FAIL falsifier, {same('falsify')}/{N} K/V-swapped states still matched: the gate cannot see the bridge (P11)")
    sys.exit(1)

misses = [r for r in rows if r.get("primary") is not None]
for r in misses:
    who = "llama.cpp restore path (control L missed it too)" if r.get("ctrlL") not in (None, "-") and r.get("ctrlL") != "void" else ("UNATTRIBUTED (control L unusable)" if r.get("ctrlL") == "void" else "bridge or our numerics (control L reproduced this prompt)")
    print(f"  miss {r['p']}: first divergence {r['primary']}, attributed to {who}")
if pre:
    print(f"RESULT {model}: PREFLIGHT OK, harness ran end to end, no voids" + (", falsifier failed as it must" if falsify else ""))
    sys.exit(0)
if N < 20:
    print(f"RESULT {model}: QUICK run of {N}, not a claim (P18)")
    sys.exit(1)
if not misses:
    print(f"RESULT {model}: PASS as written, 20/20 first 32 ids identical, 0 voids")
    sys.exit(0)
early = sum(1 for r in misses if r["primary"] <= 2)
below = have_n and prim < same("ctrlN") - 2
if early > N / 2 or below:
    print(f"RESULT {model}: NOT MET and KILL LINE, layout defect by the frozen rule (early divergences {early}/{N}, primary {prim} vs control N {same('ctrlN') if have_n else 'n/a'})")
elif have_n and len(misses) < N / 2 and all(r["primary"] >= 3 for r in misses):
    print(f"RESULT {model}: NOT MET as written ({prim}/20), classified NUMERICS by the frozen rule; the kill line does not fire")
else:
    print(f"RESULT {model}: NOT MET as written ({prim}/20), UNCLASSIFIED by the frozen rule; no cause is proposed")
sys.exit(1)
