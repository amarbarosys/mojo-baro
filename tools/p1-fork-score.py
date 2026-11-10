#!/usr/bin/env python3
"""usage: p1-fork-score.py OUT FORMAT N NFALS [F32_OUT]

Scores bench/p1-fork-gate.sh by the rules frozen in bench/p1-fork-protocol.md. Nothing here is
tunable; changing a threshold means amending the note first. Exit 0 only on PASS as written.
F32_OUT (the f32 run's directory) lets an int8 miss be attributed to quantization."""
import json
import sys
from pathlib import Path

out, fmt, N, NF = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
f32_out = Path(sys.argv[5]) if len(sys.argv) > 5 else None
RATES = {"100mbit": 100.0, "1gbit": 1000.0, "10gbit": 10000.0}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def first_div(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


refs = load(out / "refs.json")
if not refs or len(refs) != N:
    print(f"RESULT {fmt}: VOID, no complete refs.json"); sys.exit(1)
voids, fails = [], []
for p, r in refs.items():
    if r["cold_cached"] not in (0, None):
        voids.append(f"{p}: the single-node reference was not cold (cached {r['cold_cached']})")
ctrl = {p: first_div(r["cold"], r["ctrlS"]) if isinstance(r["ctrlS_cached"], int) and r["ctrlS_cached"] >= r["n_prompt"] - 1 else "unusable" for p, r in refs.items()}

print(f"\n# gate 2 identity half: format {fmt}, {N} prompts, fork A -> B answered by B, against A's cold single-node ids")
for prof, nominal in RATES.items():
    m = load(out / f"rate-{prof}.json")
    if not m:
        fails.append(f"no rate read-back for {prof}"); continue
    ok = 0.7 * nominal <= m["mbit_s"] <= 1.1 * nominal
    print(f"link {prof}: measured {m['mbit_s']:.0f} Mbit/s at the receiver ({'in' if ok else 'OUT OF'} the 0.7 to 1.1 band)")
    if not ok:
        fails.append(f"link {prof} measured {m['mbit_s']:.0f} Mbit/s: the arm-defining rate is not in effect")

f32 = {prof: load(f32_out / f"forks-{prof}.json") for prof in RATES} if f32_out else {}
table, per_rate, kill = {}, {}, []
for prof in RATES:
    rows = load(out / f"forks-{prof}.json")
    if not rows or len(rows) != N:
        voids.append(f"{prof}: forks file missing or short"); continue
    same = 0
    for p, r in rows.items():
        ref = refs[p]
        if r.get("http") != 200:
            voids.append(f"{prof} {p}: HTTP {r.get('http')} {json.dumps(r.get('body'))[:120]}"); table.setdefault(p, {})[prof] = "http"; continue
        if r["pos"] != ref["n_prompt"] - 1 or not isinstance(r["cached"], int) or r["cached"] < r["pos"]:
            voids.append(f"{prof} {p}: node B did not restore the import (pos {r['pos']}, cached {r['cached']}, |P| {ref['n_prompt']})"); table.setdefault(p, {})[prof] = "void"; continue
        if r["format"] != fmt:
            fails.append(f"{prof} {p}: state format {r['format']}, the arm is {fmt}")
        d = first_div(ref["cold"], r["tokens"])
        table.setdefault(p, {})[prof] = d
        table[p][prof + "_vsS"] = first_div(ref["ctrlS"], r["tokens"])
        if d is None:
            same += 1
            continue
        if ctrl[p] == d:
            why = "engine restore path, control S misses at the same index (E14 class); not the cross-node move"
        elif fmt == "int8" and f32.get(prof) and f32[prof].get(p, {}).get("http") == 200 and first_div(ref["cold"], f32[prof][p]["tokens"]) is None:
            why = "int8 state quantization (the f32 arm matched this prompt at this rate)"
        else:
            why = "CROSS-NODE DEFECT: control S reproduced the cold ids and the forked state did not"
            kill.append(f"{prof} {p}@{d}")
        print(f"  miss {prof} {p}: first divergence {d}: {why}")
    per_rate[prof] = same

fmtc = lambda v: "=" if v is None else str(v)  # noqa: E731
print("prompt            |P|  ctrlS  100mbit  1gbit  10gbit   (first divergence from A's cold ids, = identical)")
for p, r in refs.items():
    t = table.get(p, {})
    print(f"{p:<17} {r['n_prompt']:>3}  {fmtc(ctrl[p]):>5}  {fmtc(t.get('100mbit', '-')):>7}  {fmtc(t.get('1gbit', '-')):>5}  {fmtc(t.get('10gbit', '-')):>6}")
print(f"identical of {N}: control S {sum(1 for v in ctrl.values() if v is None)}" + "".join(f", {prof} {per_rate.get(prof, 'n/a')}" for prof in RATES))
print("reported only, fork ids identical to control S: " + ", ".join(f"{prof} {sum(1 for p in refs if table.get(p, {}).get(prof + '_vsS', 1) is None)}" for prof in RATES))
rate_dep = [p for p in refs if len({str(table.get(p, {}).get(prof)) for prof in RATES}) > 1]
print(f"prompts whose result depends on the link rate: {len(rate_dep)} {' '.join(rate_dep)}")
sizes = sorted({r.get("state_bytes") for prof in RATES for r in (load(out / f'forks-{prof}.json') or {}).values() if r.get("http") == 200})
if sizes:
    print(f"state bytes on the wire: {sizes[0]} to {sizes[-1]}")

fal = load(out / "falsify.json")
if not fal or len(fal) != NF:
    fails.append("falsifier file missing or short")
else:
    flip_ok = sum(1 for r in fal.values() if r.get("flip_http") == 409 and r.get("flip_body", {}).get("error") == "state_identity" and r["flip_body"].get("field") == "payload_sha")
    accepted = sum(1 for p, r in fal.items() if r.get("swap_http") == 200 and isinstance(r.get("swap_cached"), int) and r["swap_cached"] >= refs[p]["n_prompt"] - 1)
    wrong = sum(1 for p, r in fal.items() if r.get("swap_http") == 200 and first_div(refs[p]["cold"], r["swap_tokens"]) is not None)
    print(f"falsifier flip (one payload byte): {flip_ok} of {NF} refused with 409 state_identity payload_sha, relayed by node A")
    print(f"falsifier swapkv (K and V exchanged, sha re-signed): {accepted} of {NF} accepted AND restored by node B, {wrong} of {NF} produced wrong ids")
    if flip_ok != NF:
        fails.append("a corrupted state was not refused with the 409 payload_sha path")
    if accepted != NF:
        fails.append("the re-signed wrong state was not accepted and restored, so the ids falsifier did not test what it claims")
    elif wrong < NF - 1:
        fails.append("a wrong state that node B restored still produced the right ids: the gate cannot see a wrong state (P11)")

if voids:
    print(f"RESULT {fmt}: VOID, {len(voids)} void(s); a void is a failure (P10)")
    for v in voids[:12]:
        print(f"  void {v}")
    sys.exit(1)
if fails:
    print(f"RESULT {fmt}: FAIL, the harness did not prove itself")
    for f in fails:
        print(f"  fail {f}")
    sys.exit(1)
if N < 20:
    print(f"RESULT {fmt}: QUICK run of {N}, not a claim (P18)"); sys.exit(1)
if all(per_rate.get(prof) == N for prof in RATES):
    print(f"RESULT {fmt}: PASS as written, {N}/{N} at 100 Mbit, 1 Gbit and 10 Gbit, 0 voids, both falsifiers failed as they must"); sys.exit(0)
if kill:
    print(f"RESULT {fmt}: NOT MET and KILL LINE, cross-node defect by the frozen rule: {' '.join(kill)}")
else:
    print(f"RESULT {fmt}: NOT MET as written ({', '.join(f'{prof} {per_rate.get(prof)}/{N}' for prof in RATES)}); every miss attributed by the frozen rule to a cause other than the cross-node move; the kill line does not fire")
sys.exit(1)
