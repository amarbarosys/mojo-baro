#!/usr/bin/env python3
"""Score bench/q4x-gate.sh: per prompt, share of positions where ours / ctrl argmax equals REF.

usage: q4x-gate-score.py OUT PROMPT.txt...   exit 0 PASS, 1 FAIL
Bar (frozen 2026-10-09, before the first run): mean(ours) >= mean(ctrl) - 2 pp.
"""
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); o_all = []; c_all = []
for p in sys.argv[2:]:
    b = pathlib.Path(p).stem
    ref = json.load(open(out / f"{b}.ref"))
    ours = [int(x) for x in (out / f"{b}.ours").read_text().split()]
    ctrl = [int(x) for x in (out / f"{b}.ctrl").read_text().split()]
    n = len(ref)
    o = sum(a == r for a, r in zip(ours, ref)) / n
    c = sum(a == r for a, r in zip(ctrl, ref)) / n
    o_all.append(o); c_all.append(c)
    print(f"{b:28s} ours {o*100:6.2f}%  ctrl {c*100:6.2f}%  n={n}")
mo = sum(o_all) / len(o_all); mc = sum(c_all) / len(c_all)
ok = mo >= mc - 0.02
print(f"MEAN ours {mo*100:.2f}%  ctrl {mc*100:.2f}%  bar ours >= ctrl - 2pp  -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
