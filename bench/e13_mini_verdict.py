#!/usr/bin/env python3
"""E13-mini verdict (AMDHQ docs/design/latent-os/06-experiments.md, E13-mini).

usage: bench/e13_mini_verdict.py SCORED_JSON TRAIN_REPORT_JSON
Math items only, arms "0" and "L8-proj", paired on correct_exact.
Void: arm 0 more than 5 pp off round 5's 10/100. SIGNAL: d >= +10 pp and
d - 1.645 SE > 0, with d = (b - c)/n, SE = sqrt((b + c) - (b - c)^2/n)/n.
Anything else: NO SIGNAL (never KILL).
"""
import json, math, sys

scored = json.load(open(sys.argv[1]))
train = json.load(open(sys.argv[2]))
n = a0 = ap = b = c = 0
for it in scored["items"]:
    if it["type"] != "math":
        continue
    arms = {x["arm"]: x for x in it["arms"]}
    z = bool(arms["0"]["correct_exact"]); p = bool(arms["L8-proj"]["correct_exact"])
    n += 1; a0 += z; ap += p; b += p and not z; c += z and not p
d = (b - c) / n
se = math.sqrt((b + c) - (b - c) ** 2 / n) / n
lb = d - 1.645 * se
void = abs(a0 - 10 * n / 100) > 5 * n / 100
verdict = "VOID (arm 0 off round 5)" if void else ("SIGNAL" if d >= 0.10 and lb > 0 else "NO SIGNAL")
print(f"n={n} arm0={a0} L8-proj={ap} b(proj only)={b} c(arm0 only)={c} d={100*d:+.1f}pp SE={100*se:.1f}pp lower95={100*lb:+.1f}pp")
print(f"train: steps={train.get('steps_run')} first={train.get('first_loss')} last={train.get('last_loss')} holdout_first={train.get('holdout_first')} holdout_last={train.get('holdout_last')} s/step={train.get('step_time_s_mean')}")
print("VERDICT", verdict)
