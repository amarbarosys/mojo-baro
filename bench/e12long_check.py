#!/usr/bin/env python3
"""Per-item E12-long batch check: T vs KV accuracy, handoff hash, receiver ids, timing.

usage: bench/e12long_check.py TASKS.json RAW1.raw.json [RAW2.raw.json ...]
"""
import json
import math
import statistics
import sys

sys.path.insert(0, "bench")
import e8_score


def main():
    tasks = {t["id"]: t for t in json.load(open(sys.argv[1]))}
    rows = []
    for raw in sys.argv[2:]:
        rows += json.load(open(raw))["items"]
    ok = {"T": 0, "KV": 0}
    b = c = 0  # discordant: KV right and T wrong / T right and KV wrong
    bad = []
    rT, rK, mi = [], [], []
    for it in rows:
        a = {x["arm"]: x for x in it["arms"]}
        t, k = a["T"], a["KV"]
        sT = e8_score.score_arm(tasks[it["id"]], t)[0]
        sK = e8_score.score_arm(tasks[it["id"]], k)[0]
        ok["T"] += sT
        ok["KV"] += sK
        b += sK and not sT
        c += sT and not sK
        hash_eq = t["handoff_hash"] == k["handoff_hash"]
        ids_eq = t["generated_ids"] == k["generated_ids"]
        if not (hash_eq and ids_eq) or sT != sK:
            bad.append(it["id"])
        rT.append(t["receiver_s"])
        rK.append(k["receiver_s"])
        mi.append(1000 * (k["mint_s"] + k["ingest_s"]))
        print(f"{it['id']:28s} T {int(sT)} KV {int(sK)} hash {'=' if hash_eq else 'X'} "
              f"ids {'=' if ids_eq else 'X'} recv {t['receiver_s']:.2f}/{k['receiver_s']:.2f} s "
              f"mint+ingest {mi[-1]:.0f} ms")
    n = len(rows)
    print(f"n={n}  T {ok['T']}/{n}  KV {ok['KV']}/{n}  "
          f"median recv T {statistics.median(rT):.2f} s KV {statistics.median(rK):.2f} s  "
          f"median mint+ingest {statistics.median(mi):.0f} ms  mismatched: {bad or 'none'}")
    # E12 paired Wald CI on KV - T (06-experiments.md, E12 and E12-long thresholds).
    d = (b - c) / n
    se = math.sqrt((b + c) - (b - c) ** 2 / n) / n
    lo, hi = 100 * (d - 1.645 * se), 100 * (d + 1.645 * se)

    def pp(x):
        return ("-" if x < 0 else "+") + f"{abs(x):.1f}"

    inside = "yes" if lo > -5 and hi < 5 else "no"
    print(f"gate: n={n} b={b} c={c} d={pp(100 * d)} pp  90% CI [{pp(lo)}, {pp(hi)}] pp  "
          f"inside +/-5 pp: {inside}")


if __name__ == "__main__":
    main()
