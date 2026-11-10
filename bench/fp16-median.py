#!/usr/bin/env python3
"""Reduce N repeat samples of one bench binary to the median run's JSON.

One sample cannot detect its own contamination. This card also drives the
displays, so a compositor burst removes throughput from whatever window it
lands in: the noise is one-sided, a single draw is biased low, and lengthening
the measured region shrinks the effect without removing it. A single-sample
sweep is what left 4096^3 in docs/BASELINE.md ~9% low (90705 published, 91106
on one fresh sample, 99224 as a 5-round median).

Emits the median sample's own JSON (never a synthetic average -- the row must
stay a real run with its real receipts) annotated with `samples` and `spread`.
Exits non-zero when spread exceeds the limit, per PROTOCOL-RULES P6.
"""

import json, statistics, sys

samples_path, limit = sys.argv[1], float(sys.argv[2])
rows = [json.loads(l) for l in open(samples_path) if l.lstrip().startswith("{")]
if not rows:
    sys.exit(f"fp16-median.py: no JSON samples in {samples_path}")

g = sorted(r["gflops"] for r in rows)
med = statistics.median(g)
spread = (g[-1] - g[0]) / med if med else 9.0
row = min(rows, key=lambda r: abs(r["gflops"] - med))
row["samples"] = [round(x) for x in g]
row["spread"] = round(spread, 4)
print(json.dumps(row))

verdict = "OK" if spread <= limit else "SPREAD"
print(f"  {row['m']}: samples {row['samples']} spread {spread*100:.2f}% {verdict}",
      file=sys.stderr)
if verdict == "SPREAD":
    sys.exit(f"fp16-median.py: {row['m']} spread {spread*100:.2f}% exceeds "
             f"{limit*100:.2f}% -- instrument too noisy to publish (P6)")
