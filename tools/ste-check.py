#!/usr/bin/env python3
"""Oracle: count sentences longer than the ASD-STE100 limits (25 words, 20 for list items and numbered steps).
usage: tools/ste-check.py FILE [--max N]   prints totals and the long sentences; exit 1 if any sentence is over the limit."""
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
text = re.sub(r"```.*?```", " ", text, flags=re.S)
rows = []
for line in text.splitlines():
    s = line.strip()
    if not s: continue
    lim = 20 if re.match(r"^(\d+\.|[-*])\s", s) else 25
    s = re.sub(r"^(\d+\.|[-*])\s+", "", s)
    for sent in re.split(r"(?<=[.!?:])\s+", s):
        n = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'_./-]*", sent))
        if n: rows.append((n, lim, sent))
over = [r for r in rows if r[0] > r[1]]
print(f"sentences {len(rows)}, over limit {len(over)}, longest {max((r[0] for r in rows), default=0)}")
for n, lim, s in over: print(f"  {n}>{lim}: {s[:110]}")
sys.exit(1 if over else 0)
