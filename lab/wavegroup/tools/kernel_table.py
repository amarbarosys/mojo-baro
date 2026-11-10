#!/usr/bin/env python3
# Per-kernel device time from a rocprofv3 --kernel-trace CSV: total, count, mean, share. Names cut to the function name.
# usage: tools/kernel_table.py TRACE.csv [skip_first_n_dispatches]
import csv, re, sys
from collections import defaultdict
rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r["Start_Timestamp"]))[int(sys.argv[2]) if len(sys.argv) > 2 else 0:]
t = defaultdict(lambda: [0, 0])
for r in rows:
  name = re.sub(r"[(<\[].*", "", r["Kernel_Name"]).split("::")[-1][:40]
  t[name][0] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"]); t[name][1] += 1
span = int(rows[-1]["End_Timestamp"]) - int(rows[0]["Start_Timestamp"])
busy = sum(v[0] for v in t.values())
print(f"dispatches {len(rows)}  span {span/1e6:.2f} ms  busy {busy/1e6:.2f} ms ({busy/span:.0%})")
for k, (ns, n) in sorted(t.items(), key=lambda kv: -kv[1][0]):
  print(f"{k:40s} {ns/1e6:8.3f} ms {n:6d} x {ns/n/1e3:7.2f} us  {ns/busy:5.1%}")
