#!/usr/bin/env python3
# usage: bench/moe-tier-stamp.py LOG [LOG...]
#
# Parses the BARO_TIER_STAMP=1 per-layer table an engine run prints
# ("tier stamp: <layer> <open_us> <readback_us> <lru_us> <pread_us> <copy_us>
# <writeback_us> <row_us>") and the "tier stamp: sum_s ... fetch_s ..." line,
# checks the buckets sum to fetch_ns within 5% (item 1's check), and prints
# the dominant bucket by total time across layers.
import sys

BUCKETS = ["open", "readback", "lru", "pread", "copy", "writeback"]


def parse(path):
    rows = []
    sum_s = fetch_s = None
    with open(path) as f:
        for line in f:
            if not line.startswith("tier stamp:"):
                continue
            parts = line.split()
            if parts[2] == "layer":
                continue
            if parts[2] == "sum_s":
                sum_s = float(parts[3])
                fetch_s = float(parts[5])
                continue
            layer = int(parts[2])
            vals = [float(x) for x in parts[3:10]]
            rows.append((layer, vals))
    return rows, sum_s, fetch_s


def main():
    if len(sys.argv) < 2:
        print("usage: moe-tier-stamp.py LOG [LOG...]", file=sys.stderr)
        return 2
    for path in sys.argv[1:]:
        rows, sum_s, fetch_s = parse(path)
        if not rows:
            print(f"{path}: no tier stamp rows (run with BARO_TIER_STAMP=1)")
            continue
        totals = [0.0] * len(BUCKETS)
        for _, vals in rows:
            for i in range(len(BUCKETS)):
                totals[i] += vals[i]
        print(f"== {path} ({len(rows)} layers) ==")
        for name, us in zip(BUCKETS, totals):
            print(f"  {name:10s} {us/1e3:8.3f} ms")
        row_sum_us = sum(totals)
        print(f"  {'row_sum':10s} {row_sum_us/1e3:8.3f} ms")
        if sum_s is not None:
            dev = abs(sum_s - fetch_s) / fetch_s * 100 if fetch_s else float("nan")
            ok = "PASS" if dev <= 5.0 else "FAIL"
            print(
                f"  sum_s {sum_s:.6f}  fetch_s {fetch_s:.6f}  "
                f"deviation {dev:.2f}%  {ok} (P: within 5%)"
            )
        dominant = BUCKETS[totals.index(max(totals))]
        print(f"  dominant bucket: {dominant} ({max(totals)/1e3:.3f} ms total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
