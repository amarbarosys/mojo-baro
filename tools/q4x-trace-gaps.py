#!/usr/bin/env python3
"""Per-token busy / idle split of two rocprofv3 kernel traces that differ only in generated length.

usage: q4x-trace-gaps.py DIR_SHORT TOKENS_SHORT DIR_LONG TOKENS_LONG [--floor-us 20] [--host-us 200]

DIR holds *kernel_trace.csv (and optionally *memory_copy_trace.csv). The difference of the two runs divided by the token
difference removes load, prefill and warm fill. Idle gaps (next start - previous end) are binned:
  floor  < --floor-us   launch-floor gaps
  mid    floor..host    enqueue-bound gaps
  host   >= --host-us   host waits (doorbell poll, table build, miss DMA enqueue)
and the host bin is attributed to the kernel that precedes it.
"""
import csv, glob, sys, collections


def load(d):
    f = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)
    if not f:
        sys.exit(f"FAIL trace: no kernel_trace.csv under {d}")
    rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(f[0])))
    cp = glob.glob(f"{d}/**/*memory_copy_trace.csv", recursive=True)
    copies = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in csv.DictReader(open(cp[0]))] if cp else []
    return rows, copies


def stats(rows, copies, floor, host):
    o = dict(disp=len(rows), busy=sum(e - s for s, e, _ in rows), floor=0, mid=0, host=0, nfloor=0, nmid=0, nhost=0,
             copy=sum(e - s for s, e in copies), ncopy=len(copies), span=rows[-1][1] - rows[0][0])
    byk, nbyk, kb, kn = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for s_, e_, name in rows:
        kb[name] += e_ - s_
        kn[name] += 1
    last_end, prev = rows[0][1], rows[0][2]
    for s, e, name in rows[1:]:
        g = s - last_end
        if g > 0:
            b = "floor" if g < floor else "mid" if g < host else "host"
            o[b] += g
            o["n" + b] += 1
            if b == "host":
                byk[prev] += g
                nbyk[prev] += 1
        last_end = max(last_end, e)
        prev = name
    o["byk"], o["nbyk"], o["kb"], o["kn"] = byk, nbyk, kb, kn
    return o


def main():
    a = sys.argv[1:]
    fl = int(a[a.index("--floor-us") + 1]) * 1000 if "--floor-us" in a else 20000
    ho = int(a[a.index("--host-us") + 1]) * 1000 if "--host-us" in a else 200000
    d1, n1, d2, n2 = a[0], int(a[1]), a[2], int(a[3])
    s1, s2 = stats(*load(d1), fl, ho), stats(*load(d2), fl, ho)
    dn = n2 - n1
    ms = lambda k: (s2[k] - s1[k]) / dn / 1e6
    n = lambda k: (s2[k] - s1[k]) / dn
    print(f"tokens {n1} -> {n2} (diff {dn})")
    print(f"dispatches/token {n('disp'):.1f}  busy ms/token {ms('busy'):.3f}  span ms/token {ms('span'):.3f}  busy/span {ms('busy') / ms('span') * 100:.1f}%")
    print(f"idle ms/token: floor(<{fl // 1000}us) {ms('floor'):.3f} in {n('nfloor'):.0f} gaps | mid {ms('mid'):.3f} in {n('nmid'):.0f} | host(>={ho // 1000}us) {ms('host'):.3f} in {n('nhost'):.0f}")
    print(f"memcopies/token {n('ncopy'):.1f}  copy ms/token {ms('copy'):.3f}")
    for k in sorted(set(s1["kb"]) | set(s2["kb"]), key=lambda k: s1["kb"][k] - s2["kb"][k])[:10]:
        print(f"kernel {(s2['kb'][k] - s1['kb'][k]) / dn / 1e6:7.3f} ms/token {(s2['kn'][k] - s1['kn'][k]) / dn:6.1f} calls  {k[:70]}")
    keys = set(s1["byk"]) | set(s2["byk"])
    for k in sorted(keys, key=lambda k: s1["byk"][k] - s2["byk"][k])[:6]:
        print(f"host gap after {k[:60]:60s} {(s2['byk'][k] - s1['byk'][k]) / dn / 1e6:.3f} ms/token in {(s2['nbyk'][k] - s1['nbyk'][k]) / dn:.1f} gaps")


main()
