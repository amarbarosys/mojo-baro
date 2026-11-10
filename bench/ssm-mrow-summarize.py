#!/usr/bin/env python3
"""Summarize bench/ssm-mrow-run.sh output: per-stage ms per window at each m,
ratio vs m = 1, per-row efficiency, read-backs per run, and the delta
kernel's device time per launch from the rocprofv3 traces.

usage: bench/ssm-mrow-summarize.py OUTDIR
"""
import csv, glob, re, statistics, sys

out = sys.argv[1]
MS = [1, 2, 4, 8]


def parse(path):
    s = open(path).read()
    r = {}
    r["spec"] = re.search(r"^BARO_SPEC: (\w+)", s, re.M)
    r["spec"] = r["spec"].group(1) if r["spec"] else None
    m = re.search(r"mtp: drafted (\d+)\s+accepted (\d+)\s+k (\d+)", s)
    r["mtp"] = tuple(int(x) for x in m.groups()) if m else None
    r["tokens"] = int(re.search(r"^tokens: (\d+)", s, re.M).group(1)) if re.search(r"^tokens: (\d+)", s, re.M) else None
    r["ptoks"] = int(re.search(r"prompt tokens: (\d+)", s).group(1)) if re.search(r"prompt tokens: (\d+)", s) else None
    mg = re.search(r"^BARO_MEGA: (\w+)", s, re.M)
    r["mega"] = mg.group(1) if mg else None
    r["stages"] = {}
    for k, v in re.findall(r"^ssm-kernel: (\S+) ([0-9.e-]+)", s, re.M):
        r["stages"][k] = float(v)
    for k, v in re.findall(r"^ffn-kernel: (\S+) ([0-9.e-]+)", s, re.M):
        r["stages"][k] = float(v)
    for k, v in re.findall(r"^profile: (attn|ssm|ffn|head) ([0-9.e-]+)", s, re.M):
        r["stages"][k] = float(v)
    return r


def windows(r, m):
    if m == 1:
        return (r["tokens"] or 64) - 1
    d, a, k = r["mtp"]
    return d / k


def void(r, m):
    if r["mega"] != "False":
        return "BARO_MEGA is not False (megakernel bypasses the stage timers)"
    if m == 1 and r["spec"] != "False":
        return "spec on in the m=1 arm"
    if m > 1 and (r["spec"] != "True" or not r["mtp"] or r["mtp"][2] != m - 1):
        return "spec/k read-back does not match m"
    return None


tables = {}
for mode in (2, 4, 1):
    per = {}
    for m in MS:
        runs = sorted(glob.glob(f"{out}/p{mode}-m{m}-r*.log"))
        parsed = [parse(p) for p in runs]
        for p, r in zip(runs, parsed):
            v = void(r, m)
            if v:
                print(f"VOID {p}: {v}")
        kept = parsed[1:] if len(parsed) > 1 else parsed
        stages = {}
        for st in kept[0]["stages"]:
            vals = [r["stages"][st] * 1e3 / windows(r, m) for r in kept if st in r["stages"]]
            med = statistics.median(vals)
            spread = (max(vals) - min(vals)) / med * 100 if med > 0 and len(vals) > 1 else 0.0
            stages[st] = (med, spread)
        per[m] = stages
        if parsed:
            r0 = parsed[-1]
            print(f"mode {mode} m={m}: runs {len(parsed)} mega={r0['mega']} spec={r0['spec']} mtp={r0['mtp']} tokens={r0['tokens']} prompt={r0['ptoks']} windows={windows(r0, m):.1f}")
    tables[mode] = per
    print()
    print(f"## profile mode {mode}: ms per window (median of kept runs, spread %), ratio vs m=1, per-row efficiency m/ratio")
    names = list(per[1].keys())
    print("| stage | " + " | ".join(f"m={m}" for m in MS) + " |")
    print("|---|" + "---|" * len(MS))
    for st in names:
        cells = []
        for m in MS:
            med, sp = per[m].get(st, (float("nan"), 0))
            base = per[1][st][0]
            ratio = med / base if base else float("nan")
            cells.append(f"{med:.3f} ({sp:.1f}%) {ratio:.2f}x eff {m / ratio:.2f}" if m > 1 else f"{med:.3f} ({sp:.1f}%)")
        print(f"| {st} | " + " | ".join(cells) + " |")
    tot = {m: sum(v[0] for v in per[m].values()) for m in MS}
    print("| total | " + " | ".join(f"{tot[m]:.3f} {tot[m] / tot[1]:.2f}x eff {m / (tot[m] / tot[1]):.2f}" if m > 1 else f"{tot[m]:.3f}" for m in MS) + " |")
    if mode == 2 and "delta" in per[8]:
        print("delta share of the serialized SSM sub-block: " + ", ".join(f"m={m} {per[m]['delta'][0] / tot[m]:.3f}" for m in MS))
    print()

# rocprofv3 cross-check: device time per launch of the delta kernel
print("## rocprofv3 device time per launch of amar_ssm_delta_step (median over launches, us)")
for m in MS:
    csvs = glob.glob(f"{out}/trace-m{m}/**/*kernel_trace.csv", recursive=True)
    if not csvs:
        print(f"m={m}: no trace")
        continue
    durs = {}
    with open(csvs[0]) as f:
        rd = csv.DictReader(f)
        for row in rd:
            name = row.get("Kernel_Name", "")
            if "delta_step" in name or "delta_chunk" in name:
                t = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3
                durs.setdefault(name[:60], []).append(t)
    for name, ts in durs.items():
        ts.sort()
        print(f"m={m}: {name} launches {len(ts)} median {statistics.median(ts):.2f} us p10 {ts[len(ts)//10]:.2f} p90 {ts[9*len(ts)//10]:.2f}")
