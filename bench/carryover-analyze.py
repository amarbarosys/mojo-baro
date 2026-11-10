#!/usr/bin/env python3
"""Aggregate bench/carryover-run.sh logs (.work/carryover/runs/S-*.log).

Per run: tok/s, decode_s, sysfs snapshots (gpu_metrics v1.3), and from the
STAMP lines the device-side timeline of every trunk window: each stamped
site's span, the gaps between consecutive sites (cyclic: last site's gap to
the next window's first site), and the whole window's span. Block-0 cycles /
block-0 wall = the shader clock the kernel actually ran at (eff = sum over
every block's wave-0 cycles / sum of their wall spans; a block lifetime over
300 us is excluded in-kernel and counted as unsafe).

Site names default to gate/up/down (bench/carryover-stamp.py's SITES, site
index 0/1/2); pass --site-names to match a different SITES list.

The per-site sums (sum_<name>_ms) and the window sum (sum_win_ms) are exact
regardless of dispatch order -- each STAMP line carries its own site index,
so these are a straight group-by. The pairwise gap breakdown (sum_<a>-><b>_ms)
assumes real dispatches visit the sites in index order 0,1,2,...,cyclically;
that holds for an always-sequential SITES list (e.g. gate->up->down) but not
for one with mutually-exclusive branches (e.g. this repo's attn-only vs
ssm-only sites, M3 2026-09-15) -- there most real inter-kernel gaps do not
match any labeled pair and go uncounted in the breakdown. Trust the per-site
and window sums always; trust the pairwise breakdown only when the SITES
list is a single sequential chain.

Promoted from .work/carryover/analyze.py (2026-09-15 carry-over probe).
"""
import argparse
import collections
import glob
import hashlib
import os
import re
import statistics as S
import struct

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_RUNS = os.path.join(ROOT, ".work/carryover/runs")
DEFAULT_SITE_NAMES = ["gate", "up", "down"]


def med(x):
    return S.median(x) if x else float("nan")


def gm(hexs):
    b = bytes(int(v) for v in hexs.strip(",").split(",") if v != "")
    if len(b) < 120:
        return None
    u16 = lambda o: struct.unpack_from("<H", b, o)[0]
    return dict(avg_gfxclk=u16(40), cur_gfxclk=u16(54), avg_uclk=u16(44), cur_uclk=u16(58), power_w=u16(22),
                gfx_act=u16(16), t_hot=u16(6), throttle=struct.unpack_from("<I", b, 68)[0], indep_throttle=struct.unpack_from("<Q", b, 112)[0])


def parse_runs(runs_dir, pattern):
    runs = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        tag = os.path.basename(f)[:-4]
        arm = tag.split("-")[1]
        r = dict(tag=tag, arm=arm, sysfs={}, st=[])
        for line in open(f):
            if line.startswith("tok/s_total"):
                r["tps"] = float(re.search(r"tok/s_gen: ([0-9.]+)", line).group(1))
            elif line.startswith("tokens:"):
                r["decode_s"] = float(re.search(r"decode_s: ([0-9.]+)", line).group(1))
            elif line.startswith("SYSFS"):
                m = re.match(r"SYSFS (\S+) read_us (\S+) dpm (.*) gpu_metrics (\S+)", line)
                r["sysfs"][m.group(1)] = dict(read_us=float(m.group(2)), dpm=m.group(3).strip(), **(gm(m.group(4)) or {}))
            elif line.startswith("STAMP "):
                p = line.split()
                seq, win, layer, site, m = map(int, p[1:6])
                mn, mx, srt, scy, arr, b0rt, b0cy, uns = map(int, p[6:14])
                r["st"].append(dict(seq=seq, win=win, layer=layer, site=site, m=m, mn=mn, mx=mx, arrivals=arr, unsafe=uns,
                                     span=(mx - mn) / 100.0, sum_rt=srt / 100.0, sum_cyc=scy, b0=b0rt / 100.0, cyc=b0cy))
            elif line.startswith("GENERATED"):
                r["gen"] = hashlib.sha256(line.encode()).hexdigest()[:12]
        runs[tag] = r
    return runs


def aggregate(runs, site_names):
    nsites = len(site_names)
    # Full names, not first letters: a heterogeneous site set (this repo's
    # SITES has "att_*" and "ssm_*" siblings) collides on first letter, e.g.
    # "att_qf"->"att_k" and "att_v"->"att_out" would both read "a2a". Full
    # names keep every (site[i], site[i+1]) pair a distinct key.
    gap_names = [f"{site_names[i]}->{site_names[(i + 1) % nsites]}" for i in range(nsites)]
    agg = {}
    for tag, r in runs.items():
        st = r["st"]
        if not st:
            agg[tag] = dict(arm=r["arm"], tps=r.get("tps"), decode_s=r.get("decode_s"), gen=r.get("gen"), n=0)
            continue
        bywin = {}
        for s in st:
            bywin.setdefault(s["win"], []).append(s)
        sums = {n: 0.0 for n in site_names}
        sums.update({g: 0.0 for g in gap_names})
        sums["win"] = 0.0
        spans = {n: [] for n in site_names}
        effs = {n: [] for n in site_names}
        b0effs = {n: [] for n in site_names}
        gaps = {g: [] for g in gap_names}
        wrap_unsafe = 0
        ms = set()
        nwin = 0
        arrivals = collections.Counter()
        for w, L in bywin.items():
            L.sort(key=lambda s: s["seq"])
            nwin += 1
            for s in L:
                ms.add(s["m"])
                name = site_names[s["site"]]
                sums[name] += s["span"]
                spans[name].append(s["span"])
                wrap_unsafe += s["unsafe"]
                arrivals[(name, s["arrivals"])] += 1
                if s["sum_rt"] > 0:
                    effs[name].append(s["sum_cyc"] / s["sum_rt"])
                if s["b0"] > 0:
                    b0effs[name].append(s["cyc"] / s["b0"])
            for a, b in zip(L, L[1:]):
                g = (b["mn"] - a["mx"]) / 100.0
                if b["site"] == (a["site"] + 1) % nsites:
                    k = gap_names[a["site"]]
                    sums[k] += g
                    gaps[k].append(g)
            sums["win"] += (L[-1]["mx"] - L[0]["mn"]) / 100.0
        agg[tag] = dict(arm=r["arm"], tps=r.get("tps"), decode_s=r.get("decode_s"), gen=r.get("gen"), n=len(st), nwin=nwin, ms=sorted(ms),
                         wrap_unsafe=wrap_unsafe, arrivals=dict(arrivals), med_b0eff={k: med(v) for k, v in b0effs.items()},
                         sums={k: v / 1e3 for k, v in sums.items()}, med_span={k: med(v) for k, v in spans.items()},
                         med_eff={k: med(v) for k, v in effs.items()}, med_gap={k: med(v) for k, v in gaps.items()}, sysfs=r["sysfs"])
    return agg, site_names, gap_names


def report(agg, site_names, gap_names):
    print("== per run (sums in ms; spans/gaps medians in us; eff = block-0 MHz)")
    for tag, a in agg.items():
        print(f"{tag}: tps={a['tps']} decode_s={a['decode_s']} gen={a['gen']} launches={a['n']}", end="")
        if a["n"]:
            print(f" windows={a['nwin']} m={a['ms']} wrap_unsafe={a['wrap_unsafe']} arrivals={a['arrivals']} b0_eff_MHz=" + " ".join(f"{k}={v:.0f}" for k, v in a["med_b0eff"].items()))
            print("   sums_ms " + " ".join(f"{k}={v:.2f}" for k, v in a["sums"].items()))
            print("   med_span_us " + " ".join(f"{k}={v:.2f}" for k, v in a["med_span"].items()) + " | med_gap_us " + " ".join(f"{k}={v:.2f}" for k, v in a["med_gap"].items()) + " | eff_MHz " + " ".join(f"{k}={v:.0f}" for k, v in a["med_eff"].items()))
            for k in ("pre_run", "pre_decode", "post_decode"):
                s = a["sysfs"].get(k)
                if s:
                    print(f"   sysfs {k}: read_us={s['read_us']:.0f} avg_gfxclk={s.get('avg_gfxclk')} cur_gfxclk={s.get('cur_gfxclk')} avg_uclk={s.get('avg_uclk')} cur_uclk={s.get('cur_uclk')} power_w={s.get('power_w')} gfx_act={s.get('gfx_act')} t_hot={s.get('t_hot')} throttle={s.get('throttle')} indep={s.get('indep_throttle')} dpm=[{s['dpm']}]")
        else:
            print()
    arms = sorted(set(a["arm"] for a in agg.values()))
    sum_keys = site_names + gap_names + ["win"]
    print("== per arm (median over runs)")
    for arm in arms:
        A = [a for a in agg.values() if a["arm"] == arm]
        tps = [a["tps"] for a in A if a["tps"]]
        print(f"{arm}: n={len(A)} tps med={med(tps):.2f} min..max={min(tps):.2f}..{max(tps):.2f} spread={(max(tps) / min(tps) - 1) * 100:.2f}% decode_s med={med([a['decode_s'] for a in A if a['decode_s']]):.4f}")
        if all(a["n"] for a in A):
            for k in sum_keys:
                v = [a["sums"][k] for a in A]
                print(f"   sum_{k}_ms med={med(v):.2f} min..max={min(v):.2f}..{max(v):.2f}")
            for k in site_names:
                print(f"   med_span_{k}_us={med([a['med_span'][k] for a in A]):.2f} eff_MHz_{k}={med([a['med_eff'][k] for a in A]):.0f}")
            for k in ("pre_run", "pre_decode", "post_decode"):
                for fld in ("avg_gfxclk", "cur_gfxclk", "avg_uclk", "power_w", "t_hot"):
                    v = [a["sysfs"][k][fld] for a in A if k in a["sysfs"] and fld in a["sysfs"][k]]
                    if v:
                        print(f"   sysfs {k} {fld}: med={med(v)} min..max={min(v)}..{max(v)}", end="")
                print()
    if len(arms) == 2 and all(a["n"] for a in agg.values()):
        c, d = arms[0], arms[1]
        C = [a for a in agg.values() if a["arm"] == c]
        D = [a for a in agg.values() if a["arm"] == d]
        print(f"== {d} minus {c} (medians over runs, ms per run)")
        for k in sum_keys:
            vc = med([a["sums"][k] for a in C])
            vd = med([a["sums"][k] for a in D])
            rc = [a["sums"][k] for a in C]
            rd = [a["sums"][k] for a in D]
            disj = "disjoint" if (min(rd) > max(rc) or max(rd) < min(rc)) else "overlap"
            pct = f"{(vd / vc - 1) * 100:+.2f}%" if vc else "n/a (zero)"
            print(f"   {k}: {vd - vc:+.2f} ms ({pct}) {disj}")
        print(f"   decode_s: {med([a['decode_s'] for a in D]) - med([a['decode_s'] for a in C]):+.4f} s")
        for k in site_names:
            ec = med([a["med_eff"][k] for a in C])
            ed = med([a["med_eff"][k] for a in D])
            print(f"   eff_MHz_{k}: {c}={ec:.0f} {d}={ed:.0f} ratio={ed / ec:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", nargs="?", default=DEFAULT_RUNS)
    ap.add_argument("pattern", nargs="?", default="S-*.log")
    ap.add_argument("--site-names", default=",".join(DEFAULT_SITE_NAMES),
                     help="comma-separated names for STAMP site index 0,1,2,... (default: gate,up,down)")
    args = ap.parse_args()
    site_names = args.site_names.split(",")
    runs = parse_runs(args.runs_dir, args.pattern)
    agg, site_names, gap_names = aggregate(runs, site_names)
    report(agg, site_names, gap_names)


if __name__ == "__main__":
    main()
