#!/usr/bin/env python3
"""Scratch and spill census of the AMDGPU code objects embedded in a Mojo binary,
aggregated per kernel family, optionally judged against a baseline binary's census.

    tools/isa-spills.py ENGINE                      -> JSON census on stdout
    tools/isa-spills.py ENGINE --baseline CENSUS.json
        exit 0: no kernel family has more scratch or more spills than the baseline,
                and no family absent from the baseline has any;
        exit 1: regressions listed on stdout, or no code objects found.

A family is the kernel name with its trailing 16-hex instantiation hash removed,
so the same source compiled into several MROWS variants is one row: scratch is
the max over the variants, spills the sum. The loop gate uses this for its stage
4 since 2026-09-08: the champion's own sources carry spilling delta-step
variants, so an absolute "no spills anywhere" rule can pass nothing.
"""
import argparse, json, re, struct, subprocess, sys, tempfile

READELF = "/opt/rocm/llvm/bin/llvm-readelf"
FAMILY = re.compile(r"_[0-9a-f]{16}$")


def code_objects(path):
    d = open(path, "rb").read()
    i = 0
    while True:
        i = d.find(b"\x7fELF", i)
        if i < 0:
            return
        if struct.unpack_from("<H", d, i + 18)[0] == 224:
            shoff, = struct.unpack_from("<Q", d, i + 40)
            se, sn = struct.unpack_from("<HH", d, i + 58)
            yield d[i:i + shoff + se * sn]
        i += 4


def census(path):
    fam = {}
    n = 0
    with tempfile.NamedTemporaryFile(suffix=".co") as tmp:
        for co in code_objects(path):
            n += 1
            tmp.seek(0); tmp.truncate(); tmp.write(co); tmp.flush()
            notes = subprocess.run([READELF, "--notes", tmp.name], capture_output=True, text=True).stdout
            name = None
            for line in notes.splitlines():
                m = re.match(r"\s*\.name:\s+(\S+)", line)
                if m:
                    name = FAMILY.sub("", m.group(1))
                    fam.setdefault(name, {"scratch": 0, "spills": 0, "n": 0})["n"] += 1
                    continue
                if name is None:
                    continue
                m = re.match(r"\s*\.(private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count):\s+(\d+)", line)
                if m:
                    v = int(m.group(2))
                    if m.group(1) == "private_segment_fixed_size":
                        fam[name]["scratch"] = max(fam[name]["scratch"], v)
                    else:
                        fam[name]["spills"] += v
    return n, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--baseline", help="census JSON of the champion build")
    a = ap.parse_args()
    n, fam = census(a.engine)
    if n == 0:
        print("no AMDGPU code objects found")
        return 1
    if not a.baseline:
        json.dump({"code_objects": n, "families": fam}, sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    base = json.load(open(a.baseline))["families"]
    bad = []
    for k, v in sorted(fam.items()):
        b = base.get(k, {"scratch": 0, "spills": 0})
        if v["scratch"] > b["scratch"] or v["spills"] > b["spills"]:
            bad.append(f"{k}: scratch {b['scratch']}->{v['scratch']} spills {b['spills']}->{v['spills']}"
                       + ("" if k in base else " (new family)"))
    print(f"code_objects={n} families={len(fam)} regressions={len(bad)}")
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
