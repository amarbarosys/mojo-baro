#!/usr/bin/env python3
"""ISA receipt for a Mojo-built binary: per gfx code object, kernel resource
usage (VGPR/SGPR/LDS/scratch/spills) and an instruction histogram.

usage: tools/isa-receipt.py <binary> [--hist N] [--dump DIR]
Extracts every embedded e_machine=224 (AMDGPU) ELF, reads its metadata notes
with llvm-readelf and disassembles with llvm-objdump --mcpu=gfx1100.
"""
import argparse, collections, os, re, struct, subprocess, sys, tempfile

LLVM = "/opt/rocm/llvm/bin"
KEYS = (".vgpr_count", ".sgpr_count", ".group_segment_fixed_size",
        ".private_segment_fixed_size", ".vgpr_spill_count", ".sgpr_spill_count",
        ".max_flat_workgroup_size")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--hist", type=int, default=25)
    ap.add_argument("--dump", help="keep .co files and .s here")
    ap.add_argument("--summary", metavar="SUBSTR", help="one line per kernel whose name contains SUBSTR: vgpr, spills, lds, wg size; exit 1 on a spill")
    a = ap.parse_args()
    out = a.dump or tempfile.mkdtemp(prefix="isa-")
    os.makedirs(out, exist_ok=True)
    n = 0
    rows = []
    for blob in code_objects(a.binary):
        p = os.path.join(out, f"co{n}.co")
        open(p, "wb").write(blob)
        notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", p],
                               capture_output=True, text=True).stdout
        name = None
        vals = {}
        for line in notes.splitlines():
            s = line.strip()
            if s.startswith(".name:"):
                name = s.split(":", 1)[1].strip()
            for k in KEYS:
                if s.startswith(k + ":"):
                    vals[k[1:]] = s.split(":", 1)[1].strip()
                    if not a.summary:
                        print(f"co{n} {name} {k[1:]}={vals[k[1:]]}")
        if a.summary:
            if name and a.summary in name:
                rows.append((name, vals))
            n += 1
            continue
        asm = subprocess.run([f"{LLVM}/llvm-objdump", "-d", "--mcpu=gfx1100", p],
                             capture_output=True, text=True).stdout
        if a.dump:
            open(os.path.join(out, f"co{n}.s"), "w").write(asm)
        hist = collections.Counter()
        for line in asm.splitlines():
            m = re.match(r"\s*([a-z_0-9]+)\s", line)
            if m and ("_" in m.group(1)):
                hist[m.group(1)] += 1
        total = sum(hist.values())
        print(f"co{n} instructions={total}")
        for op, c in hist.most_common(a.hist):
            print(f"  {c:6d}  {op}")
        n += 1
    if n == 0:
        print("no AMDGPU code objects found", file=sys.stderr)
        sys.exit(1)
    if a.summary:
        spilled = 0
        for name, v in rows:
            sp = int(v.get("vgpr_spill_count", "0")) + int(v.get("sgpr_spill_count", "0"))
            spilled += sp > 0
            print(f"{name[-40:]:40s} vgpr {v.get('vgpr_count', '?'):>3s} sgpr {v.get('sgpr_count', '?'):>3s} spill {sp:3d} "
                  f"lds {int(v.get('group_segment_fixed_size', '0')) // 1024:3d}K wg {v.get('max_flat_workgroup_size', '?')}")
        print(f"{len(rows)} instances match '{a.summary}', {spilled} with spills")
        if spilled:
            sys.exit(1)
    if a.dump:
        print(f"dumped to {out}")


if __name__ == "__main__":
    main()
