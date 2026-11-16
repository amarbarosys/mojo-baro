#!/usr/bin/env python3
"""Oracle: per-layer comparison of the spark engine against llama.cpp on ONE prompt (the 0.5B gap, BASELINE.md 0.5B rows).

  tools/layer-compare-oracle.py OURS_LAYERS.bin H REF.log [CONTROL.log]

OURS_LAYERS.bin  BARO_DUMP_LAYERS output of serve/spark.mojo: per layer, H f32, the residual after the layer at the last
                 prompt position.
REF.log          llama-eval-callback log of the same prompt (fp16 GGUF, the highest-fidelity reference).
CONTROL.log      same for the quantized GGUF the pack was built from: the noise floor a correct quantized engine shows.
llama prints, per tensor, the first and last three columns of the first and last three rows, so the last token's columns
0..2 and H-3..H-1 are the six comparable values per layer (`l_out-N`). tools/llama-oracle.py parses the same log but
keeps only the per-tensor sum over all rows, which cannot be compared with one position, hence this parser.
Prints one row per layer: max |ours - ref| and the same for the control, each relative to max |ref|. The first layer
where ours leaves the control's band is the first divergent layer.
"""
import re, struct, sys

NUM = r"([-0-9.e+]+)"
ROW = re.compile(r"\s*\[\s*" + NUM + r",\s*" + NUM + r",\s*" + NUM + r",\s*\.\.\.,\s*" + NUM + r",\s*" + NUM + r",\s*" + NUM + r"\s*\]")


def l_out_last_row(path):
    rows, name, buf = {}, None, []
    for line in open(path, errors="replace"):
        m = re.search(r"\bl_out-(\d+) = ", line)
        if m:
            name, buf = int(m.group(1)), []
            continue
        if name is None:
            continue
        if line.strip().startswith("sum ="):
            if len(buf) >= 6:
                rows[name] = buf[-6:]
            name = None
            continue
        m = ROW.match(line)
        if m:
            buf.extend(float(x) for x in m.groups())
    return rows


def ours(path, h):
    raw = open(path, "rb").read()
    out = {}
    for i in range(len(raw) // (4 * h)):
        v = struct.unpack(f"{h}f", raw[i * h * 4:(i + 1) * h * 4])
        out[i] = [v[0], v[1], v[2], v[h - 3], v[h - 2], v[h - 1]]
    return out


def err(a, b):
    scale = max(abs(x) for x in b) or 1.0
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    h = int(sys.argv[2])
    o, r = ours(sys.argv[1], h), l_out_last_row(sys.argv[3])
    c = l_out_last_row(sys.argv[4]) if len(sys.argv) > 4 else {}
    if not r or not o:
        print("FAIL parse: no layers read (ours %d, ref %d)" % (len(o), len(r)))
        sys.exit(1)
    print("layer  ours_vs_ref  control_vs_ref   ours[0:3] | ref[0:3]")
    first = None
    for i in sorted(o):
        if i not in r:
            continue
        eo = err(o[i], r[i])
        ec = err(c[i], r[i]) if i in c else float("nan")
        flag = ""
        if c and i in c and eo > 3 * ec + 0.01 and first is None:
            first, flag = i, "  <-- first outside 3x control + 0.01"
        print(f"{i:5d}  {eo:11.4f}  {ec:14.4f}   {' '.join(f'{x:8.4f}' for x in o[i][:3])} | {' '.join(f'{x:8.4f}' for x in r[i][:3])}{flag}")
    print("first divergent layer: " + (str(first) if first is not None else "none outside the control band"))


main()
