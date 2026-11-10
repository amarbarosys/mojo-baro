#!/usr/bin/env python3
"""usage: slot-kv-diff.py REF.slot OURS.slot [--per-layer]

How far is the K/V our engine exported (through tools/state-to-llama-slot) from the K/V
llama.cpp computed itself for the same tokens? Both are llama-server slot files with the same
cells, so this is an elementwise comparison, no model needed. Relative error per tensor is
||ours - ref|| / ||ref||. A layout defect shows as error near or above 1 (unrelated values) or
as a few broken layers or positions; kernel numerics shows as a small, smooth error everywhere.
A verifier: it shares no code with the tool that wrote OURS."""
import struct
import sys

import numpy as np


def load(path):
    d = open(path, "rb").read()
    magic, ver, ntok = struct.unpack_from("<III", d, 0)
    assert magic == 0x67677371 and ver == 3, f"{path}: not a seq state v3 file"
    o = 12 + 4 * ntok
    n_stream, cells = struct.unpack_from("<II", d, o)
    o += 8
    for cell in (12, 24):
        p = o + cells * cell
        v_trans, n_layer = struct.unpack_from("<II", d, p)
        ty, rsz = struct.unpack_from("<iQ", d, p + 8)
        if v_trans == 0 and 0 < n_layer < 512 and ty in (0, 1) and 0 < rsz < (1 << 20):
            o = p + 8
            break
    else:
        raise SystemExit(f"FAIL slot-kv-diff: cannot locate the KV section in {path}")
    out = []
    for _side in range(2):
        layers = []
        for _ in range(n_layer):
            ty, rsz = struct.unpack_from("<iQ", d, o)
            o += 12
            dt, es = ("<f4", 4) if ty == 0 else ("<f2", 2)
            x = np.frombuffer(d, dtype=dt, count=cells * (rsz // es), offset=o).astype(np.float64)
            layers.append(x.reshape(cells, rsz // es))
            o += cells * rsz
        out.append(layers)
    return cells, out


def main():
    ref_path, ours_path = sys.argv[1], sys.argv[2]
    cr, ref = load(ref_path)
    co, ours = load(ours_path)
    if cr != co or len(ref[0]) != len(ours[0]) or ref[0][0].shape != ours[0][0].shape:
        raise SystemExit(f"FAIL slot-kv-diff: shapes differ: cells {cr} vs {co}")
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))  # noqa: E731
    res = {}
    for side, name in enumerate("KV"):
        per = [rel(ours[side][li], ref[side][li]) for li in range(len(ref[side]))]
        res[name] = per
        worst = int(np.argmax(per))
        # per-position error, to see whether one cell (e.g. position 0) carries the gap
        pos = [rel(np.concatenate([ours[side][li][t] for li in range(len(per))]), np.concatenate([ref[side][li][t] for li in range(len(per))])) for t in range(cr)]
        print(f"{name}: median {np.median(per):.4f}  max {max(per):.4f} at layer {worst}  layer0 {per[0]:.4f}  last {per[-1]:.4f}  | worst position {int(np.argmax(pos))} ({max(pos):.4f}), position 0 {pos[0]:.4f}")
    if "--per-layer" in sys.argv:
        for li in range(len(res["K"])):
            print(f"  layer {li:2d}  K {res['K'][li]:.4f}  V {res['V'][li]:.4f}")


if __name__ == "__main__":
    main()
