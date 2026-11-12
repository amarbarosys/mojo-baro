#!/usr/bin/env python3
"""latent-bridge: frozen Flash weights the trainer needs, dequantized by gguf-py (one-off read of a
Python-only library).

  OUT/head.f32      output.weight, [248320, 2560] f32 row-major (row v = logit v), route A's frozen head
  OUT/wo47.f32      blk.47.attn_output.weight, [2560, 6144] f32 (row i = output i), route B's projection
  OUT/ctarget-NAME.f32  for each id in SEL (one per line, dump order): 32 rows of 2560 f32, row j = mean
                    token_embd row over chunk j of the item's rationale ids (route C-surrogate target)

usage: tools/latent-bridge-weights.py SHARD1.gguf ITEMS.tsv OUT [NAME=SEL.txt ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Projects/imports/llama.cpp-master/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

NSLOT = 32


def tensors(shard1):
    out = {}
    for p in sorted(Path(shard1).parent.glob(Path(shard1).name.replace("00001-of", "*-of"))):
        for t in GGUFReader(str(p)).tensors:
            out[t.name] = t
    return out


def deq(t):
    n_out, k = int(t.shape[1]), int(t.shape[0])
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(n_out, k)


def main():
    shard1, items, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    ts = tensors(shard1)
    for name, fn in (("output.weight", "head.f32"), ("blk.47.attn_output.weight", "wo47.f32")):
        w = deq(ts[name])
        w.tofile(out / fn)
        print(f"{fn} {name} {ts[name].tensor_type.name} shape {w.shape} max|w| {np.abs(w).max():.4g}")
    rat = {}
    for line in open(items):
        f = line.rstrip("\n").split("\t")
        rat[f[0]] = [int(x) for x in f[6].split()]
    emb = deq(ts["token_embd.weight"])
    for spec in sys.argv[4:]:
        name, sel = spec.split("=", 1)
        ids = [s.strip() for s in open(sel) if s.strip()]
        y = np.zeros((len(ids), NSLOT, emb.shape[1]), np.float32)
        for i, iid in enumerate(ids):
            r = rat[iid]
            for j, chunk in enumerate(np.array_split(np.array(r), NSLOT)):
                y[i, j] = emb[chunk].mean(0) if len(chunk) else emb[r[min(j * len(r) // NSLOT, len(r) - 1)]]
        y.tofile(out / f"ctarget-{name}.f32")
        print(f"ctarget-{name}.f32 items {len(ids)} rows {NSLOT} median row norm {np.median(np.linalg.norm(y, axis=-1)):.4g}")


if __name__ == "__main__":
    main()
