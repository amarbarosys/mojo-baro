#!/usr/bin/env python3
"""Tiny spark2_5-shaped GGUF that drives every branch of tools/spark-pack.{py,mojo}: token_embd and the layer
matrices cycle through Q8_0, Q4_K, Q5_K and Q6_K (random blocks with finite f16 scales), norms are F32. Fixture for
tools/spark-pack-check.sh only; the tensor names and layer count are spark's, the sizes are not.

Usage: tools/spark-pack-fixture.py OUT.gguf [SEED]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
from gguf import GGUFWriter, GGMLQuantizationType as Q  # noqa: E402

BLK = {Q.Q8_0: (32, 34, [0]), Q.Q4_K: (256, 144, [0, 2]), Q.Q5_K: (256, 176, [0, 2]), Q.Q6_K: (256, 210, [208])}


def blocks(rng, qt, rows, ne0):
    per, nb, f16_at = BLK[qt]
    b = rng.integers(0, 256, size=(rows, ne0 // per, nb), dtype=np.uint8)
    for o in f16_at:
        b[:, :, o:o + 2] = rng.uniform(-0.05, 0.05, size=(rows, ne0 // per)).astype(np.float16).view(np.uint8).reshape(rows, ne0 // per, 2)
    return b.reshape(rows, -1)


def main():
    out, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rng = np.random.default_rng(seed)
    w = GGUFWriter(out, "spark2_5")
    cyc = [Q.Q8_0, Q.Q4_K, Q.Q5_K, Q.Q6_K]
    ne0, k = 256, 0

    def mat(name, rows, qt=None):
        nonlocal k
        qt = qt or cyc[k % 4]
        k += 1
        w.add_tensor(name, blocks(rng, qt, rows, ne0), raw_dtype=qt)

    mat("token_embd.weight", 8, Q.Q5_K)
    for i in range(36):
        b = f"blk.{i}."
        w.add_tensor(b + "attn_norm.weight", rng.standard_normal(ne0).astype(np.float32))
        for n in ("attn_qkv.weight", "attn_gate.weight", "attn_output.weight"):
            mat(b + n, 2)
        w.add_tensor(b + "ffn_norm.weight", rng.standard_normal(ne0).astype(np.float32))
        for n in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            mat(b + n, 2)
    w.add_tensor("output_norm.weight", rng.standard_normal(ne0).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
