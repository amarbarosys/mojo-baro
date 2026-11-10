#!/usr/bin/env python3
"""Oracle for i-quant kernel tests: dump one expert of a real GGUF tensor.

usage: iq-oracle.py GGUF_SHARD TENSOR EXPERT OUTDIR   (EXPERT -1: a 2-D tensor, all rows; A:B experts A..B-1 stacked as rows)
writes OUTDIR/{w.bin, deq.f32, x.bf16, y.f64, meta.txt}: raw block bytes of the
expert, gguf-py dequant (the reference; Q2_0, absent from gguf-py, decoded here from llama.cpp's dequantize_row_q2_0), a seeded bf16 input row and the f64 GEMV.
"""
import sys, pathlib
import numpy as np
import gguf
from gguf.quants import dequantize

shard, name, spec, out = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
lo, hi = (int(v) for v in spec.split(":")) if ":" in spec else (int(spec), int(spec) + 1)
expert = int(spec) if ":" not in spec else lo
out.mkdir(parents=True, exist_ok=True)
t = next(t for t in gguf.GGUFReader(shard).tensors if t.name == name)
k, n = int(t.shape[0]), int(t.shape[1])
raw = np.asarray(t.data).reshape(-1) if expert < 0 else np.asarray(t.data).reshape(int(t.shape[2]), -1)[lo:hi].reshape(-1)
if expert >= 0: n *= hi - lo
def q2_0(raw):
    b = raw.reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = (b[:, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
    return ((q.reshape(-1, 64).astype(np.int8) - 1).astype(np.float32) * d).reshape(-1)


deq = (q2_0(raw) if int(t.tensor_type) == 42 else dequantize(raw, t.tensor_type)).astype(np.float32).reshape(n, k)
rng = np.random.default_rng(7)
x32 = rng.standard_normal(k).astype(np.float32)
xb = (x32.view(np.uint32) >> 16).astype(np.uint16)
xf = (xb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
y = deq.astype(np.float64) @ xf
raw.tofile(out / "w.bin"); deq.tofile(out / "deq.f32"); xb.tofile(out / "x.bf16"); y.tofile(out / "y.f64")
(out / "meta.txt").write_text(f"{t.tensor_type.name} k={k} n={n} row_bytes={raw.size // n}\n")
print(f"OK {name} e{expert} {t.tensor_type.name} k={k} n={n} bytes={raw.size} -> {out}")
