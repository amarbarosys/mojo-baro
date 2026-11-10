#!/usr/bin/env python3
"""Numpy float64 oracle for kernels/test_kq_prefill_gemm.mojo (kq batched prefill, bench/llama-stint-protocol.md run 9).

Per dumped tensor: ref16 = A_f16 @ W_f16^T over the device's own dequantised f16 bytes (the operands the WMMA GEMM
read) and refx = A @ dequant(raw Q4_K / Q6_K)^T through gguf-py (the exact values the per-row GEMV uses). Gates
every element of the WMMA output on ref16 and of the GEMV output on refx with |got - ref| <= 2^-8 |ref| + 2^-19 S,
S = sum_k |a_k| |w_k| of that element (32 f32 ulps of the absolute sum: the accumulation slack of a K = 4096..12288 dot;
the KATT floor 1e-5 was float32 slack for O(1) attention outputs and fails near-zero outputs of 45-scale sums, so that
frozen verdict is printed beside it, not gated). WMMA against refx is reported, not gated (f16 weight rounding).
usage: tools/kq-prefill-ref.py [DIR]   (default .work/kqpf); exit 1 on any violation.
"""
import sys
from pathlib import Path

import numpy as np
from gguf.constants import GGMLQuantizationType as Q
from gguf.quants import dequantize

TOL_REL, TOL_SUM, KATT_ABS = 2.0 ** -8, 2.0 ** -19, 1e-5


def bf16(p, shape):
    u = np.fromfile(p, dtype=np.uint16).reshape(shape).astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def gate(name, got, ref, asum):
    err = np.abs(got - ref)
    bar = TOL_REL * np.abs(ref) + TOL_SUM * asum
    bad = int((err > bar).sum())
    katt = int((err > TOL_REL * np.abs(ref) + KATT_ABS).sum())
    rel = float((err / np.maximum(np.abs(ref), 1e-30))[np.abs(ref) > 0.1 * np.abs(ref).max()].max())
    print(f"  {name:14s} max|err| {err.max():.3e}  rel(large) {rel:.2e}  worst err/bar {float((err / bar).max()):.3f}"
          f"  violations {bad}/{got.size}  (frozen KATT floor 1e-5: {katt})")
    return bad


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".work/kqpf")
    metas = sorted(d.glob("*.meta"))
    if not metas:
        sys.exit(f"FAIL: no .meta in {d}")
    bad = 0
    for meta in metas:
        tag = meta.stem
        f = meta.read_text().split()
        gt, n, k, mp = (int(x) for x in f[:4])
        ms = [int(x) for x in f[4:]]
        ws = np.fromfile(d / f"{tag}.ws.f16", dtype=np.float16).reshape(n, k).astype(np.float64)
        raw = np.fromfile(d / f"{tag}.w.bin", dtype=np.uint8).reshape(n, -1)
        wx = dequantize(raw, Q.Q4_K if gt == 12 else Q.Q6_K).astype(np.float64)
        if wx.shape != (n, k):
            sys.exit(f"FAIL {tag}: gguf dequant shape {wx.shape}, want {(n, k)}")
        print(f"{tag}: kind {gt} N {n} K {k}  max|w16 - wx| {np.abs(ws - wx).max():.3e}")
        a = bf16(d / f"{tag}.a.bf16", (mp, k))
        for m in ms:
            ref16 = a[:m] @ ws.T
            refx = a[:m] @ wx.T
            asum = np.abs(a[:m]) @ np.abs(wx).T
            wmma = np.fromfile(d / f"{tag}.m{m}.wmma.f32", dtype=np.float32).reshape(m, n).astype(np.float64)
            gemv = np.fromfile(d / f"{tag}.m{m}.gemv.f32", dtype=np.float32).reshape(m, n).astype(np.float64)
            lds = np.fromfile(d / f"{tag}.m{m}.lds.f32", dtype=np.float32).reshape(m, n).astype(np.float64)
            print(f" m {m}")
            bad += gate("wmma vs ref16", wmma, ref16, asum)
            bad += gate("lds  vs ref16", lds, ref16, asum)
            bad += gate("gemv vs refx", gemv, refx, asum)
            gate("wmma vs refx*", wmma, refx, asum)
            gate("lds  vs refx*", lds, refx, asum)
            gate("lds  vs wmma*", lds, wmma, asum)
    if bad:
        sys.exit(f"FAIL kq-prefill-ref: {bad} elements outside |err| <= 2^-8 |ref| + 2^-19 sum|a w|")
    print("PASS kq-prefill-ref")


if __name__ == "__main__":
    main()
