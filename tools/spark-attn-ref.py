#!/usr/bin/env python3
"""Numpy float64 oracle for kernels/test_spark_attn.mojo (bench/dense-protocol.md, KATT).

Recomputes the NEOX rope on the appended K row and the gated decode attention per head
from the dumped inputs, then gates every bf16 output element on
|got - ref| <= 2^-8 * |ref| + 1e-5 (preregistered prediction 1). The appended cache row
is reported as a diagnostic: V must be copied exactly, K is compared to the float64 rope.
usage: tools/spark-attn-ref.py [DIR]   (default .work/katt); exit 1 on any violation.
"""
import sys
from pathlib import Path

import numpy as np

CASES = [(64, 32, 8), (128, 28, 4), (256, 16, 4)]
R, POS, WIN, BASE = 2, 299, 100, 1e4
TT = POS + R


def load(p, dt, shape):
    a = np.fromfile(p, dtype=dt)
    if a.size != int(np.prod(shape)):
        sys.exit(f"FAIL: {p} has {a.size} elements, want {shape}")
    return a.reshape(shape)


def bf16(p, shape):
    u = load(p, np.uint16, shape).astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def rne_bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".work/katt")
    bad = 0
    for hd, nqh, nkvh in CASES:
        f = lambda n: d / f"hd{hd}_{n}.bin"
        q = load(f("q"), np.float32, (R * nqh, hd)).astype(np.float64)
        kd = load(f("kd"), np.float32, (nkvh, TT, hd)).astype(np.float64)
        vd = load(f("vd"), np.float32, (nkvh, TT, hd)).astype(np.float64)
        g = load(f("gate"), np.float32, (R * nqh,)).astype(np.float64)  # one gate per query row
        app = load(f("kvapp"), np.float32, (2, nkvh, hd)).astype(np.float64)
        nrot, half = hd, hd // 2
        j = np.arange(half, dtype=np.float64)
        th = POS * BASE ** (-2.0 * j / nrot)
        c, s = np.cos(th), np.sin(th)
        k = kd[:, POS, :].copy()
        x0, x1 = k[:, :half].copy(), k[:, half:nrot].copy()
        k[:, :half], k[:, half:nrot] = x0 * c - x1 * s, x0 * s + x1 * c
        kerr = np.abs(app[0] - k).max()
        vexact = bool((app[1] == vd[:, POS, :]).all())
        print(f"HD {hd}: appended K max|err| vs f64 rope {kerr:.3e}, V copied exactly: {vexact}")
        if not vexact:
            bad += 1
        kd[:, POS, :] = k
        grp = nqh // nkvh
        sig = 1 / (1 + np.exp(-g))
        for win, name in ((0, "full"), (WIN, "swa")):
            got = bf16(f(f"o_{name}"), (R * nqh, hd))
            ref = np.empty_like(got)
            for r in range(R):
                t = POS + 1 + r
                lo = t - win if win > 0 and t > win else 0
                for h in range(nqh):
                    kv = h // grp
                    sc = kd[kv, lo:t] @ q[r * nqh + h] / np.sqrt(hd)
                    p = np.exp(sc - sc.max())
                    ref[r * nqh + h] = (p @ vd[kv, lo:t]) / p.sum() * sig[r * nqh + h]
            err = np.abs(got - ref)
            viol = int((err > 2.0 ** -8 * np.abs(ref) + 1e-5).sum())
            exact = float((got == rne_bf16(ref)).mean())
            print(f"HD {hd} {name}: max|err| {err.max():.3e}, violations {viol}/{got.size}, "
                  f"bf16-exact {exact:.4f}")
            bad += viol
    print("PASS: spark attention parity at HD 64/128/256" if bad == 0 else f"FAIL: {bad} violations")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
