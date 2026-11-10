#!/usr/bin/env python3
"""dump-diff: locate the first divergence between two BARO_DUMP files.

BARO_DUMP=path makes the engine write X (the residual) after every layer's
sub-block and after its ffn, for every decode token, on both the launch path
(BARO_MEGA=0) and the megakernel path. Layout: [token][2*layer + half][H] f32.

usage: tools/dump-diff.py A.bin B.bin [--layers 32] [--hidden 4096]
Prints the first divergent (token, layer, half), how many elements differ, the
max |d|, whether the difference looks like a pure scale (rmsnorm) or per-element
(GEMM input rounding), then one line per token with its first divergent slot.
"""
import sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--layers", type=int, default=32)
ap.add_argument("--hidden", type=int, default=4096)
ap.add_argument("--all", action="store_true", help="print every token's first divergent slot")
o = ap.parse_args()
H, L = o.hidden, o.layers; S = 2 * L
a = np.fromfile(o.a, dtype=np.float32); b = np.fromfile(o.b, dtype=np.float32)
n = min(len(a), len(b)) // (S * H)
a = a[:n * S * H].reshape(n, S, H); b = b[:n * S * H].reshape(n, S, H)
print(f"tokens {n}, slots/token {S} (layer*2 + half; half 0 = after attn/ssm sub-block, 1 = after ffn)")
found = False
for t in range(n):
    for k in range(S):
        d = b[t, k] - a[t, k]
        if np.abs(d).max() > 0 or np.isnan(d).any():
            l, half = k // 2, ("after-ffn" if k % 2 else "after-sub-block")
            kind = "attn" if (l + 1) % 4 == 0 else "ssm"
            ratio = np.divide(b[t, k], a[t, k], out=np.ones_like(a[t, k]), where=a[t, k] != 0)
            prev = a[t, k - 1] if k > 0 else None
            print(f"FIRST DIVERGENCE: token {t} layer {l} ({kind}) {half}: {(d != 0).sum()}/{H} differ, max|d| {np.abs(d).max():.3g}, max|a| {np.abs(a[t, k]).max():.3g}")
            print(f"  ratio b/a over differing elements: min {ratio[d != 0].min():.6f} max {ratio[d != 0].max():.6f}  ({'pure scale -> rmsnorm/scalar' if np.ptp(ratio[d != 0]) < 1e-5 else 'per-element -> input rounding / FMA form'})")
            if prev is not None:
                ca, cb = a[t, k] - prev, b[t, k] - prev
                r2 = np.divide(cb, ca, out=np.ones_like(ca), where=ca != 0)
                print(f"  contribution ratio (b-prev)/(a-prev): p5 {np.percentile(r2, 5):.5f} p50 {np.percentile(r2, 50):.5f} p95 {np.percentile(r2, 95):.5f}")
            found = True
            break
    if found:
        break
if not found:
    print("identical over all dumped tokens and slots")
    sys.exit(0)
if o.all:
    for t in range(n):
        ks = [k for k in range(S) if np.abs(b[t, k] - a[t, k]).max() > 0]
        if ks:
            print(f"token {t}: first divergent slot {ks[0]} (layer {ks[0] // 2}), {len(ks)} slots differ")
sys.exit(1)
