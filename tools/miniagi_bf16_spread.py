"""bf16 parity bar: is Mojo's bf16 grad inside torch's own bf16 draw-to-draw spread? (frozen 2026-10-10, exchange/miniagi-bf16-bar-prereg-2026-10-10.md)

Usage: LAT_D=.. LAT_H=8 ORACLE_CASE=.. .work/oracle-venv/bin/python -P tools/miniagi_bf16_spread.py CASE_DIR MOJO_BF16_DUMP [N=16]
Torch draws: KV-formulation bf16 grads on params perturbed 1e-7 rel (tools/miniagi_kv_floor.py), N draws.
Groups embed, layer0, layer1, final_gain, pause (a group whose fp32 grad is all zero is skipped). Per group, PASS when median_k rel(mojo, draw_k) <= max_{i<j} rel(draw_i, draw_j); rel = max|x - y| / max|fp32 grad|.
Prints one line per group and `PASS` or `FAIL k/4 names`, exits 1 on FAIL.
"""
import sys

import numpy as np
import torch

case, dump, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
sys.argv = ["x", case]
exec(open("tools/miniagi_kv_floor.py").read().split("g32, ref = grad")[0])
g32, _ = grad(case, False)
t16 = [grad(case, True, k)[0] for k in range(n)]
m16 = torch.from_numpy(np.fromfile(dump, np.float32))
groups = dict(embed=(0, V * D), layer0=(V * D, V * D + o.LAYER), layer1=(V * D + o.LAYER, o.GF), final_gain=(o.GF, o.PAUSE), pause=(o.PAUSE, o.NP))
groups = {k: v for k, v in groups.items() if g32[v[0]:v[1]].abs().max() > 0}
rel = lambda x, y, s, e: ((x[s:e] - y[s:e]).abs().max() / g32[s:e].abs().max()).item()
bad = []
for name, (s, e) in groups.items():
    tt = max(rel(t16[i], t16[j], s, e) for i in range(n) for j in range(i + 1, n))
    mt = sorted(rel(m16, t, s, e) for t in t16)[n // 2]
    ok = mt <= tt
    bad += [] if ok else [name]
    print(f"{'PASS' if ok else 'FAIL'} {name} mojo-vs-torch median {mt:.4f} torch-vs-torch max {tt:.4f}")
print("PASS" if not bad else f"FAIL {len(bad)}/{len(groups)} {' '.join(bad)}")
sys.exit(1 if bad else 0)
