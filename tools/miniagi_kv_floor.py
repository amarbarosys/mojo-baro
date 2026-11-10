"""bf16 rounding floor in the KV-segment formulation the Mojo model uses (one graph copy per row, K/V cached). Row kinds TOK, SOFT, LAT (previous row's final-norm hidden), PSE (pause vector).

Usage: LAT_D=.. LAT_H=8 .work/oracle-venv/bin/python -P tools/miniagi_kv_floor.py CASE_DIR   [N]   (prints `group floor` lines; N > 0 adds
N bf16 draws on params perturbed by 1e-7 relative, the spread a single floor draw hides)
The recompute oracle (miniagi_latent_oracle.py) re-runs the prefix for every latent row, so with ORACLE_BF16=1 each copy
rounds its own upstream grad; Mojo sums a row's grad over all later rows first and rounds once. This floor rounds where
Mojo does. Its fp32 grads must match the case's grad.f32 (`check` line) or the formulation is wrong.
"""
import importlib.util
import sys

import numpy as np
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("o", "tools/miniagi_latent_oracle.py")
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
D, H, HD, L, V = o.D, o.H, o.HD, o.L, o.V


def rope_at(x, p0):
    t = torch.arange(p0, p0 + x.shape[1], dtype=torch.float64)
    inv = o.THETA ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = torch.outer(t, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None, :].float()
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None, :].float()
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def segment(flat, x, cache, p0):
    b, t, _ = x.shape
    mm = o.mm
    for l in range(L):
        w = o.views(flat, l)
        n = o.rms(x, w["g1"])
        q = rope_at(mm(n, w["wq"]).view(b, t, H, HD), p0).transpose(1, 2)
        k = rope_at(mm(n, w["wk"]).view(b, t, H, HD), p0).transpose(1, 2)
        v = mm(n, w["wv"]).view(b, t, H, HD).transpose(1, 2)
        K = torch.cat([cache[l][0], k], 2) if p0 else k
        Vv = torch.cat([cache[l][1], v], 2) if p0 else v
        cache[l] = (K, Vv)
        mask = torch.arange(p0 + t)[None, :] <= torch.arange(p0, p0 + t)[:, None]
        a = F.scaled_dot_product_attention(q, K, Vv, attn_mask=mask)
        x = x + mm(a.transpose(1, 2).reshape(b, t, D), w["wo"])
        n = o.rms(x, w["g2"])
        x = x + mm(F.silu(mm(n, w["w1"])) * mm(n, w["w3"]), w["w2"])
    return o.rms(x, flat[o.GF:o.GF + D])


def loss_kv(flat, ids, tgt, w, kinds):
    b, S = ids.shape
    E = flat[: V * D].view(V, D)
    cache, zs, hs, r, j = [None] * L, [], [], 0, 0
    while r < S:
        if kinds[r] == o.TOK:
            e = r
            while e < S and kinds[e] == o.TOK:
                e += 1
            x = E[ids[:, r:e]]
        elif kinds[r] == o.SOFT:
            e = r + 1
            x = (torch.softmax(zs[-1][:, -1] / o.TEMP, -1) @ E)[:, None]
        elif kinds[r] == o.LAT:
            e = r + 1
            x = hs[-1][:, -1:]
        elif kinds[r] == o.PSE:
            e = r + 1
            x = flat[o.PAUSE + j * D: o.PAUSE + (j + 1) * D].expand(b, 1, D)
            j += 1
        else:
            raise SystemExit(f"FAIL kind {kinds[r]} not supported")
        h = segment(flat, x, cache, r)
        hs.append(h)
        zs.append(o.mm(h, E.T))
        r = e
    z = torch.cat(zs, 1)
    ce = F.cross_entropy(z.reshape(-1, V), tgt.clamp(min=0).reshape(-1), reduction="none").view(b, S)
    return (ce * w * (tgt >= 0)).sum()


def grad(case, bf16, seed=-1):
    o.mm = o._MM.apply if bf16 else torch.matmul
    ld = lambda n: torch.from_numpy(np.fromfile(f"{case}/{n}.f32", np.float32))
    b, kinds = int(ld("b")[0]), [int(x) for x in ld("kinds")]
    S = len(kinds)
    p = ld("params").clone()
    if seed >= 0:
        p = p * (1 + 1e-7 * torch.randn(p.shape, generator=torch.Generator().manual_seed(seed)))
    p.requires_grad_()
    loss_kv(p, ld("ids").view(S, b).T.long(), ld("tgt").view(S, b).T.long(), ld("w").view(S, b).T, kinds).backward()
    return p.grad, ld("grad")


g32, ref = grad(sys.argv[1], False)
g16, _ = grad(sys.argv[1], True)
groups = dict(embed=(0, V * D), layer0=(V * D, V * D + o.LAYER), layer1=(V * D + o.LAYER, o.GF), final_gain=(o.GF, o.PAUSE), pause=(o.PAUSE, o.NP))
print("check fp32 kv vs recompute", max(((g32[s:e] - ref[s:e]).abs().max() / ref[s:e].abs().max()).item() for s, e in groups.values()))
for n, (s, e) in groups.items():
    print(n, ((g16[s:e] - g32[s:e]).abs().max() / g32[s:e].abs().max()).item())
draws = [grad(sys.argv[1], True, k)[0] for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 0)]
for n, (s, e) in groups.items():
    if draws:
        f = sorted(((d[s:e] - g32[s:e]).abs().max() / g32[s:e].abs().max()).item() for d in draws)
        print("draws", n, "min", f[0], "median", f[len(f) // 2], "max", f[-1])
