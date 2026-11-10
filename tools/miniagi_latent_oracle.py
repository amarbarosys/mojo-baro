"""Torch (CPU) oracle for miniagi/latent: Coconut by recompute, the definition the KV-segment model must match.

Usage: .work/oracle-venv/bin/python -P tools/miniagi_latent_oracle.py OUTDIR
ORACLE_CASE=small (S=15, b=3, LAT/PSE/SOFT rows, aux 0.5) or cert (the certified Phase C shape: S=41, rows 34-40 SOFT, b=12, aux 0).
Weights follow miniagi/latent/lmodel.mojo row_weights (rows feeding LAT or SOFT are aux rows); test_latent checks them.
SOFT rows get softmax(z / TEMP) E (env ORACLE_TEMP, default 0.5) of the previous row (soft thinking), differentiable. Each latent row r gets the final-norm hidden of row r-1 from a full re-run over rows 0..r-1 (no cache), so the
Mojo model's segment/KV-accumulator backward is checked against an independent formulation.
Loss = sum over rows of w_row * CE_row: main rows share 1/n_main, rows that feed a latent row (readout aux) 0.5/n_aux.
Layout must match miniagi/latent/lmodel.mojo: [E V*D][L layers][final gain D][pause NTH*D], weights x @ W.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import os
D = int(os.environ.get("LAT_D", "256"))
V, FF, L, NTH, EPS, THETA = int(os.environ.get("LAT_V", "32")), D * 43 // 16, int(os.environ.get("LAT_L", "2")), 7, 1e-6, 10000.0
H = int(os.environ.get("LAT_H", "4"))
HD = D // H
LAYER = 2 * D + 4 * D * D + 3 * D * FF
GF = V * D + L * LAYER
PAUSE = GF + D
NP = PAUSE + NTH * D
TOK, LAT, PSE, SOFT = 0, 1, 2, 4
TEMP = float(os.environ.get("ORACLE_TEMP", "0.5"))
CASE = os.environ.get("ORACLE_CASE", "small")


class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _r(a) @ _r(b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        return _r(g) @ _r(b).transpose(-1, -2), _r(a).transpose(-1, -2) @ _r(g)


_r = lambda t: t.bfloat16().float()
mm = _MM.apply if os.environ.get("ORACLE_BF16") == "1" else torch.matmul


def views(flat, l):
    b = V * D + l * LAYER
    take = lambda o, *s: flat[b + o: b + o + torch.Size(s).numel()].view(*s)
    dd, df = D * D, D * FF
    return dict(g1=take(0, D), wq=take(D, D, D), wk=take(D + dd, D, D), wv=take(D + 2 * dd, D, D), wo=take(D + 3 * dd, D, D),
                g2=take(D + 4 * dd, D), w1=take(2 * D + 4 * dd, D, FF), w3=take(2 * D + 4 * dd + df, D, FF),
                w2=take(2 * D + 4 * dd + 2 * df, FF, D))


def rope(x):
    t = torch.arange(x.shape[1], dtype=torch.float64)
    inv = THETA ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = torch.outer(t, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None, :].float()
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None, :].float()
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


def hidden(flat, x):
    b, t, _ = x.shape
    for l in range(L):
        w = views(flat, l)
        n = rms(x, w["g1"])
        q = rope(mm(n, w["wq"]).view(b, t, H, HD))
        k = rope(mm(n, w["wk"]).view(b, t, H, HD))
        v = mm(n, w["wv"]).view(b, t, H, HD)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + mm(o.transpose(1, 2).reshape(b, t, D), w["wo"])
        n = rms(x, w["g2"])
        x = x + mm(F.silu(mm(n, w["w1"])) * mm(n, w["w3"]), w["w2"])
    return rms(x, flat[GF:GF + D])


def loss_of(flat, ids, tgt, w, kinds):
    b, S = ids.shape
    E = flat[: V * D].view(V, D)
    rows, j = [], 0
    for r in range(S):
        if kinds[r] == TOK:
            rows.append(E[ids[:, r]])
        elif kinds[r] == PSE:
            rows.append(flat[PAUSE + j * D: PAUSE + (j + 1) * D].expand(b, D))
            j += 1
        elif kinds[r] == SOFT:
            rows.append(torch.softmax(mm(hidden(flat, torch.stack(rows, 1))[:, -1], E.T) / TEMP, -1) @ E)
        else:
            rows.append(hidden(flat, torch.stack(rows, 1))[:, -1])
    z = mm(hidden(flat, torch.stack(rows, 1)), E.T)
    ce = F.cross_entropy(z.reshape(-1, V), tgt.clamp(min=0).reshape(-1), reduction="none").view(b, S)
    return (ce * w * (tgt >= 0)).sum()


COND_MAX, COND_TRIES = 2e-5, 20


def conditioning(flat, ids, tgt, w, kinds):
    global mm
    keep, mm = mm, torch.matmul
    gs = []
    for dt in (torch.float32, torch.float64):
        p = flat.to(dt).clone().requires_grad_()
        loss_of(p, ids, tgt, w.to(dt), kinds).backward()
        gs.append(p.grad.double())
    mm = keep
    g32, g64 = gs
    spans = [(0, V * D)] + [(V * D + l * LAYER, V * D + (l + 1) * LAYER) for l in range(L)] + [(GF, PAUSE), (PAUSE, NP)]
    return max(((g32[s:e] - g64[s:e]).abs().max() / g64[s:e].abs().max()).item() for s, e in spans if g64[s:e].abs().max() > 0)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(int(os.environ.get("ORACLE_SEED", "7")))
    if CASE == "text":
        b, kinds, aux_w = 12, [TOK] * 41, 0.0
    elif CASE == "cert":
        b, kinds, aux_w = 12, [TOK] * 34 + [SOFT] * NTH, 0.0
    else:
        b, kinds, aux_w = 3, [TOK] * 7 + [LAT, LAT, PSE, TOK, LAT, SOFT, SOFT, TOK], 0.5
    S = len(kinds)
    for draw in range(COND_TRIES):
        flat = torch.randn(NP, generator=g) * 0.05 * (256 / D) ** 0.5
        for l in range(L):
            w = views(flat, l)
            w["g1"].copy_(1 + 0.1 * torch.randn(D, generator=g))
            w["g2"].copy_(1 + 0.1 * torch.randn(D, generator=g))
        flat[GF:GF + D] = 1 + 0.1 * torch.randn(D, generator=g)
        ids = torch.randint(0, min(V, 21) if V == 32 else V, (b, S), generator=g)
        tgt = torch.randint(0, min(V, 21) if V == 32 else V, (b, S), generator=g)
        for r in range(S):
            if kinds[r] != TOK:
                ids[:, r] = -1
        tgt[:, [0, 1, 2, 9]] = -1
        aux = torch.tensor([aux_w > 0 and r + 1 < S and kinds[r + 1] in (LAT, SOFT) for r in range(S)])
        main_rows = (tgt >= 0) & ~aux
        w = main_rows.float() / main_rows.sum()
        if aux_w > 0:
            w = w + aux_w * ((tgt >= 0) & aux).float() / ((tgt >= 0) & aux).sum()
        cond = conditioning(flat, ids, tgt, w, kinds)
        if cond <= COND_MAX:
            break
        print(f"draw {draw} ill-conditioned: torch fp32 vs fp64 {cond:.2e} > {COND_MAX:.0e}, redrawing", file=sys.stderr)
    else:
        sys.exit(f"FAIL oracle: no well-conditioned case in {COND_TRIES} draws")
    p = flat.clone().requires_grad_()
    loss = loss_of(p, ids, tgt, w, kinds)
    loss.backward()
    pm = lambda t: t.T.contiguous().reshape(-1).float()
    tensors = dict(params=flat, ids=pm(ids), tgt=pm(tgt), w=pm(w), kinds=torch.tensor(kinds).float(), loss=loss.reshape(1), grad=p.grad,
                   b=torch.tensor([float(b)]), temp=torch.tensor([TEMP]), aux=torch.tensor([aux_w]))
    for k, t in tensors.items():
        t.detach().float().contiguous().numpy().tofile(out / f"{k}.f32")
    print(f"draw {draw} cond {cond:.1e} loss {loss.item():.6f} |grad| {p.grad.norm().item():.6f} written to", out)


if __name__ == "__main__":
    main()
