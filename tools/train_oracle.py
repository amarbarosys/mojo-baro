"""Torch (CPU) oracle for miniagi/train/: writes inputs and expected outputs/grads per op case.

Usage: python -P tools/train_oracle.py cases OUTDIR   (-P: tools/profile.py shadows stdlib profile)
Env: its own venv, never the repo .venv (torch CPU has no cp314 wheel; a uv sync rebuilt .venv on 3.12
and broke every built binary's runpath, 2026-10-09):
  uv venv .work/oracle-venv --python 3.12
  uv pip install --python .work/oracle-venv/bin/python numpy torch --index-url https://download.pytorch.org/whl/cpu
Each case is OUTDIR/<case>/<tensor>.f32 (raw little-endian float32; int inputs stored as float).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

T, D, H, HD, V, EPS, THETA = 64, 96, 4, 24, 256, 1e-6, 10000.0


def rope(x, pos0=0):
    t = torch.arange(pos0, pos0 + x.shape[0], dtype=torch.float64)
    inv = THETA ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = torch.outer(t, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :].float()
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :].float()
    x1, x2 = x[..., : HD // 2], x[..., HD // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# Parity model; layout must match miniagi/train/model.mojo (offsets below, weights as x @ W with W [in, out]).
MV, MD, MH, MF, ML, MT = 256, 256, 4, 688, 2, 128
MHD = MD // MH
LAYER = 2 * MD + 4 * MD * MD + 3 * MD * MF


def layer_views(flat, l):
    b = MV * MD + l * LAYER
    take = lambda o, *shape: flat[b + o : b + o + torch.Size(shape).numel()].view(*shape)
    dd, df = MD * MD, MD * MF
    return dict(
        g1=take(0, MD), wq=take(MD, MD, MD), wk=take(MD + dd, MD, MD), wv=take(MD + 2 * dd, MD, MD),
        wo=take(MD + 3 * dd, MD, MD), g2=take(MD + 4 * dd, MD), w1=take(2 * MD + 4 * dd, MD, MF),
        w3=take(2 * MD + 4 * dd + df, MD, MF), w2=take(2 * MD + 4 * dd + 2 * df, MF, MD),
    )


def n_params():
    return MV * MD + ML * LAYER + MD


def init_flat(seed=1):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(n_params(), generator=g) * 0.02
    for l in range(ML):
        w = layer_views(flat, l)
        w["g1"].fill_(1.0)
        w["g2"].fill_(1.0)
        w["wo"].mul_(1 / (2 * ML) ** 0.5)
        w["w2"].mul_(1 / (2 * ML) ** 0.5)
    flat[-MD:].fill_(1.0)
    return flat


def rope_m(x):
    t = torch.arange(x.shape[0], dtype=torch.float64)
    inv = THETA ** (-torch.arange(0, MHD, 2, dtype=torch.float64) / MHD)
    ang = torch.outer(t, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :].float()
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :].float()
    x1, x2 = x[..., : MHD // 2], x[..., MHD // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


def model_loss(flat, tokens):
    e = flat[: MV * MD].view(MV, MD)
    x = e[tokens[:-1]]
    for l in range(ML):
        w = layer_views(flat, l)
        n = rms(x, w["g1"])
        q, k, v = (rope_m((n @ w[m]).view(-1, MH, MHD)) if m != "wv" else (n @ w[m]).view(-1, MH, MHD) for m in ("wq", "wk", "wv"))
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        x = x + o.transpose(0, 1).reshape(-1, MD) @ w["wo"]
        n = rms(x, w["g2"])
        x = x + (F.silu(n @ w["w1"]) * (n @ w["w3"])) @ w["w2"]
    z = rms(x, flat[-MD:]) @ e.T
    return F.cross_entropy(z, tokens[1:])


def model_case():
    flat = init_flat().requires_grad_()
    tokens = torch.randint(0, MV, (MT + 1,), generator=torch.Generator().manual_seed(2))
    loss = model_loss(flat, tokens)
    loss.backward()
    return dict(init=flat, tokens=tokens.float(), loss=loss.reshape(1), grad=flat.grad)


LR, B1, B2, AEPS, WD, STEPS = 3e-3, 0.9, 0.95, 1e-8, 0.1, 10


def adamw(p, grads, moment_dtype):
    """AdamW as torch.optim.AdamW computes it, moments stored in moment_dtype after every update."""
    m = torch.zeros_like(p, dtype=moment_dtype)
    v = torch.zeros_like(p, dtype=moment_dtype)
    for t, g in enumerate(grads, 1):
        p = p * (1 - LR * WD)
        m = (B1 * m.float() + (1 - B1) * g).to(moment_dtype)
        v = (B2 * v.float() + (1 - B2) * g * g).to(moment_dtype)
        p = p - LR * (m.float() / (1 - B1**t)) / ((v.float() / (1 - B2**t)).sqrt() + AEPS)
    return p


def adamw_case(g):
    p0 = torch.randn(1000, generator=g)
    grads = torch.randn(STEPS, 1000, generator=g) * torch.logspace(-6, 0, 1000)
    ref = p0.clone().requires_grad_()
    opt = torch.optim.AdamW([ref], lr=LR, betas=(B1, B2), eps=AEPS, weight_decay=WD)
    for gr in grads:
        ref.grad = gr.clone()
        opt.step()
    mine = adamw(p0, grads, torch.float32)
    assert torch.allclose(mine, ref.detach(), rtol=1e-6, atol=1e-7), "hand AdamW != torch.optim.AdamW"
    return dict(p0=p0, grads=grads, p=adamw(p0, grads, torch.bfloat16))


def cases():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    out = {}

    x, w, dy = r(T, D).requires_grad_(), r(D).requires_grad_(), r(T, D)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    y.backward(dy)
    out["rmsnorm"] = dict(x=x, w=w, dy=dy, y=y, dx=x.grad, dw=w.grad)

    a, b, dy = r(T, D).requires_grad_(), r(T, D).requires_grad_(), r(T, D)
    y = F.silu(a) * b
    y.backward(dy)
    out["swiglu"] = dict(a=a, b=b, dy=dy, y=y, da=a.grad, db=b.grad)

    x, dy = r(T, H, HD).requires_grad_(), r(T, H, HD)
    y = rope(x)
    y.backward(dy)
    out["rope"] = dict(x=x, dy=dy, y=y, dx=x.grad)

    e, dy = r(V, D).requires_grad_(), r(T, D)
    idx = torch.randint(0, 40, (T,), generator=g)
    y = e[idx]
    y.backward(dy)
    out["embed"] = dict(e=e, idx=idx.float(), dy=dy, y=y, de=e.grad)

    z, tgt = r(T, V).requires_grad_(), torch.randint(0, V, (T,), generator=g)
    loss = F.cross_entropy(z, tgt)
    loss.backward()
    out["ce"] = dict(z=z, tgt=tgt.float(), loss=loss.reshape(1), dz=z.grad)
    q, k, v, dy = r(T, H, HD).requires_grad_(), r(T, H, HD).requires_grad_(), r(T, H, HD).requires_grad_(), r(T, H, HD)
    y = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True).transpose(0, 1)
    y.backward(dy)
    out["attn"] = dict(q=q, k=k, v=v, dy=dy, y=y, dq=q.grad, dk=k.grad, dv=v.grad)
    out["model"] = model_case()
    out["adamw"] = adamw_case(g)
    return out


def train(init, data, steps, lr, out, t=512):
    """Same loop as miniagi/train/main.mojo: consecutive windows, AdamW with bf16 moments, wd 0.1."""
    global LR, MT
    LR, MT = lr, t
    flat = torch.from_file(init, size=n_params(), dtype=torch.float32).clone()
    data = torch.frombuffer(bytearray(Path(data).read_bytes()), dtype=torch.uint8).long()
    m = torch.zeros(n_params(), dtype=torch.bfloat16)
    v = torch.zeros(n_params(), dtype=torch.bfloat16)
    lines = []
    for s in range(steps):
        p = flat.clone().requires_grad_()
        loss = model_loss(p, data[s * t : s * t + t + 1])
        loss.backward()
        g, k = p.grad, s + 1
        flat = flat * (1 - LR * WD)
        m = (B1 * m.float() + (1 - B1) * g).bfloat16()
        v = (B2 * v.float() + (1 - B2) * g * g).bfloat16()
        flat = flat - LR * (m.float() / (1 - B1**k)) / ((v.float() / (1 - B2**k)).sqrt() + AEPS)
        lines.append(f"{s} {loss.item()!r}\n")
    Path(out).write_text("".join(lines))
    print("written to", out)


def noise(init, data, steps, lr, out, eps=1e-7):
    """Torch run from init perturbed by relative eps: the fp32 chaos floor any trainer is compared against."""
    a = torch.from_file(init, size=n_params(), dtype=torch.float32)
    b = a * (1 + eps * torch.randn(n_params(), generator=torch.Generator().manual_seed(3)))
    tmp = Path(out).with_suffix(".init.f32")
    b.numpy().tofile(tmp)
    train(str(tmp), data, steps, lr, out)


def main():
    if len(sys.argv) == 7 and sys.argv[1] == "noise":
        return noise(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    if len(sys.argv) in (7, 8) and sys.argv[1] == "train":
        t = int(sys.argv[7]) if len(sys.argv) == 8 else 512
        return train(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], t)
    if len(sys.argv) != 3 or sys.argv[1] != "cases":
        sys.exit("usage: train_oracle.py cases OUTDIR | train INIT DATA STEPS LR OUT [T] | noise INIT DATA STEPS LR OUT")
    root = Path(sys.argv[2])
    for name, tensors in cases().items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for k, t in tensors.items():
            t.detach().float().contiguous().numpy().tofile(d / f"{k}.f32")
    print("written to", root)


if __name__ == "__main__":
    main()
