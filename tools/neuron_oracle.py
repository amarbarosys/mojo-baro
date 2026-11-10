"""Torch (CPU, float64) gradient oracle for the neuron rows of miniagi/latent (nlook.mojo, lmodel.mojo ENT / NEU kinds).

Usage: .work/oracle-venv/bin/python -P tools/neuron_oracle.py OUTDIR
Case: b=3, kinds [ENT, TOK x4, NEU, PSE, NEU, NEU, TOK], memory m=128 rows of which mv=100 valid, aux weight 0.5, one
readout target -1 per hop. Layout after the latent parameters: [W_a D x EB][W_b EB x D], x @ W.
Loss = sum_rows w_row * CE_row + aw * sum_hops mean_i(-log a[mt]) over the hops that have a target.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F

import miniagi_latent_oracle as O

D, V, NP, PAUSE = O.D, O.V, O.NP, O.PAUSE
EB = 384
NA, NB = NP, NP + D * EB
NPT = NP + 2 * D * EB
TOK, PSE, ENT, NEU = 0, 2, 5, 6
LS = 4.0


def loss_of(flat, ids, tgt, w, kinds, ent, mt, K, Vm, mv, aw):
    b, S = ids.shape
    E = flat[: V * D].view(V, D)
    WA, WB = flat[NA:NA + D * EB].view(D, EB), flat[NB:NB + EB * D].view(EB, D)
    rows, j, h, aux = [], 0, 0, flat.new_zeros(())
    for r in range(S):
        if kinds[r] == TOK:
            rows.append(E[ids[:, r]])
        elif kinds[r] == PSE:
            rows.append(flat[PAUSE + j * D: PAUSE + (j + 1) * D].expand(b, D))
            j += 1
        elif kinds[r] == ENT:
            rows.append(ent @ WB)
        else:
            nf = O.hidden(flat, torch.stack(rows, 1))[:, -1]
            lg = LS * ((nf @ WA) @ K[:mv].T)
            a = torch.softmax(lg, -1)
            rows.append((a @ Vm[:mv]) @ WB)
            t = mt[h]
            ok = t >= 0
            if ok.any():
                aux = aux + aw * (-torch.log_softmax(lg, -1)[torch.arange(b), t.clamp(min=0)] * ok).sum() / b
            h += 1
    z = O.hidden(flat, torch.stack(rows, 1)) @ E.T
    ce = F.cross_entropy(z.reshape(-1, V), tgt.clamp(min=0).reshape(-1), reduction="none").view(b, S)
    return (ce * w * (tgt >= 0)).sum() + aux


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(int(os.environ.get("ORACLE_SEED", "7")))
    b, m, mv, aw = int(os.environ.get("ORACLE_B", "3")), 128, 100, 0.5
    kinds = [ENT, TOK, TOK, TOK, TOK, NEU, PSE, NEU, NEU, TOK]
    if os.environ.get("ORACLE_NONEU"):
        kinds = [TOK, TOK, TOK, TOK, TOK, TOK, PSE, TOK, TOK, TOK]
    S = len(kinds)
    flat = torch.randn(NPT, generator=g, dtype=torch.float64) * 0.05 * (256 / D) ** 0.5
    for l in range(O.L):
        w = O.views(flat, l)
        w["g1"].copy_(1 + 0.1 * torch.randn(D, generator=g, dtype=torch.float64))
        w["g2"].copy_(1 + 0.1 * torch.randn(D, generator=g, dtype=torch.float64))
    flat[O.GF:O.GF + D] = 1 + 0.1 * torch.randn(D, generator=g, dtype=torch.float64)
    ids = torch.randint(0, 21, (b, S), generator=g)
    tgt = torch.randint(0, 21, (b, S), generator=g)
    for r in range(S):
        if kinds[r] != TOK:
            ids[:, r] = -1
    tgt[:, :4] = -1
    tgt[:, 5] = -1
    w = (tgt >= 0).double() / (tgt >= 0).sum()
    ent = torch.nn.functional.normalize(torch.randn(b, EB, generator=g, dtype=torch.float64), dim=-1)
    K = torch.zeros(m, EB, dtype=torch.float64)
    Vm = torch.zeros(m, EB, dtype=torch.float64)
    K[:mv] = torch.nn.functional.normalize(torch.randn(mv, EB, generator=g, dtype=torch.float64), dim=-1)
    Vm[:mv] = torch.nn.functional.normalize(torch.randn(mv, EB, generator=g, dtype=torch.float64), dim=-1)
    mt = torch.randint(0, mv, (3, b), generator=g)
    mt[:, 1] = -1
    p = flat.clone().requires_grad_()
    loss = loss_of(p, ids, tgt, w, kinds, ent, mt, K, Vm, mv, aw)
    loss.backward()
    pm = lambda t: t.T.contiguous().reshape(-1).double()
    tensors = dict(params=flat, ids=pm(ids), tgt=pm(tgt), w=pm(w), kinds=torch.tensor(kinds).double(), loss=loss.reshape(1), grad=p.grad,
                   b=torch.tensor([float(b)]), ent=ent.reshape(-1), mt=mt.reshape(-1).double(), K=K.reshape(-1), V=Vm.reshape(-1),
                   meta=torch.tensor([float(m), float(mv), LS, aw]))
    for k, t in tensors.items():
        t.detach().float().contiguous().numpy().tofile(out / f"{k}.f32")
    sp = [("emb", 0, V * D), ("layers", V * D, O.GF), ("final+pause", O.GF, NP), ("W_a", NA, NB), ("W_b", NB, NPT)]
    print(f"loss {loss.item():.6f} |grad| {p.grad.norm().item():.6f}", " ".join(f"{n}:{p.grad[s:e].abs().max().item():.2e}" for n, s, e in sp), "written to", out)


if __name__ == "__main__":
    main()
