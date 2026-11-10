#!/usr/bin/env python3
"""tools/mtp-head/train.py: train a small draft head for a model that ships
none (exploration, no gate). Attention-free EAGLE-style head:
  f = Head(h_t, embed(tok_{t+1}))  ->  logits = f @ lm_head^T
h_t is the target's post-final-norm hidden state, embed and lm_head are the
target's own frozen tensors. Labels are the TARGET'S argmax at t+1 (what a
greedy verifier would accept), not the text. f also regresses h_{t+1} so the
head can be fed its own output for the 2nd and 3rd draft token.
Reports held-out agreement for draft depth 1, 2, 3 (depth k feeds f back in,
with the target's own tokens as the token input, i.e. given the chain so far
was accepted).
Usage: train.py DATA_DIR FROZEN_DIR OUT [EPOCHS=3] [WIDTH=4096]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda"


class Block(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.norm, self.up, self.gate, self.down = nn.RMSNorm(d), nn.Linear(d, w, bias=False), nn.Linear(d, w, bias=False), nn.Linear(w, d, bias=False)

    def forward(self, x):
        y = self.norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class Head(nn.Module):
    def __init__(self, d, w, blocks=2):
        super().__init__()
        self.nh, self.ne = nn.RMSNorm(d), nn.RMSNorm(d)
        self.fuse = nn.Linear(2 * d, d, bias=False)
        self.blocks = nn.ModuleList(Block(d, w) for _ in range(blocks))
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, h, e):
        x = self.fuse(torch.cat([self.nh(h), self.ne(e)], -1))
        for b in self.blocks:
            x = b(x)
        return h + self.out(x)          # predict the change from h_t to h_{t+1}


def load(data):
    hs, ids, keep = [], [], []
    for p in sorted(Path(data).glob("shard-*.npz")):
        z = np.load(p)
        h, i, st = z["h"], z["ids"], z["starts"]
        ok = np.ones(len(i), bool)
        for s in list(st[1:]) + [len(i)]:          # samples need t..t+4 inside one prompt
            ok[max(s - 4, 0):s] = False
        hs.append(h); ids.append(i); keep.append(ok)
    return np.concatenate(hs), np.concatenate(ids), np.concatenate(keep)


def main():
    data, frozen, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
    out.mkdir(parents=True, exist_ok=True)
    h_np, ids_np, keep = load(data)
    H = torch.from_numpy(h_np).to(dev)                       # f16 [T, D]
    ids = torch.from_numpy(ids_np.astype(np.int64)).to(dev)
    E = torch.from_numpy(np.load(frozen / "embed.npy")).to(dev)
    W = torch.from_numpy(np.load(frozen / "lm_head.npy")).to(dev)
    T, D = H.shape
    with torch.no_grad():                                    # target's own greedy token after position t
        tgt = torch.cat([(H[a:a + 8192].float() @ W.float().T).argmax(-1) for a in range(0, T, 8192)])
    text_match = (tgt[:-1] == ids[1:]).float().mean().item()
    idx = torch.from_numpy(np.nonzero(keep)[0]).to(dev)
    cut = int(len(idx) * 0.95)
    tr, ho = idx[:cut], idx[cut:]
    print(f"tokens {T} train {len(tr)} heldout {len(ho)} | target argmax equals text {text_match:.3f}", flush=True)

    head = Head(D, width).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    bs, steps = 2048, epochs * (len(tr) // 2048)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)

    def step_fn(t, train):
        h, feats, loss, accs = H[t].float(), None, 0.0, []
        for k in range(3 if not train else 2):               # depth 1..: feed own feature back
            e = E[ids[t + k + 1]].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = head(h, e)
                logits = f @ W.to(torch.bfloat16).T
            label = tgt[t + k + 1]
            if train:
                loss = loss + F.cross_entropy(logits.float(), label) + 0.1 * F.smooth_l1_loss(f.float(), H[t + k + 1].float())
            accs.append((logits.argmax(-1) == label).float().mean().item())
            h = f.float()
        return loss, accs

    t0 = time.time()
    for s in range(steps):
        t = tr[torch.randint(len(tr), (bs,), device=dev)]
        loss, accs = step_fn(t, True)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"step {s}/{steps} loss {loss.item():.3f} train d1 {accs[0]:.3f} d2 {accs[1]:.3f} {time.time() - t0:.0f}s", flush=True)
    head.eval()
    with torch.no_grad():
        accs = np.mean([step_fn(ho[a:a + bs], False)[1] for a in range(0, len(ho) - bs, bs)], 0)
    res = {"heldout_agreement_depth1": float(accs[0]), "depth2": float(accs[1]), "depth3": float(accs[2]),
           "params_M": sum(p.numel() for p in head.parameters()) / 1e6, "tokens": T, "epochs": epochs, "width": width}
    torch.save(head.state_dict(), out / "head.pt")
    (out / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL train: {error}", file=sys.stderr)
        sys.exit(1)
