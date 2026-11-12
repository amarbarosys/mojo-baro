"""Torch (CPU, fp32) oracle for miniagi/bridge: synthetic tiny dataset + gradient check against the trainer.

  python -P tools/bridge_oracle.py gen DIR [seed] [real]  tiny (or real-dim synthetic) dataset in the P3 dump layout (cfg.txt sets the dims)
  python -P tools/bridge_oracle.py check CASE DATA [TOL] [SPREAD]  recompute loss/grads for CASE/batch.txt, compare CASE/grad.f32
Venv: the train oracle's (tools/train_oracle.py docstring), never the repo .venv. GEMM inputs are rounded to bf16 like
lops tgemm_k; attention, norms and losses stay f32. Exit 0 = every tensor rel-L2 <= TOL (default 1e-4).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

NAMES = "wq wk wv wo w1 w3 w2 wp1 wp2 wqa wka wva woa wkb wvb wc z0 esrc gm g1 g2 gz gan ga gb".split()
TAIL = 12
EPS = 1e-6
KEYS = "d nh ff nz vocab nsrc s1 s2 nhb nkvb hdb qlo R".split()
REAL = dict(d=2560, nh=20, ff=6912, nz=32, vocab=248320, nsrc=3, s1=4096, s2=2560, nhb=24, nkvb=2, hdb=256, qlo=64, R=4)
TINY = dict(d=64, nh=4, ff=128, nz=8, vocab=300, nsrc=3, s1=96, s2=64, nhb=4, nkvb=2, hdb=16, qlo=4, R=4)


def cfg_load(root):
    c = dict(REAL)
    p = Path(root) / "cfg.txt"
    if p.exists():
        for line in p.read_text().split("\n"):
            f = line.split()
            if len(f) >= 2 and f[0] in c:
                c[f[0]] = int(f[1])
    return c


def sizes(c):
    d, kbw, vbw = c["d"], c["nkvb"] * (c["hdb"] - c["qlo"]), c["nkvb"] * c["hdb"]
    return dict(zip(NAMES, [d * d, d * d, d * d, d * d, d * c["ff"], d * c["ff"], c["ff"] * d, c["s1"] * d, c["s2"] * d, d * d, d * d, d * d, d * d,
                            d * kbw, d * vbw, d * d, c["nz"] * d, 3 * d, d, d, d, d, d, d, c["nhb"]]))


def shapes(c):
    d, kbw, vbw = c["d"], c["nkvb"] * (c["hdb"] - c["qlo"]), c["nkvb"] * c["hdb"]
    return dict(wq=(d, d), wk=(d, d), wv=(d, d), wo=(d, d), w1=(d, c["ff"]), w3=(d, c["ff"]), w2=(c["ff"], d), wp1=(c["s1"], d), wp2=(c["s2"], d),
                wqa=(d, d), wka=(d, d), wva=(d, d), woa=(d, d), wkb=(d, kbw), wvb=(d, vbw), wc=(d, d), z0=(c["nz"], d), esrc=(3, d),
                gm=(d,), g1=(d,), g2=(d,), gz=(d,), gan=(d,), ga=(d,), gb=(c["nhb"],))


def f32(path, like=None):
    """Raw f32 file; a missing dump (partial P3 chain) reads as zeros shaped like `like`."""
    path = Path(path)
    if not path.exists() and like is not None:
        return np.zeros_like(like)
    return np.fromfile(path, dtype=np.float32)


class MM(torch.autograd.Function):
    """bf16-rounded operands on every GEMM, forward and backward, as lops tgemm_k does; f32 accumulate."""

    @staticmethod
    def forward(ctx, a, b):
        ab, bb = a.bfloat16().float(), b.bfloat16().float()
        ctx.save_for_backward(ab, bb)
        return ab @ bb

    @staticmethod
    def backward(ctx, g):
        ab, bb = ctx.saved_tensors
        gb = g.bfloat16().float()
        return gb @ bb.T, ab.T @ gb


def mm(a, b):
    return MM.apply(a, b)


SNAP = None  # check(): dumped act name -> [mojo forward or None, mojo grad or None, offset]
TIE = 1e-5  # a near-tie: |oracle - mojo| <= TIE * max|oracle| on an element whose bf16 rounding differs


def _tie(t, m):
    take = ((t - m).abs() <= TIE * t.abs().max()) & (t.bfloat16() != m.bfloat16())
    SNAP["_n"] += int(take.sum())
    return torch.where(take, m, t)


class Snap(torch.autograd.Function):
    """Take mojo's side of each bf16 rounding tie, forward and backward, so one f32 order flip does not
    propagate through later GEMMs. Real differences (above TIE) pass through untouched."""

    @staticmethod
    def forward(ctx, x, f, b):
        ctx.b = b
        return x.clone() if f is None else _tie(x, f)

    @staticmethod
    def backward(ctx, g):
        return (g if ctx.b is None else _tie(g, ctx.b)), None, None


def snap_mem(name, x, i):
    """Memory rows of item i. The dump holds them in the trainer's row-map order (with padding), so the rows are
    paired by content once, on m, and K, V and the grads reuse that pairing."""
    if SNAP is None or SNAP.get(name) is None or SNAP[name][0] is None and SNAP[name][1] is None:
        return x
    d = x.shape[-1]
    if name == "m":
        mo = torch.from_numpy(SNAP["m"][0].reshape(-1, d))
        dist = torch.cdist(x.detach(), mo)
        SNAP["_idx"][i] = dist.argmin(1)
        SNAP["_mem"] = max(SNAP["_mem"], float(dist.min(1).values.norm() / x.detach().norm()))
    ix = SNAP["_idx"][i]
    sl = lambda a: None if a is None else torch.from_numpy(a.reshape(-1, d))[ix].clone()
    return Snap.apply(x, sl(SNAP[name][0]), sl(SNAP[name][1]))


def snap(name, x):
    if SNAP is None or name not in SNAP:
        return x
    e = SNAP[name]
    o, n = e[2], x.numel()
    e[2] += n
    sl = lambda a: None if a is None else torch.from_numpy(a[o:o + n].reshape(x.shape).copy())
    return Snap.apply(x, sl(e[0]), sl(e[1]))


def rms(x, w):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * w


def attn(q, k, v, hd_k, hd_v, grp):
    """q [nq, nh*hdk], k [nk, nkv*hdk], v [nk, nkv*hdv] -> [nq, nh*hdv]; softmax over nk per head."""
    nq, nk = q.shape[0], k.shape[0]
    nh = q.shape[1] // hd_k
    nkv = k.shape[1] // hd_k
    outs = []
    for h in range(nh):
        kv = h // grp
        s = q[:, h * hd_k:(h + 1) * hd_k] @ k[:, kv * hd_k:(kv + 1) * hd_k].T * hd_k ** -0.5
        p = torch.softmax(s, -1)
        outs.append(p @ v[:, kv * hd_v:(kv + 1) * hd_v])
    return torch.cat(outs, -1)


class Data:
    def __init__(self, root, split, c, ident):
        root = Path(root)
        self.c = c
        rows = [l.split("\t") for l in (root / "data/items.tsv").read_text().split("\n") if l.count("\t") >= 8]
        self.items = {r[0]: r for r in rows}
        self.sel = [s for s in (root / f"data/sel-{split}.txt").read_text().split() if s]
        d = root / "dumps"
        s1n = "flash-student" if ident else "qwythos9b"
        s2n = "flash-student" if ident else "qwen35-4b"
        self.hs = f32(d / f"flash-student-{split}.hid").reshape(-1, c["d"])
        self.ht = f32(d / f"flash-teacher-{split}.hid", self.hs).reshape(-1, c["d"])
        self.as_ = f32(d / f"flash-student-{split}.att")
        self.at = f32(d / f"flash-teacher-{split}.att", self.as_)
        self.h1 = f32(d / f"{s1n}-{split}.hid", np.zeros(self.hs.shape[0] * c["s1"], np.float32)).reshape(-1, c["s1"])
        self.h2 = f32(d / f"{s2n}-{split}.hid", np.zeros(self.hs.shape[0] * c["s2"], np.float32)).reshape(-1, c["s2"])
        self.ct = f32(root / f"weights/ctarget-{split}.f32").reshape(-1, c["nz"], c["d"])
        self.idx = {}
        for name, per_att in (("flash-student", 1), ("flash-teacher", 1), (s1n, 0), (s2n, 0)):
            oh = oa = 0
            offs = []
            ip = d / f"{name}-{split}.idx"
            if not ip.exists():
                ip = d / f"flash-student-{split}.idx"
            for line in ip.read_text().split("\n"):
                f = line.split()
                if len(f) < 3:
                    continue
                offs.append((oh, oa))
                oh += int(f[1])
                oa += int(f[2])
            self.idx[name] = offs
        self.names = ("flash-student", "flash-teacher", s1n, s2n)

    def item(self, k):
        """k = position in sel order. Returns dict of per-item arrays (pre rows per source, answer rows, ...)."""
        c = self.c
        r = self.items[self.sel[k]]
        pre, post, ans = r[5].split(), r[7].split(), r[8].split()
        lp, lq, la = len(pre), len(post), len(ans)
        qw = c["nhb"] * c["hdb"]
        hs0, _ = self.idx[self.names[0]][k]
        ht0, _ = self.idx[self.names[1]][k]
        _, at0 = self.idx[self.names[0]][k]
        s10, _ = self.idx[self.names[2]][k]
        s20, _ = self.idx[self.names[3]][k]
        tail = TAIL - la
        att = self.as_[at0 * (qw + c["d"]):(at0 + TAIL) * (qw + c["d"])]
        attt = self.at[at0 * (qw + c["d"]):(at0 + TAIL) * (qw + c["d"])]
        return dict(lp=lp, la=la, x0=self.hs[hs0:hs0 + lp], x1=self.h1[s10:s10 + lp], x2=self.h2[s20:s20 + lp],
                    hA=self.hs[hs0 + lp + lq - 1:hs0 + lp + lq - 1 + la], hT=self.ht[ht0 + tail:ht0 + TAIL],
                    qB=att[:TAIL * qw].reshape(TAIL, qw)[tail:], aoS=att[TAIL * qw:].reshape(TAIL, c["d"])[tail:],
                    aoT=attt[TAIL * qw:].reshape(TAIL, c["d"])[tail:], ct=self.ct[k], tgt=np.array([int(a) for a in ans]))


def forward(c, P, items, head, wo47, flags):
    """Loss of one micro-batch with the trainer's definitions (per-item lists, sum over items where the trainer sums)."""
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    nsrc, ra, rb, rc, single, own, lam = flags["nsrc"], flags["ra"], flags["rb"], flags["rc"], flags["single"], flags["own"], flags["lam"]
    d, nz, hd, R = c["d"], c["nz"], c["d"] // c["nh"], c["R"]
    mb = len(items)
    zs = []
    for i, it in enumerate(items):
        z = P["z0"]
        if not own:
            srcs = [T(it["x0"])]
            if nsrc >= 2:
                srcs.append(mm(T(it["x1"]), P["wp1"]))
            if nsrc >= 3:
                srcs.append(mm(T(it["x2"]), P["wp2"]))
            m = snap_mem("m", torch.cat([rms(s + P["esrc"][j], P["gm"]) for j, s in enumerate(srcs)], 0), i)
            if single:
                m = torch.zeros_like(m)
            K = snap_mem("kk", mm(m, P["wk"]), i)
            V = snap_mem("vv", mm(m, P["wv"]), i)
        for r in range(R):
            n1 = rms(z, P["g1"])
            q = mm(n1, P["wq"])
            if own:
                K, V = mm(n1, P["wk"]), mm(n1, P["wv"])
            z = z + mm(attn(q, K, V, hd, hd, 1), P["wo"])
            n2 = rms(z, P["g2"])
            a, b = mm(n2, P["w1"]), mm(n2, P["w3"])
            z = z + mm(torch.nn.functional.silu(a) * b, P["w2"])
        zs.append(rms(snap("z", z), P["gz"]))
    out = dict(ce=0.0, msea=0.0, mseb=0.0, msec=0.0, cosc=0.0)
    na = sum(it["la"] for it in items)
    keep = dict(zn=snap("zn", torch.cat(zs)))
    keep["zn"].retain_grad()
    zs = list(keep["zn"].split(nz))
    if ra:
        hps, hts, ces, oas, uas, qas, kas, vas = [], [], [], [], [], [], [], []
        for it, zn in zip(items, zs):
            hA = T(it["hA"])
            hn = rms(hA, P["gan"])
            qa, ka, va = snap("qa", mm(hn, P["wqa"])), snap("ka", mm(zn, P["wka"])), snap("va", mm(zn, P["wva"]))
            o = snap("oa", attn(qa, ka, va, hd, hd, 1))
            u = snap("ua", mm(o, P["woa"]))
            hp = snap("hp", hA + P["ga"] * u)
            logits = mm(hp, head.T)
            ces.append(torch.nn.functional.cross_entropy(logits, T(it["tgt"]).long(), reduction="sum"))
            for t_ in (hp, o, u, qa, ka, va):
                t_.retain_grad()
            hps.append(hp)
            hts.append(T(it["hT"]))
            oas.append(o)
            uas.append(u)
            qas.append(qa)
            kas.append(ka)
            vas.append(va)
        hp, ht = torch.cat(hps), torch.cat(hts)
        keep.update(oa=oas, ua=uas, hp=hps, qa=qas, ka=kas, va=vas)
        out["ce"] = sum(ces) / na
        out["msea"] = ((hp - ht) ** 2).sum() / (ht ** 2).sum() if lam > 0 else 0.0
    if rb:
        qw, hdb, qlo, grp = c["nhb"] * c["hdb"], c["hdb"], c["qlo"], c["nhb"] // c["nkvb"]
        ys, aos, aot = [], [], []
        for it, zn in zip(items, zs):
            qB = T(it["qB"]).reshape(-1, c["nhb"], hdb)[:, :, qlo:].reshape(-1, c["nhb"] * (hdb - qlo))
            delta = attn(qB, mm(zn, P["wkb"]), mm(zn, P["wvb"]), hdb - qlo, hdb, grp)
            dg = (delta.reshape(-1, c["nhb"], hdb) * P["gb"][None, :, None]).reshape(-1, qw)
            ys.append(T(it["aoS"]) + mm(dg, wo47.T))
            aot.append(T(it["aoT"]))
        y, t = torch.cat(ys), torch.cat(aot)
        out["mseb"] = ((y - t) ** 2).sum() / (t ** 2).sum()
    if rc:
        cc = torch.cat([mm(zn, P["wc"]) for zn in zs])
        ct = torch.cat([T(it["ct"]) for it in items])
        out["msec"] = ((cc - ct) ** 2).sum() / (ct ** 2).sum()
        dot = (cc * ct).sum(-1)
        den = torch.sqrt((cc * cc).sum(-1) * (ct * ct).sum(-1)) + 1e-20
        out["cosc"] = (1 - dot / den).mean()
    la_ = out["ce"] + lam * out["msea"] if ra else 0.0
    lb_ = out["mseb"] if rb else 0.0
    lc_ = out["msec"] + out["cosc"] if rc else 0.0
    return flags["wa"] * la_ + flags["wb"] * lb_ + flags["wc"] * lc_, dict(out, loss_a=la_, loss_b=lb_, loss_c=lc_, keep=keep)


def gen(root, seed=1, real=False):
    """real: REAL dims with Lp 30..229 and La 3..11 (shape test only; 64/32/4 items)."""
    rng = np.random.default_rng(seed)
    c = dict(REAL if real else TINY)
    lp_lo, lp_hi, la_hi, counts = (30, 230, 12, (64, 32, 4)) if real else (8, 21, 5, (40, 16, 4))
    root = Path(root)
    for sub in ("data", "weights", "dumps"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "cfg.txt").write_text("".join(f"{k} {v}\n" for k, v in c.items()))
    d, qw = c["d"], c["nhb"] * c["hdb"]
    (root / "weights/head.f32").write_bytes((rng.standard_normal((c["vocab"], d)) * 0.05).astype(np.float32).tobytes())
    (root / "weights/wo47.f32").write_bytes((rng.standard_normal((d, qw)) * 0.05).astype(np.float32).tobytes())
    rows = []
    n = 0
    for split, cnt in zip(("train", "val", "test"), counts):
        fs_hid = open(root / f"dumps/flash-student-{split}.hid", "wb")
        ft_hid = open(root / f"dumps/flash-teacher-{split}.hid", "wb")
        fs_att = open(root / f"dumps/flash-student-{split}.att", "wb")
        ft_att = open(root / f"dumps/flash-teacher-{split}.att", "wb")
        s1 = open(root / f"dumps/qwythos9b-{split}.hid", "wb")
        s2 = open(root / f"dumps/qwen35-4b-{split}.hid", "wb")
        idx = {k: [] for k in ("flash-student", "flash-teacher", "qwythos9b", "qwen35-4b")}
        cts = []
        sel = []
        for k in range(cnt):
            lp, lq, la = int(rng.integers(lp_lo, lp_hi)), 5, int(rng.integers(1, la_hi))
            lr_ = int(rng.integers(3, 9))
            tok = lambda m: " ".join(str(int(x)) for x in rng.integers(0, c["vocab"], m))
            iid = f"t{n:05d}"
            n += 1
            rows.append("\t".join([iid, "synth", split, str(k), "0", tok(lp), tok(lr_), tok(lq), tok(la)]))
            sel.append(iid)
            T = lp + lq + la
            fs_hid.write((rng.standard_normal((T, d)) * 2).astype(np.float32).tobytes())
            ft_hid.write((rng.standard_normal((TAIL, d)) * 2).astype(np.float32).tobytes())
            for f in (fs_att, ft_att):
                f.write((rng.standard_normal((TAIL, qw))).astype(np.float32).tobytes())
                f.write((rng.standard_normal((TAIL, d))).astype(np.float32).tobytes())
            s1.write((rng.standard_normal((T, c["s1"])) * 3).astype(np.float32).tobytes())
            s2.write((rng.standard_normal((T, c["s2"])) * 2).astype(np.float32).tobytes())
            cts.append((rng.standard_normal((c["nz"], d)) * 0.1).astype(np.float32))
            idx["flash-student"].append(f"{iid} {T} {TAIL}")
            idx["flash-teacher"].append(f"{iid} {TAIL} {TAIL}")
            idx["qwythos9b"].append(f"{iid} {T} 0")
            idx["qwen35-4b"].append(f"{iid} {T} 0")
        for f in (fs_hid, ft_hid, fs_att, ft_att, s1, s2):
            f.close()
        for k, lines in idx.items():
            (root / f"dumps/{k}-{split}.idx").write_text("\n".join(lines) + "\n")
        (root / f"data/sel-{split}.txt").write_text("\n".join(sel) + "\n")
        (root / f"weights/ctarget-{split}.f32").write_bytes(np.stack(cts).tobytes())
    (root / "data/items.tsv").write_text("\n".join(rows) + "\n")
    print("written to", root)


def check(case, data, tol=1e-4, spread=0.0, snap_ties=True):
    """spread > 0: also recompute with params * (1 + spread * N(0,1)) and print torch-vs-torch rel L2 (the bf16 tie floor)."""
    case, c = Path(case), cfg_load(data)
    flags = {}
    for line in (case / "batch.txt").read_text().split("\n"):
        f = line.split()
        if len(f) >= 2:
            flags[f[0]] = f[1:] if f[0] == "items" else float(f[1])
    for k in ("nsrc", "ra", "rb", "rc", "single", "own", "ident"):
        flags[k] = int(flags[k])
    split = "val" if flags.get("split", 0.0) == 1.0 else "train"
    flags.setdefault("wa", 1.0)
    flags.setdefault("wb", 1.0)
    flags.setdefault("wc", 1.0)
    c["nsrc"] = flags["nsrc"]
    if flags["ident"]:
        c["s1"] = c["s2"] = c["d"]
    raw = f32(case / "params.f32")
    got = f32(case / "grad.f32")
    sz, sh = sizes(c), shapes(c)
    P, G, o = {}, {}, 0
    for n in NAMES:
        P[n] = torch.from_numpy(raw[o:o + sz[n]].reshape(sh[n]).copy()).requires_grad_(True)
        G[n] = got[o:o + sz[n]].reshape(sh[n])
        o += sz[n]
    assert o == len(raw) == len(got), (o, len(raw), len(got))
    ds = Data(data, split, c, flags["ident"])
    pos = {iid: k for k, iid in enumerate(ds.sel)}
    rows = [l.split("\t") for l in (Path(data) / "data/items.tsv").read_text().split("\n") if l.count("\t") >= 8]
    items = [ds.item(pos[rows[int(i)][0]]) for i in flags["items"]]
    head = torch.from_numpy(f32(Path(data) / "weights/head.f32").reshape(c["vocab"], c["d"]))
    wo47 = torch.from_numpy(f32(Path(data) / "weights/wo47.f32").reshape(c["d"], c["nhb"] * c["hdb"]))
    global SNAP
    rd = lambda k: f32(case / f"act-{k}.f32") if (case / f"act-{k}.f32").exists() else None
    SNAP = {k: [rd(k), rd("d" + k), 0] for k in ("z", "zn", "qa", "ka", "va", "oa", "ua", "hp", "m", "kk", "vv")} if snap_ties else None
    if SNAP is not None:
        SNAP.update(_n=0, _mem=0.0, _idx={})
    loss, parts = forward(c, P, items, head, wo47, flags)
    loss.backward()
    if SNAP is not None:
        print(f"bf16 ties taken from mojo: {SNAP['_n']}  memory rows rel_l2 vs mojo {SNAP['_mem']:.2e}")
    SNAP = None
    keep = parts.pop("keep")
    cat = lambda k: torch.cat(keep[k]) if k in keep else None
    catg = lambda k: torch.cat([t.grad for t in keep[k]]) if k in keep else None
    acts = dict(zn=keep["zn"], oa=cat("oa"), ua=cat("ua"), hp=cat("hp"), dhp=catg("hp"), dua=catg("ua"), doa=catg("oa"), dqa=catg("qa"),
                dka=catg("ka"), dva=catg("va"), dzn=keep["zn"].grad)
    for n, t in acts.items():
        f = case / f"act-{n}.f32"
        if t is None or not f.exists():
            continue
        t = t.detach().numpy().reshape(-1)
        m = f32(f)[:len(t)]
        nt = float(np.linalg.norm(t))
        flips = int((torch.from_numpy(m).bfloat16().view(torch.int16) != torch.from_numpy(t.copy()).bfloat16().view(torch.int16)).sum())
        print(f"act {n:4} rel_l2 {float(np.linalg.norm(m - t) / nt) if nt > 0 else float(np.linalg.norm(m)):10.3e} |t| {nt:.4e} maxabs {float(np.abs(m - t).max()):.3e} bf16_flips {flips}/{len(t)}")
    S = {}
    if spread > 0:
        g = torch.Generator().manual_seed(7)
        Q = {n: (P[n].detach() * (1 + spread * torch.randn(P[n].shape, generator=g))).requires_grad_(True) for n in NAMES}
        l2, _ = forward(c, Q, items, head, wo47, flags)
        l2.backward()
        S = {n: Q[n].grad.detach().numpy().reshape(-1) if Q[n].grad is not None else np.zeros(sz[n], np.float32) for n in NAMES}
    worst = 0.0
    bad = []
    print(f"{'tensor':6} {'rel_l2':>10} {'|oracle|':>12} {'|mojo|':>12}" + (f" {'torch-spread':>12}" if S else ""))
    for n in NAMES:
        t = P[n].grad.detach().numpy().reshape(-1) if P[n].grad is not None else np.zeros(sz[n], np.float32)
        m = G[n].reshape(-1)
        nt = float(np.linalg.norm(t))
        rel = float(np.linalg.norm(m - t) / nt) if nt > 0 else float(np.linalg.norm(m))
        worst = max(worst, rel)
        flag = "" if rel <= tol else "  <-- FAIL"
        if flag:
            bad.append(n)
        sp = f" {float(np.linalg.norm(S[n] - t) / nt) if nt > 0 else 0.0:12.3e}" if S else ""
        print(f"{n:6} {rel:10.3e} {nt:12.5e} {float(np.linalg.norm(m)):12.5e}{sp}{flag}")
    for k in ("loss_a", "loss_b", "loss_c", "ce", "msea", "mseb", "msec", "cosc"):
        mv = flags.get(k, float("nan"))
        ov = float(parts[k].detach()) if torch.is_tensor(parts[k]) else float(parts[k])
        print(f"{k:7} oracle {ov:.6e} mojo {mv:.6e} rel {abs(mv - ov) / max(abs(ov), 1e-12):.2e}")
    print(f"worst rel_l2 {worst:.3e} tol {tol:g} -> {'PASS' if not bad else 'FAIL ' + ' '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1, len(sys.argv) > 4 and sys.argv[4] == "real")
    elif sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4, float(sys.argv[5]) if len(sys.argv) > 5 else 0.0))
    else:
        raise SystemExit("usage: gen DIR [seed] | check CASE DATA [TOL]")
