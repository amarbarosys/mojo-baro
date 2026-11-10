#!/usr/bin/env python3
"""P4b oracle: the exact model. One teacher-forced forward of Ornith-1.5-9B (qwen35: 24 gated-delta layers + 8 attention layers, dense SwiGLU
FFN, 32 layers, the MTP layer 32 ignored) in numpy float64 over prompt + forced ids, weights = the GGUF's own quantized values dequantized by
llama.cpp's gguf-py (read by path, read only) one layer at a time. For every forced position it writes the argmax id, the top-1/top-2 logit
margin and the rank of the forced id: the reference that agreement(ours, exact) and agreement(llama, exact) are scored against.

  exact-ref.py --gguf G --unit UNIT.ref.json --out OUT.json [--emb exact|bf16] [--layers N] [--max-forced K]

UNIT.ref.json: {"prompt_ids": [...], "ref_ids": [...]} (a P4 unit, or any file of that shape). Layer maths = tools/model-ref.py (matches the
engine), batched over tokens and in float64 without its bf16 rounding emulation (`rne`). --emb bf16 rounds the embedding rows to bf16 first,
which is what the engine pack does to token_embd (Q4_K in the GGUF, bf16 in the pack): a variant, not the exact model.
CPU only, run through `gpu-wait run --room cpu` (about 10 to 15 min per 4K-token unit, RAM about 8 GB). Never a gate verdict by itself.
"""
import argparse, json, os, sys, time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/Projects/imports/llama.cpp/gguf-py"))
from gguf import GGUFReader, quants  # noqa: E402

H, FFN, CONV, KDIM, NH_K, NH_V, S, HD, NQH, NKVH, NROT, EPS, NL = 4096, 12288, 8192, 2048, 16, 32, 128, 256, 16, 4, 64, 1e-6, 32


class Model:
    def __init__(self, path):
        self.t = {x.name: x for x in GGUFReader(path, "r").tensors}

    def w(self, name, rows=None):
        """float64 weight, numpy shape (out, in) as the GGUF stores it (y = x @ w.T); 1-D for norms and biases."""
        t = self.t[name]
        data = t.data if rows is None else t.data[rows]
        if t.tensor_type.name in ("F32", "F16"):
            return np.asarray(data, dtype=np.float64)
        return quants.dequantize(data, t.tensor_type).astype(np.float64)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def bf16(x):
    u = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32).astype(np.float64)


def rope(x, pos):
    """YaRN rope on the first NROT dims (halves rotated against each other), as tools/model-ref.py yarn_rope; x: [T, heads, HD], pos: [T]."""
    base, scale, nctx = 1e7, 0.25, 262144
    corr = lambda beta: NROT * np.log(nctx / (beta * 2 * np.pi)) / (2 * np.log(base))  # noqa: E731
    low, high = np.clip(np.floor(corr(32.0)), 0, NROT - 1), np.clip(np.ceil(corr(1.0)), 0, NROT - 1)
    j = np.arange(NROT // 2, dtype=np.float64)
    te = pos[:, None] * base ** (-2.0 * j / NROT)
    ramp = np.clip((j - low) / max(high - low, 0.001), 0, 1)
    th = scale * te * (1 - ramp) + te * ramp
    m = 1 + 0.1 * np.log(1 / scale)
    c, s = (np.cos(th) * m)[:, None, :], (np.sin(th) * m)[:, None, :]
    y = x.copy()
    x0, x1 = x[..., : NROT // 2], x[..., NROT // 2 : NROT]
    y[..., : NROT // 2] = x0 * c - x1 * s
    y[..., NROT // 2 : NROT] = x0 * s + x1 * c
    return y


def ssm_layer(M, i, x):
    b, T = f"blk.{i}.", x.shape[0]
    cur = rms(x, M.w(b + "attn_norm.weight"))
    qkv = cur @ M.w(b + "attn_qkv.weight").T
    z = cur @ M.w(b + "attn_gate.weight").T
    beta = 1.0 / (1.0 + np.exp(-(cur @ M.w(b + "ssm_beta.weight").T)))
    g = np.log1p(np.exp(cur @ M.w(b + "ssm_alpha.weight").T + M.w(b + "ssm_dt.bias"))) * M.w(b + "ssm_a")
    cw = M.w(b + "ssm_conv1d.weight")  # (CONV, 4): tap 0 = oldest of the 4-token window, tap 3 = current
    pad = np.concatenate([np.zeros((3, CONV)), qkv], axis=0)
    conv = silu(sum(cw[:, k] * pad[k : k + T] for k in range(4)))
    q, k, v = conv[:, :KDIM].reshape(T, NH_K, S), conv[:, KDIM : 2 * KDIM].reshape(T, NH_K, S), conv[:, 2 * KDIM :].reshape(T, NH_V, S)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + EPS)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + EPS)
    q, k = np.tile(q, (1, 2, 1)) / np.sqrt(S), np.tile(k, (1, 2, 1))  # 16 -> 32 heads: whole block repeated, as the engine
    o, st = np.zeros((T, NH_V, S)), np.zeros((NH_V, S, S))
    for t in range(T):  # the gated delta rule: sequential in t, batched over the 32 heads
        st = st * np.exp(g[t])[:, None, None]
        d = (v[t] - np.einsum("hij,hi->hj", st, k[t])) * beta[t][:, None]
        st = st + np.einsum("hi,hj->hij", k[t], d)
        o[t] = np.einsum("hij,hi->hj", st, q[t])
    gated = (rms(o, M.w(b + "ssm_norm.weight")) * silu(z.reshape(T, NH_V, S))).reshape(T, H)
    return x + gated @ M.w(b + "ssm_out.weight").T


def attn_layer(M, i, x):
    b, T = f"blk.{i}.", x.shape[0]
    cur = rms(x, M.w(b + "attn_norm.weight"))
    qf = (cur @ M.w(b + "attn_q.weight").T).reshape(T, NQH, 2 * HD)
    q, gate = qf[..., :HD], qf[..., HD:].reshape(T, H)
    k = (cur @ M.w(b + "attn_k.weight").T).reshape(T, NKVH, HD)
    v = (cur @ M.w(b + "attn_v.weight").T).reshape(T, NKVH, HD)
    pos = np.arange(T, dtype=np.float64)
    q, k = rope(rms(q, M.w(b + "attn_q_norm.weight")), pos), rope(rms(k, M.w(b + "attn_k_norm.weight")), pos)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    o = np.zeros((T, NQH, HD))
    for h in range(NQH):
        sc = (q[:, h] @ k[:, h // 4].T) / np.sqrt(HD) + mask
        p = np.exp(sc - sc.max(axis=1, keepdims=True))
        o[:, h] = (p / p.sum(axis=1, keepdims=True)) @ v[:, h // 4]
    att = o.reshape(T, H) / (1.0 + np.exp(-gate))
    return x + att @ M.w(b + "attn_output.weight").T


def forward(M, ids, emb, layers):
    ids = np.asarray(ids)
    x = M.w("token_embd.weight", rows=ids)
    if emb == "bf16":
        x = bf16(x)
    for i in range(layers):
        t0 = time.time()
        x = attn_layer(M, i, x) if (i + 1) % 4 == 0 else ssm_layer(M, i, x)
        cur = rms(x, M.w(f"blk.{i}.post_attention_norm.weight"))
        act = silu(cur @ M.w(f"blk.{i}.ffn_gate.weight").T) * (cur @ M.w(f"blk.{i}.ffn_up.weight").T)
        x = x + act @ M.w(f"blk.{i}.ffn_down.weight").T
        print(f"layer {i} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return rms(x, M.w("output_norm.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", choices=["exact", "bf16"], default="exact")
    ap.add_argument("--layers", type=int, default=NL)
    ap.add_argument("--max-forced", type=int, default=0)
    a = ap.parse_args()
    u = json.load(open(a.unit))
    prompt, forced = u["prompt_ids"], u["ref_ids"][: a.max_forced or None]
    M = Model(a.gguf)
    xn = forward(M, prompt + forced[:-1], a.emb, a.layers)[len(prompt) - 1 :]  # position P-1+i predicts forced[i]
    out = []
    wo = M.t["output.weight"]
    logits = np.zeros((xn.shape[0], wo.data.shape[0]))
    for r0 in range(0, wo.data.shape[0], 8192):  # the head in vocabulary slices
        logits[:, r0 : r0 + 8192] = xn @ quants.dequantize(wo.data[r0 : r0 + 8192], wo.tensor_type).astype(np.float64).T
    for i, f in enumerate(forced):
        order = np.argsort(-logits[i])[:2]
        out.append({"i": i, "forced": int(f), "argmax": int(order[0]), "margin": float(logits[i, order[0]] - logits[i, order[1]]),
                    "forced_rank": int((logits[i] > logits[i, f]).sum())})
    agree = sum(o["argmax"] == o["forced"] for o in out)
    json.dump({"emb": a.emb, "layers": a.layers, "n": len(out), "agree": agree, "positions": out}, open(a.out, "w"))
    print(f"exact-ref: {len(out)} forced positions, exact argmax equals the forced id at {agree} ({100.0 * agree / max(1, len(out)):.2f}%), emb={a.emb}, layers={a.layers}")


if __name__ == "__main__":
    main()
