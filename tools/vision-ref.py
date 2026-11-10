#!/usr/bin/env python3
"""vision-ref: numpy reference of the qwen3vl_merger vision encoder, checked stage by stage against
llama.cpp (tools/mtmd-oracle.cpp). It is the executable spec the Mojo encoder is built against.

  vision-ref.py image OUT.rgb [N]             write the deterministic N x N test image (default 768)
  vision-ref.py run MMPROJ.gguf IMAGE.rgb NX NY OUTDIR [--gelu tanh|erf]
                                              run the encoder, write every stage as <name>.f32
  vision-ref.py check REFDIR ORACLEDIR        compare stages, print max abs error and cosine per stage

Stages carry llama.cpp's graph names (patch_bias, inp_pos_emb, ln1-0, Qcur_rope-0, attn_out-0,
ffn_inp-0, ffn_out-0, layer_out-N, embd) so the check is a join on file names.
The bar is cos > 0.9999 through layer_out-13, 0.9995 at layer_out-26 (activations near 1e4 amplify
float noise) and 0.9998 on embd. The oracle must run with flash attention off (its default).
"""
import math
import os
import sys

import numpy as np


def test_image(n):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    r = (0.5 + 0.5 * np.sin(7 * x + 3 * y)) * 255
    g = (x * y) * 255
    b = (((x * 16).astype(int) + (y * 16).astype(int)) % 2) * 200 + ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.09) * 55
    return np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)


class Proj:
    def __init__(self, path):
        import gguf

        r = gguf.GGUFReader(path)
        self.kv = {}
        for k, f in r.fields.items():
            if len(f.data) == 1:
                v = f.parts[f.data[0]]
                self.kv[k] = bytes(v).decode() if f.types[0] == gguf.GGUFValueType.STRING else v[0]
            else:
                self.kv[k] = np.array([f.parts[i][0] for i in f.data])
        self.t = {}
        for t in r.tensors:
            a = np.array(t.data)
            if t.tensor_type.name == "BF16":
                a = bf16_to_f32(a.view(np.uint16))
            elif t.tensor_type.name == "F16":
                a = a.view(np.float16).astype(np.float32)
            self.t[t.name] = a.astype(np.float32).reshape([int(d) for d in reversed(t.shape)])

    def i(self, k):
        return int(self.kv["clip.vision." + k])


def bf16_to_f32(u16):
    u = u16.astype(np.uint32)
    sign, exp, man = (u >> 15) & 1, (u >> 7) & 0xFF, u & 0x7F
    out = (sign << 31) | (exp << 23) | (man << 16)
    return out.astype(np.uint32).view(np.float32)


def layernorm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu(x, kind):
    if kind == "erf":
        erf = np.vectorize(math.erf)
        return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)
    return (0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))).astype(np.float32)


def block_order(ph, pw, merge):
    """Token order: 2x2 merge blocks row by row, inside a block (dy, dx) row-major. Returns (ys, xs)."""
    ys, xs = [], []
    for y in range(0, ph, merge):
        for x in range(0, pw, merge):
            for dy in range(merge):
                for dx in range(merge):
                    ys.append(y + dy)
                    xs.append(x + dx)
    return np.array(ys), np.array(xs)


def vision_rope(q, ys, xs, d_head):
    """ggml GGML_ROPE_TYPE_VISION, n_dims = d_head/2, sections d_head/4 each: pair (i, i + d_head/2);
    i < d_head/4 turns with y, the rest with x, frequency index restarting at each section."""
    half, quarter = d_head // 2, d_head // 4
    scale = 10000.0 ** (-2.0 / half)
    freqs = scale ** np.arange(quarter, dtype=np.float64)
    theta = np.concatenate([ys[:, None] * freqs[None, :], xs[:, None] * freqs[None, :]], 1)  # [n, half]
    c, s = np.cos(theta)[:, None, :].astype(np.float32), np.sin(theta)[:, None, :].astype(np.float32)
    a, b = q[..., :half], q[..., half:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


def run(mmproj, img_path, nx, ny, out, gelu_kind):
    os.makedirs(out, exist_ok=True)
    p = Proj(mmproj)
    H, nh, L, patch, merge = p.i("embedding_length"), p.i("attention.head_count"), p.i("block_count"), p.i("patch_size"), p.i("spatial_merge_size")
    eps, dh = float(p.kv["clip.vision.attention.layer_norm_epsilon"]), H // nh
    mean, std = p.kv["clip.vision.image_mean"].astype(np.float32), p.kv["clip.vision.image_std"].astype(np.float32)
    img = np.fromfile(img_path, np.uint8).reshape(ny, nx, 3).astype(np.float32) / 255.0
    img = (img - mean) / std
    ph, pw = ny // patch, nx // patch
    ys, xs = block_order(ph, pw, merge)

    def save(name, a):
        np.ascontiguousarray(a, np.float32).tofile(f"{out}/{name}.f32")

    # patch embedding: two conv kernels (the temporal pair) summed on a still image
    w = p.t["v.patch_embd.weight"] + p.t["v.patch_embd.weight.1"]  # [H, 3, 16, 16]
    patches = img.reshape(ph, patch, pw, patch, 3).transpose(0, 2, 4, 1, 3).reshape(ph, pw, 3 * patch * patch)
    x = patches @ w.reshape(H, -1).T  # [ph, pw, H]
    x = x[ys, xs] + p.t["v.patch_embd.bias"]
    save("patch_bias", x)
    pos = p.t["v.position_embd.weight"]  # [2304, H], a 48 x 48 grid
    g = int(round(math.sqrt(pos.shape[0])))
    if (ph, pw) != (g, g):
        raise SystemExit(f"FAIL pos-embd: grid {ph}x{pw} needs the bilinear resize, only {g}x{g} is implemented")
    x = x + pos.reshape(g, g, H)[ys, xs]
    save("inp_pos_emb", x)

    n = x.shape[0]
    for il in range(L):
        t = lambda s: p.t[f"v.blk.{il}.{s}"]
        h = layernorm(x, t("ln1.weight"), t("ln1.bias"), eps)
        if il == 0:
            save("ln1-0", h)
        qkv = h @ t("attn_qkv.weight").T + t("attn_qkv.bias")
        q, k, v = [qkv[:, i * H : (i + 1) * H].reshape(n, nh, dh) for i in range(3)]
        q, k = vision_rope(q, ys, xs, dh), vision_rope(k, ys, xs, dh)
        if il == 0:
            save("Qcur_rope-0", q)
            save("Kcur_rope-0", k)
        att = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(dh)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(n, H) @ t("attn_out.weight").T + t("attn_out.bias")
        if il == 0:
            save("attn_out-0", o)
        x = x + o
        if il == 0:
            save("ffn_inp-0", x)
        h = layernorm(x, t("ln2.weight"), t("ln2.bias"), eps)
        f = gelu(h @ t("ffn_up.weight").T + t("ffn_up.bias"), gelu_kind) @ t("ffn_down.weight").T + t("ffn_down.bias")
        if il == 0:
            save("ffn_out-0", f)
        x = x + f
        if il in (0, 1, 13, 26):
            save(f"layer_out-{il}", x)
    x = layernorm(x, p.t["v.post_ln.weight"], p.t["v.post_ln.bias"], eps)
    m = x.reshape(n // (merge * merge), H * merge * merge)
    e = gelu(m @ p.t["mm.0.weight"].T + p.t["mm.0.bias"], gelu_kind) @ p.t["mm.2.weight"].T + p.t["mm.2.bias"]
    save("embd", e)
    print(f"ref: {n} patches, {e.shape[0]} image tokens of width {e.shape[1]}, gelu {gelu_kind}")


def check(ref, oracle):
    bad = 0
    names = sorted(f[:-4] for f in os.listdir(oracle) if f.endswith(".f32") and os.path.exists(f"{ref}/{f}"))
    order = ["patch_bias", "inp_pos_emb", "ln1-0", "Qcur_rope-0", "Kcur_rope-0", "attn_out-0", "ffn_inp-0", "ffn_out-0",
             "layer_out-0", "layer_out-1", "layer_out-13", "layer_out-26", "embd"]
    for nme in [o for o in order if o in names]:
        a, b = np.fromfile(f"{ref}/{nme}.f32", np.float32), np.fromfile(f"{oracle}/{nme}.f32", np.float32)
        if a.size != b.size:
            print(f"FAIL {nme}: size {a.size} vs {b.size}")
            bad += 1
            continue
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        err = float(np.abs(a - b).max())
        rel = err / float(np.abs(b).max())
        bar = 0.9995 if nme == "layer_out-26" else 0.9998 if nme == "embd" else 0.9999
        ok = cos > bar
        bad += 0 if ok else 1
        print(f"{'OK  ' if ok else 'FAIL'} {nme:14s} cos {cos:.6f} > {bar}  max|err| {err:.3e}  rel {rel:.2e}")
    if not names:
        print("FAIL check: no common stages")
        bad = 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["image"]:
        test_image(int(a[2]) if len(a) > 2 else 768).tofile(a[1])
    elif a[:1] == ["run"]:
        kind = a[a.index("--gelu") + 1] if "--gelu" in a else "tanh"
        run(a[1], a[2], int(a[3]), int(a[4]), a[5], kind)
    elif a[:1] == ["check"]:
        check(a[1], a[2])
    else:
        print(__doc__)
        sys.exit(1)
