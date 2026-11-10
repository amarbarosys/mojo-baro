#!/usr/bin/env python3
"""Compare two engine state files section by section (oracle).

Usage: tools/state_diff.py [--moe] [--kvg N] REFERENCE.state OTHER.state

Reads BAROST01 (f32 KV) and BAROST03 (the int8 KV pool verbatim: per page, layer and KV head a
128 x HD int8 block plus a 512-byte tail of one f32 scale per token row), dequantised to f32. --moe
uses the qwen35moe KV shape (10 attention layers, 2 KV heads), default the dense one (8, 4). --kvg N
reads an engine built with -D BARO_KVG=N (one scale per N values, HD/N scales per row; default 256). K and V
also print one row per attention layer (rel L2, max abs), so a quantisation error that lives in one
layer stands out from the mean, beside "quant-only": the reference K/V quantised the int8 pool's way
(per token row, amax / 127, round) and compared with itself. quant-only is the step's own error;
the rest of rel L2 is error carried in from earlier layers (the two engines' hidden states differ).
Norms are summed in f64: f32 sums over ~83M squares undercounted the K norm by 11% (fixed 2026-10-11).

Prints, for conv, SSM state, K and V (positions below pos only), the relative L2
error ||other - ref|| / ||ref|| and the max absolute difference. Used by
bench/llama-handoff.sh to check a llama.cpp-converted state against the engine's
own state at the same position: a layout error (transpose, head or layer order)
shows up as an O(1) relative error, rounding and kernel differences as small ones.
Exit 1 if the headers (pos, sizes, tokens) disagree.
"""
import sys

import numpy as np

KVPAGE, HD = 128, 256
N_ATT, NKVH = 8, 4
KVG = 256


def deq_int8(raw):
    g = HD // KVG
    blk = raw.reshape(-1, KVPAGE * HD + KVPAGE * 4 * g)
    q = blk[:, :KVPAGE * HD].view(np.int8).reshape(-1, KVPAGE, g, KVG).astype(np.float32)
    sc = blk[:, KVPAGE * HD:].copy().view("<f4").reshape(-1, KVPAGE, g, 1)
    return (q * sc).reshape(-1)


def requant(x):
    r = x.reshape(-1, KVG).astype(np.float32); sc = np.abs(r).max(1, keepdims=True) / 127
    sc[sc == 0] = 1
    return (np.clip(np.round(r / sc), -127, 127) * sc).reshape(-1)


def load(path):
    b = np.memmap(path, dtype=np.uint8, mode="r")
    magic = bytes(b[:8]); assert magic in (b"BAROST01", b"BAROST03"), (path, magic)
    pos, conv_n, ssm_n, kvn = np.frombuffer(b[8:40], dtype="<i8")
    salt = bytes(b[40:72])
    o = 72
    toks = np.frombuffer(b[o:o + 4 * pos], dtype="<i4"); o += 4 * pos
    f = np.frombuffer(b[o:o + 4 * (conv_n + ssm_n)], dtype="<f4"); o += 4 * (conv_n + ssm_n)
    conv = f[:conv_n]; ssm = f[conv_n:]
    if magic == b"BAROST01":
        kv = np.frombuffer(b[o:o + 8 * kvn], dtype="<f4"); k, v = kv[:kvn], kv[kvn:]
    else:
        k, v = deq_int8(b[o:o + kvn]), deq_int8(b[o + kvn:o + 2 * kvn])
    return dict(pos=int(pos), salt=salt, toks=toks, conv=conv, ssm=ssm, k=k, v=v, magic=magic.decode())


def kv_pages(x):
    return x.reshape(-1, N_ATT, NKVH, KVPAGE, HD)


def kv_valid(x, pos, layer=None):
    pages = kv_pages(x)
    if layer is not None: pages = pages[:, layer:layer + 1]
    t = np.arange(pages.shape[0] * KVPAGE).reshape(-1, KVPAGE)
    mask = (t < pos)[:, None, None, :, None]
    return pages[np.broadcast_to(mask, pages.shape)]


def main():
    global N_ATT, NKVH, KVG
    args = sys.argv[1:]
    if args[:1] == ["--moe"]: N_ATT, NKVH = 10, 2; args = args[1:]
    if args[:1] == ["--kvg"]: KVG = int(args[1]); args = args[2:]
    if len(args) != 2:
        sys.exit(__doc__)
    a, b = load(args[0]), load(args[1])
    ok = a["pos"] == b["pos"] and a["conv"].size == b["conv"].size and a["ssm"].size == b["ssm"].size \
        and a["k"].size == b["k"].size and np.array_equal(a["toks"], b["toks"]) and a["salt"] == b["salt"]
    print(f"{a['magic']} vs {b['magic']}; pos {a['pos']} vs {b['pos']}; tokens equal {np.array_equal(a['toks'], b['toks'])}; salt equal {a['salt'] == b['salt']}")
    if not ok:
        print("FAIL: headers disagree")
        sys.exit(1)
    for name in ("conv", "ssm", "k", "v"):
        x, y = a[name], b[name]
        if name in ("k", "v"):
            x, y = kv_valid(x, a["pos"]), kv_valid(y, a["pos"])
        x, y = x.astype(np.float64), y.astype(np.float64)
        rel = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
        print(f"{name:>4}: rel L2 {rel:.3e}  max abs {float(np.max(np.abs(y - x))):.3e}  ref norm {float(np.linalg.norm(x)):.3e}")
    for name in ("k", "v"):
        for li in range(N_ATT):
            x, y = kv_valid(a[name], a["pos"], li), kv_valid(b[name], a["pos"], li)
            q = requant(x).astype(np.float64); x, y = x.astype(np.float64), y.astype(np.float64)
            nx = max(np.linalg.norm(x), 1e-30); rel = float(np.linalg.norm(y - x) / nx)
            qo = float(np.linalg.norm(q - x) / nx)
            print(f"  {name} layer {li}: rel L2 {rel:.3e}  quant-only {qo:.3e}  max abs {float(np.max(np.abs(y - x))):.3e}  ref norm {float(np.linalg.norm(x)):.3e}")


if __name__ == "__main__":
    main()
