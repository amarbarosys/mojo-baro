#!/usr/bin/env python3
"""C19 stage 1 (AMDHQ 06 "## C19:"): CacheGen's shipped codec on our KV, CPU oracle.

Reproduces the codec in github.com/UChi-JCL/CacheGen at 6bed34c,
LMCache/lmcache/storage_backend/serde/cachegen_{encoder,decoder}.py, as
run_cachegen.py drives it (QUANT_LEVEL 1 or 2, chunk_size = whole context):

  quantize  per (layer, token): max1 = amax|x| over the 1,024 channels (fp16,
            their KV is fp16), q = round(x * (MAX / max1) + MAX), MAX = bins//2-1
  cdf       per (layer, channel) over all tokens, process_batch semantics,
            int16 via _convert_to_int_and_normalize (16-bit, + arange)
  streams   per (layer, channel) per 256 tokens (CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK)
  dequant   (q - MAX) / MAX * max1, cast to fp16

No arithmetic coder runs here: each stream's length is the code length under
the integer CDF, sum of -log2(p), reported as ceil(bits/8) (lower) and
ceil(bits/8) + 4 (upper). Bins are mapped to our 8 attention layers by model
depth (3, 7, ..., 31), as frozen in the prereg.

usage: c19_cachegen.py F32_STATE INT8_STATE OUTDIR [--write-recon]
Writes OUTDIR/bytes.json and, with --write-recon, OUTDIR/<arm>.baro (BAROST01,
f32) per arm after a decode check (symbols and maxes written to disk, read
back, dequantized, compared bit for bit with the encoder's reconstruction).
"""
import json
import math
import pathlib
import sys

import numpy as np

NAT, NKVH, HD, KVPAGE = 8, 4, 256, 128
C = NKVH * HD
DEPTHS = [4 * a + 3 for a in range(NAT)]
STREAM_TOKENS = 256
# Mistral-7B CacheGenConfig (cachegen_basics.py), QUANT_LEVEL 1 and 2.
LEVELS = {
    "cg-L1": dict(kl=(10, 20), kb=(16, 16, 12), vl=2, vb=(16, 12)),
    "cg-L2": dict(kl=(10, 20), kb=(32, 16, 16), vl=2, vb=(32, 16)),
}


def bins_for(level):
    cfg = LEVELS[level]
    kb, vb = [], []
    for d in DEPTHS:
        kb.append(cfg["kb"][0] if d < cfg["kl"][0] else cfg["kb"][1] if d < cfg["kl"][1] else cfg["kb"][2])
        vb.append(cfg["vb"][0] if d < cfg["vl"] else cfg["vb"][1])
    return kb, vb


def read_state(path):
    raw = np.fromfile(path, dtype=np.uint8)
    magic = raw[:8].tobytes()
    pos, conv_n, ssm_n, kv_n = (int(x) for x in raw[8:40].view(np.int64))
    off = 40 + 32 + 4 * pos
    head = raw[:off]
    conv_ssm = raw[off:off + 4 * (conv_n + ssm_n)]
    off += 4 * (conv_n + ssm_n)
    return dict(raw=raw, magic=magic, pos=pos, conv_n=conv_n, ssm_n=ssm_n, kv_n=kv_n,
                head=head, conv_ssm=conv_ssm, kv_off=off)


def kv_tokens(flat, pos):
    pages = flat.size // (NAT * NKVH * KVPAGE * HD)
    x = flat.reshape(pages, NAT, NKVH, KVPAGE, HD).transpose(1, 0, 3, 2, 4)
    return x.reshape(NAT, pages * KVPAGE, C)[:, :pos, :]


def put_tokens(flat, recon):
    pages = flat.size // (NAT * NKVH * KVPAGE * HD)
    x = flat.reshape(pages, NAT, NKVH, KVPAGE, HD).transpose(1, 0, 3, 2, 4).reshape(NAT, pages * KVPAGE, C).copy()
    x[:, :recon.shape[1], :] = recon
    return x.reshape(NAT, pages, KVPAGE, NKVH, HD).transpose(1, 0, 3, 2, 4).reshape(-1)


def quantize(x16, bins):
    mx = (bins // 2 - 1)
    max1 = np.abs(x16).max(axis=-1, keepdims=True)
    if np.any(max1 == 0):
        raise SystemExit("FAIL quantize: a token vector is all zero (division by zero in their codec too)")
    q = np.rint(x16.astype(np.float32) * (np.float32(mx) / max1.astype(np.float32)) + np.float32(mx))
    return q.astype(np.int8), max1


def dequantize(q, bins, max1):
    mx = np.float32(bins // 2 - 1)
    return ((q.astype(np.float32) - mx) / mx * max1.astype(np.float32)).astype(np.float16)


def int_cdf(counts, n):
    lp = counts.shape[-1]
    cdf = np.cumsum(counts / n, axis=-1, dtype=np.float32)
    cdf = np.roll(cdf, 1, axis=-1)
    cdf[..., 0] = 0
    cdf = np.rint(cdf * np.float32(2 ** 16 - (lp - 1))).astype(np.int64) + np.arange(lp)
    return cdf


def stream_bits(q, lp):
    t, c = q.shape
    idx = (np.arange(c, dtype=np.int64)[None, :] * lp + q.astype(np.int64)).ravel()
    counts = np.bincount(idx, minlength=c * lp).reshape(c, lp)
    cdf = int_cdf(counts.astype(np.float32), t)
    upper = np.concatenate([cdf[:, 1:], np.full((c, 1), 2 ** 16, dtype=np.int64)], axis=1)
    pmf = upper - cdf
    if np.any(pmf[counts > 0] <= 0):
        raise SystemExit("FAIL cdf: a used symbol has zero coded probability")
    bits_tab = -np.log2(pmf / 2 ** 16)
    per_sym = bits_tab[np.arange(c)[None, :], q.astype(np.int64)]
    nst = math.ceil(t / STREAM_TOKENS)
    pad = nst * STREAM_TOKENS - t
    per_sym = np.pad(per_sym, ((0, pad), (0, 0)))
    bits = per_sym.reshape(nst, STREAM_TOKENS, c).sum(axis=1)
    return bits, nst


def encode_arm(level, k16, v16, pos):
    kb, vb = bins_for(level)
    lp_k, lp_v = max(kb) + 1, max(vb) + 1
    out = dict(level=level, k_bins=kb, v_bins=vb, streams_lo=0, streams_hi=0, nstreams=0)
    qs, maxes, recon = [], [], []
    for kind, x16, bl, lp in (("K", k16, kb, lp_k), ("V", v16, vb, lp_v)):
        for a in range(NAT):
            q, m = quantize(x16[a], bl[a])
            if q.min() < 0 or q.max() >= lp:
                raise SystemExit(f"FAIL quantize: {kind}{a} symbol out of range [{q.min()}, {q.max()}]")
            bits, nst = stream_bits(q, lp)
            out["streams_lo"] += int(np.ceil(bits / 8).sum())
            out["streams_hi"] += int(np.ceil(bits / 8).sum()) + 4 * bits.size
            out["nstreams"] += bits.size
            qs.append(q)
            maxes.append(m)
            recon.append(dequantize(q, bl[a], m))
    nst = math.ceil(pos / STREAM_TOKENS)
    out["maxes"] = 2 * NAT * pos * 2
    out["cdf"] = NAT * C * (lp_k + lp_v) * 2
    out["lengths"] = 2 * NAT * C * nst * 4
    out["kv_lo"] = out["streams_lo"] + out["maxes"] + out["cdf"] + out["lengths"]
    out["kv_hi"] = out["streams_hi"] + out["maxes"] + out["cdf"] + out["lengths"]
    return out, qs, maxes, recon


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_recon = "--write-recon" in sys.argv
    if len(args) != 3:
        raise SystemExit(__doc__)
    f32_path, int8_path, outdir = args
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    st = read_state(f32_path)
    if st["magic"] != b"BAROST01":
        raise SystemExit(f"FAIL input: {f32_path} is {st['magic']!r}, want BAROST01")
    i8 = read_state(int8_path)
    if i8["magic"] != b"BAROST02" or i8["pos"] != st["pos"] or not np.array_equal(i8["head"][8:], st["head"][8:]):
        raise SystemExit("FAIL input: int8 file is not the same prefix as the f32 file")
    pos, kv_n = st["pos"], st["kv_n"]
    kflat = st["raw"][st["kv_off"]:st["kv_off"] + 4 * kv_n].view(np.float32)
    vflat = st["raw"][st["kv_off"] + 4 * kv_n:st["kv_off"] + 8 * kv_n].view(np.float32)
    k32, v32 = kv_tokens(kflat, pos), kv_tokens(vflat, pos)
    k16, v16 = k32.astype(np.float16), v32.astype(np.float16)
    fixed = int(st["head"].size + st["conv_ssm"].size)
    i8_kv = int(i8["raw"].size) - fixed
    report = dict(pos=pos, f32_file=int(st["raw"].size), int8_file=int(i8["raw"].size), fixed_bytes=fixed,
                  int8_kv=i8_kv, arms={})
    for level in LEVELS:
        out, qs, maxes, recon = encode_arm(level, k16, v16, pos)
        out["payload_lo"] = fixed + out["kv_lo"]
        out["payload_hi"] = fixed + out["kv_hi"]
        out["kv_ratio_vs_int8_lo"] = i8_kv / out["kv_hi"]
        out["kv_ratio_vs_int8_hi"] = i8_kv / out["kv_lo"]
        out["payload_ratio_vs_int8_lo"] = report["int8_file"] / out["payload_hi"]
        out["payload_ratio_vs_int8_hi"] = report["int8_file"] / out["payload_lo"]
        ref = np.concatenate([k32, v32]).astype(np.float64)
        rec = np.stack(recon).astype(np.float64)
        err = []
        for li in range(2 * NAT):
            e = rec[li] - ref[li]
            err.append(float(np.sqrt((e * e).mean() / (ref[li] * ref[li]).mean())))
        out["rel_rms_per_layer_KthenV"] = err
        if write_recon:
            np.savez(outdir / f"{level}-symbols.npz", q=np.stack(qs), m=np.stack(maxes))
            back = np.load(outdir / f"{level}-symbols.npz")
            kb, vb = bins_for(level)
            bl = kb + vb
            for li in range(2 * NAT):
                if not np.array_equal(dequantize(back["q"][li], bl[li], back["m"][li]).view(np.uint16), recon[li].view(np.uint16)):
                    raise SystemExit(f"FAIL decode check: {level} layer {li} differs after the round trip")
            out["decode_check"] = "PASS: symbols + maxes read back, dequantized bit-equal"
            kn = put_tokens(kflat, np.stack(recon[:NAT]).astype(np.float32))
            vn = put_tokens(vflat, np.stack(recon[NAT:]).astype(np.float32))
            path = outdir / f"{level}.baro"
            with open(path, "wb") as f:
                f.write(st["head"].tobytes())
                f.write(st["conv_ssm"].tobytes())
                f.write(kn.astype(np.float32).tobytes())
                f.write(vn.astype(np.float32).tobytes())
            if path.stat().st_size != st["raw"].size:
                raise SystemExit(f"FAIL recon: {path} is {path.stat().st_size} bytes, source {st['raw'].size}")
            out["recon_file"] = str(path)
        report["arms"][level] = out
        print(f"{level} pos={pos} payload {out['payload_lo']}..{out['payload_hi']} B vs int8 {report['int8_file']} B"
              f" (x{out['payload_ratio_vs_int8_lo']:.2f}..x{out['payload_ratio_vs_int8_hi']:.2f})"
              f" kv x{out['kv_ratio_vs_int8_lo']:.2f}..x{out['kv_ratio_vs_int8_hi']:.2f}"
              f" rel_rms max {max(out['rel_rms_per_layer_KthenV']):.4f}")
    (outdir / "bytes.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PASS c19 stage 1: {outdir / 'bytes.json'}")


if __name__ == "__main__":
    main()
