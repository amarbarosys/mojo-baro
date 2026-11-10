#!/usr/bin/env python3
"""Build the spark2_5 engine pack from a Q8_0 GGUF (llama-quantize --allow-requantize).

Same pack format as tools/engine-pack.py --q8: q8 = int8 [out, in] followed by fp16
block-32 scales [out, in/32], converted 1:1 from the GGUF's own Q8_0 blocks (no
re-rounding, so the engine and llama.cpp run identical integers and scales).
token_embd is stored dequantised f32 (exact, as llama.cpp's get_rows does) for the
lookup and again as q8 under "output.weight" for the tied LM head.

Order per layer: attn_norm f32, attn_qkv q8 [6144,2560], attn_gate q8 [16,2560],
attn_output q8 [2560,4096], ffn_norm f32, ffn_gate q8, ffn_up q8, ffn_down q8.

K-quant input (Q4_K / Q5_K / Q6_K, e.g. the vendor Q4_K_M): the current llama-quantize refuses arch spark2_5
(2026-10-12), so such tensors are dequantised with gguf-py here; token_embd stays exact f32 and every matrix is
requantised to Q8_0 blocks (d = amax / 127), the same arithmetic as llama-quantize --allow-requantize up to its
rounding of exact ties.

Usage: tools/spark-pack.py MODEL.gguf OUTDIR      (Q8_0 or K-quant)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

N_LAYERS = 36


def q8_split(t, raw=None):
    raw = np.asarray(t.data).reshape(-1) if raw is None else raw.reshape(-1)
    ne0 = int(t.shape[0])
    rows = raw.size // (ne0 // 32 * 34)
    blk = raw.reshape(rows, ne0 // 32, 34)
    d = blk[:, :, :2].copy().view(np.float16).reshape(rows, ne0 // 32)
    qs = blk[:, :, 2:].copy().view(np.int8).reshape(rows, ne0)
    return qs, d


def main():
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(src)
    T = {t.name: t for t in r.tensors}
    order = ["token_embd.weight"]
    for i in range(N_LAYERS):
        b = f"blk.{i}."
        order += [b + n for n in ("attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                                   "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                                   "ffn_up.weight", "ffn_down.weight")]
    order += ["output_norm.weight", "output.weight"]
    lines, off = [], 0
    with open(outdir / "pack.bin", "wb") as out:
        for name in order:
            t = T["token_embd.weight" if name == "output.weight" else name]
            tt = t.tensor_type.name
            if tt == "F32":
                raw = np.asarray(t.data).astype(np.float32).tobytes()
                n_elem = raw.__len__() // 4
                dt = "f32"
            elif tt == "Q8_0":
                qs, d = q8_split(t)
                n_elem = qs.size
                if name == "token_embd.weight":
                    x = (qs.astype(np.float32).reshape(qs.shape[0], -1, 32) * d.astype(np.float32)[:, :, None])
                    raw = x.reshape(qs.shape[0], -1).astype(np.float32).tobytes()
                    dt = "f32"
                else:
                    raw = qs.tobytes() + d.tobytes()
                    dt = "q8"
            elif tt in ("Q4_K", "Q5_K", "Q6_K"):
                x = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(-1, int(t.shape[0]))
                n_elem = x.size
                if name == "token_embd.weight":
                    raw = x.tobytes()
                    dt = "f32"
                else:
                    qs, d = q8_split(t, quantize(x, GGMLQuantizationType.Q8_0))
                    raw = qs.tobytes() + d.tobytes()
                    dt = "q8"
            else:
                raise SystemExit(f"{name}: unsupported {tt}")
            out.write(raw)
            lines.append(f"{name} {dt} {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(lines) + "\n")
    print(f"packed {len(order)} tensors, {off/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
