#!/usr/bin/env python3
"""Q0 gate for bench/ornith-protocol.md: engine-pack.py's K-quant path
(dequantise Q4_K/Q6_K via gguf-py, round to bf16, then the existing --q8
path) reproduces the source tensor within q8 rounding.

For a sample of q8-packed tensors, packed-q8-dequantised is compared
element-wise against gguf-py's own dequantise of the ORIGINAL K-quant bytes
(the exact ground truth) — not against our bf16-rounded intermediate, so
the tolerance covers both the q8 rounding step and the bf16 pass
engine-pack.py takes first.

Usage: tools/test_engine_pack_kquant.py MODEL.gguf PACKDIR [name ...]
"""
import sys
from pathlib import Path

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

DEFAULT_NAMES = ["blk.1.ffn_down.weight", "blk.3.attn_q.weight", "output.weight",
                  "blk.32.ffn_up.weight"]


def source_f32(f, data_start, toff, ttype, shape):
    qtype = GGMLQuantizationType(ttype)
    block, tsize = GGML_QUANT_SIZES[qtype]
    n_elem = int(np.prod(shape))
    assert n_elem % block == 0, (shape, block)
    f.seek(data_start + toff)
    raw = np.frombuffer(f.read(n_elem // block * tsize), dtype=np.uint8)
    return gguf_dequantize(raw, qtype).reshape(shape).astype(np.float32)


def main():
    model, packdir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or DEFAULT_NAMES
    f, infos, data_start, _ = ge.parse(model)
    pidx = {}
    for line in (packdir / "index.txt").read_text().splitlines():
        n, dt, off, ne = line.split()
        pidx[n] = (dt, int(off), int(ne))
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    ok = True
    for name in names:
        dims, ttype, toff = infos[name]
        shape = list(reversed(dims))
        w_true = source_f32(f, data_start, toff, ttype, shape)
        dt, off, ne = pidx[name]
        assert dt == "q8", (name, dt)
        n_out, n_in = shape
        assert n_out * n_in == ne, (name, n_out, n_in, ne)
        q = pack[off:off + ne].view(np.int8).reshape(n_out, n_in // 32, 32)
        d = pack[off + ne:off + ne + (ne // 32) * 2].view(np.float16).reshape(n_out, n_in // 32, 1)
        w_pack = (q.astype(np.float32) * d.astype(np.float32)).reshape(n_out, n_in)
        d_bc = np.broadcast_to(d.astype(np.float32), (n_out, n_in // 32, 32)).reshape(n_out, n_in)
        err = np.abs(w_pack - w_true)
        # q8 rounding step (d/2) plus one bf16-rounding ulp of the f32->bf16
        # pass engine-pack.py takes before quantising.
        tol = d_bc * 0.5 + np.abs(w_true) * 2.0 ** -7 + 1e-8
        bad = int((err > tol).sum())
        print(f"{name}: max err {err.max():.6g}  max tol {tol.max():.6g}  over-tol {bad}/{w_true.size}")
        ok = ok and bad == 0
    print("K-quant pack gate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
