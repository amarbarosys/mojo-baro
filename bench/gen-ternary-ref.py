#!/usr/bin/env python3
"""T0 host reference for the ternary GEMV instrument (bench_coldcache_ternary.mojo).

Dequantizes blk.0.ffn_gate.weight straight from the production engine packs
(.work/engine-pack-{q2b3,tq1,tq2}/pack.bin) using the verbatim C codec in
tools/ternary-ref.c (via tools/b3s-check.py's ctypes binding), then computes
the fp64 dot against row 0 of the existing .work/gguf/blk_0_ffn_gate_weight.a.bin
(the same A row bench_coldcache_q8row.mojo uses). Writes
.work/gguf/blk_0_ffn_gate_weight.<fam>.ref1.bin, fp32 [N], one per family.

Usage: bench/gen-ternary-ref.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from importlib.util import spec_from_file_location, module_from_spec


def _load(name):
    spec = spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("b3s-check")
ep = _load("engine-pack")

N, K = 12288, 4096
TENSOR = "blk.0.ffn_gate.weight"
A_PATH = ROOT / ".work" / "gguf" / "blk_0_ffn_gate_weight.a.bin"


def pack_entry(fam):
    packdir = ROOT / ".work" / f"engine-pack-{fam}"
    idx = {}
    for line in (packdir / "index.txt").read_text().splitlines():
        n, dt, off, ne = line.split()
        idx[n] = (dt, int(off), int(ne))
    dt, off, ne = idx[TENSOR]
    assert dt == fam, (TENSOR, dt, fam)
    assert ne == N * K, (ne, N, K)
    return packdir / "pack.bin", off


def main():
    a16 = np.frombuffer(A_PATH.read_bytes()[: K * 2], dtype=np.uint16)
    a64 = ep.bf16_to_f32(a16).astype(np.float64)

    lib = bc.ref_lib()
    for fam in ("q2b3", "tq1", "tq2"):
        blk, nbytes, bsize, _, _ = bc.FAM[fam]
        nb = K // blk
        pack_path, off = pack_entry(fam)
        pack = np.memmap(pack_path, dtype=np.uint8, mode="r")
        pq = np.frombuffer(pack[off : off + N * nb * nbytes], dtype=np.uint8).reshape(N, -1)
        pd = np.frombuffer(
            pack[off + N * nb * nbytes : off + N * nb * nbytes + N * nb * 2],
            dtype=np.float16,
        ).reshape(N, nb)
        y = bc.c_dequantize(lib, fam, pq, pd)
        c = (a64 @ y.astype(np.float64).T).astype(np.float32)
        out = ROOT / ".work" / "gguf" / f"blk_0_ffn_gate_weight.{fam}.ref1.bin"
        out.write_bytes(c.tobytes())
        print(fam, "N=" + str(N), "K=" + str(K), "->", out, c.nbytes, "bytes")


if __name__ == "__main__":
    main()
