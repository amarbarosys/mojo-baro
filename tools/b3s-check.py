#!/usr/bin/env python3
"""Q0 gate for the ternary pack families (q2b3 = Q2_B3/B3S, tq1 = TQ1_0,
tq2 = TQ2_0): the engine pack entry must be byte-equal to the reference C
quantization of the same tensor. The reference is tools/ternary-ref.c, the
verbatim quantize_row_*_ref / dequantize_row_* from ggml, built on demand
into .work/ternary-ref.so and driven through ctypes.

Usage:
  tools/b3s-check.py --selftest
      Python quantizers in tools/engine-pack.py vs the C reference, bit for
      bit, on random ternary-native rows and random fp32 rows for all three
      families; also re-diffs the C bodies against .work/b3s-ref/ when that
      staging dir is present.
  tools/b3s-check.py PACKDIR MODEL.gguf name [name ...]
      For each pack name (dtype q2b3/tq1/tq2): payload bytes and fp16
      scales of the pack entry vs the C reference run on the bf16 tensor
      read from MODEL.gguf.
  tools/b3s-check.py --quant Q.gguf PACKDIR MODEL.gguf name [name ...]
      As above, plus the same tensor's blocks from a llama-quantize file
      (TQ1_0 type 34, TQ2_0 type 35, or a B3S-fork Q2_B3 type 43).
  tools/b3s-check.py --fixture M MODEL.gguf name [name ...]
      Write kernels/test_ternary_gemm.mojo inputs to .work/gguf/: per name
      and family <safe>.<fam>.bin (payload [N, nb*B]), <safe>.<fam>.scales.bin
      (fp16 [N, nb]), <safe>.<fam>.c.bin (fp32 [M, N] = A @ dequant(W)^T with
      the C dequantizer), and one shared <safe>.tern.a.bin (bf16 [M, K],
      seed 7).
"""
import ctypes
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec


def _load(name):
    spec = spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ge = _load("gguf-extract")
ep = _load("engine-pack")

# family -> (block weights, payload bytes, block bytes, scale-first, ggml type)
FAM = {
    "q2b3": (128, 26, 28, True, 43),
    "tq1": (256, 52, 54, False, 34),
    "tq2": (256, 64, 66, False, 35),
}
CFUN = {
    "q2b3": ("quantize_row_q2_b3_ref", "dequantize_row_q2_b3"),
    "tq1": ("quantize_row_tq1_0_ref", "dequantize_row_tq1_0"),
    "tq2": ("quantize_row_tq2_0_ref", "dequantize_row_tq2_0"),
}


def ref_lib():
    src = ROOT / "tools" / "ternary-ref.c"
    so = ROOT / ".work" / "ternary-ref.so"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        so.parent.mkdir(exist_ok=True)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    lib = ctypes.CDLL(str(so))
    for q, d in CFUN.values():
        getattr(lib, q).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        getattr(lib, d).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    return lib


def c_quantize(lib, fam, x32):
    """x32: fp32 [N, K] -> (payload [N, nb*B] uint8, scales [N, nb] fp16)."""
    blk, nbytes, bsize, scale_first, _ = FAM[fam]
    n, k = x32.shape
    assert k % blk == 0
    x = np.ascontiguousarray(x32, dtype=np.float32)
    out = np.zeros(n * (k // blk) * bsize, dtype=np.uint8)
    getattr(lib, CFUN[fam][0])(x.ctypes.data, out.ctypes.data, n * k)
    return split_blocks(fam, out, n)


def c_dequantize(lib, fam, payload, scales):
    blk, nbytes, bsize, scale_first, _ = FAM[fam]
    n, nb = scales.shape
    raw = join_blocks(fam, payload, scales)
    y = np.zeros(n * nb * blk, dtype=np.float32)
    getattr(lib, CFUN[fam][1])(raw.ctypes.data, y.ctypes.data, n * nb * blk)
    return y.reshape(n, nb * blk)


def split_blocks(fam, raw, n):
    blk, nbytes, bsize, scale_first, _ = FAM[fam]
    b = np.frombuffer(np.ascontiguousarray(raw), dtype=np.uint8).reshape(n, -1, bsize)
    if scale_first:
        d, q = b[:, :, :2], b[:, :, 2:]
    else:
        q, d = b[:, :, :nbytes], b[:, :, nbytes:]
    return (np.ascontiguousarray(q).reshape(n, -1),
            np.ascontiguousarray(d).reshape(n, -1).view(np.float16))


def join_blocks(fam, payload, scales):
    blk, nbytes, bsize, scale_first, _ = FAM[fam]
    n, nb = scales.shape
    q = payload.reshape(n, nb, nbytes)
    d = np.ascontiguousarray(scales).view(np.uint8).reshape(n, nb, 2)
    parts = (d, q) if scale_first else (q, d)
    return np.ascontiguousarray(np.concatenate(parts, axis=2)).reshape(-1)


def f32_to_bf16(x32):
    return (np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def compare(label, fam, pq, pd, rq, rd):
    dq = int((pq != rq).sum())
    dd = int((pd.view(np.uint16) != rd.view(np.uint16)).sum())
    print(f"{label}: payload mismatch {dq}/{pq.size}  scales mismatch {dd}/{pd.size}")
    return dq == 0 and dd == 0


def verbatim_check():
    """Re-diff the C bodies in tools/ternary-ref.c against the staged sources."""
    staged = ROOT / ".work" / "b3s-ref"
    if not staged.exists():
        print("verbatim check: .work/b3s-ref absent, skipped")
        return True
    ours = (ROOT / "tools" / "ternary-ref.c").read_text()
    srcs = {
        "quantize_row_q2_b3_ref": "ggml-quants.c", "dequantize_row_q2_b3": "ggml-quants.c",
        "quantize_row_tq1_0_ref": "mainline-ggml-quants.c", "quantize_row_tq2_0_ref": "mainline-ggml-quants.c",
        "dequantize_row_tq1_0": "mainline-ggml-quants.c", "dequantize_row_tq2_0": "mainline-ggml-quants.c",
    }
    ok = True
    for fn, f in srcs.items():
        pat = re.compile(r"^void " + fn + r"\(.*?^}\n", re.M | re.S)
        a, b = pat.search(ours), pat.search((staged / f).read_text())
        same = a is not None and b is not None and a.group(0) == b.group(0)
        print(f"verbatim {fn}: {'OK' if same else 'DIFFERS'}")
        ok = ok and same
    return ok


def selftest():
    lib = ref_lib()
    ok = verbatim_check()
    rng = np.random.default_rng(11)
    for fam in FAM:
        blk = FAM[fam][0]
        k = blk * 8
        n = 64
        # ternary-native rows: {-s, 0, +s} with a random per-row scale
        s = rng.uniform(0.01, 4.0, size=(n, 1)).astype(np.float32)
        tern = (rng.integers(-1, 2, size=(n, k)).astype(np.float32) * s)
        # random fp32 rows, plus rows with zero blocks and ties at +-0.5*amax
        rnd = rng.standard_normal((n, k)).astype(np.float32)
        rnd[3, :blk] = 0
        rnd[5, :] = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=k).astype(np.float32)
        for label, x in (("ternary", tern), ("random", rnd)):
            w16 = f32_to_bf16(x)
            pq, pd = ep.TERNARY[fam](w16)
            rq, rd = c_quantize(lib, fam, ep.bf16_to_f32(w16))
            ok = compare(f"{fam} {label} py-vs-C", fam, pq, pd, rq, rd) and ok
            y = c_dequantize(lib, fam, rq, rd)
            if label == "ternary":
                exact = np.array_equal(y, np.sign(x) * rd.astype(np.float32).repeat(blk, axis=1))
                print(f"{fam} ternary C dequant exact: {exact}")
                ok = ok and exact
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


def read_bf16(gguf, name):
    f, infos, data_start, _ = ge.parse(gguf)
    dims, ttype, toff = infos[name]
    tname, esize = ge.GGML_BYTES[ttype]
    assert tname == "bf16" and len(dims) == 2, (name, tname, dims)
    n_elem = int(np.prod(dims))
    f.seek(data_start + toff)
    raw = f.read(n_elem * esize)
    return np.frombuffer(raw, dtype=np.uint16).reshape(list(reversed(dims)))


def read_quant(qgguf, name, fam):
    f, infos, data_start, _ = ge.parse(qgguf)
    dims, ttype, toff = infos[name]
    blk, nbytes, bsize, _, gtype = FAM[fam]
    assert ttype == gtype, (name, ttype, gtype)
    shape = list(reversed(dims))
    nb = shape[1] // blk
    f.seek(data_start + toff)
    raw = np.frombuffer(f.read(shape[0] * nb * bsize), dtype=np.uint8)
    return split_blocks(fam, raw, shape[0])


def gate(packdir, gguf, names, qgguf=None):
    lib = ref_lib()
    idx = {}
    for line in (packdir / "index.txt").read_text().splitlines():
        n, dt, off, ne = line.split()
        idx[n] = (dt, int(off), int(ne))
    pack = np.memmap(packdir / "pack.bin", dtype=np.uint8, mode="r")
    ok = True
    for name in names:
        dt, off, ne = idx[name]
        assert dt in FAM, (name, dt)
        blk, nbytes, bsize, _, _ = FAM[dt]
        w16 = read_bf16(gguf, name)
        n, k = w16.shape
        nb = k // blk
        pq = np.frombuffer(pack[off:off + n * nb * nbytes], dtype=np.uint8).reshape(n, -1)
        pd = np.frombuffer(pack[off + n * nb * nbytes:off + n * nb * nbytes + n * nb * 2],
                           dtype=np.float16).reshape(n, nb)
        rq, rd = c_quantize(lib, dt, ep.bf16_to_f32(w16))
        ok = compare(f"{name} [{dt}] pack-vs-C", dt, pq, pd, rq, rd) and ok
        if qgguf is not None:
            gq, gd = read_quant(qgguf, name, dt)
            ok = compare(f"{name} [{dt}] pack-vs-llama-quantize", dt, pq, pd, gq, gd) and ok
    print("Q0 gate:", "PASS" if ok else "FAIL")
    return ok


def fixture(m, gguf, names):
    lib = ref_lib()
    out = ROOT / ".work" / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        safe = name.replace(".", "_").replace("/", "_")
        w16 = read_bf16(gguf, name)
        n, k = w16.shape
        rng = np.random.default_rng(7)
        a16 = f32_to_bf16(rng.standard_normal((m, k)).astype(np.float32))
        a32 = ep.bf16_to_f32(a16)
        (out / f"{safe}.tern.a.bin").write_bytes(a16.tobytes())
        for fam in FAM:
            q, d = c_quantize(lib, fam, ep.bf16_to_f32(w16))
            y = c_dequantize(lib, fam, q, d)
            c = (a32.astype(np.float64) @ y.astype(np.float64).T).astype(np.float32)
            (out / f"{safe}.{fam}.bin").write_bytes(q.tobytes())
            (out / f"{safe}.{fam}.scales.bin").write_bytes(d.tobytes())
            (out / f"{safe}.{fam}.c.bin").write_bytes(c.tobytes())
            print(f"{safe}.{fam}: payload {q.nbytes} B, scales {d.nbytes} B, C [{m}, {n}]")


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if argv[:1] == ["--fixture"]:
        fixture(int(argv[1]), Path(argv[2]), argv[3:])
        return
    qgguf = None
    if argv[:1] == ["--quant"]:
        qgguf = Path(argv[1])
        argv = argv[2:]
    sys.exit(0 if gate(Path(argv[0]), Path(argv[1]), argv[2:], qgguf) else 1)


if __name__ == "__main__":
    main()
