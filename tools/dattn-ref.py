#!/usr/bin/env python3
"""Gate 2 of bench/dattn-protocol.md: the generic decode attention (.work/test_dattn) vs an
fp64 numpy reference of softmax(Q K^T * scale) V over the same f16 (or f32) KV bytes.

Inputs are regenerated here from the same deterministic formula the Mojo side uses
(op_bench's fill: ((i * 2654435761) % 2001) / 1000 - 1, V seeded +1000003), rounded to
the KV dtype exactly as the kernel's host fill does, so only the kernel's output crosses
the process boundary. Pass = max |O - ref| <= 2e-3 * max |ref| on every case.

usage: tools/dattn-ref.py [--bin .work/test_dattn] [--out .work/dattn] [--quick]
Runs inside one gpu-wait job (re-execs itself through gpu-wait when not already admitted).
"""
import argparse, itertools, os, subprocess, sys
import numpy as np

SHAPES = {
    "S0": (256, 16, 4, np.float32),
    "S1": (256, 16, 4, np.float16),
    "S2": (64, 40, 8, np.float16),
    "S3": (128, 28, 4, np.float16),
}
LENGTHS = [1, 127, 128, 129, 4096]
PATHS = [("exact", 1), ("split", 1), ("split", 8), ("split", 64)]
NLDS = [2, 4, 8]
ROTS = [0, 1]
MS = [1, 3]
QSCALES = [1, 8]
TOL = 2e-3


def hash_val(i):
    i = np.asarray(i, dtype=np.uint64)
    return ((i * np.uint64(2654435761)) % np.uint64(2001)).astype(np.float64) / 1000.0 - 1.0


def inputs(shape, T, qscale, M):
    HD, NQH, NKVH, kvt = SHAPES[shape]
    TT = T + M - 1
    q = (hash_val(np.arange(M * NQH * HD)).astype(np.float32) * np.float32(qscale)).reshape(M * NQH, HD)
    idx = np.arange(NKVH * TT * HD)
    k = hash_val(idx).astype(np.float32).astype(kvt).reshape(NKVH, TT, HD)
    v = hash_val(idx + 1000003).astype(np.float32).astype(kvt).reshape(NKVH, TT, HD)
    return q, k, v


def reference(shape, q, k, v, T, M):
    HD, NQH, NKVH, _ = SHAPES[shape]
    G = NQH // NKVH
    q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    out = np.zeros((M * NQH, HD))
    scale = 1.0 / np.sqrt(HD)
    for r in range(M):
        n = T + r
        for h in range(NQH):
            kvh = h // G
            s = k64[kvh, :n] @ q64[r * NQH + h] * scale
            p = np.exp(s - s.max())
            out[r * NQH + h] = (p @ v64[kvh, :n]) / p.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=".work/test_dattn")
    ap.add_argument("--out", default=".work/dattn")
    ap.add_argument("--quick", action="store_true", help="S1 only, T in (129, 4096)")
    a = ap.parse_args()
    if not os.environ.get("GPU_WAITING_ROOM_JOB"):
        os.execvp("gpu-wait", ["gpu-wait", "run", "--vram", "4", "--timeout", "1800", "--",
                               sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
    os.makedirs(a.out, exist_ok=True)
    shapes = ["S1"] if a.quick else list(SHAPES)
    lengths = [129, 4096] if a.quick else LENGTHS
    worst, failed, n = 0.0, [], 0
    for shape, T, (path, ns), nld, rot, M, qscale in itertools.product(shapes, lengths, PATHS, NLDS, ROTS, MS, QSCALES):
        if path == "exact" and (nld != 4 or rot == 1):
            continue
        if M > 1 and (nld != 4 or rot == 1):
            continue
        q, k, v = inputs(shape, T, qscale, M)
        ref = reference(shape, q, k, v, T, M)
        f = f"{a.out}/{shape}_{T}_{path}{ns}_n{nld}_r{rot}_m{M}_q{qscale}.bin"
        r = subprocess.run([a.bin, shape, str(T), path, str(ns), str(nld), str(rot), str(M), str(qscale), f],
                           capture_output=True, text=True)
        echo = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if r.returncode != 0 or not echo.startswith("echo "):
            failed.append((shape, T, path, ns, nld, rot, M, qscale, "RUN FAILED: " + r.stderr.strip()[-200:]))
            continue
        got = np.fromfile(f, dtype=np.float32).reshape(ref.shape).astype(np.float64)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        worst = max(worst, err)
        n += 1
        ok = err <= TOL and np.isfinite(got).all()
        print(f"{'ok  ' if ok else 'FAIL'} {shape} T={T:<5} {path} ns={ns:<3} nld={nld} rot={rot} m={M} q={qscale} relmax={err:.2e}  [{echo[5:]}]")
        if not ok:
            failed.append((shape, T, path, ns, nld, rot, M, qscale, f"relmax {err:.3e}"))
    print(f"\n{n} cases, worst relmax {worst:.3e}, tolerance {TOL:.0e}, failures {len(failed)}")
    for x in failed:
        print("  FAIL", *x)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
