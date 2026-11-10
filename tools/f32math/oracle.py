#!/usr/bin/env python3
"""Oracle for tools/f32mathcli.mojo: Float32 exp(x) and log(x) correctly rounded from mpmath at 100 bits (nearest of
the two Float32 neighbours, so no double rounding through Float64). Same line protocol: hex Float32 bits in,
`exp log` bits out, `nan` for NaN. `--gen N SEED` writes test inputs; `--ulp ANSWERS ORACLE [MAXULP]` prints per
column max ulp error and exits 1 above MAXULP."""
import random
import struct
import sys

import mpmath

mpmath.mp.prec = 100
F32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def b2f(s):
    return struct.unpack("<f", struct.pack("<I", int(s, 16)))[0]


def f2b(y):
    return "nan" if y != y else hex(struct.unpack("<I", struct.pack("<f", y))[0])


def ordered(s):
    b = int(s, 16)
    return -(b & 0x7FFFFFFF) if b >> 31 else b


def r32(v):
    """Correctly rounded Float32 of an mpf: compare the two neighbours of the Float32 cast."""
    if v > F32_MAX * (1 + 2.0 ** -25):
        return float("inf")
    if v == 0:
        return 0.0
    c = struct.unpack("<f", struct.pack("<f", float(v)))[0]
    b = struct.unpack("<I", struct.pack("<f", c))[0]
    best = c
    for nb in (b - 1, b + 1):
        if 0 <= nb < 0x7F800000 or 0x80000000 <= nb < 0xFF800000:
            y = struct.unpack("<f", struct.pack("<I", nb))[0]
            if abs(mpmath.mpf(y) - v) < abs(mpmath.mpf(best) - v):
                best = y
    return best


def explog(x):
    if x != x:
        return x, x
    e = 0.0 if x == float("-inf") else (x if x == float("inf") else r32(mpmath.exp(mpmath.mpf(x))))
    if x == 0:
        lg = float("-inf")
    elif x < 0:
        lg = float("nan")
    elif x == float("inf"):
        lg = x
    else:
        lg = r32(mpmath.log(mpmath.mpf(x)))
    return e, lg



def main():
    if sys.argv[1:2] == ["--ulp"]:
        a = [l.split() for l in open(sys.argv[2]) if l.strip()]
        o = [l.split() for l in open(sys.argv[3]) if l.strip()]
        xs = [l.strip() for l in open(sys.argv[5])] if len(sys.argv) > 5 else None
        lim = float(sys.argv[4]) if len(sys.argv) > 4 else 1
        if len(a) != len(o):
            sys.exit(f"FAIL ulp: {len(a)} answers for {len(o)} oracle lines")
        bad = 0
        for j, name in enumerate(["exp", "log"]):
            worst, exact, where, over1 = 0, 0, "", 0
            for i, (ra, ro) in enumerate(zip(a, o)):
                x, y = ra[j], ro[j]
                if x == y:
                    exact += 1
                    continue
                d = float("inf") if "nan" in (x, y) else abs(ordered(x) - ordered(y))
                over1 += d > 1
                if d > worst:
                    worst, where = d, f"line {i + 1}" + (f" x={b2f(xs[i]):.9g} got {b2f(x) if x != 'nan' else x} want {b2f(y) if y != 'nan' else y}" if xs else "")
            print(f"{name}: max {worst:g} ulp ({where or 'none'}), correctly rounded {exact}/{len(a)} ({100 * exact / len(a):.2f}%), >1 ulp {over1}")
            bad += worst > lim
        sys.exit(1 if bad else 0)
    elif sys.argv[1:2] == ["--gen"]:
        n, r = int(sys.argv[2]), random.Random(int(sys.argv[3]))
        xs = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, float("inf"), float("-inf"), float("nan"), 88.72, 88.73, -87.33, -103.9,
              -104.0, 1e-45, 1.1754944e-38, F32_MAX]
        xs += [r.uniform(-30, 0) for _ in range(n // 4)]  # softmax after max subtraction
        xs += [r.uniform(-104, 89) for _ in range(n // 4)]  # full exp range
        xs += [10 ** r.uniform(-45, 38) for _ in range(n // 4)] + [1 + r.uniform(-1e-3, 1e-3) for _ in range(n // 8)]
        xs += [r.uniform(0, 1) for _ in range(n // 8)]  # log of probabilities
        print("\n".join(f2b(struct.unpack("<f", struct.pack("<f", x))[0]) if x == x else "0x7fc00000" for x in xs))
    else:
        for line in sys.stdin.read().split():
            e, lg = explog(b2f(line))
            print(f2b(e), f2b(lg))


if __name__ == "__main__":
    main()
