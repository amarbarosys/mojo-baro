#!/usr/bin/env python3
"""usage: state-bytes-diff.py A.state B.state [--swapped]

Did node B keep the bytes node A exported? Compares two f32 BAROST01 states (bare or behind the
256-byte LAT1 header) section by section: pos, slot sizes, tokens, conv, SSM, and K and V for the
positions BELOW pos. Positions at and above pos are skipped on purpose: each node writes them itself
while it decodes, so they say nothing about the import.

--swapped expects B's K to equal A's V and B's V to equal A's K, exactly. That is the arm that
discriminates: a node that ignored the import and re-prefilled would produce A's own K and V, so
plain equality alone cannot prove the import was used; a swapped result can only come from the
imported bytes. Exit 0 when every section matches as expected, 1 with a FAIL line otherwise.
Layout (serve/engine.mojo save_state, tools/llama-slot-to-state.mojo): page-major pool,
[page][att layer 8][kv head 4][t % 128][256]. A verifier; nothing at serve time uses it."""
import sys

import numpy as np

N_ATT, NKVH, KVPAGE, HD = 8, 4, 128, 256


def load(path):
    d = open(path, "rb").read()
    if d[:4] == b"LAT1":
        d = d[256:]
    if d[:8] != b"BAROST01":
        raise SystemExit(f"FAIL state-bytes-diff: {path} is not an f32 BAROST01 state ({d[:8]!r})")
    pos, conv_n, ssm_n, kvn = (int.from_bytes(d[8 + 8 * k : 16 + 8 * k], "little") for k in range(4))
    o = 72
    tokens = d[o : o + 4 * pos]; o += 4 * pos
    conv = d[o : o + 4 * conv_n]; o += 4 * conv_n
    ssm = d[o : o + 4 * ssm_n]; o += 4 * ssm_n
    if len(d) != o + 8 * kvn:
        raise SystemExit(f"FAIL state-bytes-diff: {path} length {len(d)} is not {o + 8 * kvn}")
    shape = (kvn // (N_ATT * NKVH * KVPAGE * HD), N_ATT, NKVH, KVPAGE, HD)
    k = np.frombuffer(d, dtype="<u4", count=kvn, offset=o).reshape(shape)
    v = np.frombuffer(d, dtype="<u4", count=kvn, offset=o + 4 * kvn).reshape(shape)
    return {"pos": pos, "salt": d[40:72], "tokens": tokens, "conv": conv, "ssm": ssm, "k": k, "v": v}


def below(x, pos):  # [page][layer][head][t][hd] -> the rows for positions 0..pos-1, bit patterns
    pages = x.shape[0]
    flat = x.transpose(0, 3, 1, 2, 4).reshape(pages * KVPAGE, N_ATT, NKVH, HD)
    return flat[:pos]


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    swapped = "--swapped" in sys.argv
    bad = [n for n in ("pos", "salt", "tokens", "conv", "ssm") if a[n] != b[n]]
    ak, av, bk, bv = (below(x, a["pos"]) for x in (a["k"], a["v"], b["k"], b["v"])) if a["pos"] == b["pos"] else (None,) * 4
    if ak is not None:
        want_k, want_v = (av, ak) if swapped else (ak, av)
        for name, got, want in (("K", bk, want_k), ("V", bv, want_v)):
            if not np.array_equal(got, want):
                bad.append(f"{name} ({int((got != want).sum())} of {got.size} words differ)")
        if swapped and np.array_equal(ak, av):
            bad.append("A's K equals A's V, so the swapped arm cannot discriminate")
    mode = "B.K == A.V and B.V == A.K" if swapped else "B == A"
    if bad:
        print(f"FAIL state-bytes-diff ({mode}): differs in {', '.join(bad)}")
        sys.exit(1)
    n = a["pos"] * N_ATT * NKVH * HD
    print(f"OK {mode}: pos {a['pos']}, salt, tokens, conv {len(a['conv'])} B, ssm {len(a['ssm'])} B, K and V {n} words each below pos, all bit-identical")


if __name__ == "__main__":
    main()
