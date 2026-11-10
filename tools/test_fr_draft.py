#!/usr/bin/env python3
"""Oracle for tools/fr-draft.mojo (FR-Spec pack, bench/mtp-protocol.md).

Usage: tools/test_fr_draft.py SRC_PACK FR_PACK IDS_FILE

Checks, independently of the Mojo tool:
  1. FR_PACK/index.txt is SRC_PACK/index.txt plus exactly two trailing lines,
     output.weight.frdraft q4 and frdraft.ids i32, at the offsets the file holds.
  2. The first len(SRC pack.bin) bytes of FR pack.bin equal SRC pack.bin.
  3. Row r of the reduced head (nibbles and scales) equals row ids[r] of
     SRC output.weight, for every r.
  4. The id map equals IDS_FILE as little-endian int32.
  5. Every side file of SRC_PACK is present and identical.
"""
import filecmp
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, fr, ids_file = map(Path, sys.argv[1:])
    s_lines = (src / "index.txt").read_text().splitlines()
    f_lines = (fr / "index.txt").read_text().splitlines()
    assert f_lines[:len(s_lines)] == s_lines, "source index lines changed"
    extra = f_lines[len(s_lines):]
    assert len(extra) == 2, extra
    hname, hdt, hoff, hn = extra[0].split()
    iname, idt, ioff, ik = extra[1].split()
    assert (hname, hdt, iname, idt) == ("output.weight.frdraft", "q4", "frdraft.ids", "i32"), extra
    hoff, hn, ioff, ik = int(hoff), int(hn), int(ioff), int(ik)

    idx = {l.split()[0]: l.split() for l in s_lines}
    _, dt, off, n = idx["output.weight"]
    off, n = int(off), int(n)
    h = int(idx["output_norm.weight"][3])
    vocab = n // h
    ids = np.array(ids_file.read_text().split(), dtype=np.int64)
    k = ids.size
    assert dt == "q4" and ik == k and hn == k * h, (dt, ik, k, hn)

    s = np.memmap(src / "pack.bin", dtype=np.uint8, mode="r")
    f = np.memmap(fr / "pack.bin", dtype=np.uint8, mode="r")
    step = 1 << 28
    for a in range(0, s.size, step):
        b = min(a + step, s.size)
        assert np.array_equal(s[a:b], f[a:b]), f"trunk differs near byte {a}"
    assert hoff == s.size, (hoff, s.size)

    rq, rs = h // 2, (h // 32) * 2
    sq = s[off:off + vocab * rq].reshape(vocab, rq)
    sd = s[off + vocab * rq:off + vocab * rq + vocab * rs].reshape(vocab, rs)
    fq = f[hoff:hoff + k * rq].reshape(k, rq)
    fd = f[hoff + k * rq:hoff + k * rq + k * rs].reshape(k, rs)
    assert np.array_equal(fq, sq[ids]), "reduced nibble rows differ"
    assert np.array_equal(fd, sd[ids]), "reduced scale rows differ"
    assert ioff == hoff + k * (rq + rs), (ioff, hoff, k)
    got = np.frombuffer(f[ioff:ioff + 4 * k].tobytes(), dtype="<i4")
    assert np.array_equal(got, ids), "id map differs"
    assert f.size == ioff + 4 * k, (f.size, ioff + 4 * k)

    for p in src.iterdir():
        if p.is_file() and p.name not in ("pack.bin", "index.txt"):
            assert filecmp.cmp(p, fr / p.name, shallow=False), f"side file {p.name}"
    print(f"PASS: FR pack {fr}: trunk identical, {k} of {vocab} rows byte-exact, id map exact, side files present")


if __name__ == "__main__":
    main()
