#!/usr/bin/env python3
"""C19 stage 3: export one arm's symbols, maxes and integer CDFs as raw files for
tools/c19_rc.mojo (AMDHQ 06 "## C19:", stage 3 amendment).

usage: c19_export.py S1_DIR LEVEL F32_STATE OUTDIR
Reads S1_DIR/LEVEL-symbols.npz (tools/c19_cachegen.py --write-recon) and writes:
  meta.txt      nlayers ntokens nchannels lpmax, then per layer: bins lp
  q.bin         int8  [nlayers][ntokens][nchannels]  (the symbols)
  max.bin       fp16  [nlayers][ntokens]
  cdf.bin       int32 [nlayers][nchannels][lpmax + 1], cdf[lp..] = 65536
  fixed.bin     the source file's header, ids, conv and SSM bytes (BAROST01 prefix)
The CDFs are stage 1's (c19_cachegen.int_cdf over the whole context).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import c19_cachegen as cg  # noqa: E402


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    s1, level, f32_path, out = sys.argv[1:]
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(pathlib.Path(s1) / f"{level}-symbols.npz")
    q, m = z["q"], z["m"]
    nl, t, c = q.shape
    kb, vb = cg.bins_for(level)
    bins = kb + vb
    lps = [max(kb) + 1] * cg.NAT + [max(vb) + 1] * cg.NAT
    lpmax = max(lps)
    cdf = np.full((nl, c, lpmax + 1), 2 ** 16, dtype=np.int32)
    for li in range(nl):
        idx = (np.arange(c, dtype=np.int64)[None, :] * lps[li] + q[li].astype(np.int64)).ravel()
        counts = np.bincount(idx, minlength=c * lps[li]).reshape(c, lps[li])
        cdf[li, :, :lps[li]] = cg.int_cdf(counts.astype(np.float32), t)
    q.astype(np.int8).tofile(out / "q.bin")
    m.reshape(nl, t).astype(np.float16).tofile(out / "max.bin")
    cdf.tofile(out / "cdf.bin")
    st = cg.read_state(f32_path)
    np.concatenate([st["head"], st["conv_ssm"]]).tofile(out / "fixed.bin")
    with open(out / "meta.txt", "w") as f:
        f.write(f"{nl} {t} {c} {lpmax}\n")
        for li in range(nl):
            f.write(f"{bins[li]} {lps[li]}\n")
    print(f"exported {level}: {nl} layers x {t} tokens x {c} channels, lpmax {lpmax}, fixed {st['head'].size + st['conv_ssm'].size} B")


if __name__ == "__main__":
    main()
