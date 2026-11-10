#!/usr/bin/env python3
"""Oracle for bench/p4-glitch-capture.sh: compares K same-prompt runs of the spark engine.

Each run dir holds `tokens` (the generated ids) and row-<step>.bin (the f32 logits row
the argmax read at that step). For every run this reports the first step whose token
leaves the majority stream, the first step whose ROW leaves the reference run's row
bitwise, and at the token-deviation step: how many logits differ, which indices, and
whether the emitted token is the argmax of the run's own row. That last field separates
"the row was already wrong" (upstream of the head) from "the row was right and argmax
picked something else".

usage: p4-row-diff.py CAPTURE_DIR | --selftest
"""
import collections
import glob
import os
import sys
import tempfile

import numpy as np


def load_run(d):
    toks = [int(x) for x in open(os.path.join(d, "tokens")).read().split()]
    return toks


def row(d, step):
    return np.fromfile(os.path.join(d, "row-%d.bin" % step), dtype=np.float32)


def bits(a):
    return a.view(np.uint32)


def analyze(capdir, out=sys.stdout):
    runs = sorted(glob.glob(os.path.join(capdir, "run*")), key=lambda p: int(os.path.basename(p)[3:]))
    if len(runs) < 2:
        print("FAIL p4-row-diff: need at least 2 runs in", capdir, file=out)
        return 2
    toks = {r: load_run(r) for r in runs}
    n = min(len(t) for t in toks.values())
    majority = []
    for s in range(n):
        # majority over runs that still share the majority prefix: a run that already
        # left the stream votes on a different context and is excluded
        voters = [r for r in runs if toks[r][:s] == majority]
        majority.append(collections.Counter(toks[r][s] for r in voters).most_common(1)[0][0])
    clean = [r for r in runs if toks[r][:n] == majority]
    if not clean:
        print("FAIL p4-row-diff: no run follows the majority stream end to end", file=out)
        return 2
    ref = clean[0]
    print("runs %d  steps %d  clean %d  reference %s" % (len(runs), n, len(clean), os.path.basename(ref)), file=out)
    print("run,first_token_dev,first_row_dev,rows_differing_at_first_row_dev,max_abs_diff,token,majority_token,"
          "argmax_own_row,argmax_ref_row,own_logit_token,ref_logit_token,nonfinite_in_row", file=out)
    deviants = 0
    rowdev_runs = 0
    for r in runs:
        tdev = next((s for s in range(n) if toks[r][s] != majority[s]), None)
        rdev = None
        ndiff = 0
        mad = 0.0
        last = n if tdev is None else tdev + 1
        for s in range(last):
            a, b = row(r, s), row(ref, s)
            neq = bits(a) != bits(b)
            if neq.any():
                rdev, ndiff = s, int(neq.sum())
                with np.errstate(invalid="ignore"):
                    mad = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                break
        name = os.path.basename(r)
        if rdev is not None:
            rowdev_runs += 1
        if tdev is None:
            print("%s,none,%s,%d,%g,,,,,,," % (name, "none" if rdev is None else rdev, ndiff, mad), file=out)
            continue
        deviants += 1
        a, b = row(r, tdev), row(ref, tdev)
        tok = toks[r][tdev]
        print("%s,%d,%s,%d,%g,%d,%d,%d,%d,%g,%g,%d" % (
            name, tdev, "none" if rdev is None else rdev, ndiff, mad, tok, majority[tdev],
            int(np.nanargmax(a)), int(np.nanargmax(b)), float(a[tok]), float(b[tok]),
            int((~np.isfinite(a)).sum())), file=out)
        if rdev is not None:
            a2, b2 = row(r, rdev), row(ref, rdev)
            idx = np.nonzero(bits(a2) != bits(b2))[0]
            head = ", ".join("%d:%g->%g" % (i, b2[i], a2[i]) for i in idx[:8])
            print("#  %s row step %d differs at %d indices, span %d..%d, first: %s" % (
                name, rdev, len(idx), int(idx[0]), int(idx[-1]), head), file=out)
    print("SUMMARY deviant_token_runs=%d/%d runs_with_any_row_difference=%d/%d" % (
        deviants, len(runs), rowdev_runs, len(runs)), file=out)
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as t:
        rng = np.random.default_rng(0)
        base = [rng.standard_normal(64).astype(np.float32) for _ in range(4)]
        for k in range(3):
            d = os.path.join(t, "run%d" % (k + 1))
            os.mkdir(d)
            rows = [x.copy() for x in base]
            if k == 2:
                rows[2][7] = 99.0
            for s, x in enumerate(rows):
                x.tofile(os.path.join(d, "row-%d.bin" % s))
            open(os.path.join(d, "tokens"), "w").write(" ".join(str(int(np.argmax(x))) for x in rows))
        import io
        buf = io.StringIO()
        rc = analyze(t, buf)
        text = buf.getvalue()
        ok = rc == 0 and "run3,2,2,1," in text and "deviant_token_runs=1/3" in text and "run1,none,none" in text
        print("selftest", "PASS" if ok else "FAIL\n" + text)
        return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(selftest() if sys.argv[1] == "--selftest" else analyze(sys.argv[1]))
