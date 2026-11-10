#!/usr/bin/env python3
"""Oracle for the BARO_TRACE_SUM build of serve/spark.mojo (bench/p4-soak.sh with
P4_SOAK_ENV=BARO_TRACE_SUM_DIR=...): compares the per-request checksum tables.

A table is uint32[positions, 3 * layers + 1]: per layer the QKV-after-bias, residual-after-
attention and residual-after-FFN checksums, then the logits row. Same prompt, same process,
T=0 means every table must be identical. For each request that is not, this prints the FIRST
differing cell in execution order (position, then layer, then stage), which names the kernel
group that produced the transient wrong value, and how far the damage spread.

usage: p4-trace-diff.py DIR --layers N [--prompt-len P] | --selftest
"""
import argparse
import collections
import glob
import os
import sys
import tempfile

import numpy as np

STAGES = ["qkv(rms+qkv gemv+bias)", "post-attn(rope+kv+attn+o gemv)", "post-ffn(rms+gate/up/down gemv)"]


def cell_name(flat, stride, layers, prompt_len):
    pos, c = divmod(int(flat), stride)
    phase = "" if prompt_len is None else (" prefill" if pos < prompt_len - 1 else " decode step %d" % (pos - prompt_len + 1))
    if c == 3 * layers:
        return "pos %d%s logits(out norm+head gemv)" % (pos, phase)
    return "pos %d%s layer %d %s" % (pos, phase, c // 3, STAGES[c % 3])


def analyze(d, layers, prompt_len, out=sys.stdout):
    stride = 3 * layers + 1
    files = sorted(glob.glob(os.path.join(d, "trace-*.bin")), key=lambda p: int(os.path.basename(p)[6:-4]))
    if len(files) < 2:
        print("FAIL p4-trace-diff: need at least 2 trace files in", d, file=out)
        return 2
    tabs = {f: np.fromfile(f, dtype=np.uint32) for f in files}
    size = collections.Counter(len(t) for t in tabs.values()).most_common(1)[0][0]
    if size % stride:
        print("FAIL p4-trace-diff: table size %d is not a multiple of stride %d" % (size, stride), file=out)
        return 2
    same = [f for f in files if len(tabs[f]) == size]
    key = collections.Counter(tabs[f].tobytes() for f in same).most_common(1)[0][0]
    ref = np.frombuffer(key, dtype=np.uint32)
    unwritten = int((ref == 0).sum())
    print("requests %d  positions %d  stride %d  majority %d  zero_cells_in_majority %d" % (
        len(files), size // stride, stride, sum(1 for f in same if tabs[f].tobytes() == key), unwritten), file=out)
    bad = 0
    for f in files:
        t = tabs[f]
        name = os.path.basename(f)
        if len(t) != size:
            bad += 1
            print("%s: length %d differs from majority %d (a changed token count)" % (name, len(t), size), file=out)
            continue
        diff = np.nonzero(t != ref)[0]
        if len(diff) == 0:
            continue
        bad += 1
        first = diff[0]
        pos0 = first // stride
        in_first_pos = int((diff // stride == pos0).sum())
        print("%s: FIRST %s | differing cells %d, of which %d in that position | positions touched %d" % (
            name, cell_name(first, stride, layers, prompt_len), len(diff), in_first_pos,
            len(np.unique(diff // stride))), file=out)
    print("SUMMARY deviant_requests=%d/%d" % (bad, len(files)), file=out)
    return 0


def selftest():
    import io
    layers, stride = 2, 7
    with tempfile.TemporaryDirectory() as t:
        base = np.arange(1, 5 * stride + 1, dtype=np.uint32)
        for k in range(1, 4):
            x = base.copy()
            if k == 3:
                x[2 * stride + 4:] += 9
            x.tofile(os.path.join(t, "trace-%d.bin" % k))
        buf = io.StringIO()
        rc = analyze(t, layers, 2, buf)
        text = buf.getvalue()
        ok = rc == 0 and "trace-3.bin: FIRST pos 2 decode step 1 layer 1 post-attn" in text and "deviant_requests=1/3" in text
        print("selftest", "PASS" if ok else "FAIL\n" + text)
        return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--prompt-len", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.dir or not a.layers:
        ap.error("DIR and --layers are required")
    sys.exit(analyze(a.dir, a.layers, a.prompt_len))
