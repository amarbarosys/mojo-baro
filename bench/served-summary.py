#!/usr/bin/env python3
"""Receipt for bench/served-prompts.sh: one-shot arm against served arm.

usage: bench/served-summary.py OUTDIR

Separate from the harness so it can be re-run on an existing run directory
without touching the GPU: the arms' artifacts (per-prompt one-shot logs and
served.jsonl) are the data, and a parser bug is fixed and re-run against them
rather than costing another stint.

Exit 1 on any void row or identity failure (P10: a void is a failed arm, not a
skipped one).
"""
import json
import pathlib
import re
import statistics as st
import sys

TOKS = re.compile(r"tok/s_gen: ([0-9.]+)")


def main():
    out = pathlib.Path(sys.argv[1])
    served = {}
    for line in (out / "served.jsonl").read_text().splitlines():
        d = json.loads(line)
        served[d["p"]] = d
    rows = []
    for log in sorted(out.glob("*.oneshot.log")):
        p = log.name[: -len(".oneshot.log")]
        txt = log.read_text()
        m = TOKS.search(txt)
        gen = next((l.split(":", 1)[1].split() for l in txt.splitlines() if l.startswith("GENERATED:")), None)
        s = served.get(p)
        if m is None or gen is None or s is None:
            rows.append((p, None, None, None, "VOID"))
            continue
        ident = "PASS" if [int(x) for x in gen] == s["tokens"] else "FAIL"
        wall = (len(s["tokens"]) - 1) / s["wall_s"] if s["wall_s"] > 0 else 0.0
        rows.append((p, float(m.group(1)), s["timings"]["tok_s_gen"], wall, ident))

    hdr = f"{'prompt':<18}{'oneshot':>9}{'served':>9}{'servedWall':>12}  identity"
    print(hdr)
    print("-" * len(hdr))
    for p, a, b, w, i in rows:
        f = lambda v, n: (f"{v:{n}.2f}" if v is not None else "n/a".rjust(n))
        print(f"{p:<18}{f(a, 9)}{f(b, 9)}{f(w, 12)}  {i}")
    (out / "results.txt").write_text(
        hdr + "\n" + "\n".join(f"{p} {a} {b} {w} {i}" for p, a, b, w, i in rows) + "\n")

    void = [r[0] for r in rows if r[4] == "VOID"]
    ok = [r for r in rows if r[4] != "VOID"]
    fails = [r[0] for r in ok if r[4] != "PASS"]
    if ok:
        A = [r[1] for r in ok]
        B = [r[2] for r in ok]
        W = [r[3] for r in ok]
        sp = lambda x: (max(x) - min(x)) / st.median(x) * 100
        print(f"one-shot median {st.median(A):.2f} tok/s_gen spread {sp(A):.1f}%  |  "
              f"served median {st.median(B):.2f} spread {sp(B):.1f}%  |  "
              f"ratio {st.median(B) / st.median(A):.3f}")
        print(f"served wall-clock median {st.median(W):.2f} tok/s (HTTP round trip included, "
              f"prefill and queueing in it)  |  identity fails: {fails or 'none'}")
        pre = [s["timings"]["prefill_s"] for s in served.values()]
        dec = [s["timings"]["decode_s"] for s in served.values()]
        wall = [s["wall_s"] for s in served.values()]
        print(f"served medians: prefill {st.median(pre) * 1e3:.1f} ms  decode {st.median(dec):.3f} s  "
              f"wall {st.median(wall):.3f} s  overhead {(st.median(wall) - st.median(dec) - st.median(pre)) * 1e3:.1f} ms")
    if void:
        print(f"FAIL: {len(void)} void row(s): {void}")
        return 1
    if fails:
        print(f"FAIL: served tokens differ from one-shot on {fails}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
