#!/usr/bin/env python3
"""bench/expert-overlap.py: how many routed experts do nearby decode tokens
share? Exploration for latent-draft speculation on qwen35moe: a verify window
of R rows reads the UNION of the rows' experts, so the row price of a fused
multi-row kernel is set by this overlap, not by draft acceptance.

Usage:
  gpu-wait run ... -- bench/expert-overlap.py run ENGINE TIERPACK OUT [TIERCAP]
  bench/expert-overlap.py report OUT        (CPU only)

run: 20 bench/mtp-prompts fixtures, 64 greedy tokens each, BARO_EXPERTS trace.
report: per distance d=1..3, mean fraction of a token's experts also routed
by the token d steps earlier (same layer); and union size of a 2/3/4-row
window relative to one row. Random-routing baseline printed beside it.
"""
import json, os, subprocess, sys
from pathlib import Path


def run(engine, pack, out, cap):
    if not os.environ.get("GPU_WAITING_ROOM_JOB"):
        raise RuntimeError("run through gpu-wait")
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path("bench/mtp-prompts").glob("p*.tokens"))
    if len(paths) != 20:
        raise RuntimeError("expected 20 prompt fixtures")
    reqs = [dict(id=i, prompt=list(map(int, p.read_text().split())), n=64,
                 spec=False, temperature=0) for i, p in enumerate(paths)]
    wire = "".join(json.dumps(r) + "\n" for r in reqs)
    env = dict(os.environ, BARO_SERVE="1", BARO_PACK=str(pack.resolve()),
               BARO_TIER=cap, BARO_EXPERTS=str((out / "trace.txt").resolve()),
               BARO_SPEC="0", BARO_NGRAM="0", BARO_MEGA="0", BARO_CKPT="0")
    with (out / "engine.out").open("w") as so, (out / "engine.err").open("w") as se:
        subprocess.run([str(engine.resolve())], input=wire, text=True, env=env,
                       stdout=so, stderr=se, check=True, timeout=900)


def report(out):
    seqs, cur, topk = [], None, 0
    for line in (out / "trace.txt").read_text().splitlines():
        if line.startswith("#"):
            cur = {}
            seqs.append(cur)
            continue
        f = list(map(int, line.split()))
        ids = frozenset(e for e in f[2:] if e >= 0)
        topk = max(topk, len(ids))
        cur[(f[0], f[1])] = ids
    if not seqs or not topk:
        raise RuntimeError("empty trace")
    n_exp = max(e for s in seqs for ids in s.values() for e in ids) + 1
    res = {"sequences": len(seqs), "topk": topk, "experts_seen": n_exp,
           "random_shared_fraction": topk / n_exp}
    for d in (1, 2, 3):
        fr = [len(ids & s[(t - d, ly)]) / len(ids)
              for s in seqs for (t, ly), ids in s.items() if (t - d, ly) in s and ids]
        res[f"shared_fraction_d{d}"] = sum(fr) / len(fr)
    for r in (2, 3, 4):
        un = []
        for s in seqs:
            for (t, ly), ids in s.items():
                rows = [s.get((t + k, ly)) for k in range(r)]
                if all(rows):
                    un.append(len(frozenset().union(*rows)) / topk)
        res[f"union_rows{r}_vs_one_row"] = sum(un) / len(un)
    (out / "overlap.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "run":
            run(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
                sys.argv[5] if len(sys.argv) > 5 else "64")
        elif sys.argv[1] == "report":
            report(Path(sys.argv[2]))
        else:
            raise RuntimeError("usage: run ENGINE TIERPACK OUT [CAP] | report OUT")
    except Exception as error:
        print(f"FAIL expert-overlap: {error}", file=sys.stderr)
        sys.exit(1)
