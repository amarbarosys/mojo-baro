#!/usr/bin/env python3
"""Run the E8 evaluator in batches, scoring after each one.

A single 120-item run is ~51 min on one GPU (measured 2026-09-09: 25.4 s/item
across 5 arms). Batching turns that into checkpoints: if an arm errors or the
task set turns out not to discriminate, that shows up after the first batch
instead of at minute 51.

Batches are stratified so each carries the same json:math ratio as the whole
set; taking them in file order would put all 20 json items in batch 1.

Already-finished batches are skipped, so an interrupted run resumes by
re-invoking with the same --prefix.

usage:
  ./.venv/bin/python bench/e8_batch.py --batches 4
  ./.venv/bin/python bench/e8_batch.py --batches 4 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

TASKS = os.environ.get("BARO_E8_TASKS", "bench/data/e8_tasks.json")
ARMS = "0,T,L8-raw,L8-soft,L32-soft"


def stratified_batches(tasks, n):
    by_type = defaultdict(list)
    for t in tasks:
        by_type[t["type"]].append(t["id"])
    batches = [[] for _ in range(n)]
    for _, ids in sorted(by_type.items()):
        for i, tid in enumerate(ids):
            batches[i % n].append(tid)
    return batches


def load_items(path):
    with open(path) as f:
        return json.load(f)["items"]


def tally(items):
    """Per-arm exact-correct counts, recomputed with the scorer's own logic."""
    sys.path.insert(0, "bench")
    import e8_score

    per = defaultdict(lambda: [0, 0])
    for it in items:
        for a in it["arms"]:
            exact, _subset, _note = e8_score.score_arm(it, a)
            per[a["arm"]][1] += 1
            if exact:
                per[a["arm"]][0] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--prefix", default=f"results/e8/round4-{time.strftime('%Y-%m-%d')}")
    ap.add_argument("--arms", default=ARMS)
    # Env wins over the old hardcoded defaults: E12-long set BARO_E8_TMAX and
    # this script silently reset it to 1088 (rc=250 on the first 8k item).
    ap.add_argument("--ans-max", default=os.environ.get("BARO_E8_ANS_MAX", "512"))
    ap.add_argument("--tmax", default=os.environ.get("BARO_E8_TMAX", "1088"))
    ap.add_argument("--stop-after", type=int, default=0,
                    help="run at most N batches this invocation (0 = all); re-invoke with the same --prefix to continue, finished batches are skipped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(TASKS) as f:
        tasks = json.load(f)
    batches = stratified_batches(tasks, args.batches)

    print(f"{len(tasks)} tasks -> {args.batches} batches "
          f"({', '.join(str(len(b)) for b in batches)} items), arms {args.arms}")
    for i, b in enumerate(batches, 1):
        kinds = defaultdict(int)
        for tid in b:
            kinds[tid.split("_")[0]] += 1
        print(f"  batch {i}: {len(b)} items ({dict(kinds)})")
    if args.dry_run:
        return

    env = dict(os.environ, BARO_E8_ANS_MAX=args.ans_max, BARO_E8_TMAX=args.tmax)
    all_items = []
    t_start = time.time()

    for i, ids in enumerate(batches, 1):
        out = f"{args.prefix}-b{i:02d}"
        raw = f"{out}.raw.json"
        if os.path.exists(raw):
            print(f"\n=== batch {i}/{len(batches)}: already done, skipping ===")
        else:
            print(f"\n=== batch {i}/{len(batches)}: {len(ids)} items ===", flush=True)
            t0 = time.time()
            r = subprocess.run(
                ["bash", "bench/latent-handoff.sh", "--ids", ",".join(ids),
                 "--arms", args.arms, "--out", out],
                env=env,
            )
            if r.returncode != 0 or not os.path.exists(raw):
                print(f"batch {i} FAILED (rc={r.returncode}); stopping", file=sys.stderr)
                sys.exit(1)
            print(f"batch {i} took {(time.time()-t0)/60:.1f} min", flush=True)

        all_items += load_items(raw)
        per = tally(all_items)
        done = len(all_items)
        print(f"--- running tally after batch {i}: {done} items ---")
        for arm in args.arms.split(","):
            if arm in per:
                ok, n = per[arm]
                print(f"    {arm:10s} {ok:3d}/{n:<3d}  {100.0*ok/n:5.1f}%")
        spread = [per[a][0] for a in args.arms.split(",") if a in per]
        if spread:
            print(f"    spread across arms: {max(spread)-min(spread)} items")

        if args.stop_after and i >= args.stop_after:
            remaining = len(batches) - i
            print(f"\nstopping after batch {i} as asked; {remaining} batches left.")
            print(f"continue with: ./.venv/bin/python bench/e8_batch.py "
                  f"--batches {args.batches} --prefix {args.prefix}")
            return

    merged = f"{args.prefix}-merged"
    with open(f"{args.prefix}-b01.raw.json") as f:
        head = json.load(f)
    head["items"] = all_items
    with open(f"{merged}.raw.json", "w") as f:
        json.dump(head, f)
    subprocess.run(["python3", "bench/e8_score.py", f"{merged}.raw.json", TASKS,
                    f"{merged}.json", f"{merged}.md"], check=True)
    print(f"\ntotal {(time.time()-t_start)/60:.1f} min")
    print(f"merged: {merged}.json {merged}.md")


if __name__ == "__main__":
    main()
