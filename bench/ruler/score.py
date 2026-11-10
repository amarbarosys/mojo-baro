#!/usr/bin/env python3
"""bench/ruler/score.py: RULER-style scoring over run.py's output.

Metric: string_match_all (RULER's scripts/eval/synthetic/constants.py) for
every task in this subset -- per example, the fraction of `answers` found as
a case-insensitive substring of the completion, averaged and x100.

Effective length: the largest size whose task average >= 0.85 x that task's
own 4k average (RULER's convention uses a fixed anchor, Llama2-7B's 4k score
85.6; we don't have that reference model, so we anchor on the model's own 4k
score and print both the anchor and the threshold).

Usage: bench/ruler/score.py RESPONSE_DIR [--tasks t1,t2] [--sizes 4096,...]
RESPONSE_DIR layout (written by run.py): <task>_<size>/<id>.response.txt
"""
import argparse
import json
from pathlib import Path

from gen import SIZES, TASKS


def string_match_all(pred, answers):
    if not answers:
        return 0.0
    pred_l = pred.lower()
    hits = sum(1 for a in answers if a.lower() in pred_l)
    return hits / len(answers)


def score_file(prompts_path, responses_dir):
    scores = []
    with open(prompts_path) as f:
        for line in f:
            row = json.loads(line)
            resp_path = responses_dir / f"{row['id']}.response.txt"
            pred = resp_path.read_text() if resp_path.exists() else ""
            scores.append(string_match_all(pred, row["answers"]))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("response_dir")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--json", default=None, help="write the table as JSON to this path")
    a = ap.parse_args()

    prompts_dir = Path(a.prompts)
    response_dir = Path(a.response_dir)
    tasks = a.tasks.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]

    table = {}  # task -> size -> avg*100
    for task in tasks:
        table[task] = {}
        for size in sizes:
            prompts_path = prompts_dir / f"{task}_{size}.jsonl"
            responses_subdir = response_dir / f"{task}_{size}"
            if not prompts_path.exists() or not responses_subdir.exists():
                continue
            scores = score_file(prompts_path, responses_subdir)
            table[task][size] = round(100 * sum(scores) / len(scores), 2) if scores else None

    print(f"{'task':<16}" + "".join(f"{s:>8}" for s in sizes) + "   effective_len  anchor(4k)")
    effective = {}
    for task in tasks:
        row = table.get(task, {})
        anchor = row.get(4096)
        threshold = 0.85 * anchor if anchor is not None else None
        eff = None
        for size in sizes:
            v = row.get(size)
            if v is None or threshold is None:
                continue
            if v >= threshold:
                eff = size
        effective[task] = eff
        cells = "".join(f"{row.get(s, '-'):>8}" if row.get(s) is not None else f"{'-':>8}" for s in sizes)
        print(f"{task:<16}{cells}   {eff if eff else '-':>13}  {anchor if anchor is not None else '-'}")

    out = {"table": table, "effective_length": effective, "threshold_frac": 0.85}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"written to {a.json}")


if __name__ == "__main__":
    main()
