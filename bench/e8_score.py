#!/usr/bin/env python3
"""E8 HARNESS scoring oracle (exchange/e8-lane-plan-2026-09-09.md, item HARNESS).

Reads the raw dump the Mojo binary writes (bench_latent_handoff.mojo: per item,
per arm -- generated ids, decoded answer_text, `scored_text` (answer_text with
<think>...</think> and ``` fences stripped, and for json tasks the first
balanced {...} object extracted -- computed once in Mojo, reused here so the
two languages never derive different "the answer" strings) and, for json
tasks, the schema-valid flag computed in Mojo via grammar/'s Automaton/
Matcher on scored_text) plus bench/data/e8_tasks.json (expected answers), and
fills in per arm:
  correct_exact  -> math: last integer in scored_text == expected
                    json: parsed scored_text == expected (exact dict match)
  correct_subset -> math: same as correct_exact (no subset concept for a scalar)
                    json: every expected key present in parsed with an equal
                    value; extra keys in parsed are ignored
Neither correctness field depends on schema_valid (round 2: schema_valid is
strict -- e.g. rejects an extra key the answer got right, or non-compact
JSON -- so it stays a separate, still-reported signal; the coordinator picks
which of {schema_valid, correct_exact, correct_subset} the gate uses).
Never launches the GPU; never recomputes the Mojo side's stripping or schema
check.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

LAST_INT_RE = re.compile(r"-?\d+")


def extract_last_int(text):
    matches = LAST_INT_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_json_obj(text):
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def subset_match(parsed, expected):
    if not isinstance(parsed, dict) or not isinstance(expected, dict):
        return parsed == expected
    for k, v in expected.items():
        if k not in parsed or parsed[k] != v:
            return False
    return True


def score_arm(task, arm):
    if arm.get("error"):
        return False, False, "arm error: " + arm["error"]
    scored = arm.get("scored_text", "")
    if task["type"] == "math":
        got = extract_last_int(scored)
        want = task["expected"]
        if got is None:
            return False, False, "no integer found in scored_text"
        ok = got == want
        reason = "" if ok else f"got {got} want {want}"
        return ok, ok, reason
    if task["type"] == "json":
        parsed = parse_json_obj(scored)
        if parsed is None:
            return False, False, "scored_text did not parse as JSON"
        exact = parsed == task["expected"]
        subset = subset_match(parsed, task["expected"])
        reason = "" if exact else f"parsed {parsed!r} vs expected {task['expected']!r}"
        return exact, subset, reason
    if task["type"] == "ruler":
        # RULER string_match_all: every answer a case-insensitive substring.
        low = scored.lower()
        missing = [a for a in task["expected"] if a.lower() not in low]
        ok = not missing
        return ok, ok, "" if ok else f"missing {missing!r}"
    return False, False, "unknown task type " + task["type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_json")
    ap.add_argument("tasks_json")
    ap.add_argument("out_json")
    ap.add_argument("out_md")
    ap.add_argument("--vram-before", type=int, default=None)
    ap.add_argument("--vram-after", type=int, default=None)
    ap.add_argument("--topology2-status", default=None)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw_json).read_text())
    tasks = {t["id"]: t for t in json.loads(Path(args.tasks_json).read_text())}

    arm_names = []
    per_arm_exact = {}
    per_arm_subset = {}
    per_arm_total = {}
    per_arm_producer_s = {}
    per_type_arm_exact = {}
    per_type_arm_subset = {}
    per_type_arm_total = {}

    for item in raw["items"]:
        task = tasks.get(item["id"])
        if task is None:
            print("WARNING: item", item["id"], "not found in", args.tasks_json, file=sys.stderr)
            continue
        for arm in item["arms"]:
            name = arm["arm"]
            if name not in arm_names:
                arm_names.append(name)
                per_arm_exact[name] = 0
                per_arm_subset[name] = 0
                per_arm_total[name] = 0
                per_arm_producer_s[name] = []
            exact, subset, reason = score_arm(task, arm)
            arm["correct_exact"] = exact
            arm["correct_subset"] = subset
            arm["score_reason"] = reason
            per_arm_total[name] += 1
            if exact:
                per_arm_exact[name] += 1
            if subset:
                per_arm_subset[name] += 1
            if arm.get("producer_s"):
                per_arm_producer_s[name].append(arm["producer_s"])
            key = (task["type"], name)
            per_type_arm_total[key] = per_type_arm_total.get(key, 0) + 1
            if exact:
                per_type_arm_exact[key] = per_type_arm_exact.get(key, 0) + 1
            if subset:
                per_type_arm_subset[key] = per_type_arm_subset.get(key, 0) + 1

    raw["vram_before_bytes"] = args.vram_before
    raw["vram_after_bytes"] = args.vram_after
    raw["topology2_status"] = args.topology2_status

    Path(args.out_json).write_text(json.dumps(raw, indent=2))

    types = sorted({t["type"] for t in tasks.values()})
    lines = []
    lines.append(f"# E8 HARNESS results -- {raw.get('topology')}")
    lines.append("")
    lines.append(f"pack: `{raw.get('pack')}`  transport: {raw.get('transport')}")
    if args.vram_before is not None and args.vram_after is not None:
        lines.append(
            f"VRAM before both loads: {args.vram_before / 1e9:.2f} GB, "
            f"after both loads: {args.vram_after / 1e9:.2f} GB "
            f"(delta {(args.vram_after - args.vram_before) / 1e9:.2f} GB)"
        )
    if args.topology2_status:
        lines.append(f"Topology 2: {args.topology2_status}")
    lines.append("")
    lines.append("## Accuracy per arm per task type")
    lines.append("")
    lines.append(
        "json cells show `exact/subset` (`correct_exact`/`correct_subset`, "
        "round 2 defect 2); other task types have no subset concept so "
        "`correct_exact == correct_subset` and the cell shows one count."
    )
    lines.append("")
    header = "| arm | " + " | ".join(types) + " | overall (exact/subset) |"
    lines.append(header)
    lines.append("|---|" + "---|" * (len(types) + 1))
    for name in arm_names:
        row = [name]
        for t in types:
            key = (t, name)
            n = per_type_arm_total.get(key, 0)
            if not n:
                row.append("-")
                continue
            ec = per_type_arm_exact.get(key, 0)
            sc = per_type_arm_subset.get(key, 0)
            row.append(f"{ec}/{n} / {sc}/{n}" if t == "json" else f"{ec}/{n}")
        on = per_arm_total[name]
        if on:
            row.append(f"{per_arm_exact[name]}/{on} / {per_arm_subset[name]}/{on}")
        else:
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Producer time (median, s)")
    lines.append("")
    lines.append("| arm | median producer_s | n |")
    lines.append("|---|---|---|")
    for name in arm_names:
        vals = per_arm_producer_s[name]
        med = f"{statistics.median(vals):.3f}" if vals else "-"
        lines.append(f"| {name} | {med} | {len(vals)} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("wrote", args.out_json, "and", args.out_md)


if __name__ == "__main__":
    main()
