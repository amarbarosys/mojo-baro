#!/usr/bin/env python3
"""PASS/MISS verdict for one model's quality row against its frozen band.
bench/quality-protocol.md item 2 + the 2026-09-16 amendment (spark-family
perplexity deferred). Reads bench/quality-bands.json (the machine copy of
the protocol's predicted-band table), the ppl result jsons for both arms
(optional: omitted for the 6 spark-family models this round), and the
task-eval result jsons for both arms; writes one combined result.json.
"""
import argparse
import json
import re
from pathlib import Path


def llama_ppl(log_path):
    text = Path(log_path).read_text(errors="replace")
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    if not m:
        raise ValueError(f"no PPL line found in {log_path}")
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="key into bench/quality-bands.json")
    ap.add_argument("--bands", default="bench/quality-bands.json")
    ap.add_argument("--ppl-ours", default=None)
    ap.add_argument("--ppl-llama-log", default=None)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bands = json.loads(Path(args.bands).read_text())[args.key]

    ours_task = json.loads((Path(args.task_dir) / "ours.json").read_text())
    llama_task = json.loads((Path(args.task_dir) / "llama.json").read_text())
    delta_pp = ours_task["exact_pct"] - llama_task["exact_pct"]
    delta_ok = abs(delta_pp) <= bands["delta_pp"]

    result = {
        "key": args.key,
        "model": bands["model"],
        "task_ours_exact_pct": ours_task["exact_pct"],
        "task_llama_exact_pct": llama_task["exact_pct"],
        "delta_pp": delta_pp,
        "delta_pp_band": bands["delta_pp"],
        "delta_pp_ok": delta_ok,
    }

    if args.ppl_ours and args.ppl_llama_log:
        ppl_ours = json.loads(Path(args.ppl_ours).read_text())["ppl"]
        ppl_llama, ppl_llama_err = llama_ppl(args.ppl_llama_log)
        ratio = ppl_ours / ppl_llama
        ratio_lo, ratio_hi = bands["ppl_ratio"]
        ratio_ok = ratio_lo <= ratio <= ratio_hi
        verdict = "PASS" if (ratio_ok and delta_ok) else "MISS"
        if bands.get("basis") == "none":
            verdict += "-informational"
        result.update({
            "ppl_ours": ppl_ours,
            "ppl_llama": ppl_llama,
            "ppl_llama_err": ppl_llama_err,
            "ppl_ratio": ratio,
            "ppl_ratio_band": bands["ppl_ratio"],
            "ppl_ratio_ok": ratio_ok,
            "verdict": verdict,
        })
    else:
        result.update({
            "ppl_ours": None,
            "ppl_llama": None,
            "ppl_ratio": None,
            "verdict": "perplexity pending (task eval only " + ("PASS" if delta_ok else "MISS") + ")",
        })

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
