#!/usr/bin/env python3
"""Paired greedy MoE ngram feasibility gate; run inside gpu-wait.

Usage: python bench/moe-ngram.py CONTROL CANDIDATE PACK OUT [--smoke]
The control is a clean HEAD build. Candidate spec-off/on requests alternate.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


def main():
    control, candidate, pack, out = map(Path, sys.argv[1:5])
    smoke = "--smoke" in sys.argv[5:]
    out.mkdir(parents=True, exist_ok=False)
    if not os.environ.get("GPU_WAITING_ROOM_JOB"):
        raise RuntimeError("run through gpu-wait")
    paths = sorted(Path("bench/mtp-prompts").glob("p*.tokens"))
    if len(paths) != 20:
        raise RuntimeError("expected 20 prompt fixtures")
    if smoke:
        paths = paths[:2]
    prompts = [list(map(int, p.read_text().split())) for p in paths]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (control, candidate)]
    if hashes[0] == hashes[1]:
        raise RuntimeError("control and candidate binaries are identical")
    env = dict(os.environ, BARO_SERVE="1", BARO_PACK=str(pack.resolve()),
               BARO_SPEC="0", BARO_NGRAM="0", BARO_MEGA="0", BARO_CKPT="0",
               BARO_SPEC_K="2", BARO_SPEC_DBG="1", BARO_PROFILE="0")
    results = {}
    coverage = {"reject_first": 0, "reject_later": 0, "accept_all": 0}
    receipts = {"control_sha256": hashes[0], "candidate_sha256": hashes[1],
                "pack": str(pack.resolve()), "prompts": [p.name for p in paths],
                "smoke": smoke, "k": 2, "n": 64}
    (out / "arms.json").write_text(json.dumps(receipts, indent=2) + "\n")
    for label, engine in (("control", control), ("candidate", candidate)):
        requests = []
        # Unscored warmup, then identical 64-token requests in paired order.
        requests.append(dict(id=9999, prompt=prompts[0], n=8, spec=False, temperature=0))
        for i, prompt in enumerate(prompts):
            arms = (False,) if label == "control" else ((False, True) if i % 2 == 0 else (True, False))
            for spec in arms:
                requests.append(dict(id=2 * i + int(spec), prompt=prompt, n=64,
                                     spec=spec, temperature=0, top_p=1.0))
        wire = "".join(json.dumps(r) + "\n" for r in requests)
        (out / f"{label}.requests.jsonl").write_text(wire)
        with (out / f"{label}.out").open("w") as stdout, (out / f"{label}.err").open("w") as stderr:
            subprocess.run([str(engine.resolve())], input=wire, text=True, env=env,
                           stdout=stdout, stderr=stderr, check=True, timeout=300)
        text = (out / f"{label}.out").read_text()
        rows, tokens = {}, {}
        for line in text.splitlines():
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if "error" in row:
                raise RuntimeError(f"{label} engine error: {row}")
            rid = row.get("id")
            if "tok" in row:
                tokens.setdefault(rid, []).append(row["tok"])
            if row.get("done"):
                rows[rid] = row
        for request in requests:
            rid = request["id"]
            if rid not in rows or len(tokens.get(rid, [])) != request["n"]:
                raise RuntimeError(f"{label} request {rid}: missing completion or tokens")
            if request["spec"] and rows[rid].get("draft_kind") != "ngram":
                raise RuntimeError("ngram arm was not enabled")
            if not request["spec"] and rows[rid].get("drafted", 0):
                raise RuntimeError("control speculated")
        if label == "candidate":
            for m, accepted in re.findall(r"win pos=\d+ m=(\d+) n_acc=(\d+)", text):
                m, accepted = int(m), int(accepted)
                key = "accept_all" if accepted == m - 1 else "reject_first" if accepted == 0 else "reject_later"
                coverage[key] += 1
        results[label] = (rows, tokens)
    cr, ct = results["control"]
    nr, nt = results["candidate"]
    for i in range(len(prompts)):
        if ct[2 * i] != nt[2 * i] or ct[2 * i] != nt[2 * i + 1]:
            raise RuntimeError(f"token parity failed: {paths[i].name}, logs in {out}")
    report = {"identity_prompts": len(prompts), "coverage": coverage, "arms": {}}
    for label, rows in (("control", [cr[2*i] for i in range(len(prompts))]),
                        ("candidate_off", [nr[2*i] for i in range(len(prompts))]),
                        ("ngram", [nr[2*i+1] for i in range(len(prompts))])):
        rates = [r["tok_s"] for r in rows]
        report["arms"][label] = dict(median_tok_s=statistics.median(rates), min_tok_s=min(rates), max_tok_s=max(rates))
    spec = [nr[2*i+1] for i in range(len(prompts))]
    for field in ("drafted", "accepted", "windows", "verify_rows", "draft_s", "verify_s"):
        report[field] = sum(r[field] for r in spec)
    if report["drafted"] == 0:
        raise RuntimeError("no proposals verified; feasibility test did not exercise verifier")
    report["acceptance"] = report["accepted"] / report["drafted"]
    report["speedup"] = report["arms"]["ngram"]["median_tok_s"] / report["arms"]["candidate_off"]["median_tok_s"]
    report["verified_ms_per_window"] = 1000 * report["verify_s"] / report["windows"]
    report["verification_ms_per_emitted"] = 1000 * report["verify_s"] / (report["windows"] + report["accepted"])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL moe-ngram: {error}; logs: {sys.argv[4] if len(sys.argv) > 4 else 'arguments missing'}", file=sys.stderr)
        sys.exit(1)
