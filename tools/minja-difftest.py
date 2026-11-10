#!/usr/bin/env python3
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
"""Gate: .work/minja-cli must reproduce tests/minja-corpus/expected/<template>/<case>.txt
byte-for-byte for every template x case, and must fail where jinja2 raised
(<case>.error). MINJA_NOW is pinned to the corpus clock. Exit 1 on any diff.
Usage: tools/difftest.py [--cli .work/minja-cli] [--only TEMPLATE] [--negative]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/minja-corpus"
FIXED_NOW = "1788868800"  # 2026-09-08 12:00:00 UTC, matches tools/minja-corpus.py


def render(cli, tpl_path, case, info):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False)
        path = f.name
    env = dict(os.environ, MINJA_NOW=FIXED_NOW)
    r = subprocess.run([cli, str(tpl_path), path, info["bos_token"], info["eos_token"], info["pad_token"]],
                       capture_output=True, env=env)
    os.unlink(path)
    out = r.stdout.decode("utf-8", errors="replace")
    if r.returncode != 0:
        return None, (out + r.stderr.decode("utf-8", errors="replace")).strip()[-400:]
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=str(ROOT / ".work/minja-cli"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--negative", action="store_true", help="flip enable_thinking in the oracle files' lookup: must FAIL")
    a = ap.parse_args()
    cases = json.loads((CORPUS / "cases.json").read_text())
    templates = json.loads((CORPUS / "templates.json").read_text())
    fails = total = 0
    for name, info in templates.items():
        if a.only and a.only not in name:
            continue
        bad = []
        for cname, case in cases.items():
            case = dict(case)
            case.setdefault("add_generation_prompt", True)
            exp_txt = CORPUS / "expected" / name / f"{cname}.txt"
            exp_err = CORPUS / "expected" / name / f"{cname}.error"
            if a.negative and cname in ("think_on", "think_off"):
                exp_txt = CORPUS / "expected" / name / ("think_off.txt" if cname == "think_on" else "think_on.txt")
            out, err = render(a.cli, CORPUS / f"{name}.jinja", case, info)
            total += 1
            if exp_err.exists():
                if err is None:
                    bad.append((cname, f"expected error ({exp_err.read_text()[:60]}), got render"))
                continue
            exp = exp_txt.read_text()
            if err is not None:
                bad.append((cname, f"engine error: {err}"))
            elif out != exp:
                i = next((k for k in range(min(len(out), len(exp))) if out[k] != exp[k]), min(len(out), len(exp)))
                bad.append((cname, f"diff at {i}: expected {exp[max(0,i-30):i+40]!r} got {out[max(0,i-30):i+40]!r}"))
        fails += len(bad)
        print(f"{name:40s} {'OK' if not bad else 'FAIL ' + str(len(bad))}")
        for cname, why in bad[:4]:
            print(f"    {cname}: {why}")
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {total} renders, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
