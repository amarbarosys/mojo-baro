#!/usr/bin/env python3
"""Oracle: render two chat templates with jinja2 (the reference engine) on every case and compare bytes.
usage: tools/template-equiv-jinja2.py OLD.jinja NEW.jinja [CASEDIR]   exit 1 if any case differs or errors differently"""
import json, sys
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment

def env():
    e = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def raise_exception(m): raise ValueError(m)
    e.globals["raise_exception"] = raise_exception
    e.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
    return e

def render(path, case):
    t = env().from_string(Path(path).read_text())
    try:
        return t.render(**case), None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"

old, new = sys.argv[1], sys.argv[2]
d = Path(sys.argv[3] if len(sys.argv) > 3 else "tests/template-cases")
bad = n = 0
for c in sorted(d.glob("*.json")):
    n += 1
    case = json.loads(c.read_text())
    a, ea = render(old, case); b, eb = render(new, case)
    same = (a == b and ea == eb)
    print(("same " if same else "DIFF ") + c.stem + ("" if same else f"   old_err={ea} new_err={eb}"))
    bad += not same
print(f"cases {n}, differing {bad}")
sys.exit(1 if bad else 0)
