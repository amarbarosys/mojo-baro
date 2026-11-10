#!/usr/bin/env python3
"""Gate for docs/settings-template.toml: the template lists exactly the settings
the source reads, and each entry is documented.

Source settings: getenv("BARO_*") and get_defined_*["BARO_*"] in *.mojo under
serve/ kernels/ tools/, and env::var("BARO_*") in serve/src/*.rs.
Template: one [settings.BARO_NAME] table per setting with keys kind
("env" | "build" | "server-env"), type, default, doc (non-empty), models (list).

usage: tools/settings-check.py [TEMPLATE]   exit 0 = PASS, 1 = FAIL
"""
import re, sys, tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
template = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "docs/settings-template.toml"

found = {}
for d in ("serve", "kernels", "tools"):
    for f in (root / d).rglob("*.mojo"):
        text = f.read_text(errors="replace")
        for n in re.findall(r'getenv\("(BARO_[A-Z0-9_]+)"', text):
            found.setdefault(n, "env")
        for n in re.findall(r'get_defined_(?:string|int|bool)\["(BARO_[A-Z0-9_]+)"', text):
            found[n] = "build"
for f in (root / "serve/src").glob("*.rs"):
    for n in re.findall(r'env::var\("(BARO_[A-Z0-9_]+)"', f.read_text(errors="replace")):
        found.setdefault(n, "server-env")

if not template.exists():
    print(f"FAIL settings-check: {template} does not exist ({len(found)} settings in source)")
    sys.exit(1)
try:
    entries = tomllib.loads(template.read_text()).get("settings", {})
except tomllib.TOMLDecodeError as e:
    print(f"FAIL settings-check: {template} is not valid TOML: {e}")
    sys.exit(1)

problems = []
for n in sorted(set(found) - set(entries)):
    problems.append(f"missing from template: {n} ({found[n]})")
for n in sorted(set(entries) - set(found)):
    if entries[n].get("kind") != "external":   # read by the MAX runtime, not our source
        problems.append(f"in template but not read by source: {n}")
for n, e in sorted(entries.items()):
    if n not in found:
        continue
    for key in ("kind", "type", "default", "doc", "models"):
        if key not in e:
            problems.append(f"{n}: missing key '{key}'")
    if e.get("kind") not in (None, found[n]):
        problems.append(f"{n}: kind '{e.get('kind')}' but source reads it as '{found[n]}'")
    if "doc" in e and not str(e["doc"]).strip():
        problems.append(f"{n}: empty doc")
    if "models" in e and not isinstance(e["models"], list):
        problems.append(f"{n}: models must be a list")

if problems:
    print(f"FAIL settings-check: {len(problems)} problem(s), {len(entries)} entries vs {len(found)} in source")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print(f"PASS settings-check: {len(found)} settings, all documented")
