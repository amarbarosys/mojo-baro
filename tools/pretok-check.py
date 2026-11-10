#!/usr/bin/env python3
"""Compare serve/tokenizer.mojo's pre-tokenizer regexes with llama.cpp's (serve/pretok-table.json).

usage: tools/pretok-check.py           exit 1 on any difference for a pre type the tokenizer implements

Parses `_compile_pre` in serve/tokenizer.mojo: each `if/elif self.pre == "a" or self.pre == "b":` branch
and its `Pattern(...)` literals (the CONTR alias expanded). llama.cpp spells the contraction group as
`(?:'[sS]|'[tT]|...)` where ours uses `(?i:'s|'t|...)`; both are normalised to the latter before
comparing. Names llama.cpp has that the tokenizer lacks are listed, not failed: the tokenizer raises on
them at load time ("no pre-tokenizer regex on file"), which is loud already.
"""
import json, pathlib, re, sys
table = json.loads(pathlib.Path("serve/pretok-table.json").read_text())["names"]
src = pathlib.Path("serve/tokenizer.mojo").read_text()
body = src[src.index("def _compile_pre"):]
body = body[:body.index("\n    def ", 10)]
contr = re.search(r"comptime CONTR = r\"([^\"]+)\"", body).group(1)
ours = {}
for m in re.finditer(r"(?:if|elif) (.+?):\n((?:\s+self\.patterns\.append\(Pattern\(.*\)\)\n)+)", body):
    names = re.findall(r"self\.pre == \"([^\"]+)\"", m.group(1))
    pats = [p.replace("CONTR + r\"", "").replace("r\"", "") for p in re.findall(r"Pattern\((.*)\)\)", m.group(2))]
    pats = [contr + p[:-1] if "CONTR" in q else p[:-1] for p, q in zip(pats, re.findall(r"Pattern\((.*)\)\)", m.group(2)))]
    for n in names: ours[n] = pats
CANON = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
def norm(r):
    r = r.replace("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])", CANON)
    r = r.replace("'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])", CANON)
    r = r.replace("\r", "\\r").replace("\n", "\\n")      # C escapes in the C++ literal vs regex escapes in ours
    return r.replace('\\"', '"')                            # a quoted quote inside a class is the quote

known = {}
kf = pathlib.Path("serve/pretok-known-diffs.txt")
if kf.exists():
    for line in kf.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            n, _, why = line.partition(":"); known[n.strip()] = why.strip()
bad = 0
for n, pats in sorted(ours.items()):
    if n not in table: print(f"  ours-only {n}: llama.cpp has no pre type of that name"); continue
    theirs = [norm(r) for r in table[n]["regexes"]]; mine = [norm(p) for p in pats]
    if theirs == mine: print(f"  SAME {n} ({len(mine)} regex)")
    elif n in known: print(f"  KNOWN-DIFF {n}: {known[n]}")
    else:
        bad += 1; print(f"  DIFF {n}:")
        for i in range(max(len(theirs), len(mine))):
            a = theirs[i] if i < len(theirs) else "(none)"; b = mine[i] if i < len(mine) else "(none)"
            if a != b: print(f"    llama.cpp[{i}]: {a}\n    ours[{i}]:      {b}")
    fl = table[n]["flags"]
    if "ignore_merges" in fl: print(f"        llama.cpp flags {fl}")
uncovered = sorted(n for n in table if n not in ours and table[n]["regexes"])
print(f"pretok-check: {len(ours)} implemented, {bad} differ; {len(uncovered)} llama.cpp names not implemented (load-time error on those files)")
sys.exit(1 if bad else 0)
