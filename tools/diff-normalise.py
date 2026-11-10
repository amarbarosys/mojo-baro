#!/usr/bin/env python3
"""Rewrite a unified diff's @@ hunk counts from the hunk body.

Iteration 002 lost 4/4 candidates at `apply`, every one to the same shape: a
header declaring 7 context lines above and below while supplying 4. `patch`
rejects the hunk as malformed before any compile, so a correct edit is thrown
away for arithmetic. Counting hunk lines is not the skill under test.

This changes counts only. Content, order, and the +/-/context classification of
every line are untouched, so a hunk that would have applied still applies to the
same place; a hunk that describes a change to nonexistent code still fails, just
at compile instead of at parse. Line NUMBERS are left alone: `patch` locates a
hunk by its context and reports an offset, which is the mechanism that already
tolerates the region-slice's absolute numbering.

Empty lines inside a hunk are read as context. Models routinely emit a bare
newline where the format wants " \\n", and trailing whitespace is not the skill
under test either.

    tools/diff-normalise.py IN.diff OUT.diff   -> prints what it changed
"""
from __future__ import annotations

import re
import sys

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def normalise(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    out: list[str] = []
    notes: list[str] = []

    i = 0
    while i < len(lines):
        m = HUNK.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        old_start, old_len, new_start, new_len, tail = m.groups()
        header_i = i
        i += 1

        body: list[str] = []
        while i < len(lines):
            l = lines[i]
            if HUNK.match(l) or l.startswith(("--- ", "+++ ", "diff ", "index ")):
                break
            body.append(l)
            i += 1

        # Trailing blank lines belong to the fence, not the hunk: a hunk that
        # ends in context does not need them, and counting them would push the
        # hunk past the end of the file.
        while body and body[-1].strip() == "":
            body.pop()

        n_old = n_new = 0
        fixed_body: list[str] = []
        for l in body:
            if l.startswith("\\"):          # "\ No newline at end of file"
                fixed_body.append(l)
                continue
            if l == "":                      # bare newline where " " was meant
                l = " "
            c = l[0]
            if c == "-":
                n_old += 1
            elif c == "+":
                n_new += 1
            else:
                n_old += 1
                n_new += 1
            fixed_body.append(l)

        declared = (int(old_len) if old_len is not None else 1,
                    int(new_len) if new_len is not None else 1)
        if declared != (n_old, n_new):
            notes.append(f"line {header_i + 1}: declared -{declared[0]},+{declared[1]} "
                         f"-> counted -{n_old},+{n_new}")
        out.append(f"@@ -{old_start},{n_old} +{new_start},{n_new} @@{tail}")
        out.extend(fixed_body)

    return "\n".join(out) + "\n", notes


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    fixed, notes = normalise(text)
    open(dst, "w").write(fixed)
    for n in notes:
        print(n)
    print(f"hunks rewritten: {len(notes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
