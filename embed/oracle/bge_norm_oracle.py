# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
"""Oracle side of tools/wordpiece.mojo: one row per codepoint whose BertNormalizer output (clean_text, CJK padding,
NFD accent strip, lowercase) is not the codepoint itself, or that BertPreTokenizer splits off as punctuation.
Row: codepoint TAB P (punctuation) or - TAB replacement as space-separated hex codepoints (empty = removed)."""
import sys
import unicodedata as u


def cjk(c):
    return any(a <= c <= b for a, b in ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
                                        (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)))


def norm(c):
    ch = chr(c)
    if c in (0, 0xFFFD) or (ch not in "\t\n\r" and u.category(ch) in ("Cc", "Cf", "Cn", "Co")):
        return ""
    if ch in "\t\n\r" or u.category(ch) == "Zs":
        return " "
    if cjk(c):
        return " " + ch + " "
    s = "".join(x for x in u.normalize("NFD", ch) if u.category(x) != "Mn")
    return s.lower()


def punct(c):
    return (33 <= c <= 47) or (58 <= c <= 64) or (91 <= c <= 96) or (123 <= c <= 126) or u.category(chr(c)).startswith("P")


for c in range(0x110000):
    if 0xD800 <= c <= 0xDFFF or u.category(chr(c)) in ("Cn", "Co") or cjk(c):  # unassigned / private use / CJK: wordpiece.mojo tests ranges
        continue
    n = norm(c)
    p = len(n) == 1 and punct(ord(n))
    if n != chr(c) or (p and c > 126):
        sys.stdout.write(f"{c:x}\t{'P' if p else '-'}\t{' '.join(f'{ord(x):x}' for x in n)}\n")
