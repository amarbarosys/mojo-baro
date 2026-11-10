#!/usr/bin/env python3
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
"""Pre-tokenizer coverage harvest: top-N HF text-generation models vs this repo's families.

Downloads tokenizer.json only (no weights) for the top N models by downloads, runs
each model's own pre-tokenizer (HF `tokenizers`, the oracle) on a corpus sample, and
finds the mojo-baro family (serve/tokenizer.mojo `_compile_pre`) whose uregex pattern
chain splits the sample identically, using Python `regex` with split_keep semantics.
Output: .work/pretok/models.tsv (repo, model type, pre-tokenizer kinds, family|NEW|...)
and a summary on stdout. Usage: tools/pretok_harvest.py [--n 250] [--baro PATH]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import regex
from huggingface_hub import hf_hub_download, list_models
from tokenizers import Tokenizer

OUT = Path(".work/pretok")
CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"


def baro_families(path):
    """{family name: [pattern, ...]} from the if/elif chain in _compile_pre."""
    src = Path(path).read_text().split("def _compile_pre", 1)[1].split("def _pieces", 1)[0]
    fams, cur = {}, None
    for line in src.splitlines():
        names = re.findall(r'self\.pre == "([^"]+)"', line)
        if names:
            cur = names[0]
            fams[cur] = {"aliases": names, "patterns": []}
        m = re.search(r'Pattern\((CONTR \+ )?r"(.*)"\)\)\s*$', line)
        if m and cur:
            fams[cur]["patterns"].append((CONTR if m.group(1) else "") + m.group(2).replace('\\"', '"'))
    return fams


def split_keep(pieces, pat):
    out = []
    for s in pieces:
        pos = 0
        for m in pat.finditer(s):
            if m.start() == m.end():
                continue
            if m.start() > pos:
                out.append(s[pos:m.start()])
            out.append(m.group())
            pos = m.end()
        if pos < len(s):
            out.append(s[pos:])
    return out


def kinds(pt):
    if pt is None:
        return ["none"]
    if pt["type"] == "Sequence":
        return [k for p in pt["pretokenizers"] for k in kinds(p)]
    if pt["type"] == "Split":
        return ["Split:regex" if "Regex" in pt["pattern"] else "Split:literal"]
    return [pt["type"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--baro", default="serve/tokenizer.mojo")
    ap.add_argument("--corpus", default="../mojo-uregex/.work/bench/corpus.txt")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fams = baro_families(a.baro)
    compiled = {k: [regex.compile(p) for p in v["patterns"]] for k, v in fams.items() if v["patterns"]}
    lines = [l for l in Path(a.corpus).read_text().splitlines() if l][:400]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte_dec = {chr(c): b for b, c in zip(bs, cs)}
    fam_pieces = {}
    for k, pats in compiled.items():
        fam_pieces[k] = []
        for l in lines:
            p = [l]
            for pat in pats:
                p = split_keep(p, pat)
            fam_pieces[k].append(p)

    rows, new_specs = [], {}
    models = list(list_models(pipeline_tag="text-generation", sort="downloads", limit=a.n))
    for i, m in enumerate(models):
        repo = m.id
        try:
            f = hf_hub_download(repo, "tokenizer.json", cache_dir=str(OUT / "hf"))
        except Exception as e:
            rows.append((repo, "-", "-", "NO_TOKENIZER_JSON:" + type(e).__name__)); continue
        try:
            spec = json.load(open(f))
            tok = Tokenizer.from_file(f)
        except Exception as e:
            rows.append((repo, "-", "-", "LOAD_FAIL:" + type(e).__name__)); continue
        mtype = spec.get("model", {}).get("type", "?")
        k = kinds(spec.get("pre_tokenizer"))
        if spec.get("pre_tokenizer") is None or all(x in ("Metaspace", "none", "WhitespaceSplit", "Split:literal") for x in k):
            rows.append((repo, mtype, "+".join(k), "NO_REGEX")); continue
        pt = tok.pre_tokenizer
        bytelevel = "ByteLevel" in k
        oracle = []
        for l in lines:
            ps = [p for p, _ in pt.pre_tokenize_str(l)]
            if bytelevel:
                ps = [bytes(byte_dec.get(c, 63) for c in p).decode("utf-8", "replace") for p in ps]
            oracle.append(ps)
        hit = next((name for name, fp in fam_pieces.items() if fp == oracle), None)
        sig = hashlib.sha1(json.dumps(spec["pre_tokenizer"], sort_keys=True).encode()).hexdigest()[:8]
        rows.append((repo, mtype, "+".join(k), hit or "NEW:" + sig))
        if not hit:
            new_specs[sig] = spec["pre_tokenizer"]
        print(f"{i + 1}/{len(models)} {repo} {rows[-1][3]}", file=sys.stderr)

    with open(OUT / "models.tsv", "w") as fh:
        fh.write("repo\tmodel\tpre_tokenizer\tfamily\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    c = Counter(r[3].split(":")[0] for r in rows)
    print(f"models {len(rows)}")
    for k, v in c.most_common():
        print(f"{v:4d}  {k}")
    print(f"model types: {dict(Counter(r[1] for r in rows))}")
    json.dump(new_specs, open(OUT / "new.json", "w"), indent=1)
    print(f"distinct NEW pre-tokenizers: {len(new_specs)} (specs in {OUT / 'new.json'})")
    print(f"written to {OUT / 'models.tsv'}")


if __name__ == "__main__":
    main()
