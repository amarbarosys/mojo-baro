#!/usr/bin/env python3
"""The byte-level tokenizer serve/front/fakes.mojo `FakeText` implements, as an HF tokenizer.json.

Byte b is token b + 3; <pad>=0 <bos>=1 <eos>=2 <|im_start|>=259 <|im_end|>=260; add_special prepends <bos>.
No tokenizer-meta.json chat template, so the Rust front renders its ChatML default (what FakeText renders).
usage: make_tokenizer.py OUTDIR
"""
import json
import os
import sys


def byte_chars():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def added(i, content):
    return {"id": i, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": True}


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
chars = byte_chars()
vocab = {chars[b]: b + 3 for b in range(256)}
specials = [(0, "<pad>"), (1, "<bos>"), (2, "<eos>"), (259, "<|im_start|>"), (260, "<|im_end|>")]
tok = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [added(i, c) for i, c in specials],
    "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                 {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {"<bos>": {"id": "<bos>", "ids": [1], "tokens": ["<bos>"]}},
    },
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
              "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
              "vocab": vocab, "merges": []},
}
json.dump(tok, open(os.path.join(out, "tokenizer.json"), "w"), ensure_ascii=False)
json.dump({"eos_token_id": 2, "bos_token_id": 1}, open(os.path.join(out, "tokenizer-meta.json"), "w"))
os.makedirs(os.path.join(out, "pack"), exist_ok=True)
open(os.path.join(out, "pack", "README"), "w").write("fake pack for the HTTP oracle\n")
