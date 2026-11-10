#!/usr/bin/env python3
"""usage: gguf-geometry.py MODEL.gguf  ->  --att N --nkvh N --hd N --ext 0

The geometry flags tools/state-to-llama-slot.mojo needs for an attention-only model, read
from the GGUF header through the gguf-kv iTool (never guessed, never hardcoded)."""
import glob
import sys
from pathlib import Path

hits = glob.glob(str(Path.home() / "iTools/*/gguf-kv"))
if not hits:
    print("FAIL gguf-geometry: the gguf-kv iTool is missing (itools-doctor)")
    sys.exit(1)
sys.path.insert(0, hits[0])
from gguf_kv import read_kv  # noqa: E402

kv = read_kv(Path(sys.argv[1]))
arch = kv["general.architecture"]
g = lambda k: kv.get(f"{arch}.{k}")  # noqa: E731
att, nkvh, heads = g("block_count"), g("attention.head_count_kv"), g("attention.head_count")
hd = g("attention.key_length") or (g("embedding_length") // heads if heads else None)
if not (att and nkvh and hd):
    print(f"FAIL gguf-geometry: {arch} header lacks block_count/head_count_kv/key_length")
    sys.exit(1)
print(f"--att {att} --nkvh {nkvh} --hd {hd} --ext 0")
