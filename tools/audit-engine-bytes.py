#!/usr/bin/env python3
"""Count logical unique weight bytes; this is not measured HBM traffic."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pack", type=Path)
parser.add_argument("--hidden", type=int, default=4096)
args = parser.parse_args()
groups = defaultdict(int)
indexed = 0
for line in (args.pack / "index.txt").read_text().splitlines():
    name, dtype, offset, count = line.split()
    count = int(count)
    size = {"bf16": lambda n: 2 * n, "f32": lambda n: 4 * n,
            "q8": lambda n: n + n // 32 * 2,
            "q4": lambda n: n // 2 + n // 32 * 2}[dtype](count)
    assert int(offset) == indexed, "Index is not a contiguous pack"
    indexed += size
    group = ("embedding" if name == "token_embd.weight" else
             "draft_q4_output" if name.endswith(".q4draft") else
             "draft_layer" if name.startswith("blk.32.") else
             "trunk_and_output")
    groups[group] += size
assert indexed == (args.pack / "pack.bin").stat().st_size
print(json.dumps({
    "pack_bytes": indexed,
    "groups_bytes": dict(groups),
    "no_spec_unique_weight_bytes": groups["trunk_and_output"] + 2 * args.hidden,
    "assumptions": "32-layer engine; bf16 embedding row; excludes KV/state/activations, "
                   "cache effects and repeated physical loads; not HBM counters",
}, indent=2))
