#!/usr/bin/env python
"""CPU-only falsifier for the frozen P5a v2 draft dump format."""
import struct
from pathlib import Path

import torch

from mtp_head import H, read_v2_dump


def write_fixture(path):
    docs = [
        ([11, 12, 13, 14], [
            (1, 12, 101, [101, 102, 103, 104, 105, 106, 107, 108], [0.4, 0.2, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03], 1.25),
            (2, 13, 202, [202, 203, 204, 205, 206, 207, 208, 209], [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04], 2.5),
        ]),
        ([21, 22, 23], [
            (1, 22, 301, [301, 302, 303, 304, 305, 306, 307, 308], [0.35, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02], -3.0),
        ]),
    ]
    with path.open("wb") as f:
        f.write(struct.pack("<I", 2))
        for tokens, rows in docs:
            f.write(struct.pack("<I", len(tokens)))
            f.write(struct.pack(f"<{len(tokens)}I", *tokens))
            f.write(struct.pack("<I", len(rows)))
            for pos, input_token, argmax, ids, probs, hidden_value in rows:
                f.write(struct.pack("<IIII", pos, input_token, argmax, 8))
                f.write(struct.pack("<8I", *ids))
                f.write(struct.pack("<8f", *probs))
                f.write(struct.pack(f"<{H}f", *([hidden_value] * H)))


def main():
    path = Path(".work/team-B/codex/p5a/roundtrip-v2.bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fixture(path)
    docs = read_v2_dump(path)
    assert len(docs) == 2
    assert docs[0]["tokens"] == [11, 12, 13, 14]
    assert [r["pos"] for r in docs[0]["records"]] == [1, 2]
    assert [r["input_token"] for r in docs[0]["records"]] == [12, 13]
    assert [r["target_argmax"] for r in docs[0]["records"]] == [101, 202]
    assert docs[0]["records"][0]["top8_ids"][0].item() == 101
    assert torch.allclose(docs[0]["records"][1]["top8_probs"].sum(), torch.tensor(1.0), atol=1e-6)
    assert docs[0]["records"][0]["h"].shape == (H,)
    assert docs[0]["records"][0]["h"][0].item() == 1.25
    assert docs[1]["records"][0]["h"][0].item() == -3.0
    print(f"PASS P5a v2 roundtrip docs={len(docs)} records=3 hidden={H}")


if __name__ == "__main__":
    main()
