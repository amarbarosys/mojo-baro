#!/usr/bin/env python3
"""Rewrite a qwen35moe pack's projection tensors into the dense q8 layout, in place
on a btrfs reflink of the raw pack (R6a byte split, bench/moe-persist-protocol.md R6.3).

usage: tools/moe-pack-q8d.py SRC_PACK_DIR DST_PACK_DIR

Every tensor keeps its byte length and offset, so only the 130 projection
regions are new extents (about 1.4 GB) and index.txt changes only in the dtype
column. Receipt: three tensors dequantised value-for-value against the raw pack.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("engine_pack", Path(__file__).resolve().parent / "engine-pack.py")
ep = module_from_spec(spec)
spec.loader.exec_module(ep)
MOE_DENSE_Q8 = ep.MOE_DENSE_Q8


def deq_raw(buf):
    blk = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 34)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    q = blk[:, 2:].view(np.int8).astype(np.float32)
    return d * q


def deq_dense(buf, n):
    q = np.frombuffer(buf[:n], dtype=np.int8).astype(np.float32).reshape(-1, 32)
    d = np.frombuffer(buf[n:], dtype=np.float16).astype(np.float32).reshape(-1, 1)
    return d * q


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "pack.bin").exists():
        subprocess.run(["cp", "--reflink=always", str(src / "pack.bin"), str(dst / "pack.bin")], check=True)
    lines = (src / "index.txt").read_text().splitlines()
    out_lines = []
    done = []
    with open(src / "pack.bin", "rb") as fi, open(dst / "pack.bin", "r+b") as fo:
        for line in lines:
            name, dt, off, n = line.split()
            off, n = int(off), int(n)
            if dt == "q8_0" and name.split(".")[-2] in MOE_DENSE_Q8:
                nbytes = (n // 32) * 34
                fi.seek(off)
                raw = fi.read(nbytes)
                blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
                dense = blk[:, 2:].tobytes() + blk[:, :2].tobytes()
                assert len(dense) == nbytes
                fo.seek(off)
                fo.write(dense)
                dt = "q8"
                done.append((name, off, n))
            out_lines.append(f"{name} {dt} {off} {n}")
    (dst / "index.txt").write_text("\n".join(out_lines) + "\n")
    print(f"rewrote {len(done)} tensors in place at {dst}")
    # receipt: first, middle, last rewritten tensor, value for value, read back from disk
    with open(src / "pack.bin", "rb") as fi, open(dst / "pack.bin", "rb") as fo:
        for name, off, n in (done[0], done[len(done) // 2], done[-1]):
            fi.seek(off)
            raw = fi.read((n // 32) * 34)
            fo.seek(off)
            back = fo.read((n // 32) * 34)
            a, b = deq_raw(raw), deq_dense(back, n)
            if a.shape != b.shape or not np.array_equal(a, b):
                print(f"FAIL receipt: {name} differs after the split")
                sys.exit(1)
            print(f"receipt {name}: {a.size} values bit-equal (max |v| {np.abs(a).max():.4g})")


if __name__ == "__main__":
    main()
