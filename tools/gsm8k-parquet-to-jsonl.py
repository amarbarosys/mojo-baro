#!/usr/bin/env python3
"""Convert the GSM8K parquet shards to jsonl so Mojo can read them.

Parquet needs pyarrow, which has no Mojo reader, so this is the one Python step
in the E8 task pipeline. It runs once per dataset download; the generator that
consumes the output is tools/generate_e8_tasks.mojo.

Usage: ./.venv/bin/python tools/gsm8k-parquet-to-jsonl.py
"""
import json
import os
import sys

import pandas as pd

BASE = os.path.expanduser(os.environ.get("E8_GSM8K_DIR", "~/Models/datasets/gsm8k/main"))


def main():
    total = 0
    for split in ("test", "train"):
        src = os.path.join(BASE, f"{split}-00000-of-00001.parquet")
        if not os.path.exists(src):
            print(f"missing {src}", file=sys.stderr)
            raise SystemExit(1)
        dst = os.path.join(BASE, f"{split}.jsonl")
        df = pd.read_parquet(src)
        with open(dst, "w") as f:
            for q, a in zip(df["question"], df["answer"]):
                # one object per line, no embedded newlines: the Mojo reader
                # splits on "\n" before parsing
                f.write(json.dumps({"question": q, "answer": a}) + "\n")
        print(f"{split}: {len(df)} rows -> {dst}")
        total += len(df)
    print(f"{total} rows converted")


if __name__ == "__main__":
    main()
