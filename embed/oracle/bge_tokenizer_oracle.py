# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
import sys
sys.path.append("$HOME/Projects/mojo/mojo-baro/.venv/lib/python3.14/site-packages")

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(sys.argv[1])
tokenizer.enable_truncation(max_length=256)
with open(sys.argv[2], encoding="utf-8") as leads:
    for i, line in enumerate(leads):
        if i == 20000:
            break
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 3:
            continue
        ids = tokenizer.encode(cols[1] + "\n" + cols[2]).ids
        print(" ".join(map(str, ids)))
