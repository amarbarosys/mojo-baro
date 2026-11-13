# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
"""ONNX CPU oracle for tools/bge.mojo: first ROWS leads of leads.tsv as title + newline + lead, 256 tokens, CLS, L2 norm.
usage: bge_onnx_oracle.py MODEL_DIR LEADS ROWS OUT.f32"""
import sys

import numpy as np
import onnxruntime as ort

sys.path.append("$HOME/Projects/mojo/mojo-baro/.venv/lib/python3.14/site-packages")
from tokenizers import Tokenizer  # noqa: E402

model, leads, rows, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok = Tokenizer.from_file(f"{model}/tokenizer.json")
tok.enable_truncation(256)
tok.enable_padding()
sess = ort.InferenceSession(f"{model}/onnx/model.onnx", providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
texts = []
for line in open(leads, encoding="utf-8"):
    c = line.rstrip("\n").split("\t")
    if len(c) >= 3:
        texts.append(f"{c[1]}\n{c[2]}")
    if len(texts) == rows:
        break
res = []
for i in range(0, len(texts), 64):
    enc = tok.encode_batch(texts[i:i + 64])
    feed = {"input_ids": np.array([e.ids for e in enc], np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], np.int64)}
    cls = sess.run(None, {k: v for k, v in feed.items() if k in names})[0][:, 0, :]
    res.append((cls / np.linalg.norm(cls, axis=1, keepdims=True)).astype("<f4"))
np.concatenate(res).tofile(out)
