# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
"""Oracle and comparison side of bench/embed-gate.sh (the bge-small embedder, serve/src/embed.rs).

  embed_gate.py onnx MODEL_DIR TEXTS.jsonl MAX_LEN OUT.f32 NTOK.txt   ONNX CPU vectors (CLS, L2) and the untruncated
                                                                      token count per text
  embed_gate.py json SERVED.json OUT.f32                               OpenAI response -> rows of f32; dims 384 and
                                                                      unit norm checked
  embed_gate.py cmp A.f32 B.f32 ROWS                                   min and mean cosine over ROWS (comma list or all)
  embed_gate.py eligible TEXTS.jsonl NTOK.txt MAXTOK LINES.txt         texts brain-ask's `embed lines` can take: one
                                                                      line, not empty, at most MAXTOK tokens (it cuts at
                                                                      256); writes them to LINES.txt, prints their rows
  embed_gate.py pick A.f32 ROWS OUT.f32                                the listed rows of A
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sp in [os.environ.get("EMBED_GATE_SITE")] + glob.glob(os.path.join(HERE, "..", "..", ".venv", "lib", "python3*", "site-packages")):
    if sp and os.path.isdir(sp):
        sys.path.append(sp)


def onnx(model, texts_path, max_len, out, ntok_path):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    texts = [json.loads(line) for line in open(texts_path, encoding="utf-8")]
    full = Tokenizer.from_file(f"{model}/tokenizer.json")
    open(ntok_path, "w").write("".join(f"{len(full.encode(t).ids)}\n" for t in texts))
    tok = Tokenizer.from_file(f"{model}/tokenizer.json")
    tok.enable_truncation(max_len)
    tok.enable_padding()
    sess = ort.InferenceSession(f"{model}/onnx/model.onnx", providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    res = []
    for t in texts:
        e = tok.encode(t)
        feed = {"input_ids": np.array([e.ids], np.int64), "attention_mask": np.array([e.attention_mask], np.int64),
                "token_type_ids": np.array([e.type_ids], np.int64)}
        cls = sess.run(None, {k: v for k, v in feed.items() if k in names})[0][:, 0, :]
        res.append((cls / np.linalg.norm(cls, axis=1, keepdims=True)).astype("<f4"))
    np.concatenate(res).tofile(out)


def served(path, out):
    d = json.load(open(path))["data"]
    a = np.array([x["embedding"] for x in sorted(d, key=lambda x: x["index"])], np.float64)
    assert a.ndim == 2 and a.shape[1] == 384, f"dims {a.shape}"
    n = np.linalg.norm(a, axis=1)
    assert np.all(np.abs(n - 1) < 1e-4), f"norm range {n.min()} {n.max()}"
    a.astype("<f4").tofile(out)
    print(f"rows {len(a)} dims 384 norm {n.min():.7f}..{n.max():.7f}")


def cmp(pa, pb, rows):
    a, b = (np.fromfile(p, "<f4").reshape(-1, 384).astype(np.float64) for p in (pa, pb))
    idx = list(range(min(len(a), len(b)))) if rows == "all" else [int(x) for x in rows.split(",") if x != ""]
    assert idx, "no rows to compare"
    assert max(idx) < len(a) and max(idx) < len(b), f"shapes {a.shape} {b.shape}"
    c = (a[idx] * b[idx]).sum(1) / (np.linalg.norm(a[idx], axis=1) * np.linalg.norm(b[idx], axis=1))
    print(f"{len(idx)} {c.min():.7f} {c.mean():.7f}")


def eligible(texts_path, ntok_path, maxtok, out):
    texts = [json.loads(line) for line in open(texts_path, encoding="utf-8")]
    ntok = [int(x) for x in open(ntok_path).read().split()]
    rows = [i for i, t in enumerate(texts) if t and "\n" not in t and "\r" not in t and ntok[i] <= int(maxtok)]
    open(out, "w", encoding="utf-8").write("".join(texts[i] + "\n" for i in rows))
    print(",".join(map(str, rows)))


def pick(pa, rows, out):
    a = np.fromfile(pa, "<f4").reshape(-1, 384)
    a[[int(x) for x in rows.split(",")]].tofile(out)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"onnx": lambda: onnx(args[0], args[1], int(args[2]), args[3], args[4]), "json": lambda: served(*args), "cmp": lambda: cmp(*args), "eligible": lambda: eligible(*args), "pick": lambda: pick(*args)}[cmd]()
