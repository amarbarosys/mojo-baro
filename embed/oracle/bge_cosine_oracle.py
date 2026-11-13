# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
"""Per-row cosine of two F32 row files (384 columns): prints min and mean. usage: bge_cosine_oracle.py A.f32 B.f32"""
import sys

import numpy as np

a, b = (np.fromfile(p, "<f4").reshape(-1, 384).astype(np.float64) for p in sys.argv[1:3])
assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
print(f"{c.min():.7f} {c.mean():.7f}")
