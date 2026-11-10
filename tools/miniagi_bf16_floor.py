"""Per-group bf16 rounding floor: max|fp32 grad - bf16-GEMM grad| / max|fp32 grad|, from two oracle cases.

Usage: python -P tools/miniagi_bf16_floor.py D CASE_FP32 CASE_BF16   (prints `group floor` lines)
"""
import sys

import numpy as np

D, a, b = int(sys.argv[1]), *(np.fromfile(f"{p}/grad.f32", np.float32) for p in sys.argv[2:4])
V, FF = 32, D * 43 // 16
LAYER = 2 * D + 4 * D * D + 3 * D * FF
GF = V * D + 2 * LAYER
for n, (s, e) in dict(embed=(0, V * D), layer0=(V * D, V * D + LAYER), layer1=(V * D + LAYER, GF), final_gain=(GF, GF + D),
                      pause=(GF + D, len(a))).items():
    print(n, np.abs(a[s:e] - b[s:e]).max() / np.abs(a[s:e]).max())
