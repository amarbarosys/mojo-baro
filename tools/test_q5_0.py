"""Oracle for tools/test_q5_0.mojo: gguf-py dequantize of every Q5_0 tensor, same order, compared bit for bit.
usage: python tools/test_q5_0.py MODEL.gguf MOJO_OUT.bin"""
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

r = GGUFReader(sys.argv[1])
ref = [dequantize(t.data, GGMLQuantizationType.Q5_0).astype(np.float32).ravel() for t in r.tensors if t.tensor_type == GGMLQuantizationType.Q5_0]
ref = np.concatenate(ref)
got = np.fromfile(sys.argv[2], dtype=np.float32)
if got.size != ref.size:
    sys.exit(f"FAIL q5_0: {got.size} floats, oracle {ref.size}")
bad = np.count_nonzero(got.view(np.uint32) != ref.view(np.uint32))
print(f"q5_0 {ref.size} floats, {bad} differ bit-wise")
sys.exit(1 if bad else 0)
