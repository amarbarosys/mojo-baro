# E13 piece 2: the projector P = Linear(H,H) -> GELU(tanh-approx) ->
# Linear(H,H), f32, applied host-side to producer A's k raw final-norm
# HIDDEN vectors before they cross to receiver B via
# apply_latent_to_receiver (docs/design/latent-os/06-experiments.md, E13).
#
# GELU uses the tanh approximation (not the exact erf form) so both sides of
# the parity check -- this file and tools/e13-projector-oracle.py's torch
# reference -- compute the identical closed-form function; nothing here
# depends on an erf implementation existing on either side.
#
# Weight file format (flat binary, f32 little-endian, row-major, PyTorch
# nn.Linear convention: weight shape [out_features, in_features]):
#   offset 0                :  W1 [H x H]   (fc1.weight)
#   offset H*H*4            :  b1 [H]       (fc1.bias)
#   offset (H*H+H)*4        :  W2 [H x H]   (fc2.weight)
#   offset (2*H*H+H)*4      :  b2 [H]       (fc2.bias)
#   total = (2*H*H + 2*H) * 4 bytes
from std.math import tanh
from std.memory import bitcast

comptime GELU_C: Float32 = 0.7978845608028654  # sqrt(2/pi)


def gelu_tanh(x: Float32) -> Float32:
    var x3 = x * x * x
    return 0.5 * x * (1.0 + tanh(GELU_C * (x + 0.044715 * x3)))


struct Projector(Movable):
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var h: Int

    def __init__(out self, var w1: List[Float32], var b1: List[Float32], var w2: List[Float32], var b2: List[Float32], h: Int):
        self.w1 = w1^
        self.b1 = b1^
        self.w2 = w2^
        self.b2 = b2^
        self.h = h


def bytes_to_f32_list(data: List[UInt8], n: Int) raises -> List[Float32]:
    if len(data) < n * 4:
        raise Error("bytes_to_f32_list: short read")
    var out = List[Float32](unsafe_uninit_length=n)
    for i in range(n):
        var b0 = UInt32(data[i * 4 + 0])
        var b1v = UInt32(data[i * 4 + 1])
        var b2v = UInt32(data[i * 4 + 2])
        var b3 = UInt32(data[i * 4 + 3])
        var bits = b0 | (b1v << 8) | (b2v << 16) | (b3 << 24)
        out[i] = bitcast[DType.float32, 1](bits)[0]
    return out^


def load_projector(path: String, h: Int) raises -> Projector:
    with open(path, "r") as f:
        var w1_bytes = f.read_bytes(h * h * 4)
        var w1 = bytes_to_f32_list(w1_bytes, h * h)
        var b1_bytes = f.read_bytes(h * 4)
        var b1 = bytes_to_f32_list(b1_bytes, h)
        var w2_bytes = f.read_bytes(h * h * 4)
        var w2 = bytes_to_f32_list(w2_bytes, h * h)
        var b2_bytes = f.read_bytes(h * 4)
        var b2 = bytes_to_f32_list(b2_bytes, h)
        return Projector(w1^, b1^, w2^, b2^, h)


def apply_projector(p: Projector, x: List[Float32]) -> List[Float32]:
    """x: H-dim vector. Returns W2 @ gelu_tanh(W1 @ x + b1) + b2 (nn.Linear
    convention: weight row o, cols 0..H-1, is out-feature o's weights)."""
    var h = p.h
    var mid = List[Float32](unsafe_uninit_length=h)
    for o in range(h):
        var acc: Float32 = p.b1[o]
        var row_off = o * h
        for i in range(h):
            acc += p.w1[row_off + i] * x[i]
        mid[o] = gelu_tanh(acc)
    var out = List[Float32](unsafe_uninit_length=h)
    for o in range(h):
        var acc: Float32 = p.b2[o]
        var row_off = o * h
        for i in range(h):
            acc += p.w2[row_off + i] * mid[i]
        out[o] = acc
    return out^


def apply_projector_k(p: Projector, xs: List[Float32], k: Int) -> List[Float32]:
    """xs: k*H flat (one HIDDEN vector per k-step, as collect_latent_raw
    lays them out). Returns k*H flat, each H-slice projected independently --
    this is what an L8-proj/L32-proj arm feeds to apply_latent_to_receiver."""
    var h = p.h
    var out = List[Float32](unsafe_uninit_length=k * h)
    for s in range(k):
        var xv = List[Float32](unsafe_uninit_length=h)
        for i in range(h):
            xv[i] = xs[s * h + i]
        var yv = apply_projector(p, xv)
        for i in range(h):
            out[s * h + i] = yv[i]
    return out^
