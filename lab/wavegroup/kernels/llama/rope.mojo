# Llama rope as llama.cpp runs a GGUF (rope type NORM): rotate adjacent pairs (2i, 2i+1), theta = pos * base^(-2i/HD) / ff[i],
# ff = the GGUF tensor rope_freqs (Llama 3 long-context scaling). Rows r*NH+h are at position pos0+r. Block HD/2.
from max.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout
from bexp import cos, sin, exp, log
from attn import kv_tab_off, KVT
from model import HD, NQH, NKVH, FREQ_BASE

comptime f32 = DType.float32

@always_inline
def rot[XL: TensorLayout, FL: TensorLayout](X: TileTensor[f32, XL, MutAnyOrigin], FF: TileTensor[f32, FL, MutAnyOrigin], row: Int, pos: Int, i: Int) -> Tuple[Float32, Float32]:
    var theta = Float32(pos) * exp(Float32(-2 * i) / Float32(HD) * log(FREQ_BASE)) / rebind[Float32](FF[i])
    var x0 = rebind[Float32](X[row, 2 * i])
    var x1 = rebind[Float32](X[row, 2 * i + 1])
    return (x0 * cos(theta) - x1 * sin(theta), x0 * sin(theta) + x1 * cos(theta))

def wg_rope_norm[XL: TensorLayout, FL: TensorLayout](X: TileTensor[f32, XL, MutAnyOrigin], FF: TileTensor[f32, FL, MutAnyOrigin], pos0: Int32, nh: Int32):
    """grid (NH, T): rotate X in place."""
    var i = Int(thread_idx.x)
    var row = Int(block_idx.y) * Int(nh) + Int(block_idx.x)
    var y = rot(X, FF, row, Int(pos0) + Int(block_idx.y), i)
    X[row, 2 * i] = rebind[X.ElementType](y[0])
    X[row, 2 * i + 1] = rebind[X.ElementType](y[1])

def wg_rope_kv[NAT: Int, QL: TensorLayout, KL: TensorLayout, FL: TensorLayout](Q: TileTensor[f32, QL, MutAnyOrigin], K: TileTensor[f32, KL, MutAnyOrigin],
        V: TileTensor[f32, KL, MutAnyOrigin], FF: TileTensor[f32, FL, MutAnyOrigin], Kc: Pointer[Scalar[KVT], MutAnyOrigin],
        Vc: Pointer[Scalar[KVT], MutAnyOrigin], tab: Pointer[Int32, MutAnyOrigin], pos0: Int32, att_i: Int32):
    """grid (NQH + NKVH, T): q heads rotate in place; k heads rotate into the paged cache, v is copied beside them."""
    comptime assert KVT == f32, "wg_rope_kv writes an f32 cache"
    var i = Int(thread_idx.x)
    var r = Int(block_idx.y)
    var h = Int(block_idx.x)
    if h < NQH:
        var y = rot(Q, FF, r * NQH + h, Int(pos0) + r, i)
        Q[r * NQH + h, 2 * i] = rebind[Q.ElementType](y[0])
        Q[r * NQH + h, 2 * i + 1] = rebind[Q.ElementType](y[1])
    else:
        var row = r * NKVH + h - NQH
        var y = rot(K, FF, row, Int(pos0) + r, i)
        var off = kv_tab_off[NAT](tab, Int(pos0) + r, Int(att_i), h - NQH) + 2 * i
        Kc[off] = rebind[Scalar[KVT]](y[0])
        Kc[off + 1] = rebind[Scalar[KVT]](y[1])
        Vc[off] = rebind[Scalar[KVT]](rebind[Float32](V[row, 2 * i]))
        Vc[off + 1] = rebind[Scalar[KVT]](rebind[Float32](V[row, 2 * i + 1]))
