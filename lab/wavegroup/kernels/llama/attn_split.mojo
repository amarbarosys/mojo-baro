# Decode attention split over the context (flash-decoding) at HD 64: one block per (q head, span of CH tokens) runs
# mojo-baro's attn_head_span and leaves (m, l, o) in Pg; wg_attn_merge rescales the spans per head. mojo-baro's own split
# (dattn) fails to launch at HD 64, and the one-block-per-head kernel leaves most of the GPU idle at decode.
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from bexp import exp
from attn import attn_head_span, KVT
from model import HD, NQH, NKVH

comptime CH = 32 # tokens per span
comptime NSMAX = 16 # spans at 512 tokens of context
comptime PS = HD + 2 # one span's partial: m, l, o[HD]

def wg_attn_split[NAT: Int, QL: TensorLayout, CL: TensorLayout](Q: TileTensor[DType.float32, QL, MutAnyOrigin], Kc: TileTensor[KVT, CL, MutAnyOrigin],
        Vc: TileTensor[KVT, CL, MutAnyOrigin], tab: Pointer[Int32, MutAnyOrigin], pg: Pointer[Float32, MutAnyOrigin], T: Int32, scale: Float32, att_i: Int32):
    """grid (NQH, ceil(T / CH)), block HD."""
    var h = Int(block_idx.x)
    var sp = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var qs = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[HD]())
    var sc = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var r = attn_head_span[NAT=NAT](Q, Kc, Vc, rebind[MutPointer[Scalar[DType.int32], MutAnyOrigin]](tab), qs, sc, red, h, h // (NQH // NKVH),
        sp * CH, min(Int(T), sp * CH + CH), tid, Int(lane_id()), scale, Int(att_i))
    var p = (h * NSMAX + sp) * PS
    if tid == 0:
        pg[p] = r[0]
        pg[p + 1] = r[1]
    pg[p + 2 + tid] = r[2]

def wg_attn_merge[OL: TensorLayout](pg: Pointer[Float32, MutAnyOrigin], O: TileTensor[DType.float32, OL, MutAnyOrigin], ns: Int32):
    """grid NQH, block HD: o = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s."""
    var h = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var m = Float32(-3.4e38)
    for s in range(Int(ns)):
        m = max(m, pg[(h * NSMAX + s) * PS])
    var l: Float32 = 0
    var o: Float32 = 0
    for s in range(Int(ns)):
        var p = (h * NSMAX + s) * PS
        var a = exp(pg[p] - m)
        l += a * pg[p + 1]
        o += a * pg[p + 2 + tid]
    O[h, tid] = rebind[O.ElementType](o / l)
