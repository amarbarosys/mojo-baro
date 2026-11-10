from max.gpu import block_dim, block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from std.math import cos, sin, tanh
from bexp import exp, log
from std.memory import bitcast
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from attn import KVT, HD, NQH, NKVH, attn_head_span, kv_off, kv_tab_off
from matmul_skinny import dtype, ROW_WAVES
from gemv_kq import gemv_epi

comptime f32 = DType.float32


def amar_embed_lookup_f32[
    TLayout: TensorLayout, OLayout: TensorLayout, KLayout: TensorLayout
](
    Table: TileTensor[f32, TLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[DType.int32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
):
    comptime assert Table.flat_rank == 2 and O.flat_rank == 2 and Toks.flat_rank == 1
    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[DType.int32]](Toks[Int(pos) + Int(block_idx.y)]))
    O[block_idx.y, idx] = rebind[O.ElementType](rebind[Scalar[f32]](Table[token, idx]))


def amar_rope_plain[
    NROT_: Int, XLayout: TensorLayout, NEOX: Bool = True, FF: Bool = False
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
    nh: Int32,
    freq_base: Float32,
    ff: Pointer[Float32, MutAnyOrigin],
):
    rope_plain_body[NROT_, NEOX=NEOX, FF=FF](X, pos, nh, freq_base, ff, Int(block_idx.x), Int(block_idx.y), Int(thread_idx.x))


@always_inline
def rope_plain_body[
    NROT_: Int, XLayout: TensorLayout, NEOX: Bool, FF: Bool
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
    nh: Int32,
    freq_base: Float32,
    ff: Pointer[Float32, MutAnyOrigin],
    h: Int,
    r: Int,
    j: Int,
):
    comptime assert X.flat_rank == 2
    var theta = Float32(Int(pos) + r) * exp(
        Float32(-2 * j) / Float32(NROT_) * log(freq_base)
    )
    comptime if FF: # rope_freqs (Llama 3 long-context scaling): theta / ff[j], as llama.cpp's freq_factors
        theta = theta / ff[Int(j)]
    var c = cos(theta)
    var s = sin(theta)
    var row = r * Int(nh) + h
    comptime if NEOX:
        var x0 = rebind[Scalar[f32]](X[row, j])
        var x1 = rebind[Scalar[f32]](X[row, j + NROT_ // 2])
        X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, j + NROT_ // 2] = rebind[X.ElementType](x0 * s + x1 * c)
    else:
        var x0 = rebind[Scalar[f32]](X[row, 2 * j])
        var x1 = rebind[Scalar[f32]](X[row, 2 * j + 1])
        X[row, 2 * j] = rebind[X.ElementType](x0 * c - x1 * s)
        X[row, 2 * j + 1] = rebind[X.ElementType](x0 * s + x1 * c)


def amar_gemv_q8[
    EPI: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    CLayout: TensorLayout, BLayout: TensorLayout, ADT: DType = DType.bfloat16
](
    A: TileTensor[ADT, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    Ob: TileTensor[ADT, BLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2
    comptime assert C.flat_rank == 1 and Ob.flat_rank == 1
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return
    comptime QV = 16
    comptime STEP = WARP_SIZE * QV
    comptime UNROLL = 4
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = SIMD[f32, QV](0)
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var qs = Array[SIMD[DType.int8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[DType.float16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            qs[u] = rebind[SIMD[DType.int8, QV]](Qv[row, kb // QV + lane])
            ds[u] = rebind[Scalar[DType.float16]](S[row, (kb + lane * QV) // 32])
        comptime for u in range(UNROLL):
            var kb = kk + u * STEP
            var w = qs[u].cast[f32]() * ds[u].cast[f32]()
            var a = rebind[SIMD[ADT, QV]](Av[0, kb // QV + lane]).cast[f32]()
            acc += w * a
        kk += UNROLL * STEP
    while kk < K:
        var kv = kk + lane * QV
        if kv < K:
            var q = rebind[SIMD[DType.int8, QV]](Qv[row, kv // QV]).cast[f32]()
            var d = rebind[Scalar[DType.float16]](S[row, kv // 32]).cast[f32]()
            var a = rebind[SIMD[ADT, QV]](Av[0, kv // QV]).cast[f32]()
            acc += q * d * a
        kk += STEP
    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        gemv_epi[EPI, ODT=ADT](C, Ob, row, total)


def amar_rope_kv_append[
    NROT_: Int, NAT: Int, CLayout: TensorLayout, NLayout: TensorLayout, HD_: Int = HD, NKVH_: Int = NKVH, NEOX: Bool = True, FF: Bool = False
](
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    K: TileTensor[f32, NLayout, MutAnyOrigin],
    V: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    freq_base: Float32,
    att_i: Int32,
    ff: Pointer[Float32, MutAnyOrigin],
):
    rope_kv_body[NROT_, NAT, HD_=HD_, NKVH_=NKVH_, NEOX=NEOX, FF=FF](Kc, Vc, K, V, tab, pos, freq_base, att_i, ff, Int(block_idx.x), Int(block_idx.y),
        Int(thread_idx.x), Int(block_idx.z)) # grid.z = rows of a batched append (K/V [rows * NKVH, HD]); 0 at decode


@always_inline
def rope_kv_body[
    NROT_: Int, NAT: Int, CLayout: TensorLayout, NLayout: TensorLayout, HD_: Int, NKVH_: Int, NEOX: Bool, FF: Bool
](
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    K: TileTensor[f32, NLayout, MutAnyOrigin],
    V: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    freq_base: Float32,
    att_i: Int32,
    ff: Pointer[Float32, MutAnyOrigin],
    h: Int,
    which: Int,
    d: Int,
    z: Int,
):
    comptime assert Kc.flat_rank == 1 and K.flat_rank == 2
    var p = Int(pos) + z
    var kr = z * NKVH_ + h
    var cb = kv_tab_off[NAT, HD_, NKVH_](tab, p, Int(att_i), Int(h)) + d
    if which == 1:
        Vc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](V[kr, d]).cast[KVT]())
        return
    var val = rebind[Scalar[f32]](K[kr, d])
    if d < NROT_:
        comptime if NEOX:
            var half = NROT_ // 2
            var j = d if d < half else d - half
            var theta = Float32(p) * exp(
                Float32(-2 * j) / Float32(NROT_) * log(freq_base)
            )
            comptime if FF:
                theta = theta / ff[j]
            var c = cos(theta)
            var s = sin(theta)
            var x0 = rebind[Scalar[f32]](K[kr, j])
            var x1 = rebind[Scalar[f32]](K[kr, j + half])
            val = x0 * c - x1 * s if d < half else x0 * s + x1 * c
        else:
            var j = d // 2
            var theta = Float32(p) * exp(
                Float32(-2 * j) / Float32(NROT_) * log(freq_base)
            )
            comptime if FF:
                theta = theta / ff[j]
            var c = cos(theta)
            var s = sin(theta)
            var x0 = rebind[Scalar[f32]](K[kr, 2 * j])
            var x1 = rebind[Scalar[f32]](K[kr, 2 * j + 1])
            val = x0 * c - x1 * s if d % 2 == 0 else x0 * s + x1 * c
    Kc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](val.cast[KVT]())


def amar_rope_qkv[
    NROT_: Int, NAT: Int, QLayout: TensorLayout, CLayout: TensorLayout, NLayout: TensorLayout, HD_: Int = HD, NQH_: Int = NQH, NKVH_: Int = NKVH,
    NEOX: Bool = True, FF: Bool = False
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    K: TileTensor[f32, NLayout, MutAnyOrigin],
    V: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    freq_base: Float32,
    att_i: Int32,
    ff: Pointer[Float32, MutAnyOrigin],
):
    """amar_rope_plain on q and amar_rope_kv_append in one launch at decode: grid NQH + 2 NKVH, block HD."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b < NQH_:
        if t < NROT_ // 2:
            rope_plain_body[NROT_, NEOX=NEOX, FF=FF](Q, pos, Int32(NQH_), freq_base, ff, b, 0, t)
        return
    var k = b - NQH_
    rope_kv_body[NROT_, NAT, HD_=HD_, NKVH_=NKVH_, NEOX=NEOX, FF=FF](Kc, Vc, K, V, tab, pos, freq_base, att_i, ff, k % NKVH_, k // NKVH_, t, 0)


def amar_bias_add[
    XLayout: TensorLayout, BLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Bias: TileTensor[f32, BLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Bias.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    X[i] = rebind[X.ElementType](rebind[Scalar[f32]](X[i]) + rebind[Scalar[f32]](Bias[i]))


def amar_attn_decode_swa_gated[
    QLayout: TensorLayout, KLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout, NAT: Int, HD_: Int = HD, NQH_: Int = NQH, NKVH_: Int = NKVH, HAS_GATE: Bool = True, ODT: DType = DType.bfloat16
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[ODT, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    win: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Gate.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH_ // NKVH_)
    var T = Int(t_len) + r
    var t_lo = 0
    if Int(win) > 0 and T > Int(win):
        t_lo = T - Int(win)
    var qrow = r * NQH_ + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_ // WARP_SIZE]())
    var res = attn_head_span[NAT=NAT, HD_=HD_, NKVH_=NKVH_](Q, Kc, Vc, tab, qs, scores, red, qrow, kvh, t_lo, T, tid, Int(lane_id()), scale, Int(att_i))
    if tid < HD_:
        var inv = 1 / res[1]
        var o = res[2] * inv
        comptime if HAS_GATE:
            var g = rebind[Scalar[f32]](Gate[qrow]) # [rows, NQH] when batched; qrow == h at decode
            O[qrow, tid] = rebind[O.ElementType]((o * (1 / (1 + exp(-g)))).cast[ODT]())
        else:
            O[qrow, tid] = rebind[O.ElementType](o.cast[ODT]())


comptime ATT_NSMAX = 64


def att_span_len(t_len: Int, win: Int) -> Int:
    """Tokens per span for the split decode attention: 32, more when the context needs over ATT_NSMAX spans."""
    var n = min(t_len, win) if win > 0 else t_len
    return max(32, (n + ATT_NSMAX * 32 - 1) // (ATT_NSMAX * 32) * 32)


def amar_attn_split_part[
    QLayout: TensorLayout, KLayout: TensorLayout, NAT: Int, HD_: Int = HD, NQH_: Int = NQH, NKVH_: Int = NKVH
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    pg: Pointer[Float32, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    win: Int32,
    span: Int32,
    scale: Float32,
    att_i: Int32,
):
    """Decode attention split over the context: grid (NQH, spans), block HD; leaves (m, l, o[HD]) per (head, span) in pg."""
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1
    var h = Int(block_idx.x)
    var sp = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var T = Int(t_len)
    var t_lo = 0
    if Int(win) > 0 and T > Int(win):
        t_lo = T - Int(win)
    var a = t_lo + sp * Int(span)
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD_ // WARP_SIZE]())
    var res = attn_head_span[NAT=NAT, HD_=HD_, NKVH_=NKVH_](Q, Kc, Vc, tab, qs, scores, red, h, h // (NQH_ // NKVH_), a, min(T, a + Int(span)),
        tid, Int(lane_id()), scale, Int(att_i))
    var p = (h * ATT_NSMAX + sp) * (HD_ + 2)
    if tid == 0:
        pg[p] = res[0]
        pg[p + 1] = res[1]
    if tid < HD_:
        pg[p + 2 + tid] = res[2]


def amar_attn_split_merge[
    GLayout: TensorLayout, OLayout: TensorLayout, HD_: Int = HD, HAS_GATE: Bool = True, ODT: DType = DType.bfloat16
](
    pg: Pointer[Float32, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[ODT, OLayout, MutAnyOrigin],
    ns: Int32,
):
    """grid NQH, block HD: o = sum_s e^(m_s - M) o_s / sum_s e^(m_s - M) l_s, then the gate and the output cast."""
    comptime assert Gate.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var m = Float32(-3.4e38)
    for s in range(Int(ns)):
        m = max(m, pg[(h * ATT_NSMAX + s) * (HD_ + 2)])
    var l: Float32 = 0
    var o: Float32 = 0
    for s in range(Int(ns)):
        var p = (h * ATT_NSMAX + s) * (HD_ + 2)
        var e = exp(pg[p] - m)
        l += e * pg[p + 1]
        o += e * pg[p + 2 + tid]
    o = o * (1 / l)
    comptime if HAS_GATE:
        var g = rebind[Scalar[f32]](Gate[h])
        O[h, tid] = rebind[O.ElementType]((o * (1 / (1 + exp(-g)))).cast[ODT]())
    else:
        O[h, tid] = rebind[O.ElementType](o.cast[ODT]())


def amar_argmax_part[
    NB: Int, XLayout: TensorLayout, VLayout: TensorLayout, ILayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Pv: TileTensor[f32, VLayout, MutAnyOrigin],
    Pi: TileTensor[DType.int32, ILayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Pv.flat_rank == 1 and Pi.flat_rank == 1
    comptime T = 256
    comptime V = 8
    var N = Int(n)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var best_v = Float32(-3.4e38)
    var best_i: Int32 = 0
    var i = (b * T + tid) * V
    while i < N:
        var v = X.ptr.load[width=V](i)
        comptime for j in range(V):
            if v[j] > best_v:
                best_v = v[j]
                best_i = Int32(i + j)
        i += NB * T * V
    var vals = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[T]())
    var idxs = stack_allocation[DType.int32, address_space = AddressSpace.SHARED](row_major[T]())
    vals[tid] = rebind[vals.ElementType](best_v)
    idxs[tid] = rebind[idxs.ElementType](best_i)
    barrier()
    var active = T
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            var v2 = rebind[Scalar[f32]](vals[tid + active])
            var i2 = rebind[Scalar[DType.int32]](idxs[tid + active])
            var v1 = rebind[Scalar[f32]](vals[tid])
            var i1 = rebind[Scalar[DType.int32]](idxs[tid])
            if v2 > v1 or (v2 == v1 and i2 < i1):
                vals[tid] = rebind[vals.ElementType](v2)
                idxs[tid] = rebind[idxs.ElementType](i2)
        barrier()
    if tid == 0:
        Pv[b] = rebind[Pv.ElementType](rebind[Scalar[f32]](vals[0]))
        Pi[b] = rebind[Pi.ElementType](rebind[Scalar[DType.int32]](idxs[0]))


def amar_argmax_final[
    NB: Int, VLayout: TensorLayout, ILayout: TensorLayout, OLayout: TensorLayout
](
    Pv: TileTensor[f32, VLayout, MutAnyOrigin],
    Pi: TileTensor[DType.int32, ILayout, MutAnyOrigin],
    Out: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    Pred: TileTensor[DType.int32, OLayout, MutAnyOrigin],
    wpos: Int32,
    forced: Int32,
):
    comptime assert Pv.flat_rank == 1 and Pi.flat_rank == 1 and Out.flat_rank == 1
    if thread_idx.x != 0:
        return
    var bv = Float32(-3.4e38)
    var bi: Int32 = 0
    comptime for t in range(NB):
        var v = rebind[Scalar[f32]](Pv[t])
        var ix = rebind[Scalar[DType.int32]](Pi[t])
        if v > bv or (v == bv and ix < bi):
            bv = v
            bi = ix
    Pred[Int(wpos)] = rebind[Pred.ElementType](bi)
    Out[Int(wpos)] = rebind[Out.ElementType](forced if forced >= 0 else bi)


def amar_trace_sum(
    src: MutPointer[Scalar[f32], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    n: Int32,
    slot: Int32,
):
    if global_idx.x != 0:
        return
    var acc = SIMD[DType.uint32, 8](2166136261)
    var i = 0
    while i + 8 <= Int(n):
        acc = (acc ^ bitcast[DType.uint32, 8](src.load[width=8](i))) * 16777619
        i += 8
    var tail = UInt32(2166136261)
    while i < Int(n):
        tail = (tail ^ bitcast[DType.uint32, 1](src.load[width=1](i))) * 16777619
        i += 1
    var folded = tail
    comptime for j in range(8):
        folded = (folded ^ acc[j]) * 16777619
    dst[Int(slot)] = folded
