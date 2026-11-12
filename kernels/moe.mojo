from arch import perm, sdot4, sudot4
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.memory import bitcast
from std.builtin.globals import global_constant
from std.bit import pop_count
from std.sys import llvm_intrinsic, get_defined_int
from std.math import fma
from bexp import exp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i8 = DType.int8

comptime N_EXP = get_defined_int["BARO_N_EXP", 256]()
comptime TOPK = get_defined_int["BARO_TOPK", 8]()
comptime E_FFN = get_defined_int["BARO_E_FFN", 512]()
comptime SH_FFN = get_defined_int["BARO_SH_FFN", 512]()
comptime MOE_H = get_defined_int["BARO_H", 2048]()

comptime MOE_WAVES = 8
comptime MOE_THREADS = MOE_WAVES * WARP_SIZE
comptime MOE_VEC = 8

comptime Q4K = 256
comptime Q4K_BYTES = 144
comptime Q6K = 256
comptime Q6K_BYTES = 210


@always_inline
def q4k_scale_min(
    w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, j: Int
) -> Tuple[Int, Int]:
    var d = Int(w[unsafe_offset=base + 4 + j % 4])
    var m = Int(w[unsafe_offset=base + 8 + j % 4])
    if j < 4:
        return (d & 0x3F, m & 0x3F)
    var md = Int(w[unsafe_offset=base + 12 + j - 4])
    return ((md & 0x0F) | ((d >> 2) & 0x30), (md >> 4) | ((m >> 2) & 0x30))


@always_inline
def q4k_value(
    w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int
) -> Scalar[f32]:
    var block = k // Q4K
    var within = k % Q4K
    var group = within // 32
    var in_group = within % 32
    var base = row_base + block * Q4K_BYTES
    var dbits = Int(w[unsafe_offset=base]) | (Int(w[unsafe_offset=base + 1]) << 8)
    var mbits = Int(w[unsafe_offset=base + 2]) | (Int(w[unsafe_offset=base + 3]) << 8)
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(dbits)))[0].cast[f32]()
    var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(mbits)))[0].cast[f32]()
    var sc, mn = q4k_scale_min(w, base, group)
    var qbyte = Int(w[unsafe_offset=base + 16 + (group // 2) * 32 + in_group])
    var q = qbyte & 0x0F if group % 2 == 0 else qbyte >> 4
    return (d * Scalar[f32](sc * q) - dm * Scalar[f32](mn)).cast[bf16]().cast[f32]()


def amar_moe_router_top8[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
):
    comptime assert L.flat_rank == 1 and IDX.flat_rank == 1 and W.flat_rank == 1
    comptime PER = N_EXP // WARP_SIZE
    var lane = Int(lane_id())
    var v = Array[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[lane + m * WARP_SIZE])
        if v[m] > lm:
            lm = v[m]
    var mx = warp.max(lm)
    var ls = Scalar[f32](0)
    comptime for m in range(PER):
        v[m] = exp(v[m] - mx)
        ls += v[m]
    var tot = warp.sum(ls)
    comptime for m in range(PER):
        v[m] = v[m] / tot
    var wsum = Scalar[f32](0)
    for j in range(TOPK):
        var best = Scalar[f32](-1)
        var bi = N_EXP
        comptime for m in range(PER):
            if v[m] > best:
                best = v[m]
                bi = lane + m * WARP_SIZE
        var gbest = warp.max(best)
        var cand = bi if best == gbest else N_EXP
        var gi = -warp.max(-cand)
        if gi == bi and best == gbest:
            comptime for m in range(PER):
                if lane + m * WARP_SIZE == gi:
                    v[m] = Scalar[f32](-1)
        if lane == 0:
            IDX[j] = rebind[IDX.ElementType](Int32(gi))
            W[j] = rebind[W.ElementType](gbest)
        wsum += gbest
    if lane == 0:
        for j in range(TOPK):
            W[j] = rebind[W.ElementType](rebind[Scalar[f32]](W[j]) / wsum)


def amar_moe_router_top1[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
):
    comptime assert L.flat_rank == 1 and IDX.flat_rank == 1 and W.flat_rank == 1
    comptime PER = N_EXP // WARP_SIZE
    var lane = Int(lane_id())
    var v = Array[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[lane + m * WARP_SIZE])
        if v[m] > lm:
            lm = v[m]
    var mx = warp.max(lm)
    var ls = Scalar[f32](0)
    comptime for m in range(PER):
        v[m] = exp(v[m] - mx)
        ls += v[m]
    var tot = warp.sum(ls)
    var best = Scalar[f32](-1)
    var bi = N_EXP
    comptime for m in range(PER):
        v[m] = v[m] / tot
        if v[m] > best:
            best = v[m]
            bi = lane + m * WARP_SIZE
    var gbest = warp.max(best)
    var cand = bi if best == gbest else N_EXP
    var gi = -warp.max(-cand)
    if lane == 0:
        IDX[0] = rebind[IDX.ElementType](Int32(gi))
        W[0] = rebind[W.ElementType](Scalar[f32](1.0))


@always_inline
def router_top8_sig_body[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    mut IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    mut W: TileTensor[f32, WLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    K: Int,
):
    comptime assert L.flat_rank == 1 and IDX.flat_rank == 1 and W.flat_rank == 1
    comptime assert X.flat_rank == 1 and G.flat_rank == 1 and O.flat_rank == 1
    comptime PER = N_EXP // WARP_SIZE
    var lane = Int(lane_id())
    var v = Array[Scalar[f32], PER](uninitialized=True)
    var lm = Scalar[f32](-3.4028234663852886e38)
    comptime for m in range(PER):
        v[m] = rebind[Scalar[f32]](L[lane + m * WARP_SIZE])
        if v[m] > lm:
            lm = v[m]
    var mx = warp.max(lm)
    var ls = Scalar[f32](0)
    comptime for m in range(PER):
        v[m] = exp(v[m] - mx)
        ls += v[m]
    var tot = warp.sum(ls)
    comptime for m in range(PER):
        v[m] = v[m] / tot
    var wsum = Scalar[f32](0)
    for j in range(TOPK):
        var best = Scalar[f32](-1)
        var bi = N_EXP
        comptime for m in range(PER):
            if v[m] > best:
                best = v[m]
                bi = lane + m * WARP_SIZE
        var gbest = warp.max(best)
        var cand = bi if best == gbest else N_EXP
        var gi = -warp.max(-cand)
        if gi == bi and best == gbest:
            comptime for m in range(PER):
                if lane + m * WARP_SIZE == gi:
                    v[m] = Scalar[f32](-1)
        if lane == 0:
            IDX[j] = rebind[IDX.ElementType](Int32(gi))
            W[j] = rebind[W.ElementType](gbest)
        wsum += gbest
    if lane == 0:
        for j in range(TOPK):
            W[j] = rebind[W.ElementType](rebind[Scalar[f32]](W[j]) / wsum)
    var Xv = X.vectorize[8]()
    var Gv = G.vectorize[8]()
    var acc = SIMD[f32, 8](0)
    var i = lane
    while i < K // 8:
        acc = fma(rebind[SIMD[f32, 8]](Xv[i]), rebind[SIMD[f32, 8]](Gv[i]), acc)
        i += WARP_SIZE
    var t = warp.sum(acc.reduce_add())
    if lane == 0:
        O[0] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-t)))


def amar_moe_router_top8_sig[
    LLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    L: TileTensor[f32, LLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    W: TileTensor[f32, WLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    k_dim: Int32,
):
    var IDX_ = IDX
    var W_ = W
    var O_ = O
    router_top8_sig_body(L, IDX_, W_, X, G, O_, Int(k_dim))


def amar_moe_sig_gate[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert X.flat_rank == 1 and G.flat_rank == 1 and O.flat_rank == 1

    var lane = Int(lane_id())
    var K = Int(k_dim)
    var Xv = X.vectorize[8]()
    var Gv = G.vectorize[8]()
    var acc = SIMD[f32, 8](0)
    var i = lane
    while i < K // 8:
        acc = fma(rebind[SIMD[f32, 8]](Xv[i]), rebind[SIMD[f32, 8]](Gv[i]), acc)
        i += WARP_SIZE
    var t = warp.sum(acc.reduce_add())
    if lane == 0:
        O[0] = rebind[O.ElementType](Scalar[f32](1) / (Scalar[f32](1) + exp(-t)))


def amar_moe_gate_up[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, GLayout: TensorLayout, ULayout: TensorLayout,
    ILayout: TensorLayout, HLayout: TensorLayout
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    WG: TileTensor[bf16, GLayout, MutAnyOrigin],
    WU: TileTensor[bf16, ULayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert Xb.flat_rank == 2 and WG.flat_rank == 2 and WU.flat_rank == 2
    comptime assert IDX.flat_rank == 1 and HO.flat_rank == 1

    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return

    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row = e * FFN + r
    var lane = Int(lane_id())
    var K = Int(k_dim)

    var Xv = Xb.vectorize[1, MOE_VEC]()
    var Gv = WG.vectorize[1, MOE_VEC]()
    var Uv = WU.vectorize[1, MOE_VEC]()
    comptime STEP = WARP_SIZE * MOE_VEC

    var ag = SIMD[f32, MOE_VEC](0)
    var au = SIMD[f32, MOE_VEC](0)
    var kk = 0
    while kk < K:
        var a = rebind[SIMD[bf16, MOE_VEC]](
            Xv[0, kk // MOE_VEC + lane]
        ).cast[f32]()
        ag += rebind[SIMD[bf16, MOE_VEC]](
            Gv[row, kk // MOE_VEC + lane]
        ).cast[f32]() * a
        au += rebind[SIMD[bf16, MOE_VEC]](
            Uv[row, kk // MOE_VEC + lane]
        ).cast[f32]() * a
        kk += STEP

    var g = warp.sum(ag.reduce_add())
    var u = warp.sum(au.reduce_add())
    if lane == 0:
        HO[wid] = rebind[HO.ElementType](
            g / (Scalar[f32](1) + exp(-g)) * u
        )


def amar_moe_down[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, DLayout: TensorLayout, ILayout: TensorLayout,
    WLayout: TensorLayout, OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: TileTensor[bf16, DLayout, MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and WD.flat_rank == 2
    comptime assert IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1

    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return

    var lane = Int(lane_id())
    var Hv = Hb.vectorize[1, MOE_VEC]()
    var Dv = WD.vectorize[1, MOE_VEC]()
    comptime STEP = WARP_SIZE * MOE_VEC

    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row = e * N + c
        var acc = SIMD[f32, MOE_VEC](0)
        var kk = 0
        while kk < FFN:
            var h = rebind[SIMD[bf16, MOE_VEC]](
                Hv[j, kk // MOE_VEC + lane]
            ).cast[f32]()
            acc += rebind[SIMD[bf16, MOE_VEC]](
                Dv[row, kk // MOE_VEC + lane]
            ).cast[f32]() * h
            kk += STEP
        out += rebind[Scalar[f32]](WT[j]) * warp.sum(acc.reduce_add())

    if lane == 0:
        O[c] = rebind[O.ElementType](out)


@always_inline
def q4k_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var acc = Scalar[f32](0)
    var k = lane
    while k < k_dim:
        acc += rebind[Scalar[bf16]](X[x_row, k]).cast[f32]() * q4k_value(W, row_base, k)
        k += WARP_SIZE
    return warp.sum(acc)


@always_inline
def q4k_dot_blocks[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // Q4K
    var acc = SIMD[f32, 16](0)
    var b = lane // 8
    var s = lane % 8
    var pair = s // 2
    var half = (s % 2) * 16
    while b < nb:
        var base = row_base + b * Q4K_BYTES
        var hdr = W.unsafe_offset(base).load[width=16]()
        var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
        var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
        var g0 = 2 * pair
        var g1 = g0 + 1
        var hw = bitcast[DType.uint32, 4](hdr)
        var p0 = q4k_sm_w(hw, g0)
        var p1 = q4k_sm_w(hw, g1)
        var sc0 = p0[0]
        var mn0 = p0[1]
        var sc1 = p1[0]
        var mn1 = p1[1]
        var qb = W.unsafe_offset(base + 16 + pair * 32 + half).load[width=16]()
        var lo = (qb & 0x0F).cast[f32]()
        var hi = (qb >> 4).cast[f32]()
        var v0 = (SIMD[f32, 16](d * Scalar[f32](sc0)) * lo - SIMD[f32, 16](dm * Scalar[f32](mn0))).cast[bf16]().cast[f32]()
        var v1 = (SIMD[f32, 16](d * Scalar[f32](sc1)) * hi - SIMD[f32, 16](dm * Scalar[f32](mn1))).cast[bf16]().cast[f32]()
        var k0 = b * Q4K + g0 * 32 + half
        var a0 = rebind[SIMD[bf16, 16]](Xv[x_row, k0 // 16]).cast[f32]()
        var a1 = rebind[SIMD[bf16, 16]](Xv[x_row, (k0 + 32) // 16]).cast[f32]()
        acc = fma(v0, a0, fma(v1, a1, acc))
        b += 4
    return warp.sum(acc.reduce_add())


@always_inline
def q4k_dot_blocks_fill[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    F: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    fill_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // Q4K
    var acc = SIMD[f32, 16](0)
    var b = lane // 8
    var s = lane % 8
    var pair = s // 2
    var half = (s % 2) * 16
    while b < nb:
        var base = row_base + b * Q4K_BYTES
        var hdr = W.unsafe_offset(base).load[width=16]()
        var fbase = fill_base + b * Q4K_BYTES
        if s == 0:
            F.unsafe_offset(fbase).store(hdr)
        var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
        var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
        var g0 = 2 * pair
        var g1 = g0 + 1
        var hw = bitcast[DType.uint32, 4](hdr)
        var p0 = q4k_sm_w(hw, g0)
        var p1 = q4k_sm_w(hw, g1)
        var sc0 = p0[0]
        var mn0 = p0[1]
        var sc1 = p1[0]
        var mn1 = p1[1]
        var qb = W.unsafe_offset(base + 16 + pair * 32 + half).load[width=16]()
        F.unsafe_offset(fbase + 16 + pair * 32 + half).store(qb)
        var lo = (qb & 0x0F).cast[f32]()
        var hi = (qb >> 4).cast[f32]()
        var v0 = (SIMD[f32, 16](d * Scalar[f32](sc0)) * lo - SIMD[f32, 16](dm * Scalar[f32](mn0))).cast[bf16]().cast[f32]()
        var v1 = (SIMD[f32, 16](d * Scalar[f32](sc1)) * hi - SIMD[f32, 16](dm * Scalar[f32](mn1))).cast[bf16]().cast[f32]()
        var k0 = b * Q4K + g0 * 32 + half
        var a0 = rebind[SIMD[bf16, 16]](Xv[x_row, k0 // 16]).cast[f32]()
        var a1 = rebind[SIMD[bf16, 16]](Xv[x_row, (k0 + 32) // 16]).cast[f32]()
        acc = fma(v0, a0, fma(v1, a1, acc))
        b += 4
    return warp.sum(acc.reduce_add())


def q4k_decode_vector[
    OLayout: TensorLayout,
](
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
):
    comptime assert O.flat_rank == 1
    var k = global_idx.x
    if k < Q4K:
        O[k] = rebind[O.ElementType](q4k_value(W, 0, k))


@always_inline
def q6k_value(
    w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int
) -> Scalar[f32]:
    var block = k // Q6K
    var within = k % Q6K
    var half = within // 128
    var group = (within % 128) // 32
    var lane = within % 32
    var base = row_base + block * Q6K_BYTES
    var ql_index = lane if group == 0 or group == 2 else lane + 32
    var ql = Int(w[unsafe_offset=base + half * 64 + ql_index])
    var qh = Int(w[unsafe_offset=base + 128 + half * 32 + lane])
    var shift = 0 if group == 0 else 2 if group == 1 else 4 if group == 2 else 6
    var q = ((ql >> (0 if group < 2 else 4)) & 0x0F) | (((qh >> shift) & 0x03) << 4)
    var scale_index = half * 8 + (lane // 16) + group * 2
    var sc = bitcast[i8, 1](SIMD[u8, 1](w[unsafe_offset=base + 192 + scale_index]))[0]
    var dbits = Int(w[unsafe_offset=base + 208]) | (Int(w[unsafe_offset=base + 209]) << 8)
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(dbits)))[0].cast[f32]()
    return d * Scalar[f32](Int(sc)) * Scalar[f32](q - 32)


@always_inline
def q6k_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var acc = Scalar[f32](0)
    var k = Int(lane_id())
    while k < k_dim:
        acc += rebind[Scalar[bf16]](X[x_row, k]).cast[f32]() * q6k_value(W, row_base, k)
        k += WARP_SIZE
    return warp.sum(acc)


@always_inline
def q8_0_value(w: MutPointer[Scalar[u8], MutAnyOrigin], row_base: Int, k: Int) -> Scalar[f32]:
    var block = k // 32
    var in_block = k % 32
    var p = w.unsafe_offset(row_base + block * 34)
    var raw = p.unsafe_bitcast[Scalar[u16]]()[]
    var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
    var q = p.unsafe_offset(2 + in_block).unsafe_bitcast[Scalar[i8]]()[]
    return (d * Scalar[f32](q)).cast[bf16]().cast[f32]()


@always_inline
def q8_0_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var base = row_base + b * 34
        var raw = W.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


comptime IQ4NL_BYTES = 18


@always_inline
def iq4nl_i8(n: SIMD[u8, 16]) -> SIMD[i8, 16]:
    var w = bitcast[DType.uint32, 4](n)
    var sel = w & 0x07070707
    var m = ((w >> 3) & 0x01010101) * 0xFF
    var o = SIMD[DType.uint32, 4](0)
    comptime for i in range(4):
        var lo = perm(UInt32(0xF6EADDCF), UInt32(0xBFAD9881), sel[i])
        var hi = perm(UInt32(0x71594535), UInt32(0x26190D01), sel[i])
        o[i] = (hi & m[i]) | (lo & ~m[i])
    return bitcast[i8, 16](o)


@always_inline
def iq4nl_lut(n: SIMD[u8, 16]) -> SIMD[f32, 16]:
    return iq4nl_i8(n).cast[f32]()


@always_inline
def iq4nl_half(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, h: Int) -> SIMD[f32, 16]:
    var raw = w.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
    var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
    var q = w.unsafe_offset(base + 2).load[width=16]()
    var n = (q >> 4) if h == 1 else (q & 15)
    return SIMD[f32, 16](d) * iq4nl_lut(n)


@always_inline
def iq4nl_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var v = iq4nl_half(W, row_base + b * IQ4NL_BYTES, h)
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


comptime KLane = Tuple[SIMD[i8, 16], SIMD[i8, 16], Scalar[f32], Scalar[f32], Scalar[f32], Scalar[f32], Int, Int]
comptime Q5K_BYTES = 176
comptime IQ4XS_BYTES = 136
comptime Q3K_BYTES = 110
comptime Q2K_BYTES = 84


@always_inline
def f16_at(w: MutPointer[Scalar[u8], MutAnyOrigin], off: Int) -> Scalar[f32]:
    return bitcast[f16, 1](SIMD[u16, 1](w.unsafe_offset(off).unsafe_bitcast[Scalar[u16]]()[]))[0].cast[f32]()


@always_inline
def q5k_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var pair = s // 2
    var half = (s % 2) * 16
    var h = bitcast[DType.uint32, 4](w.unsafe_offset(base).load[width=16]())
    var dd = bitcast[f16, 2](h[0])
    var d = dd[0].cast[f32]()
    var dm = dd[1].cast[f32]()
    var sm0 = q4k_sm_w(h, 2 * pair)
    var sm1 = q4k_sm_w(h, 2 * pair + 1)
    var qb = w.unsafe_offset(base + 48 + pair * 32 + half).load[width=16]()
    var qh = w.unsafe_offset(base + 16 + half).load[width=16]()
    var lo = (qb & 0x0F) | (((qh >> UInt8(2 * pair)) & 1) << 4)
    var hi = (qb >> 4) | (((qh >> UInt8(2 * pair + 1)) & 1) << 4)
    var k0 = pair * 64 + half
    return (bitcast[i8, 16](lo), bitcast[i8, 16](hi), d * Scalar[f32](sm0[0]), d * Scalar[f32](sm1[0]), dm * Scalar[f32](sm0[1]), dm * Scalar[f32](sm1[1]), k0, k0 + 32)


@always_inline
def iq4xs_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var sh = Int(w.unsafe_offset(base + 2).unsafe_bitcast[Scalar[u16]]()[])
    var ls = ((Int(w[unsafe_offset=base + 4 + s // 2]) >> (4 * (s % 2))) & 0xF) | (((sh >> (2 * s)) & 3) << 4)
    var dl = f16_at(w, base) * Scalar[f32](ls - 32)
    var q = w.unsafe_offset(base + 8 + s * 16).load[width=16]()
    return (iq4nl_i8(q & 15), iq4nl_i8(q >> 4), dl, dl, Scalar[f32](0), Scalar[f32](0), s * 32, s * 32 + 16)


@always_inline
def q3k_scale(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, i: Int) -> Int:
    var lo = (Int(w[unsafe_offset=base + 96 + i % 8]) >> (4 * (i // 8))) & 0xF
    var hi = (Int(w[unsafe_offset=base + 104 + i % 4]) >> (2 * (i // 4))) & 3
    return (lo | (hi << 4)) - 32


@always_inline
def q3k_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var h = s // 4
    var j = s % 4
    var d = f16_at(w, base + 108)
    var sh = UInt8(2 * j)
    var hb = UInt8(4 * h + j)
    var q0 = ((w.unsafe_offset(base + 32 + h * 32).load[width=16]() >> sh) & 3) | (((w.unsafe_offset(base).load[width=16]() >> hb) & 1) << 2)
    var q1 = ((w.unsafe_offset(base + 48 + h * 32).load[width=16]() >> sh) & 3) | (((w.unsafe_offset(base + 16).load[width=16]() >> hb) & 1) << 2)
    var i = h * 8 + 2 * j
    var k0 = h * 128 + j * 32
    return (bitcast[i8, 16](q0) - 4, bitcast[i8, 16](q1) - 4, d * Scalar[f32](q3k_scale(w, base, i)), d * Scalar[f32](q3k_scale(w, base, i + 1)), Scalar[f32](0), Scalar[f32](0), k0, k0 + 16)


@always_inline
def q2k_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var h = s // 4
    var j = s % 4
    var dd = bitcast[f16, 2](w.unsafe_offset(base + 80).unsafe_bitcast[Scalar[DType.uint32]]()[])
    var d = dd[0].cast[f32]()
    var dm = dd[1].cast[f32]()
    var scw = Int(w.unsafe_offset(base + h * 8 + 2 * j).unsafe_bitcast[Scalar[u16]]()[])
    var sc0 = scw & 0xFF
    var sc1 = scw >> 8
    var sh = UInt8(2 * j)
    var q0 = (w.unsafe_offset(base + 16 + h * 32).load[width=16]() >> sh) & 3
    var q1 = (w.unsafe_offset(base + 32 + h * 32).load[width=16]() >> sh) & 3
    var k0 = h * 128 + j * 32
    return (bitcast[i8, 16](q0), bitcast[i8, 16](q1), d * Scalar[f32](sc0 & 0xF), d * Scalar[f32](sc1 & 0xF), dm * Scalar[f32](sc0 >> 4), dm * Scalar[f32](sc1 >> 4), k0, k0 + 16)


@always_inline
def q4k_sm_w(h: SIMD[DType.uint32, 4], g: Int) -> Tuple[Int, Int]:
    var sh = UInt32(8 * (g % 4))
    var a = Int((h[1] >> sh) & 0xFF)
    var b = Int((h[2] >> sh) & 0xFF)
    var c = Int((h[3] >> sh) & 0xFF)
    if g < 4:
        return (a & 0x3F, b & 0x3F)
    return ((c & 0x0F) | ((a >> 2) & 0x30), (c >> 4) | ((b >> 2) & 0x30))


comptime IQ2XXS_GRID: Array[UInt64, 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
]


comptime IQ2XXS_BYTES = 66


@always_inline
def iq2_signed(g: UInt64, sg: Int) -> SIMD[i8, 8]:
    var v = bitcast[i8, 8](SIMD[DType.uint64, 1](g))
    var bits = SIMD[u8, 8](UInt8(sg | ((pop_count(sg) & 1) << 7))) & SIMD[u8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    return bits.ne(0).select(-v, v)


@always_inline
def gridv[N: Int, //, T: Array[UInt64, N]](i: Int) -> UInt64:
    ref t = global_constant[T]()
    return UnsafePointer(to=t).bitcast[UInt64]().unsafe_address_space_cast[AddressSpace.GLOBAL]()[unsafe_offset=i]


@always_inline
def iq2xxs_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var p = w.unsafe_offset(base + 2 + s * 8)
    var a0 = Int(p.unsafe_bitcast[Scalar[DType.uint32]]()[])
    var a1 = Int(p.unsafe_offset(4).unsafe_bitcast[Scalar[DType.uint32]]()[])
    var db = f16_at(w, base) * (Scalar[f32](0.5) + Scalar[f32](a1 >> 28)) * Scalar[f32](0.25)
    var q0 = iq2_signed(gridv[IQ2XXS_GRID](a0 & 0xFF), a1 & 127).join(iq2_signed(gridv[IQ2XXS_GRID]((a0 >> 8) & 0xFF), (a1 >> 7) & 127))
    var q1 = iq2_signed(gridv[IQ2XXS_GRID]((a0 >> 16) & 0xFF), (a1 >> 14) & 127).join(iq2_signed(gridv[IQ2XXS_GRID]((a0 >> 24) & 0xFF), (a1 >> 21) & 127))
    return (q0, q1, db, db, Scalar[f32](0), Scalar[f32](0), s * 32, s * 32 + 16)


comptime IQ1S_GRID: Array[UInt64, 2048] = [
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
]


comptime IQ1M_BYTES = 56


@always_inline
def iq1m_part(w: MutPointer[Scalar[u8], MutAnyOrigin], qs: Int, qh: Int) -> SIMD[i8, 16]:
    var h = Int(w[unsafe_offset=qh])
    var g0 = bitcast[i8, 8](SIMD[DType.uint64, 1](gridv[IQ1S_GRID](Int(w[unsafe_offset=qs]) | ((h << 8) & 0x700))))
    var g1 = bitcast[i8, 8](SIMD[DType.uint64, 1](gridv[IQ1S_GRID](Int(w[unsafe_offset=qs + 1]) | ((h << 4) & 0x700))))
    var d0 = Int8(-1) if h & 0x08 else Int8(1)
    var d1 = Int8(-1) if h & 0x80 else Int8(1)
    return (g0 * 8 + d0).join(g1 * 8 + d1)


@always_inline
def iq1m_lane(w: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, s: Int) -> KLane:
    var sc = w.unsafe_offset(base + 48).unsafe_bitcast[Scalar[u16]]()
    var u = (Int(sc[0]) >> 12) | ((Int(sc[1]) >> 8) & 0x00F0) | ((Int(sc[2]) >> 4) & 0x0F00) | (Int(sc[3]) & 0xF000)
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(u)))[0].cast[f32]()
    var sw = Int(sc[s // 2]) >> (6 * (s % 2))
    var dl1 = d * Scalar[f32](2 * (sw & 7) + 1) * Scalar[f32](0.125)
    var dl2 = d * Scalar[f32](2 * ((sw >> 3) & 7) + 1) * Scalar[f32](0.125)
    var q0 = iq1m_part(w, base + s * 4, base + 32 + s * 2)
    var q1 = iq1m_part(w, base + s * 4 + 2, base + 32 + s * 2 + 1)
    return (q0, q1, dl1, dl2, Scalar[f32](0), Scalar[f32](0), s * 32, s * 32 + 16)


@always_inline
def kq_row_dot[
    lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane,
    BYTES: Int,
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 256
    var acc = SIMD[f32, 16](0)
    var b = lane // 8
    var s = lane % 8
    while b < nb:
        var r = lane_fn(W, row_base + b * BYTES, s)
        var v0 = SIMD[f32, 16](r[2]) * r[0].cast[f32]() - SIMD[f32, 16](r[4])
        var v1 = SIMD[f32, 16](r[3]) * r[1].cast[f32]() - SIMD[f32, 16](r[5])
        var a0 = rebind[SIMD[bf16, 16]](Xv[x_row, (b * 256 + r[6]) // 16]).cast[f32]()
        var a1 = rebind[SIMD[bf16, 16]](Xv[x_row, (b * 256 + r[7]) // 16]).cast[f32]()
        acc = fma(v0, a0, fma(v1, a1, acc))
        b += 4
    return warp.sum(acc.reduce_add())


@always_inline
def kq_q8_row_dot[
    lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane,
    BYTES: Int,
    HAS_MIN: Bool,
](
    Aq: MutPointer[Scalar[i8], MutAnyOrigin],
    As: MutPointer[Scalar[f16], MutAnyOrigin],
    a_base: Int,
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    row_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    var lane = Int(lane_id())
    var nb = k_dim // 256
    var acc = Scalar[f32](0)
    var b = lane // 8
    var s = lane % 8
    var ones = Int32(0x01010101)
    while b < nb:
        var r = lane_fn(W, row_base + b * BYTES, s)
        var k0 = a_base + b * 256 + r[6]
        var k1 = a_base + b * 256 + r[7]
        var w0 = bitcast[DType.int32, 4](r[0])
        var w1 = bitcast[DType.int32, 4](r[1])
        var x0 = bitcast[DType.int32, 4](Aq.unsafe_offset(k0).load[width=16]())
        var x1 = bitcast[DType.int32, 4](Aq.unsafe_offset(k1).load[width=16]())
        var d0 = Int32(0)
        var d1 = Int32(0)
        var s0 = Int32(0)
        var s1 = Int32(0)
        comptime for i in range(4):
            d0 = sdot4(w0[i], x0[i], d0, False)
            d1 = sdot4(w1[i], x1[i], d1, False)
            comptime if HAS_MIN:
                s0 = sdot4(ones, x0[i], s0, False)
                s1 = sdot4(ones, x1[i], s1, False)
        var t0 = r[2] * Scalar[f32](d0) - r[4] * Scalar[f32](s0)
        var t1 = r[3] * Scalar[f32](d1) - r[5] * Scalar[f32](s1)
        acc = fma(As[unsafe_offset=k0 // 32].cast[f32](), t0, fma(As[unsafe_offset=k1 // 32].cast[f32](), t1, acc))
        b += 4
    return warp.sum(acc)

@always_inline
def q8d_row_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    q_base: Int,
    s_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var raw = W.unsafe_offset(s_base + b * 2).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(q_base + b * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


def moe_embed_q8_0_pos[
    OLayout: TensorLayout, KLayout: TensorLayout,
](
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Toks: TileTensor[i32, KLayout, MutAnyOrigin],
    pos: Int32,
    n: Int32,
    row_bytes: Int32,
):
    comptime assert O.flat_rank == 2 and Toks.flat_rank == 1
    var idx = global_idx.x
    if idx >= Int(n):
        return
    var token = Int(rebind[Scalar[i32]](Toks[Int(pos) + Int(block_idx.y)]))
    var row_base = token * Int(row_bytes)
    O[block_idx.y, idx] = rebind[O.ElementType](q8_0_value(W, row_base, idx))


def moe_matmul_q8d_m1[
    OLayout: TensorLayout, ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 1
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, 0, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[row] = rebind[O.ElementType](dot)


def moe_matmul_q8d_m1x[
    ALayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    w1: Int64,
    O1: MutPointer[Scalar[f32], MutAnyOrigin],
    n1: Int32,
    w2: Int64,
    O2: MutPointer[Scalar[f32], MutAnyOrigin],
    n2: Int32,
    w3: Int64,
    O3: MutPointer[Scalar[f32], MutAnyOrigin],
    n3: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var k = Int(k_dim)
    var wo = Int(w1)
    var O = O1
    var n = Int(n1)
    if row >= n:
        row -= n
        wo = Int(w2)
        O = O2
        n = Int(n2)
        if row >= n:
            row -= n
            wo = Int(w3)
            O = O3
            n = Int(n3)
            if row >= n:
                return
    var dot = q8d_row_dot(A, W.unsafe_offset(wo), 0, row * k, n * k + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[unsafe_offset=row] = dot


def moe_matmul_q8d_m1_add[
    OLayout: TensorLayout, ALayout: TensorLayout, XLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    s_off: Int32,
):
    comptime assert A.flat_rank == 2 and O.flat_rank == 1 and X.flat_rank == 1
    var row = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var k = Int(k_dim)
    var dot = q8d_row_dot(A, W, 0, row * k, Int(s_off) + row * (k // 32) * 2, k)
    if lane_id() == 0:
        O[row] = rebind[O.ElementType](dot)
        X[row] = rebind[X.ElementType](rebind[Scalar[f32]](X[row]) + dot)


def moe_gate_up_q8_0[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
    k_dim: Int32,
    row_bytes: Int32,
    up_offset: Int32,
):
    comptime assert X.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_base = (e * FFN + r) * Int(row_bytes)
    var g = q8_0_row_dot(X, W, 0, row_base, Int(k_dim))
    var u = q8_0_row_dot(X, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


def moe_down_q8_0[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
    row_bytes: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if c >= Int(n):
        return
    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = (e * Int(n) + c) * Int(row_bytes)
        out += rebind[Scalar[f32]](WT[j]) * q8_0_row_dot(Hb, WD, j, row_base, FFN)
    if lane_id() == 0:
        O[c] = rebind[O.ElementType](out)


def moe_down_q8_0_res[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    ALayout: TensorLayout, XLayout: TensorLayout,
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    A: TileTensor[f32, ALayout, MutAnyOrigin],
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    n: Int32,
    row_bytes: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and A.flat_rank == 1 and X.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if c >= Int(n):
        return
    var out = Scalar[f32](0)
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = (e * Int(n) + c) * Int(row_bytes)
        out += rebind[Scalar[f32]](WT[j]) * q8_0_row_dot(Hb, WD, j, row_base, FFN)
    if lane_id() == 0:
        X[c] = rebind[X.ElementType](
            rebind[Scalar[f32]](X[c]) + rebind[Scalar[f32]](A[c]) + out
        )


def amar_moe_gate_up_q4k[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    WG: MutPointer[Scalar[u8], MutAnyOrigin],
    WU: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[f32, HLayout, MutAnyOrigin],
    k_dim: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_row_dot(Xb, WG, 0, row_base, Int(k_dim))
    var u = q4k_row_dot(Xb, WU, 0, row_base, Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType](g / (Scalar[f32](1) + exp(-g)) * u)


def moe_gate_up_q4k_pack[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_dot_blocks(Xb, W, 0, row_base, Int(k_dim))
    var u = q4k_dot_blocks(Xb, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


def moe_gate_up_q4k_zc[
    NSEL: Int, FFN: Int,
    XLayout: TensorLayout, ILayout: TensorLayout, HLayout: TensorLayout, out_dt: DType = f32
](
    Xb: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HE: TileTensor[i32, ILayout, MutAnyOrigin],
    HG: MutPointer[Scalar[u8], MutAnyOrigin],
    HO: TileTensor[out_dt, HLayout, MutAnyOrigin],
    k_dim: Int32,
    up_offset: Int32,
    h_up_offset: Int32,
):
    comptime assert Xb.flat_rank == 2 and IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = (Int(k_dim) // Q4K) * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var he = Int(rebind[Scalar[i32]](HE[j]))
    if he == -2:
        if lane_id() == 0:
            HO[wid] = rebind[HO.ElementType](Scalar[out_dt](0))
        return
    var g: Scalar[f32]
    var u: Scalar[f32]
    if he >= 0:
        var hbase = he * FFN * row_bytes + r * row_bytes
        g = q4k_dot_blocks_fill(Xb, HG, W, 0, hbase, row_base, Int(k_dim))
        u = q4k_dot_blocks_fill(Xb, HG, W, 0, hbase + Int(h_up_offset), row_base + Int(up_offset), Int(k_dim))
    else:
        g = q4k_dot_blocks(Xb, W, 0, row_base, Int(k_dim))
        u = q4k_dot_blocks(Xb, W, 0, row_base + Int(up_offset), Int(k_dim))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[out_dt]())


def amar_moe_down_q4k[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var lane = Int(lane_id())
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q4K) * Q4K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q4k_dot_blocks(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane == 0:
        O[c] = rebind[O.ElementType](out)


def amar_moe_down_q4k_zc[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HE: TileTensor[i32, ILayout, MutAnyOrigin],
    HD: MutPointer[Scalar[u8], MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var lane = Int(lane_id())
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q4K) * Q4K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var he = Int(rebind[Scalar[i32]](HE[j]))
        if he == -2:
            continue
        var dot: Scalar[f32]
        if he >= 0:
            dot = q4k_dot_blocks_fill(Hb, HD, WD, j, he * N * row_bytes + c * row_bytes, row_base, FFN)
        else:
            dot = q4k_dot_blocks(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane == 0:
        O[c] = rebind[O.ElementType](out)


def amar_moe_down_q6k[
    NSEL: Int, FFN: Int,
    HLayout: TensorLayout, ILayout: TensorLayout, WLayout: TensorLayout,
    OLayout: TensorLayout
](
    Hb: TileTensor[bf16, HLayout, MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert Hb.flat_rank == 2 and IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var out = Scalar[f32](0)
    var row_bytes = (FFN // Q6K) * Q6K_BYTES
    for j in range(NSEL):
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        var row_base = e * N * row_bytes + c * row_bytes
        var dot = q6k_row_dot(Hb, WD, j, row_base, FFN)
        out += rebind[Scalar[f32]](WT[j]) * dot
    if lane_id() == 0:
        O[c] = rebind[O.ElementType](out)



@always_inline
def q4k_q8_visit(
    hdr: SIMD[u8, 16], qb: SIMD[u8, 16], a0: SIMD[i8, 16], a1: SIMD[i8, 16],
    d8_0: Scalar[f32], d8_1: Scalar[f32], pair: Int,
) -> Scalar[f32]:
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
    var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
    var g0 = 2 * pair
    var g1 = g0 + 1
    var hw = bitcast[DType.uint32, 4](hdr)
    var p0 = q4k_sm_w(hw, g0)
    var p1 = q4k_sm_w(hw, g1)
    var sc0 = p0[0]
    var mn0 = p0[1]
    var sc1 = p1[0]
    var mn1 = p1[1]
    var lo = bitcast[DType.int32, 4](qb & 0x0F)
    var hi = bitcast[DType.int32, 4](qb & 0xF0)
    var ai0 = bitcast[DType.int32, 4](a0)
    var ai1 = bitcast[DType.int32, 4](a1)
    var ones = Int32(0x01010101)
    var dot0 = Int32(0)
    var dot1 = Int32(0)
    var s0 = Int32(0)
    var s1 = Int32(0)
    comptime for i in range(4):
        dot0 = sdot4(lo[i], ai0[i], dot0, False)
        dot1 = sudot4(False, hi[i], True, ai1[i], dot1, False)
        s0 = sdot4(ones, ai0[i], s0, False)
        s1 = sdot4(ones, ai1[i], s1, False)
    var t0 = d * Scalar[f32](sc0 * Int(dot0)) - dm * Scalar[f32](mn0 * Int(s0))
    dot1 = dot1 >> 4
    var t1 = d * Scalar[f32](sc1 * Int(dot1)) - dm * Scalar[f32](mn1 * Int(s1))
    return fma(d8_0, t0, d8_1 * t1)


@always_inline
def q4k_q8_row_dot[NB: Int](
    Aq: MutPointer[Scalar[i8], MutAnyOrigin],
    As: MutPointer[Scalar[f16], MutAnyOrigin],
    a_base: Int,
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    row_base: Int,
) -> Scalar[f32]:
    comptime NV = (NB + 3) // 4
    var lane = Int(lane_id())
    var b0 = lane // 8
    var sub = lane % 8
    var pair = sub // 2
    var half = (sub % 2) * 16
    var q = Array[SIMD[u8, 16], NV](uninitialized=True)
    var a0 = Array[SIMD[i8, 16], NV](uninitialized=True)
    var a1 = Array[SIMD[i8, 16], NV](uninitialized=True)
    comptime for v in range(NV):
        var b = min(b0 + 4 * v, NB - 1)
        q[v] = W.unsafe_offset(row_base + b * Q4K_BYTES + 16 + pair * 32 + half).load[width=16]()
        var k0 = a_base + b * Q4K + 2 * pair * 32 + half
        a0[v] = Aq.unsafe_offset(k0).load[width=16]()
        a1[v] = Aq.unsafe_offset(k0 + 32).load[width=16]()
    var acc = Scalar[f32](0)
    comptime for v in range(NV):
        var b = b0 + 4 * v
        comptime if NB % 4 != 0:
            if b >= NB:
                continue
        var hdr = W.unsafe_offset(row_base + b * Q4K_BYTES).load[width=16]()
        var sb = (a_base + b * Q4K) // 32 + 2 * pair
        acc += q4k_q8_visit(hdr, q[v], a0[v], a1[v], As[unsafe_offset=sb].cast[f32](), As[unsafe_offset=sb + 1].cast[f32](), pair)
    return warp.sum(acc)


def moe_gate_up_q4k_dot[
    NSEL: Int, FFN: Int, NB: Int,
    ILayout: TensorLayout, HLayout: TensorLayout
](
    Aq: MutPointer[Scalar[i8], MutAnyOrigin],
    As: MutPointer[Scalar[f16], MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    HO: TileTensor[bf16, HLayout, MutAnyOrigin],
    up_offset: Int32,
):
    comptime assert IDX.flat_rank == 1 and HO.flat_rank == 1
    var wid = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    if wid >= NSEL * FFN:
        return
    var j = wid // FFN
    var r = wid % FFN
    var e = Int(rebind[Scalar[i32]](IDX[j]))
    var row_bytes = NB * Q4K_BYTES
    var row_base = e * FFN * row_bytes + r * row_bytes
    var g = q4k_q8_row_dot[NB](Aq, As, 0, W, row_base)
    var u = q4k_q8_row_dot[NB](Aq, As, 0, W, row_base + Int(up_offset))
    if lane_id() == 0:
        HO[wid] = rebind[HO.ElementType]((g / (Scalar[f32](1) + exp(-g)) * u).cast[bf16]())


@always_inline
def q8_quant_group(
    src: MutPointer[Scalar[bf16], MutAnyOrigin],
    Aq: MutPointer[Scalar[i8], MutAnyOrigin],
    As: MutPointer[Scalar[f16], MutAnyOrigin],
    g: Int,
):
    var lane = Int(lane_id())
    var x = src[unsafe_offset=g * 32 + lane].cast[f32]()
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(127) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    if q > 127:
        q = 127
    if q < -127:
        q = -127
    Aq[unsafe_offset=g * 32 + lane] = Int8(q)
    if lane == 0:
        As[unsafe_offset=g] = scale.cast[f16]()


@always_inline
def moe_down_q4k_dot_row[
    NSEL: Int, FFN: Int, NB: Int, ILayout: TensorLayout, WLayout: TensorLayout
](
    Hq: MutPointer[Scalar[i8], MutAnyOrigin],
    Hs: MutPointer[Scalar[f16], MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    c: Int,
    N: Int,
) -> Scalar[f32]:
    comptime assert IDX.flat_rank == 1 and WT.flat_rank == 1
    comptime assert NB == 2 and NSEL % 2 == 0
    comptime NV = NSEL // 2
    var lane = Int(lane_id())
    var b0 = lane // 8
    var sub = lane % 8
    var pair = sub // 2
    var half = (sub % 2) * 16
    var jsel = b0 // 2
    var b = b0 % 2
    var row_bytes = NB * Q4K_BYTES
    var q = Array[SIMD[u8, 16], NV](uninitialized=True)
    var a0 = Array[SIMD[i8, 16], NV](uninitialized=True)
    var a1 = Array[SIMD[i8, 16], NV](uninitialized=True)
    var bases = Array[Int, NV](uninitialized=True)
    comptime for v in range(NV):
        var j = 2 * v + jsel
        var e = Int(rebind[Scalar[i32]](IDX[j]))
        bases[v] = e * N * row_bytes + c * row_bytes
        q[v] = WD.unsafe_offset(bases[v] + b * Q4K_BYTES + 16 + pair * 32 + half).load[width=16]()
        var k0 = j * FFN + b * Q4K + 2 * pair * 32 + half
        a0[v] = Hq.unsafe_offset(k0).load[width=16]()
        a1[v] = Hq.unsafe_offset(k0 + 32).load[width=16]()
    var part = Array[Scalar[f32], NV](uninitialized=True)
    comptime for v in range(NV):
        var j = 2 * v + jsel
        var hdr = WD.unsafe_offset(bases[v] + b * Q4K_BYTES).load[width=16]()
        var sb = (j * FFN + b * Q4K) // 32 + 2 * pair
        part[v] = q4k_q8_visit(hdr, q[v], a0[v], a1[v], Hs[unsafe_offset=sb].cast[f32](), Hs[unsafe_offset=sb + 1].cast[f32](), pair)
    var out = Scalar[f32](0)
    comptime for j in range(NSEL):
        var mine = part[j // 2] if jsel == j % 2 else Scalar[f32](0)
        var dot = warp.sum(mine)
        out += rebind[Scalar[f32]](WT[j]) * dot
    return out


def amar_moe_down_q4k_dot[
    NSEL: Int, FFN: Int, NB: Int,
    ILayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout
](
    Hq: MutPointer[Scalar[i8], MutAnyOrigin],
    Hs: MutPointer[Scalar[f16], MutAnyOrigin],
    WD: MutPointer[Scalar[u8], MutAnyOrigin],
    IDX: TileTensor[i32, ILayout, MutAnyOrigin],
    WT: TileTensor[f32, WLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert IDX.flat_rank == 1 and WT.flat_rank == 1 and O.flat_rank == 1
    var c = Int(block_idx.x) * MOE_WAVES + Int(thread_idx.x) // WARP_SIZE
    var N = Int(n)
    if c >= N:
        return
    var out = moe_down_q4k_dot_row[NSEL, FFN, NB](Hq, Hs, WD, IDX, WT, c, N)
    if lane_id() == 0:
        O[c] = rebind[O.ElementType](out)
