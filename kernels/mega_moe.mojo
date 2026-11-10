from std.atomic import Atomic, Ordering
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import fma, log1p, rsqrt, sqrt
from bexp import exp
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from dattn import dattn_split_body, dattn_combine_body, dattn_nsplit, dkv_q, dkv_sc_off
 
from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS, ROW_VEC, SPLITK, SM
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS
from attn import HD, NQH, NKVH, KVT, KVQ8, TCAP, kv_off, kv_tab_off, NROT, attn_head_span
from model import H, FFN, QF, KV
from mega import grid_barrier, stamp, rope_cs, MEGA_G, RMS_EPS, ATT_SCALE, DATT_NLD
from moe import (
    q4k_sm_w,
    q8_0_row_dot, q4k_dot_blocks, q6k_row_dot, router_top8_sig_body,
    q4k_q8_row_dot, moe_down_q4k_dot_row, q8_quant_group,
    N_EXP, TOPK, E_FFN, SH_FFN, Q4K, Q4K_BYTES, Q6K, Q6K_BYTES,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i8 = DType.int8
comptime f16 = DType.float16
comptime u32 = DType.uint32
comptime i32 = DType.int32
comptime i64 = DType.int64
comptime ATT = NQH * HD
comptime INNER = NH_V * SSTATE
comptime Q8_ROW_H = (H // 32) * 34
comptime Q8S_H = (H // 32) * 2
comptime Q8S_ATT = (ATT // 32) * 2
comptime Q8S_INNER = (INNER // 32) * 2
comptime Q8_ROW_SH = (SH_FFN // 32) * 34
comptime Q4K_ROW_H = (H // Q4K) * Q4K_BYTES
comptime Q4K_ROW_E = (E_FFN // Q4K) * Q4K_BYTES
comptime Q6K_ROW_E = (E_FFN // Q6K) * Q6K_BYTES
comptime W_ATT = 16
comptime W_SSM = 19
comptime NPROF = 16
comptime PATT = SPLITK * SM * FFN
comptime MOE_BARRIERS = 12 * 30 + 11 * 10
comptime MOE_BARRIERS_DOT = MOE_BARRIERS + 37


@always_inline
def rms_moe(
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    gp: MutPointer[Scalar[f32], MutAnyOrigin],
    cb: MutPointer[Scalar[bf16], MutAnyOrigin],
    fp: MutPointer[Scalar[f32], MutAnyOrigin],
):
    var X = TileTensor(xp, row_major[1, H]())
    var G = TileTensor(gp, row_major[H]())
    var O = TileTensor(cb, row_major[1, H]())
    var F = TileTensor(fp, row_major[1, H]())
    var tid = Int(thread_idx.x)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    var partial: Float32 = 0
    var i = tid
    while i < H:
        var v = rebind[Scalar[f32]](X[0, i])
        partial += v * v
        i += EW_THREADS
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        sums[tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(H) + RMS_EPS)
    i = Int(block_idx.x) * EW_THREADS + tid
    while i < H:
        var h = (rebind[Scalar[f32]](X[0, i]) * scale * rebind[Scalar[f32]](G[i])).cast[bf16]()
        O[0, i] = rebind[O.ElementType](h)
        F[0, i] = rebind[F.ElementType](h.cast[f32]())
        i += Int(grid_dim.x) * EW_THREADS
    barrier()


@always_inline
def f32_row_dot[UNROLL: Int, AL: TensorLayout, WL: TensorLayout](
    A: TileTensor[f32, AL, MutAnyOrigin],
    W: TileTensor[f32, WL, MutAnyOrigin],
    row: Int, lane: Int, K: Int,
) -> Float32:
    comptime assert A.flat_rank == 2 and W.flat_rank == 2
    var Wv = W.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = SIMD[f32, ROW_VEC](0)
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = Array[SIMD[f32, ROW_VEC], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[f32, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])
        comptime for u in range(UNROLL):
            var a = rebind[SIMD[f32, ROW_VEC]](Av[0, (kk + u * STEP) // ROW_VEC + lane]).cast[f32]()
            acc += ws[u].cast[f32]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[f32, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[f32]()
        var a = rebind[SIMD[f32, ROW_VEC]](Av[0, kk // ROW_VEC + lane]).cast[f32]()
        acc += w * a
        kk += STEP
    return warp.sum(acc.reduce_add())


@always_inline
def q8_dot_u[U: Int, XLayout: TensorLayout](
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
    while b + (U - 1) * 16 < nb:
        var raws = Array[Scalar[u16], U](uninitialized=True)
        var qs = Array[SIMD[i8, 16], U](uninitialized=True)
        var xs = Array[SIMD[bf16, 16], U](uninitialized=True)
        comptime for u in range(U):
            var base = row_base + (b + u * 16) * 34
            raws[u] = W.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
            qs[u] = W.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
            xs[u] = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * (b + u * 16) + h])
        comptime for u in range(U):
            var d = bitcast[f16, 1](SIMD[u16, 1](raws[u])).cast[f32]()[0]
            var v = (SIMD[f32, 16](d) * qs[u].cast[f32]()).cast[bf16]().cast[f32]()
            var a = xs[u].cast[f32]()
            acc = fma(v, a, acc)
        b += U * 16
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


@always_inline
def q8d_dot_u[U: Int, XLayout: TensorLayout](
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
    while b + (U - 1) * 16 < nb:
        var raws = Array[Scalar[u16], U](uninitialized=True)
        var qs = Array[SIMD[i8, 16], U](uninitialized=True)
        var xs = Array[SIMD[bf16, 16], U](uninitialized=True)
        comptime for u in range(U):
            var bb = b + u * 16
            raws[u] = W.unsafe_offset(s_base + bb * 2).unsafe_bitcast[Scalar[u16]]()[]
            qs[u] = W.unsafe_offset(q_base + bb * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
            xs[u] = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * bb + h])
        comptime for u in range(U):
            var d = bitcast[f16, 1](SIMD[u16, 1](raws[u])).cast[f32]()[0]
            var v = (SIMD[f32, 16](d) * qs[u].cast[f32]()).cast[bf16]().cast[f32]()
            var a = xs[u].cast[f32]()
            acc = fma(v, a, acc)
        b += U * 16
    while b < nb:
        var raw = W.unsafe_offset(s_base + b * 2).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(q_base + b * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


@always_inline
def q4k_dot_u[U: Int, XLayout: TensorLayout](
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
    var g0 = 2 * pair
    var g1 = g0 + 1
    while b + (U - 1) * 4 < nb:
        var hdrs = Array[SIMD[u8, 16], U](uninitialized=True)
        var qbs = Array[SIMD[u8, 16], U](uninitialized=True)
        var a0s = Array[SIMD[bf16, 16], U](uninitialized=True)
        var a1s = Array[SIMD[bf16, 16], U](uninitialized=True)
        comptime for u in range(U):
            var bb = b + u * 4
            var base = row_base + bb * Q4K_BYTES
            hdrs[u] = W.unsafe_offset(base).load[width=16]()
            qbs[u] = W.unsafe_offset(base + 16 + pair * 32 + half).load[width=16]()
            var k0 = bb * Q4K + g0 * 32 + half
            a0s[u] = rebind[SIMD[bf16, 16]](Xv[x_row, k0 // 16])
            a1s[u] = rebind[SIMD[bf16, 16]](Xv[x_row, (k0 + 32) // 16])
        comptime for u in range(U):
            var hdr = hdrs[u]
            var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
            var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
            var hw = bitcast[DType.uint32, 4](hdr)
            var p0 = q4k_sm_w(hw, g0)
            var p1 = q4k_sm_w(hw, g1)
            var sc0 = p0[0]
            var mn0 = p0[1]
            var sc1 = p1[0]
            var mn1 = p1[1]
            var lo = (qbs[u] & 0x0F).cast[f32]()
            var hi = (qbs[u] >> 4).cast[f32]()
            var v0 = (SIMD[f32, 16](d * Scalar[f32](sc0)) * lo - SIMD[f32, 16](dm * Scalar[f32](mn0))).cast[bf16]().cast[f32]()
            var v1 = (SIMD[f32, 16](d * Scalar[f32](sc1)) * hi - SIMD[f32, 16](dm * Scalar[f32](mn1))).cast[bf16]().cast[f32]()
            var a0 = a0s[u].cast[f32]()
            var a1 = a1s[u].cast[f32]()
            acc = fma(v0, a0, fma(v1, a1, acc))
        b += U * 4
    while b < nb:
        var base = row_base + b * Q4K_BYTES
        var hdr = W.unsafe_offset(base).load[width=16]()
        var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
        var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
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
def dump_x(
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    slot: Int,
):
    if block_idx.x == 0:
        var i = Int(thread_idx.x)
        while i < H:
            dbg[unsafe_offset=slot * H + i] = xp[unsafe_offset=i]
            i += ROW_THREADS


@always_inline
def mega_moe_body[
    CsL: TensorLayout, SsL: TensorLayout, NL: Int, NAT: Int,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: MutPointer[Scalar[i64], MutAnyOrigin],
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    curb: MutPointer[Scalar[bf16], MutAnyOrigin],
    xg: MutPointer[Scalar[f32], MutAnyOrigin],
    fnp: MutPointer[Scalar[f32], MutAnyOrigin],
    qkv: MutPointer[Scalar[f32], MutAnyOrigin],
    zp: MutPointer[Scalar[f32], MutAnyOrigin],
    araw: MutPointer[Scalar[f32], MutAnyOrigin],
    braw: MutPointer[Scalar[f32], MutAnyOrigin],
    egp: MutPointer[Scalar[f32], MutAnyOrigin],
    betap: MutPointer[Scalar[f32], MutAnyOrigin],
    convp: MutPointer[Scalar[f32], MutAnyOrigin],
    sop: MutPointer[Scalar[f32], MutAnyOrigin],
    resb: MutPointer[Scalar[bf16], MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    qfp: MutPointer[Scalar[f32], MutAnyOrigin],
    kp: MutPointer[Scalar[f32], MutAnyOrigin],
    vp: MutPointer[Scalar[f32], MutAnyOrigin],
    qp: MutPointer[Scalar[f32], MutAnyOrigin],
    gatep: MutPointer[Scalar[f32], MutAnyOrigin],
    aop: MutPointer[Scalar[f32], MutAnyOrigin],
    aob: MutPointer[Scalar[bf16], MutAnyOrigin],
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    patt: MutPointer[Scalar[f32], MutAnyOrigin],
    idxp: MutPointer[Scalar[i32], MutAnyOrigin],
    wtp: MutPointer[Scalar[f32], MutAnyOrigin],
    sigp: MutPointer[Scalar[f32], MutAnyOrigin],
    rhp: MutPointer[Scalar[bf16], MutAnyOrigin],
    shp: MutPointer[Scalar[bf16], MutAnyOrigin],
    xqp: MutPointer[Scalar[i8], MutAnyOrigin],
    xsp: MutPointer[Scalar[f16], MutAnyOrigin],
    hqp: MutPointer[Scalar[i8], MutAnyOrigin],
    hsp: MutPointer[Scalar[f16], MutAnyOrigin],
    ctrp: MutPointer[Scalar[u32], MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, dump: Int32, att_split: Int32, moedot: Int32,
):
    comptime assert ConvState.flat_rank == 4 and SAll.flat_rank == 5
    comptime assert NQH + NKVH <= MEGA_G and NH_V <= MEGA_G
    comptime assert ROW_THREADS == EW_THREADS and ROW_THREADS == HD and SSTATE <= ROW_THREADS
    var ConvState_ = ConvState
    var SAll_ = SAll
    var X = TileTensor(xp, row_major[1, H]())
    var CurB = TileTensor(curb, row_major[1, H]())
    var Xg = TileTensor(xg, row_major[1, H]())
    var Fn2 = TileTensor(fnp, row_major[1, H]())
    var Fn1 = TileTensor(fnp, row_major[H]())
    var Qkvm = TileTensor(qkv, row_major[1, CONV]())
    var Zm = TileTensor(zp, row_major[1, INNER]())
    var Araw = TileTensor(araw, row_major[1, NH_V]())
    var Braw = TileTensor(braw, row_major[1, NH_V]())
    var Eg = TileTensor(egp, row_major[1, NH_V]())
    var Beta = TileTensor(betap, row_major[1, NH_V]())
    var Conv = TileTensor(convp, row_major[1, CONV]())
    var So = TileTensor(sop, row_major[1, NH_V, SSTATE]())
    var ResB = TileTensor(resb, row_major[1, INNER]())
    var Qfm = TileTensor(qfp, row_major[1, QF]())
    var Kflat = TileTensor(kp, row_major[1, KV]())
    var Vflat = TileTensor(vp, row_major[1, KV]())
    var Q = TileTensor(qp, row_major[NQH, HD]())
    var Gate = TileTensor(gatep, row_major[ATT]())
    var Ao = TileTensor(aop, row_major[NQH, HD]())
    var AoB = TileTensor(aob, row_major[1, ATT]())
    var Kc = TileTensor(kc, row_major[TCAP]())
    var Vc = TileTensor(vc, row_major[TCAP]())
    var Pg = TileTensor(patt, row_major[PATT]())
    var Logits = TileTensor(xg, row_major[N_EXP]())
    var Idx = TileTensor(idxp, row_major[TOPK]())
    var Wt = TileTensor(wtp, row_major[TOPK]())
    var Sig = TileTensor(sigp, row_major[1]())
    var RoutedHf = TileTensor(rhp, row_major[TOPK * E_FFN]())
    var RoutedH = TileTensor(rhp, row_major[TOPK, E_FFN]())
    var SharedHf = TileTensor(shp, row_major[SH_FFN]())
    var SharedH = TileTensor(shp, row_major[1, SH_FFN]())

    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var nwaves = nblk * ROW_WAVES
    var ctr = ctrp
    var gen = ctrp.unsafe_offset(1)
    var fail = ctrp.unsafe_offset(2)
    var p = Int(pos)
    var si = 0
    var ai = 0
    var sl = Int(slots)
    var rg = Int(ring)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[ROW_THREADS // WARP_SIZE]()
    )
    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())

    if Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](fail) != 0:
        return
    var w = 1
    for layer in range(NL):
        stamp(prof, NPROF * layer)
        var is_att = (layer + 1) % 4 == 0
        var o0 = Int(off[unsafe_offset=w])
        var o1 = Int(off[unsafe_offset=w + 1])
        var o2 = Int(off[unsafe_offset=w + 2])
        var o3 = Int(off[unsafe_offset=w + 3])
        var o4 = Int(off[unsafe_offset=w + 4])
        var o5 = Int(off[unsafe_offset=w + 5])
        var o6 = Int(off[unsafe_offset=w + 6])
        rms_moe(xp, wbuf.unsafe_offset(o0).unsafe_bitcast[Scalar[f32]](), curb, xg)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 1)
        if is_att:
            var g = bid * ROW_WAVES + wave
            while g < QF + KV + KV:
                if g < QF:
                    var t = q8d_dot_u[4](CurB, wbuf.unsafe_offset(o1), 0, g * H, QF * H + g * Q8S_H, H)
                    if lane == 0:
                        Qfm[0, g] = rebind[Qfm.ElementType](t)
                elif g < QF + KV:
                    var r = g - QF
                    var t = q8d_dot_u[4](CurB, wbuf.unsafe_offset(o2), 0, r * H, KV * H + r * Q8S_H, H)
                    if lane == 0:
                        Kflat[0, r] = rebind[Kflat.ElementType](t)
                else:
                    var r = g - QF - KV
                    var t = q8d_dot_u[4](CurB, wbuf.unsafe_offset(o3), 0, r * H, KV * H + r * Q8S_H, H)
                    if lane == 0:
                        Vflat[0, r] = rebind[Vflat.ElementType](t)
                g += nwaves
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 2)

            var Qn = TileTensor(wbuf.unsafe_offset(o4).unsafe_bitcast[Scalar[f32]](), row_major[HD]())
            var Kn = TileTensor(wbuf.unsafe_offset(o5).unsafe_bitcast[Scalar[f32]](), row_major[HD]())
            if bid < NQH:
                var h = bid
                var v = rebind[Scalar[f32]](Qfm[0, h * 2 * HD + tid])
                Gate[h * HD + tid] = rebind[Gate.ElementType](Qfm[0, h * 2 * HD + HD + tid])
                var ssq = warp.sum(v * v)
                if lane == 0:
                    sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                var total: Float32 = 0
                comptime for k in range(HD // WARP_SIZE):
                    total += rebind[Scalar[f32]](sums[k])
                Q[h, tid] = rebind[Q.ElementType](
                    v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Qn[tid])
                )
                barrier()
                if tid < NROT // 2:
                    var cs = rope_cs(tid, p)
                    var x0 = rebind[Scalar[f32]](Q[h, tid])
                    var x1 = rebind[Scalar[f32]](Q[h, tid + NROT // 2])
                    Q[h, tid] = rebind[Q.ElementType](x0 * cs[0] - x1 * cs[1])
                    Q[h, tid + NROT // 2] = rebind[Q.ElementType](x0 * cs[1] + x1 * cs[0])
            elif bid < NQH + NKVH:
                var h = bid - NQH
                var v = rebind[Scalar[f32]](Kflat[0, h * HD + tid])
                var ssq = warp.sum(v * v)
                if lane == 0:
                    sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                var total: Float32 = 0
                comptime for k in range(HD // WARP_SIZE):
                    total += rebind[Scalar[f32]](sums[k])
                Kflat[0, h * HD + tid] = rebind[Kflat.ElementType](
                    v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Kn[tid])
                )
                barrier()
                if tid < NROT // 2:
                    var cs = rope_cs(tid, p)
                    var x0 = rebind[Scalar[f32]](Kflat[0, h * HD + tid])
                    var x1 = rebind[Scalar[f32]](Kflat[0, h * HD + tid + NROT // 2])
                    Kflat[0, h * HD + tid] = rebind[Kflat.ElementType](x0 * cs[0] - x1 * cs[1])
                    Kflat[0, h * HD + tid + NROT // 2] = rebind[Kflat.ElementType](x0 * cs[1] + x1 * cs[0])
                barrier()
                comptime if KVQ8:
                    var kx = Float32(0)
                    var vx = Float32(0)
                    if tid < HD:
                        kx = rebind[Scalar[f32]](Kflat[0, h * HD + tid])
                        vx = rebind[Scalar[f32]](Vflat[0, h * HD + tid])
                        var km = warp.max(abs(kx))
                        var vm = warp.max(abs(vx))
                        if lane == 0:
                            sums[wave] = rebind[sums.ElementType](km)
                            scores[wave] = rebind[scores.ElementType](vm)
                    barrier()
                    if tid < HD:
                        var km = Float32(0)
                        var vm = Float32(0)
                        comptime for w in range(HD // WARP_SIZE):
                            km = max(km, rebind[Scalar[f32]](sums[w]))
                            vm = max(vm, rebind[Scalar[f32]](scores[w]))
                        var rb = kv_tab_off[NAT](tab, p, ai, h)
                        Kc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(kx, Float32(127) / km if km > 0 else Float32(0)))
                        Vc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(vx, Float32(127) / vm if vm > 0 else Float32(0)))
                        if tid == 0:
                            Kc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, p)] = km / 127
                            Vc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, p)] = vm / 127
                else:
                    var kb = kv_tab_off[NAT](tab, p, ai, h) + tid
                    Kc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](Kflat[0, h * HD + tid]).cast[KVT]())
                    Vc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](Vflat[0, h * HD + tid]).cast[KVT]())
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 3)

            if p + 1 <= Int(att_split):
                if bid < NQH:
                    var h = bid
                    var kvh = h // (NQH // NKVH)
                    var res = attn_head_span[NAT=NAT](Q, Kc, Vc, tab, qs, scores, sums, h, kvh, 0, p + 1, tid, lane, ATT_SCALE, ai)
                    var inv = 1 / res[1]
                    Ao[h, tid] = rebind[Ao.ElementType](res[2] * inv)
            else:
                var ns = dattn_nsplit[HD, DATT_NLD, NKVH](p + 1, 1, MEGA_G)
                if bid < NKVH * ns:
                    dattn_split_body[HD, NQH, NKVH, KVT, NAT, DATT_NLD, False](
                        Q, Kc, Vc, Ao, Pg, tab, bid // ns, bid % ns, 0, ns, p + 1, ATT_SCALE, ai, tid
                    )
                if ns > 1:
                    if not grid_barrier(ctr, gen, fail):
                        return
                    if bid < NQH:
                        dattn_combine_body[HD, MEGA_G](Pg, Ao, bid, ns, tid)
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 4)

            var i = bid * ROW_THREADS + tid
            while i < ATT:
                var gg = rebind[Scalar[f32]](Gate[i])
                AoB[0, i] = rebind[AoB.ElementType](
                    (rebind[Scalar[f32]](Ao[i // HD, i % HD]) * (1 / (1 + exp(-gg)))).cast[bf16]()
                )
                i += nblk * ROW_THREADS
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 5)

            g = bid * ROW_WAVES + wave
            while g < H:
                var t = q8d_dot_u[4](AoB, wbuf.unsafe_offset(o6), 0, g * ATT, H * ATT + g * Q8S_ATT, ATT)
                if lane == 0:
                    X[0, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, g]) + t)
                g += nwaves
            ai += 1
        else:
            var o7 = Int(off[unsafe_offset=w + 7])
            var o8 = Int(off[unsafe_offset=w + 8])
            var o9 = Int(off[unsafe_offset=w + 9])
            var Wa = TileTensor(wbuf.unsafe_offset(o3).unsafe_bitcast[Scalar[f32]](), row_major[NH_V, H]())
            var Wb = TileTensor(wbuf.unsafe_offset(o4).unsafe_bitcast[Scalar[f32]](), row_major[NH_V, H]())
            var Cw = TileTensor(wbuf.unsafe_offset(o5).unsafe_bitcast[Scalar[f32]](), row_major[CONV, 4]())
            var SsmA = TileTensor(wbuf.unsafe_offset(o6).unsafe_bitcast[Scalar[f32]](), row_major[NH_V]())
            var DtB = TileTensor(wbuf.unsafe_offset(o7).unsafe_bitcast[Scalar[f32]](), row_major[NH_V]())
            var Nw = TileTensor(wbuf.unsafe_offset(o8).unsafe_bitcast[Scalar[f32]](), row_major[SSTATE]())
            var g = bid * ROW_WAVES + wave
            while g < CONV + INNER + NH_V + NH_V:
                if g < CONV:
                    var t = q8d_dot_u[4](CurB, wbuf.unsafe_offset(o1), 0, g * H, CONV * H + g * Q8S_H, H)
                    if lane == 0:
                        Qkvm[0, g] = rebind[Qkvm.ElementType](t)
                elif g < CONV + INNER:
                    var r = g - CONV
                    var t = q8d_dot_u[4](CurB, wbuf.unsafe_offset(o2), 0, r * H, INNER * H + r * Q8S_H, H)
                    if lane == 0:
                        Zm[0, r] = rebind[Zm.ElementType](t)
                elif g < CONV + INNER + NH_V:
                    var r = g - CONV - INNER
                    var t = f32_row_dot[2](Xg, Wa, r, lane, H)
                    if lane == 0:
                        Araw[0, r] = rebind[Araw.ElementType](t)
                else:
                    var r = g - CONV - INNER - NH_V
                    var t = f32_row_dot[2](Xg, Wb, r, lane, H)
                    if lane == 0:
                        Braw[0, r] = rebind[Braw.ElementType](t)
                g += nwaves
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 2)

            if bid == 0 and tid < NH_V:
                var h = tid
                var sa = rebind[Scalar[f32]](SsmA[h])
                var db = rebind[Scalar[f32]](DtB[h])
                var br = rebind[Scalar[f32]](Braw[0, h])
                Beta[0, h] = rebind[Beta.ElementType](1 / (1 + exp(-br)))
                var asum = rebind[Scalar[f32]](Araw[0, h]) + db
                var sp = log1p(exp(asum))
                Eg[0, h] = rebind[Eg.ElementType](exp(sp * sa))
            var c = bid * ROW_THREADS + tid
            var rs = rg % sl
            var wsl = (rg + 1) % sl
            while c < CONV:
                var cw0 = rebind[Scalar[f32]](Cw[c, 0])
                var cw1 = rebind[Scalar[f32]](Cw[c, 1])
                var cw2 = rebind[Scalar[f32]](Cw[c, 2])
                var cw3 = rebind[Scalar[f32]](Cw[c, 3])
                var w0 = rebind[Scalar[f32]](ConvState_[rs, si, 0, c])
                var w1 = rebind[Scalar[f32]](ConvState_[rs, si, 1, c])
                var w2 = rebind[Scalar[f32]](ConvState_[rs, si, 2, c])
                var w3 = rebind[Scalar[f32]](Qkvm[0, c])
                var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
                Conv[0, c] = rebind[Conv.ElementType](acc / (1 + exp(-acc)))
                ConvState_[wsl, si, 0, c] = rebind[ConvState_.ElementType](w1)
                ConvState_[wsl, si, 1, c] = rebind[ConvState_.ElementType](w2)
                ConvState_[wsl, si, 2, c] = rebind[ConvState_.ElementType](w3)
                c += nblk * ROW_THREADS
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 3)

            if bid < NH_V:
                var head = bid
                var base = head * SSTATE
                var v: Float32 = 0
                if tid < SSTATE:
                    v = rebind[Scalar[f32]](Conv[0, base + tid])
                    var ssq = warp.sum(v * v)
                    if lane == 0:
                        sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                if tid < SSTATE:
                    var total: Float32 = 0
                    comptime for k in range(SSTATE // WARP_SIZE):
                        total += rebind[Scalar[f32]](sums[k])
                    var inv = rsqrt(total + SSM_EPS)
                    if head < NH_K:
                        inv = inv / sqrt(Float32(SSTATE))
                    Conv[0, base + tid] = rebind[Conv.ElementType](v * inv)
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 4)

            if bid < NH_V:
                var h = bid
                var j = tid
                var kh = h % NH_K
                if j < SSTATE:
                    kq[0, j] = rebind[kq.ElementType](Conv[0, kh * SSTATE + j])
                    kq[1, j] = rebind[kq.ElementType](Conv[0, KDIM + kh * SSTATE + j])
                barrier()
                if j < SSTATE:
                    var eg = rebind[Scalar[f32]](Eg[0, h])
                    var beta = rebind[Scalar[f32]](Beta[0, h])
                    var vj = rebind[Scalar[f32]](Conv[0, 2 * KDIM + h * SSTATE + j])
                    comptime CHK = 32
                    var sk: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = rebind[Scalar[f32]](SAll_[rs, si, h, c * CHK + ii, j])
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            sk = fma(t, rebind[Scalar[f32]](kq[1, c * CHK + ii]), sk)
                    var d = (vj - sk) * beta
                    var o: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = rebind[Scalar[f32]](SAll_[rs, si, h, c * CHK + ii, j])
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            var s = fma(rebind[Scalar[f32]](kq[1, c * CHK + ii]), d, t)
                            SAll_[wsl, si, h, c * CHK + ii, j] = rebind[SAll_.ElementType](s)
                            o = fma(s, rebind[Scalar[f32]](kq[0, c * CHK + ii]), o)
                    So[0, h, j] = rebind[So.ElementType](o)
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 5)

            if bid < NH_V:
                var h = bid
                var j = tid
                var nwj: Float32 = 0
                if j < SSTATE:
                    nwj = rebind[Scalar[f32]](Nw[j])
                var v: Float32 = 0
                if j < SSTATE:
                    v = rebind[Scalar[f32]](So[0, h, j])
                    var ssq = warp.sum(v * v)
                    if lane == 0:
                        sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                if j < SSTATE:
                    var total: Float32 = 0
                    comptime for k in range(SSTATE // WARP_SIZE):
                        total += rebind[Scalar[f32]](sums[k])
                    var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
                    var z = rebind[Scalar[f32]](Zm[0, h * SSTATE + j])
                    ResB[0, h * SSTATE + j] = rebind[ResB.ElementType](
                        (v * scale * nwj * (z / (1 + exp(-z)))).cast[bf16]()
                    )
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 6)

            g = bid * ROW_WAVES + wave
            while g < H:
                var t = q8d_dot_u[4](ResB, wbuf.unsafe_offset(o9), 0, g * INNER, H * INNER + g * Q8S_INNER, INNER)
                if lane == 0:
                    X[0, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, g]) + t)
                g += nwaves
            si += 1
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 7)
        if dump != 0:
            dump_x(dbg, xp, 2 * layer)

        var wb = w + (7 if is_att else 10)
        var on = Int(off[unsafe_offset=wb])
        var og = Int(off[unsafe_offset=wb + 1])
        var ou = Int(off[unsafe_offset=wb + 2])
        var od = Int(off[unsafe_offset=wb + 3])
        var orr = Int(off[unsafe_offset=wb + 4])
        var osg = Int(off[unsafe_offset=wb + 5])
        var osu = Int(off[unsafe_offset=wb + 6])
        var osd = Int(off[unsafe_offset=wb + 7])
        var osi = Int(off[unsafe_offset=wb + 8])
        var up_off = og_up(og, ou)
        var sh_up_off = og_up(osg, osu)
        var Router = TileTensor(wbuf.unsafe_offset(orr).unsafe_bitcast[Scalar[f32]](), row_major[N_EXP, H]())
        var Gsh = TileTensor(wbuf.unsafe_offset(osi).unsafe_bitcast[Scalar[f32]](), row_major[H]())
        var q6 = layer == 34 or layer == 38 or layer == 39

        rms_moe(xp, wbuf.unsafe_offset(on).unsafe_bitcast[Scalar[f32]](), curb, fnp)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 8)

        var g = bid * ROW_WAVES + wave
        while g < N_EXP:
            var t = f32_row_dot[2](Fn2, Router, g, lane, H)
            if lane == 0:
                Logits[g] = rebind[Logits.ElementType](t)
            g += nwaves
        if moedot != 0:
            g = bid * ROW_WAVES + wave
            while g < H // 32:
                q8_quant_group(curb, xqp, xsp, g)
                g += nwaves
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 9)

        if bid == 0 and wave == 0:
            router_top8_sig_body(Logits, Idx, Wt, Fn1, Gsh, Sig, H)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 10)

        g = bid * ROW_WAVES + wave
        while g < TOPK * E_FFN + SH_FFN:
            if g < TOPK * E_FFN:
                var j = g // E_FFN
                var r = g % E_FFN
                var e = Int(rebind[Scalar[i32]](Idx[j]))
                var row_base = e * E_FFN * Q4K_ROW_H + r * Q4K_ROW_H
                var gg: Scalar[f32]
                var u: Scalar[f32]
                if moedot != 0:
                    gg = q4k_q8_row_dot[H // Q4K](xqp, xsp, 0, wbuf.unsafe_offset(og), row_base)
                    u = q4k_q8_row_dot[H // Q4K](xqp, xsp, 0, wbuf.unsafe_offset(og), row_base + up_off)
                else:
                    gg = q4k_dot_blocks(CurB, wbuf.unsafe_offset(og), 0, row_base, H)
                    u = q4k_dot_blocks(CurB, wbuf.unsafe_offset(og), 0, row_base + up_off, H)
                if lane == 0:
                    RoutedHf[g] = rebind[RoutedHf.ElementType]((gg / (Scalar[f32](1) + exp(-gg)) * u).cast[bf16]())
            else:
                var r = g - TOPK * E_FFN
                var row_base = r * Q8_ROW_H
                var gg = q8_dot_u[4](CurB, wbuf.unsafe_offset(osg), 0, row_base, H)
                var u = q8_dot_u[4](CurB, wbuf.unsafe_offset(osg), 0, row_base + sh_up_off, H)
                if lane == 0:
                    SharedHf[r] = rebind[SharedHf.ElementType]((gg / (Scalar[f32](1) + exp(-gg)) * u).cast[bf16]())
            g += nwaves
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 11)

        if moedot != 0 and not q6:
            g = bid * ROW_WAVES + wave
            while g < TOPK * E_FFN // 32:
                q8_quant_group(rhp, hqp, hsp, g)
                g += nwaves
            if not grid_barrier(ctr, gen, fail):
                return

        g = bid * ROW_WAVES + wave
        while g < H:
            var out = Scalar[f32](0)
            if q6:
                for j in range(TOPK):
                    var e = Int(rebind[Scalar[i32]](Idx[j]))
                    var row_base = e * H * Q6K_ROW_E + g * Q6K_ROW_E
                    var dot = q6k_row_dot(RoutedH, wbuf.unsafe_offset(od), j, row_base, E_FFN)
                    out += rebind[Scalar[f32]](Wt[j]) * dot
            elif moedot != 0:
                out = moe_down_q4k_dot_row[TOPK, E_FFN, E_FFN // Q4K](hqp, hsp, wbuf.unsafe_offset(od), Idx, Wt, g, H)
            else:
                for j in range(TOPK):
                    var e = Int(rebind[Scalar[i32]](Idx[j]))
                    var row_base = e * H * Q4K_ROW_E + g * Q4K_ROW_E
                    var dot = q4k_dot_blocks(RoutedH, wbuf.unsafe_offset(od), j, row_base, E_FFN)
                    out += rebind[Scalar[f32]](Wt[j]) * dot
            var sh = Scalar[f32](0)
            sh += rebind[Scalar[f32]](Sig[0]) * q8_dot_u[4](SharedH, wbuf.unsafe_offset(osd), 0, g * Q8_ROW_SH, SH_FFN)
            if lane == 0:
                X[0, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, g]) + out + sh)
            g += nwaves
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 12)
        if dump != 0:
            dump_x(dbg, xp, 2 * layer + 1)
        if is_att:
            w += W_ATT
        else:
            w += W_SSM
    stamp(prof, NPROF * NL)


@always_inline
def og_up(a: Int, b: Int) -> Int:
    return b - a


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_mega_moe_token[
    CsL: TensorLayout, SsL: TensorLayout, NL: Int, NAT: Int,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: MutPointer[Scalar[i64], MutAnyOrigin],
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    curb: MutPointer[Scalar[bf16], MutAnyOrigin],
    xg: MutPointer[Scalar[f32], MutAnyOrigin],
    fnp: MutPointer[Scalar[f32], MutAnyOrigin],
    qkv: MutPointer[Scalar[f32], MutAnyOrigin],
    zp: MutPointer[Scalar[f32], MutAnyOrigin],
    araw: MutPointer[Scalar[f32], MutAnyOrigin],
    braw: MutPointer[Scalar[f32], MutAnyOrigin],
    egp: MutPointer[Scalar[f32], MutAnyOrigin],
    betap: MutPointer[Scalar[f32], MutAnyOrigin],
    convp: MutPointer[Scalar[f32], MutAnyOrigin],
    sop: MutPointer[Scalar[f32], MutAnyOrigin],
    resb: MutPointer[Scalar[bf16], MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    qfp: MutPointer[Scalar[f32], MutAnyOrigin],
    kp: MutPointer[Scalar[f32], MutAnyOrigin],
    vp: MutPointer[Scalar[f32], MutAnyOrigin],
    qp: MutPointer[Scalar[f32], MutAnyOrigin],
    gatep: MutPointer[Scalar[f32], MutAnyOrigin],
    aop: MutPointer[Scalar[f32], MutAnyOrigin],
    aob: MutPointer[Scalar[bf16], MutAnyOrigin],
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    patt: MutPointer[Scalar[f32], MutAnyOrigin],
    idxp: MutPointer[Scalar[i32], MutAnyOrigin],
    wtp: MutPointer[Scalar[f32], MutAnyOrigin],
    sigp: MutPointer[Scalar[f32], MutAnyOrigin],
    rhp: MutPointer[Scalar[bf16], MutAnyOrigin],
    shp: MutPointer[Scalar[bf16], MutAnyOrigin],
    xqp: MutPointer[Scalar[i8], MutAnyOrigin],
    xsp: MutPointer[Scalar[f16], MutAnyOrigin],
    hqp: MutPointer[Scalar[i8], MutAnyOrigin],
    hsp: MutPointer[Scalar[f16], MutAnyOrigin],
    ctrp: MutPointer[Scalar[u32], MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, dump: Int32, att_split: Int32, moedot: Int32,
):
    mega_moe_body[CsL, SsL, NL, NAT](
        wbuf, off, xp, curb, xg, fnp, qkv, zp, araw, braw, egp, betap, convp, sop, resb,
        ConvState, SAll, qfp, kp, vp, qp, gatep, aop, aob, kc, vc, tab, patt, idxp, wtp, sigp,
        rhp, shp, xqp, xsp, hqp, hsp, ctrp, prof, dbg, ring, slots, pos, dump, att_split, moedot,
    )
