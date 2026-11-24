from std.atomic import Atomic, Ordering
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import exp, fma, log1p, rsqrt, sqrt
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from dattn import dattn_split_body, dattn_combine_body, dattn_nsplit, dkv_q, dkv_sc_off

from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS, ROW_VEC, SPLITK, SM, q4_dot_blocks
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS
from attn import HD, NQH, NKVH, KVT, KVQ8, TCAP, kv_off, kv_tab_off, NROT, attn_head_span
from model import H, FFN, QF, KV
from mega import grid_barrier, stamp, rope_cs, MEGA_G, RMS_EPS, ATT_SCALE, DATT_NLD

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
comptime W_ATT = 11
comptime W_SSM = 14
comptime NPROF = 16
comptime PATT = SPLITK * SM * FFN
comptime Q8S_FFN = (FFN // 32) * 2
comptime NWV = EW_THREADS // WARP_SIZE


@always_inline
def rms_rows[R: Int](
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    gp: MutPointer[Scalar[f32], MutAnyOrigin],
    cb: MutPointer[Scalar[bf16], MutAnyOrigin],
    fp: MutPointer[Scalar[f32], MutAnyOrigin],
):
    var X = TileTensor(xp, row_major[R, H]())
    var G = TileTensor(gp, row_major[H]())
    var O = TileTensor(cb, row_major[R, H]())
    var F = TileTensor(fp, row_major[R, H]())
    var tid = Int(thread_idx.x)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[R * NWV]()
    )
    var partial = SIMD[f32, R](0)
    var i = tid
    while i < H:
        comptime for r in range(R):
            var v = rebind[Scalar[f32]](X[r, i])
            partial[r] += v * v
        i += EW_THREADS
    comptime for r in range(R):
        var wsum = warp.sum(partial[r])
        if lane_id() == 0:
            sums[r * NWV + tid // WARP_SIZE] = rebind[sums.ElementType](wsum)
    barrier()
    var scale = SIMD[f32, R](0)
    comptime for r in range(R):
        var total: Float32 = 0
        comptime for w in range(NWV):
            total += rebind[Scalar[f32]](sums[r * NWV + w])
        scale[r] = rsqrt(total / Float32(H) + RMS_EPS)
    i = Int(block_idx.x) * EW_THREADS + tid
    while i < H:
        var g = rebind[Scalar[f32]](G[i])
        comptime for r in range(R):
            var h = (rebind[Scalar[f32]](X[r, i]) * scale[r] * g).cast[bf16]()
            O[r, i] = rebind[O.ElementType](h)
            F[r, i] = rebind[F.ElementType](h.cast[f32]())
        i += Int(grid_dim.x) * EW_THREADS
    barrier()


@always_inline
def f32_dot_rows[UNROLL: Int, R: Int, AL: TensorLayout, WL: TensorLayout](
    A: TileTensor[f32, AL, MutAnyOrigin],
    W: TileTensor[f32, WL, MutAnyOrigin],
    row: Int, lane: Int, K: Int,
) -> SIMD[f32, R]:
    comptime assert A.flat_rank == 2 and W.flat_rank == 2
    var Wv = W.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = Array[SIMD[f32, ROW_VEC], R](fill=SIMD[f32, ROW_VEC](0))
    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = Array[SIMD[f32, ROW_VEC], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[f32, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])
        comptime for u in range(UNROLL):
            comptime for r in range(R):
                var a = rebind[SIMD[f32, ROW_VEC]](Av[r, (kk + u * STEP) // ROW_VEC + lane]).cast[f32]()
                acc[r] += ws[u].cast[f32]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[f32, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[f32]()
        comptime for r in range(R):
            var a = rebind[SIMD[f32, ROW_VEC]](Av[r, kk // ROW_VEC + lane]).cast[f32]()
            acc[r] += w * a
        kk += STEP
    var out = SIMD[f32, R](0)
    comptime for r in range(R):
        out[r] = warp.sum(acc[r].reduce_add())
    return out


@always_inline
def q8_dot_rows[U: Int, R: Int, XLayout: TensorLayout](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    xrow: Int,
    xstride: Int,
    row_base: Int,
    k_dim: Int,
) -> SIMD[f32, R]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var b = lane // 2
    var h = lane % 2
    while b + (U - 1) * 16 < nb:
        var raws = Array[Scalar[u16], U](uninitialized=True)
        var qs = Array[SIMD[i8, 16], U](uninitialized=True)
        var xs = Array[SIMD[bf16, 16], U * R](uninitialized=True)
        comptime for u in range(U):
            var base = row_base + (b + u * 16) * 34
            raws[u] = W.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
            qs[u] = W.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
            comptime for r in range(R):
                xs[u * R + r] = rebind[SIMD[bf16, 16]](Xv[xrow + r * xstride, 2 * (b + u * 16) + h])
        comptime for u in range(U):
            var d = bitcast[f16, 1](SIMD[u16, 1](raws[u])).cast[f32]()[0]
            var v = (SIMD[f32, 16](d) * qs[u].cast[f32]()).cast[bf16]().cast[f32]()
            comptime for r in range(R):
                acc[r] = fma(v, xs[u * R + r].cast[f32](), acc[r])
        b += U * 16
    while b < nb:
        var base = row_base + b * 34
        var raw = W.unsafe_offset(base).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        comptime for r in range(R):
            var a = rebind[SIMD[bf16, 16]](Xv[xrow + r * xstride, 2 * b + h]).cast[f32]()
            acc[r] = fma(v, a, acc[r])
        b += 16
    var out = SIMD[f32, R](0)
    comptime for r in range(R):
        out[r] = warp.sum(acc[r].reduce_add())
    return out


@always_inline
def wq[N: Int, K: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[u8, type_of(row_major[N, K // 2]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o).unsafe_bitcast[Scalar[u8]](), row_major[N, K // 2]())


@always_inline
def ws[N: Int, K: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f16, type_of(row_major[N, K // 32]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o + N * K // 2).unsafe_bitcast[Scalar[f16]](), row_major[N, K // 32]())


@always_inline
@always_inline
@always_inline
def dump_x(
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    slot: Int, row: Int,
):
    if block_idx.x == 0:
        var i = Int(thread_idx.x)
        while i < H:
            dbg[unsafe_offset=slot * H + i] = xp[unsafe_offset=row * H + i]
            i += ROW_THREADS


@always_inline
def pick(r: Int, a: Int, b: Int) -> Int:
    return a if r == 0 else b


@always_inline
@always_inline
def mega_dense_body[
    CsL: TensorLayout, SsL: TensorLayout, NL: Int, NAT: Int, R: Int,
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
    pos1: Int32, ring1: Int32, tabstr: Int32,
):
    comptime assert ConvState.flat_rank == 4 and SAll.flat_rank == 5
    comptime assert R == 1 or R == 2
    comptime assert R * (NQH + NKVH) <= MEGA_G and R * NH_V <= MEGA_G
    comptime assert ROW_THREADS == EW_THREADS and ROW_THREADS == HD and SSTATE <= ROW_THREADS
    comptime NSSM = NL - NL // 4
    comptime UQ = 4 // R
    comptime TE = TOPK * E_FFN
    var csp = ConvState.ptr
    var ssp = SAll.ptr
    var X = TileTensor(xp, row_major[R, H]())
    var CurB = TileTensor(curb, row_major[R, H]())
    var Xg = TileTensor(xg, row_major[R, H]())
    var Fn2 = TileTensor(fnp, row_major[R, H]())
    var Qkvm = TileTensor(qkv, row_major[R, CONV]())
    var Zm = TileTensor(zp, row_major[R, INNER]())
    var Araw = TileTensor(araw, row_major[R, NH_V]())
    var Braw = TileTensor(braw, row_major[R, NH_V]())
    var Eg = TileTensor(egp, row_major[R, NH_V]())
    var Beta = TileTensor(betap, row_major[R, NH_V]())
    var Conv = TileTensor(convp, row_major[R, CONV]())
    var So = TileTensor(sop, row_major[R, NH_V, SSTATE]())
    var ResB = TileTensor(resb, row_major[R, INNER]())
    var Qfm = TileTensor(qfp, row_major[R, QF]())
    var Kflat = TileTensor(kp, row_major[R, KV]())
    var Vflat = TileTensor(vp, row_major[R, KV]())
    var Q = TileTensor(qp, row_major[R * NQH, HD]())
    var Gate = TileTensor(gatep, row_major[R, ATT]())
    var Ao = TileTensor(aop, row_major[R * NQH, HD]())
    var AoB = TileTensor(aob, row_major[R, ATT]())
    var Kc = TileTensor(kc, row_major[TCAP]())
    var Vc = TileTensor(vc, row_major[TCAP]())
    var Pg = TileTensor(patt, row_major[R, FFN]())
    var Pu = TileTensor(wtp, row_major[R, FFN]())
    var FgB = TileTensor(rhp, row_major[R, FFN]())
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var nwaves = nblk * ROW_WAVES
    var ctr = ctrp
    var gen = ctrp.unsafe_offset(1)
    var fail = ctrp.unsafe_offset(2)
    var si = 0
    var ai = 0
    var sl = Int(slots)
    var p0 = Int(pos)
    var p1 = Int(pos1) if R == 2 else Int(pos)
    var rg0 = Int(ring)
    var rg1 = Int(ring1) if R == 2 else Int(ring)
    var tabs = Int(tabstr)
    var cs_n = sl * NSSM * 3 * CONV
    var ss_n = sl * NSSM * NH_V * SSTATE * SSTATE
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
        rms_rows[R](xp, wbuf.unsafe_offset(o0).unsafe_bitcast[Scalar[f32]](), curb, xg)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 1)
        if is_att:
            var Wqq = wq[QF, H](wbuf, o1)
            var Wqs = ws[QF, H](wbuf, o1)
            var Wkq = wq[KV, H](wbuf, o2)
            var Wks = ws[KV, H](wbuf, o2)
            var Wvq = wq[KV, H](wbuf, o3)
            var Wvs = ws[KV, H](wbuf, o3)
            var g = bid * ROW_WAVES + wave
            while g < QF + KV + KV:
                if g < QF:
                    var t = q4_dot_blocks[R, 2](CurB, Wqq, Wqs, g, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Qfm[r, g] = rebind[Qfm.ElementType](t[r])
                elif g < QF + KV:
                    var rr = g - QF
                    var t = q4_dot_blocks[R, 2](CurB, Wkq, Wks, rr, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Kflat[r, rr] = rebind[Kflat.ElementType](t[r])
                else:
                    var rr = g - QF - KV
                    var t = q4_dot_blocks[R, 2](CurB, Wvq, Wvs, rr, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Vflat[r, rr] = rebind[Vflat.ElementType](t[r])
                g += nwaves
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 2)

            var Qn = TileTensor(wbuf.unsafe_offset(o4).unsafe_bitcast[Scalar[f32]](), row_major[HD]())
            var Kn = TileTensor(wbuf.unsafe_offset(o5).unsafe_bitcast[Scalar[f32]](), row_major[HD]())
            if bid < R * NQH:
                var r = bid // NQH
                var h = bid % NQH
                var p = pick(r, p0, p1)
                var v = rebind[Scalar[f32]](Qfm[r, h * 2 * HD + tid])
                Gate[r, h * HD + tid] = rebind[Gate.ElementType](Qfm[r, h * 2 * HD + HD + tid])
                var ssq = warp.sum(v * v)
                if lane == 0:
                    sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                var total: Float32 = 0
                comptime for k in range(HD // WARP_SIZE):
                    total += rebind[Scalar[f32]](sums[k])
                Q[r * NQH + h, tid] = rebind[Q.ElementType](
                    v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Qn[tid])
                )
                barrier()
                if tid < NROT // 2:
                    var cs = rope_cs(tid, p)
                    var x0 = rebind[Scalar[f32]](Q[r * NQH + h, tid])
                    var x1 = rebind[Scalar[f32]](Q[r * NQH + h, tid + NROT // 2])
                    Q[r * NQH + h, tid] = rebind[Q.ElementType](x0 * cs[0] - x1 * cs[1])
                    Q[r * NQH + h, tid + NROT // 2] = rebind[Q.ElementType](x0 * cs[1] + x1 * cs[0])
            elif bid < R * NQH + R * NKVH:
                var r = (bid - R * NQH) // NKVH
                var h = (bid - R * NQH) % NKVH
                var p = pick(r, p0, p1)
                var tabr = tab.unsafe_offset(r * tabs)
                var v = rebind[Scalar[f32]](Kflat[r, h * HD + tid])
                var ssq = warp.sum(v * v)
                if lane == 0:
                    sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                var total: Float32 = 0
                comptime for k in range(HD // WARP_SIZE):
                    total += rebind[Scalar[f32]](sums[k])
                Kflat[r, h * HD + tid] = rebind[Kflat.ElementType](
                    v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Kn[tid])
                )
                barrier()
                if tid < NROT // 2:
                    var cs = rope_cs(tid, p)
                    var x0 = rebind[Scalar[f32]](Kflat[r, h * HD + tid])
                    var x1 = rebind[Scalar[f32]](Kflat[r, h * HD + tid + NROT // 2])
                    Kflat[r, h * HD + tid] = rebind[Kflat.ElementType](x0 * cs[0] - x1 * cs[1])
                    Kflat[r, h * HD + tid + NROT // 2] = rebind[Kflat.ElementType](x0 * cs[1] + x1 * cs[0])
                barrier()
                comptime if KVQ8:
                    var kx = Float32(0)
                    var vx = Float32(0)
                    if tid < HD:
                        kx = rebind[Scalar[f32]](Kflat[r, h * HD + tid])
                        vx = rebind[Scalar[f32]](Vflat[r, h * HD + tid])
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
                        var rb = kv_tab_off[NAT](tabr, p, ai, h)
                        Kc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(kx, Float32(127) / km if km > 0 else Float32(0)))
                        Vc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(vx, Float32(127) / vm if vm > 0 else Float32(0)))
                        if tid == 0:
                            Kc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, p)] = km / 127
                            Vc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, p)] = vm / 127
                else:
                    var kb = kv_tab_off[NAT](tabr, p, ai, h) + tid
                    Kc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](Kflat[r, h * HD + tid]).cast[KVT]())
                    Vc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](rebind[Scalar[f32]](Vflat[r, h * HD + tid]).cast[KVT]())
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 3)

            var ns0 = 0
            var ns1 = 0
            var items0 = NQH
            var items1 = NQH if R == 2 else 0
            if p0 + 1 > Int(att_split):
                ns0 = dattn_nsplit[HD, DATT_NLD, NKVH](p0 + 1, 1, MEGA_G)
                items0 = NKVH * ns0
            comptime if R == 2:
                if p1 + 1 > Int(att_split):
                    ns1 = dattn_nsplit[HD, DATT_NLD, NKVH](p1 + 1, 1, MEGA_G)
                    items1 = NKVH * ns1
            var combine = ns0 > 1 or ns1 > 1
            var it = bid
            while it < items0 + items1:
                var r = 0
                var li = it
                if li >= items0:
                    r = 1
                    li -= items0
                var T = pick(r, p0, p1) + 1
                var tabr = tab.unsafe_offset(r * tabs)
                var nsr = pick(r, ns0, ns1)
                if nsr == 0:
                    var h = li
                    var kvh = h // (NQH // NKVH)
                    var res = attn_head_span[NAT=NAT](Q, Kc, Vc, tabr, qs, scores, sums, r * NQH + h, kvh, 0, T, tid, lane, ATT_SCALE, ai)
                    var inv = 1 / res[1]
                    Ao[r * NQH + h, tid] = rebind[Ao.ElementType](res[2] * inv)
                else:
                    var ns = nsr
                    dattn_split_body[HD, NQH, NKVH, KVT, NAT, DATT_NLD, False](
                        Q, Kc, Vc, Ao, Pg, tabr, li // ns, li % ns, r, ns, T, ATT_SCALE, ai, tid
                    )
                it += nblk
                barrier()
            if combine:
                if not grid_barrier(ctr, gen, fail):
                    return
                it = bid
                while it < R * NQH:
                    var nsr = pick(it // NQH, ns0, ns1)
                    if nsr > 1:
                        dattn_combine_body[HD, MEGA_G](Pg, Ao, it, nsr, tid)
                    it += nblk
                    barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 4)

            var i = bid * ROW_THREADS + tid
            while i < R * ATT:
                var r = i // ATT
                var ii = i % ATT
                var gg = rebind[Scalar[f32]](Gate[r, ii])
                AoB[r, ii] = rebind[AoB.ElementType](
                    (rebind[Scalar[f32]](Ao[r * NQH + ii // HD, ii % HD]) * (1 / (1 + exp(-gg)))).cast[bf16]()
                )
                i += nblk * ROW_THREADS
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 5)

            var Woq = wq[H, ATT](wbuf, o6)
            var Wos = ws[H, ATT](wbuf, o6)
            g = bid * ROW_WAVES + wave
            while g < H:
                var t = q4_dot_blocks[R, 2](AoB, Woq, Wos, g, lane, 0, ATT // 32, R)
                if lane == 0:
                    comptime for r in range(R):
                        X[r, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, g]) + t[r])
                g += nwaves
            ai += 1
        else:
            var o7 = Int(off[unsafe_offset=w + 7])
            var o8 = Int(off[unsafe_offset=w + 8])
            var o9 = Int(off[unsafe_offset=w + 9])
            var Wqkvq = wq[CONV, H](wbuf, o1)
            var Wqkvs = ws[CONV, H](wbuf, o1)
            var Wzq = wq[INNER, H](wbuf, o2)
            var Wzs = ws[INNER, H](wbuf, o2)
            var WaQ = wq[NH_V, H](wbuf, o3)
            var WaS = ws[NH_V, H](wbuf, o3)
            var WbQ = wq[NH_V, H](wbuf, o4)
            var WbS = ws[NH_V, H](wbuf, o4)
            var Cw = TileTensor(wbuf.unsafe_offset(o5).unsafe_bitcast[Scalar[f32]](), row_major[CONV, 4]())
            var SsmA = TileTensor(wbuf.unsafe_offset(o6).unsafe_bitcast[Scalar[f32]](), row_major[NH_V]())
            var DtB = TileTensor(wbuf.unsafe_offset(o7).unsafe_bitcast[Scalar[f32]](), row_major[NH_V]())
            var Nw = TileTensor(wbuf.unsafe_offset(o8).unsafe_bitcast[Scalar[f32]](), row_major[SSTATE]())
            var g = bid * ROW_WAVES + wave
            while g < CONV + INNER + NH_V + NH_V:
                if g < CONV:
                    var t = q4_dot_blocks[R, 2](CurB, Wqkvq, Wqkvs, g, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Qkvm[r, g] = rebind[Qkvm.ElementType](t[r])
                elif g < CONV + INNER:
                    var rr = g - CONV
                    var t = q4_dot_blocks[R, 2](CurB, Wzq, Wzs, rr, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Zm[r, rr] = rebind[Zm.ElementType](t[r])
                elif g < CONV + INNER + NH_V:
                    var rr = g - CONV - INNER
                    var t = q4_dot_blocks[R, 2](CurB, WaQ, WaS, rr, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Araw[r, rr] = rebind[Araw.ElementType](t[r])
                else:
                    var rr = g - CONV - INNER - NH_V
                    var t = q4_dot_blocks[R, 2](CurB, WbQ, WbS, rr, lane, 0, H // 32, R)
                    if lane == 0:
                        comptime for r in range(R):
                            Braw[r, rr] = rebind[Braw.ElementType](t[r])
                g += nwaves
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 2)

            if bid < R and tid < NH_V:
                var r = bid
                var h = tid
                var sa = rebind[Scalar[f32]](SsmA[h])
                var db = rebind[Scalar[f32]](DtB[h])
                var br = rebind[Scalar[f32]](Braw[r, h])
                Beta[r, h] = rebind[Beta.ElementType](1 / (1 + exp(-br)))
                var asum = rebind[Scalar[f32]](Araw[r, h]) + db
                var sp = log1p(exp(asum))
                Eg[r, h] = rebind[Eg.ElementType](exp(sp * sa))
            var c = bid * ROW_THREADS + tid
            while c < R * CONV:
                var r = c // CONV
                var cc = c % CONV
                var rs = pick(r, rg0, rg1) % sl
                var wsl = (pick(r, rg0, rg1) + 1) % sl
                var cw0 = rebind[Scalar[f32]](Cw[cc, 0])
                var cw1 = rebind[Scalar[f32]](Cw[cc, 1])
                var cw2 = rebind[Scalar[f32]](Cw[cc, 2])
                var cw3 = rebind[Scalar[f32]](Cw[cc, 3])
                var rbase = r * cs_n + (rs * NSSM + si) * 3 * CONV + cc
                var wbase = r * cs_n + (wsl * NSSM + si) * 3 * CONV + cc
                var w0 = csp[unsafe_offset=rbase]
                var w1 = csp[unsafe_offset=rbase + CONV]
                var w2 = csp[unsafe_offset=rbase + 2 * CONV]
                var w3 = rebind[Scalar[f32]](Qkvm[r, cc])
                var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
                Conv[r, cc] = rebind[Conv.ElementType](acc / (1 + exp(-acc)))
                csp[unsafe_offset=wbase] = w1
                csp[unsafe_offset=wbase + CONV] = w2
                csp[unsafe_offset=wbase + 2 * CONV] = w3
                c += nblk * ROW_THREADS
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 3)

            if bid < R * NH_V:
                var r = bid // NH_V
                var head = bid % NH_V
                var base = head * SSTATE
                var v: Float32 = 0
                if tid < SSTATE:
                    v = rebind[Scalar[f32]](Conv[r, base + tid])
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
                    Conv[r, base + tid] = rebind[Conv.ElementType](v * inv)
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 4)

            if bid < R * NH_V:
                var r = bid // NH_V
                var h = bid % NH_V
                var j = tid
                var kh = h % NH_K
                var rs = pick(r, rg0, rg1) % sl
                var wsl = (pick(r, rg0, rg1) + 1) % sl
                if j < SSTATE:
                    kq[0, j] = rebind[kq.ElementType](Conv[r, kh * SSTATE + j])
                    kq[1, j] = rebind[kq.ElementType](Conv[r, KDIM + kh * SSTATE + j])
                barrier()
                if j < SSTATE:
                    var eg = rebind[Scalar[f32]](Eg[r, h])
                    var beta = rebind[Scalar[f32]](Beta[r, h])
                    var vj = rebind[Scalar[f32]](Conv[r, 2 * KDIM + h * SSTATE + j])
                    var rbase = r * ss_n + ((rs * NSSM + si) * NH_V + h) * SSTATE * SSTATE + j
                    var wbase = r * ss_n + ((wsl * NSSM + si) * NH_V + h) * SSTATE * SSTATE + j
                    comptime CHK = 32
                    var sk: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = ssp[unsafe_offset=rbase + (c * CHK + ii) * SSTATE]
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            sk = fma(t, rebind[Scalar[f32]](kq[1, c * CHK + ii]), sk)
                    var d = (vj - sk) * beta
                    var o: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = ssp[unsafe_offset=rbase + (c * CHK + ii) * SSTATE]
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            var s = fma(rebind[Scalar[f32]](kq[1, c * CHK + ii]), d, t)
                            ssp[unsafe_offset=wbase + (c * CHK + ii) * SSTATE] = s
                            o = fma(s, rebind[Scalar[f32]](kq[0, c * CHK + ii]), o)
                    So[r, h, j] = rebind[So.ElementType](o)
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 5)

            if bid < R * NH_V:
                var r = bid // NH_V
                var h = bid % NH_V
                var j = tid
                var nwj: Float32 = 0
                if j < SSTATE:
                    nwj = rebind[Scalar[f32]](Nw[j])
                var v: Float32 = 0
                if j < SSTATE:
                    v = rebind[Scalar[f32]](So[r, h, j])
                    var ssq = warp.sum(v * v)
                    if lane == 0:
                        sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                if j < SSTATE:
                    var total: Float32 = 0
                    comptime for k in range(SSTATE // WARP_SIZE):
                        total += rebind[Scalar[f32]](sums[k])
                    var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
                    var z = rebind[Scalar[f32]](Zm[r, h * SSTATE + j])
                    ResB[r, h * SSTATE + j] = rebind[ResB.ElementType](
                        (v * scale * nwj * (z / (1 + exp(-z)))).cast[bf16]()
                    )
                barrier()
            if not grid_barrier(ctr, gen, fail):
                return
            stamp(prof, NPROF * layer + 6)

            var Wsoq = wq[H, INNER](wbuf, o9)
            var Wsos = ws[H, INNER](wbuf, o9)
            g = bid * ROW_WAVES + wave
            while g < H:
                var t = q4_dot_blocks[R, 2](ResB, Wsoq, Wsos, g, lane, 0, INNER // 32, R)
                if lane == 0:
                    comptime for r in range(R):
                        X[r, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, g]) + t[r])
                g += nwaves
            si += 1
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 7)
        comptime if R == 1:
            if dump != 0:
                dump_x(dbg, xp, 2 * layer, 0)

        var wb = w + (7 if is_att else 10)
        var on = Int(off[unsafe_offset=wb])
        var og = Int(off[unsafe_offset=wb + 1])
        var ou = Int(off[unsafe_offset=wb + 2])
        var od = Int(off[unsafe_offset=wb + 3])

        rms_rows[R](xp, wbuf.unsafe_offset(on).unsafe_bitcast[Scalar[f32]](), curb, xg)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 8)

        var Wgq = wq[FFN, H](wbuf, og)
        var Wgs = ws[FFN, H](wbuf, og)
        var Wuq = wq[FFN, H](wbuf, ou)
        var Wus = ws[FFN, H](wbuf, ou)
        var Wdq = wq[H, FFN](wbuf, od)
        var Wds = ws[H, FFN](wbuf, od)
        var g = bid * ROW_WAVES + wave
        while g < FFN + FFN:
            if g < FFN:
                var t = q4_dot_blocks[R, 2](CurB, Wgq, Wgs, g, lane, 0, H // 32, R)
                if lane == 0:
                    comptime for r in range(R):
                        Pg[r, g] = rebind[Pg.ElementType](t[r])
            else:
                var rr = g - FFN
                var t = q4_dot_blocks[R, 2](CurB, Wuq, Wus, rr, lane, 0, H // 32, R)
                if lane == 0:
                    comptime for r in range(R):
                        Pu[r, rr] = rebind[Pu.ElementType](t[r])
            g += nwaves
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 9)

        var c = bid * ROW_THREADS + tid
        while c < R * FFN:
            var r = c // FFN
            var cc = c % FFN
            var gg = rebind[Scalar[f32]](Pg[r, cc])
            var u = rebind[Scalar[f32]](Pu[r, cc])
            var silu = gg / (1 + exp(-gg))
            FgB[r, cc] = rebind[FgB.ElementType]((silu * u).cast[bf16]())
            c += nblk * ROW_THREADS
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 10)

        g = bid * ROW_WAVES + wave
        while g < H:
            var t = q4_dot_blocks[R, 2](FgB, Wdq, Wds, g, lane, 0, FFN // 32, R)
            if lane == 0:
                comptime for r in range(R):
                    X[r, g] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, g]) + t[r])
            g += nwaves
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, NPROF * layer + 12)
        comptime if R == 2:
            if dump != 0:
                dump_x(dbg, xp, layer + 2, 0)
        else:
            if dump != 0:
                dump_x(dbg, xp, 2 * layer + 1, 0)
        if is_att:
            w += W_ATT
        else:
            w += W_SSM
    stamp(prof, NPROF * NL)


@always_inline
@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_mega_rows[
    CsL: TensorLayout, SsL: TensorLayout, NL: Int, NAT: Int, R: Int = 1,
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
    pos1: Int32, ring1: Int32, tabstr: Int32,
):
    mega_dense_body[CsL, SsL, NL, NAT, R](
        wbuf, off, xp, curb, xg, fnp, qkv, zp, araw, braw, egp, betap, convp, sop, resb,
        ConvState, SAll, qfp, kp, vp, qp, gatep, aop, aob, kc, vc, tab, patt, idxp, wtp, sigp,
        rhp, shp, xqp, xsp, hqp, hsp, ctrp, prof, dbg, ring, slots, pos, dump, att_split, moedot,
        pos1, ring1, tabstr,
    )
