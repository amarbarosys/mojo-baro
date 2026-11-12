from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import fma
from bexp import cos, exp, log, rsqrt, sin
from std.sys import get_defined_string
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import mma
from dattn import dkv_sc, dkv_sc_off, dkv_dq, dkv_q
from model import HD, NQH, NKVH, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE, IS_MOE
from std.utils import StaticTuple

comptime f32 = DType.float32
comptime KVQ = get_defined_string["BARO_KVQ", "int8" if IS_MOE else "f32"]()
comptime KVQ8 = KVQ == "int8"
comptime KVT = DType.int8 if KVQ8 else (DType.bfloat16 if KVQ == "bf16" else DType.float32)
comptime KVPAGE = 128
comptime KVPSH = 7
comptime KVPAD = KVPAGE * 4 if KVQ8 else 0
comptime KVHSTR = KVPAGE * HD + KVPAD
comptime TCAP = 1 << 24


@always_inline
def kv_off[NAT: Int, HD_: Int = HD, NKVH_: Int = NKVH](t: Int, att_i: Int, kvh: Int) -> Int:
    return (((t >> KVPSH) * NAT + att_i) * NKVH_ + kvh) * (KVPAGE * HD_ + KVPAD) + (t & (KVPAGE - 1)) * HD_


@always_inline
def kv_tab_off[NAT: Int, HD_: Int = HD, NKVH_: Int = NKVH](tab: MutPointer[Scalar[DType.int32], MutAnyOrigin], t: Int, att_i: Int, kvh: Int) -> Int:
    return ((Int(tab[t >> KVPSH]) * NAT + att_i) * NKVH_ + kvh) * (KVPAGE * HD_ + KVPAD) + (t & (KVPAGE - 1)) * HD_


def amar_head_rmsnorm[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    eps: Float32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1
    var h = block_idx.x
    var d = thread_idx.x
    var v = rebind[Scalar[f32]](X[h, d])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[d // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    X[h, d] = rebind[X.ElementType](
        v * rsqrt(total / Float32(HD) + eps) * rebind[Scalar[f32]](G[d])
    )


@always_inline
def attn_head_span[
    QLayout: TensorLayout, KLayout: TensorLayout,
    QsL: TensorLayout, ScL: TensorLayout, RdL: TensorLayout, NAT: Int, HD_: Int = HD, NKVH_: Int = NKVH
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    mut qs: TileTensor[f32, QsL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut scores: TileTensor[f32, ScL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut red: TileTensor[f32, RdL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    qrow: Int, kvh: Int, t_lo: Int, t_hi: Int, tid: Int, lane: Int, scale: Float32, att_i: Int,
) -> SIMD[f32, 4]:
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1
    comptime assert qs.flat_rank == 1 and scores.flat_rank == 1 and red.flat_rank == 1
    comptime assert HD_ % WARP_SIZE == 0
    var wave = tid // WARP_SIZE
    if tid < HD_:
        qs[tid] = rebind[qs.ElementType](Q[qrow, tid])
    barrier()
    var qv = qs.vectorize[8]()
    var kp = Kc.ptr
    var vp = Vc.ptr
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var o = Float32(0)
    var t0 = t_lo
    while t0 < t_hi:
        barrier()
        var s = Float32(-3.4e38)
        if tid < HD_:
            var t = t0 + tid
            if t < t_hi:
                var kb = kv_tab_off[NAT, HD_, NKVH_](tab, t, att_i, kvh)
                var Kr = TileTensor(kp.unsafe_offset(kb), row_major[HD_]()).vectorize[8]()
                var ksc = dkv_sc[KVT, HD_](kp, kb, t)
                var acc: Float32 = 0
                for d8 in range(HD_ // 8):
                    var k8 = dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Kr[d8]), ksc)
                    var q8 = rebind[SIMD[f32, 8]](qv[d8])
                    comptime for j in range(8):
                        acc += q8[j] * k8[j]
                s = acc * scale
            scores[tid] = rebind[scores.ElementType](s)
            var wmax = warp.max(s)
            if lane == 0:
                red[wave] = rebind[red.ElementType](wmax)
        barrier()
        var cmax = Float32(-3.4e38)
        comptime for w in range(HD_ // WARP_SIZE):
            var sc = rebind[Scalar[f32]](red[w])
            if sc > cmax:
                cmax = sc
        var m_new = max(m_run, cmax)
        var alpha = exp(m_run - m_new)
        barrier()
        if tid < HD_:
            var e = exp(s - m_new)
            scores[tid] = rebind[scores.ElementType](e)
            var wsum = warp.sum(e)
            if lane == 0:
                red[wave] = rebind[red.ElementType](wsum)
        barrier()
        var csum = Float32(0)
        comptime for w in range(HD_ // WARP_SIZE):
            csum += rebind[Scalar[f32]](red[w])
        l_run = l_run * alpha + csum
        m_run = m_new
        o = o * alpha
        if tid < HD_:
            var n = t_hi - t0
            if n > HD_:
                n = HD_
            var tt = 0
            while tt + 8 <= n:
                var v = Array[Float32, 8](uninitialized=True)
                var sc = Array[Float32, 8](uninitialized=True)
                comptime for j in range(8):
                    comptime if KVQ8:
                        var vb = kv_tab_off[NAT, HD_, NKVH_](tab, t0 + tt + j, att_i, kvh)
                        v[j] = vp[unsafe_offset=vb + tid].cast[f32]() * dkv_sc[KVT, HD_](vp, vb, t0 + tt + j)
                    else:
                        v[j] = vp[unsafe_offset=kv_tab_off[NAT, HD_, NKVH_](tab, t0 + tt + j, att_i, kvh) + tid].cast[f32]()
                    sc[j] = rebind[Scalar[f32]](scores[tt + j])
                comptime for j in range(8):
                    o += sc[j] * v[j]
                tt += 8
            while tt < n:
                comptime if KVQ8:
                    var vb = kv_tab_off[NAT, HD_, NKVH_](tab, t0 + tt, att_i, kvh)
                    o += rebind[Scalar[f32]](scores[tt]) * vp[unsafe_offset=vb + tid].cast[f32]() * dkv_sc[KVT, HD_](vp, vb, t0 + tt)
                else:
                    o += rebind[Scalar[f32]](scores[tt]) * vp[unsafe_offset=kv_tab_off[NAT, HD_, NKVH_](tab, t0 + tt, att_i, kvh) + tid].cast[f32]()
                tt += 1
        t0 += HD_
    return SIMD[f32, 4](m_run, l_run, o, 0)


@always_inline
def attn_head_body[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout,
    QsL: TensorLayout, ScL: TensorLayout, RdL: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    mut qs: TileTensor[f32, QsL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut scores: TileTensor[f32, ScL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut red: TileTensor[f32, RdL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    qrow: Int, kvh: Int, T: Int, tid: Int, lane: Int, scale: Float32, att_i: Int,
):
    comptime assert O.flat_rank == 2
    var res = attn_head_span[NAT=NAT](Q, Kc, Vc, tab, qs, scores, red, qrow, kvh, 0, T, tid, lane, scale, att_i)
    if tid < HD:
        var inv = 1 / res[1]
        O[qrow, tid] = rebind[O.ElementType](res[2] * inv)




def amar_attn_decode[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and O.flat_rank == 2
    var h = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var kvh = h // (NQH // NKVH)
    var T = Int(t_len) + r
    var qrow = r * NQH + h
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var O_ = O
    attn_head_body[NAT=NAT](Q, Kc, Vc, tab, O_, qs, scores, red, qrow, kvh, T, tid, Int(lane_id()), scale, Int(att_i))


def amar_attn_decode_gqa[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and O.flat_rank == 2
    comptime G = NQH // NKVH
    comptime NW = HD // WARP_SIZE
    var kvh = Int(block_idx.x)
    var r = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var T = Int(t_len) + r
    var q0 = r * NQH + kvh * G
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G, HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G, HD]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G, NW]())
    comptime for g in range(G):
        qs[g, tid] = rebind[qs.ElementType](Q[q0 + g, tid])
    barrier()
    var qv = qs.vectorize[1, 8]()
    var kp = Kc.ptr
    var vp = Vc.ptr
    var m_run = Array[Float32, G](fill=Float32(-3.4e38))
    var l_run = Array[Float32, G](fill=Float32(0))
    var o = Array[Float32, G](fill=Float32(0))
    var t0 = 0
    while t0 < T:
        barrier()
        var s = Array[Float32, G](fill=Float32(-3.4e38))
        var t = t0 + tid
        if t < T:
            var kb = kv_tab_off[NAT](tab, t, Int(att_i), kvh)
            var Kr = TileTensor(kp.unsafe_offset(kb), row_major[HD]()).vectorize[8]()
            var ksc = dkv_sc[KVT, HD](kp, kb, t)
            var acc = Array[Float32, G](fill=Float32(0))
            for d8 in range(HD // 8):
                var k8 = dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Kr[d8]), ksc)
                comptime for g in range(G):
                    var q8 = rebind[SIMD[f32, 8]](qv[g, d8])
                    comptime for j in range(8):
                        acc[g] += q8[j] * k8[j]
            comptime for g in range(G):
                s[g] = acc[g] * scale
        comptime for g in range(G):
            scores[g, tid] = rebind[scores.ElementType](s[g])
            var wmax = warp.max(s[g])
            if lane == 0:
                red[g, wave] = rebind[red.ElementType](wmax)
        barrier()
        var m_new = Array[Float32, G](fill=Float32(0))
        var alpha = Array[Float32, G](fill=Float32(0))
        comptime for g in range(G):
            var cmax = Float32(-3.4e38)
            comptime for w in range(NW):
                var sc = rebind[Scalar[f32]](red[g, w])
                if sc > cmax:
                    cmax = sc
            m_new[g] = max(m_run[g], cmax)
            alpha[g] = exp(m_run[g] - m_new[g])
        barrier()
        comptime for g in range(G):
            var e = exp(s[g] - m_new[g])
            scores[g, tid] = rebind[scores.ElementType](e)
            var wsum = warp.sum(e)
            if lane == 0:
                red[g, wave] = rebind[red.ElementType](wsum)
        barrier()
        comptime for g in range(G):
            var csum = Float32(0)
            comptime for w in range(NW):
                csum += rebind[Scalar[f32]](red[g, w])
            l_run[g] = l_run[g] * alpha[g] + csum
            m_run[g] = m_new[g]
            o[g] = o[g] * alpha[g]
        var n = min(T - t0, HD)
        var tt = 0
        while tt + 8 <= n:
            var v = Array[Float32, 8](uninitialized=True)
            comptime for j in range(8):
                comptime if KVQ8:
                    var vb = kv_tab_off[NAT](tab, t0 + tt + j, Int(att_i), kvh)
                    v[j] = vp[unsafe_offset=vb + tid].cast[f32]() * dkv_sc[KVT, HD](vp, vb, t0 + tt + j)
                else:
                    v[j] = vp[unsafe_offset=kv_tab_off[NAT](tab, t0 + tt + j, Int(att_i), kvh) + tid].cast[f32]()
            comptime for g in range(G):
                var sc = Array[Float32, 8](uninitialized=True)
                comptime for j in range(8):
                    sc[j] = rebind[Scalar[f32]](scores[g, tt + j])
                comptime for j in range(8):
                    o[g] += sc[j] * v[j]
            tt += 8
        while tt < n:
            var vb = kv_tab_off[NAT](tab, t0 + tt, Int(att_i), kvh)
            var v1 = vp[unsafe_offset=vb + tid].cast[f32]()
            comptime for g in range(G):
                comptime if KVQ8:
                    o[g] += rebind[Scalar[f32]](scores[g, tt]) * v1 * dkv_sc[KVT, HD](vp, vb, t0 + tt)
                else:
                    o[g] += rebind[Scalar[f32]](scores[g, tt]) * v1
            tt += 1
        t0 += HD
    comptime for g in range(G):
        var inv = 1 / l_run[g]
        O[q0 + g, tid] = rebind[O.ElementType](o[g] * inv)


def amar_gate_mul[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Gate.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var g = rebind[Scalar[f32]](Gate[i])
    X[i] = rebind[X.ElementType](
        rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))
    )


def amar_gate_mul_cast[
    XLayout: TensorLayout, GLayout: TensorLayout, OLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[DType.bfloat16, OLayout, MutAnyOrigin],
    n: Int32,
):
    comptime assert X.flat_rank == 1 and Gate.flat_rank == 1 and O.flat_rank == 1
    var i = global_idx.x
    if i >= Int(n):
        return
    var g = rebind[Scalar[f32]](Gate[i])
    O[i] = rebind[O.ElementType](
        (rebind[Scalar[f32]](X[i]) * (1 / (1 + exp(-g)))).cast[
            DType.bfloat16
        ]()
    )


def amar_qgate_split[
    FLayout: TensorLayout, QLayout: TensorLayout, GLayout: TensorLayout
](
    Qfull: TileTensor[f32, FLayout, MutAnyOrigin],
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
):
    comptime assert Qfull.flat_rank == 2 and Q.flat_rank == 2 and Gate.flat_rank == 1
    var h = block_idx.x
    var r = Int(block_idx.y)
    var d = thread_idx.x
    Q[r * NQH + h, d] = rebind[Q.ElementType](Qfull[r, h * 2 * HD + d])
    Gate[r * NQH * HD + h * HD + d] = rebind[Gate.ElementType](
        Qfull[r, h * 2 * HD + HD + d]
    )


def amar_rope_yarn[
    XLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    pos: Int32,
    nh: Int32,
):
    comptime assert X.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var j = thread_idx.x
    var theta_ex = Float32(Int(pos) + r) * exp(
        Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE)
    )
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    var c = cos(theta) * MSCALE
    var s = sin(theta) * MSCALE
    var row = r * Int(nh) + h
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT // 2])
    X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
    X[row, j + NROT // 2] = rebind[X.ElementType](x0 * s + x1 * c)


@always_inline
def kv_q8_store(p: MutPointer[Scalar[KVT], MutAnyOrigin], x: Float32, rb: Int, t: Int, d: Int):
    var mx = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var am = warp.max(abs(x))
    if lane_id() == 0:
        mx[d // WARP_SIZE] = rebind[mx.ElementType](am)
    barrier()
    var m = Float32(0)
    comptime for w in range(HD // WARP_SIZE):
        m = max(m, rebind[Scalar[f32]](mx[w]))
    var inv = Float32(127) / m if m > 0 else Float32(0)
    p[unsafe_offset=rb + d] = rebind[Scalar[KVT]](dkv_q(x, inv))
    if d == 0:
        p.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, t)] = m / 127


def amar_kv_append[
    CLayout: TensorLayout, NLayout: TensorLayout, NAT: Int
](
    Cache: TileTensor[KVT, CLayout, MutAnyOrigin],
    New: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_idx: Int32,
    att_i: Int32,
):
    comptime assert Cache.flat_rank == 1 and New.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var d = thread_idx.x
    var t = Int(t_idx) + r
    var cb = kv_tab_off[NAT](tab, t, Int(att_i), Int(h)) + Int(d)
    comptime if KVQ8:
        kv_q8_store(Cache.ptr, rebind[Scalar[f32]](New[r * NKVH + h, d]), cb - Int(d), t, Int(d))
    else:
        Cache.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](
            rebind[Scalar[f32]](New[r * NKVH + h, d]).cast[KVT]()
        )


def amar_head_rmsnorm_rope[
    XLayout: TensorLayout, GLayout: TensorLayout
](
    X: TileTensor[f32, XLayout, MutAnyOrigin],
    G: TileTensor[f32, GLayout, MutAnyOrigin],
    eps: Float32,
    pos: Int32,
    nh: Int32,
):
    comptime assert X.flat_rank == 2 and G.flat_rank == 1
    var row = Int(block_idx.x)
    var d = thread_idx.x
    var v = rebind[Scalar[f32]](X[row, d])
    var ssq = warp.sum(v * v)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    if lane_id() == 0:
        sums[d // WARP_SIZE] = rebind[sums.ElementType](ssq)
    barrier()
    var total: Float32 = 0
    comptime for w in range(HD // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    X[row, d] = rebind[X.ElementType](
        v * rsqrt(total / Float32(HD) + eps) * rebind[Scalar[f32]](G[d])
    )
    barrier()
    var j = Int(d)
    if j >= NROT // 2:
        return
    var r = row // Int(nh)
    var theta_ex = Float32(Int(pos) + r) * exp(
        Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE)
    )
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    var c = cos(theta) * MSCALE
    var s = sin(theta) * MSCALE
    var x0 = rebind[Scalar[f32]](X[row, j])
    var x1 = rebind[Scalar[f32]](X[row, j + NROT // 2])
    X[row, j] = rebind[X.ElementType](x0 * c - x1 * s)
    X[row, j + NROT // 2] = rebind[X.ElementType](x0 * s + x1 * c)


def amar_kv_append2[
    CLayout: TensorLayout, NLayout: TensorLayout, NAT: Int
](
    Kc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, CLayout, MutAnyOrigin],
    Kn: TileTensor[f32, NLayout, MutAnyOrigin],
    Vn: TileTensor[f32, NLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_idx: Int32,
    att_i: Int32,
):
    comptime assert Kc.flat_rank == 1 and Kn.flat_rank == 2
    var h = block_idx.x
    var r = Int(block_idx.y)
    var d = thread_idx.x
    var t = Int(t_idx) + r
    var cb = kv_tab_off[NAT](tab, t, Int(att_i), Int(h)) + Int(d)
    comptime if KVQ8:
        if block_idx.z == 0:
            kv_q8_store(Kc.ptr, rebind[Scalar[f32]](Kn[r * NKVH + h, d]), cb - Int(d), t, Int(d))
        else:
            kv_q8_store(Vc.ptr, rebind[Scalar[f32]](Vn[r * NKVH + h, d]), cb - Int(d), t, Int(d))
    else:
        if block_idx.z == 0:
            Kc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](
                rebind[Scalar[f32]](Kn[r * NKVH + h, d]).cast[KVT]()
            )
        else:
            Vc.ptr[unsafe_offset=cb] = rebind[Scalar[KVT]](
                rebind[Scalar[f32]](Vn[r * NKVH + h, d]).cast[KVT]()
            )


comptime PA_TK = 16
comptime PA_ROWS = 2
comptime PA_KS = HD + 4


def amar_attn_prefill[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    m: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2
    var kvh = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var M = Int(m)
    var P = Int(pos)
    var head = kvh * (NQH // NKVH) + wave % (NQH // NKVH)
    var r = tile * PA_ROWS + wave // (NQH // NKVH)
    var valid = r < M
    var qrow = r * NQH + head
    var T = P + r + 1
    var t_blk = P + min(tile * PA_ROWS + PA_ROWS, M)
    var j = lane % PA_TK
    var dpart = lane // PA_TK

    var ks = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_TK, PA_KS]())
    var vs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_TK, HD]())
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[PA_ROWS * (NQH // NKVH), HD]())
    var ksv = ks.vectorize[1, 8]()
    var vsv = vs.vectorize[1, 8]()
    var qsv = qs.vectorize[1, 8]()
    var Qv = Q.vectorize[1, 8]()
    var O_ = O

    if valid:
        qsv[wave, lane] = rebind[qsv.ElementType](Qv[qrow, lane])
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var o = SIMD[f32, 8](0)
    var lp = tid // 16
    var lc = (tid % 16) * 2
    var t0 = 0
    while t0 < t_blk:
        barrier()
        var p = t0 + lp
        if p < t_blk:
            var pb = kv_tab_off[NAT](tab, p, Int(att_i), kvh)
            var Kr = TileTensor(Kc.ptr.unsafe_offset(pb), row_major[HD]()).vectorize[8]()
            var Vr = TileTensor(Vc.ptr.unsafe_offset(pb), row_major[HD]()).vectorize[8]()
            var ksc = dkv_sc[KVT, HD](Kc.ptr, pb, p)
            var vsc = dkv_sc[KVT, HD](Vc.ptr, pb, p)
            ksv[lp, lc] = rebind[ksv.ElementType](dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Kr[lc]), ksc))
            ksv[lp, lc + 1] = rebind[ksv.ElementType](dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Kr[lc + 1]), ksc))
            vsv[lp, lc] = rebind[vsv.ElementType](dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Vr[lc]), vsc))
            vsv[lp, lc + 1] = rebind[vsv.ElementType](dkv_dq[KVT, 8](rebind[SIMD[KVT, 8]](Vr[lc + 1]), vsc))
        else:
            ksv[lp, lc] = rebind[ksv.ElementType](SIMD[f32, 8](0))
            ksv[lp, lc + 1] = rebind[ksv.ElementType](SIMD[f32, 8](0))
            vsv[lp, lc] = rebind[vsv.ElementType](SIMD[f32, 8](0))
            vsv[lp, lc + 1] = rebind[vsv.ElementType](SIMD[f32, 8](0))
        barrier()
        var part: Float32 = 0
        comptime for d8 in range(HD // 16):
            var q8 = rebind[SIMD[f32, 8]](qsv[wave, dpart * (HD // 16) + d8])
            var k8 = rebind[SIMD[f32, 8]](ksv[j, dpart * (HD // 16) + d8])
            comptime for i in range(8):
                part += q8[i] * k8[i]
        part += warp.shuffle_xor(part, UInt32(PA_TK))
        var s = Float32(-3.4e38)
        if valid and t0 + j < T:
            s = part * scale
        var tmax = warp.max(s)
        if tmax > Float32(-3.4e38):
            var m_new = max(m_run, tmax)
            var alpha = exp(m_run - m_new)
            var pj = exp(s - m_new)
            var psum = pj if dpart == 0 else Float32(0)
            l_run = l_run * alpha + warp.sum(psum)
            o = o * alpha
            comptime for jj in range(PA_TK):
                var pb = warp.shuffle_idx(pj, UInt32(jj))
                var v8 = rebind[SIMD[f32, 8]](vsv[jj, lane])
                o = fma(v8, SIMD[f32, 8](pb), o)
            m_run = m_new
        t0 += PA_TK
    if valid:
        var inv = 1 / l_run
        comptime for i in range(8):
            O_[qrow, lane * 8 + i] = rebind[O_.ElementType](o[i] * inv)


comptime f16 = DType.float16
comptime PW_WAVES = 4
comptime PW_THREADS = PW_WAVES * WARP_SIZE
comptime PW_GQ = NQH // NKVH
comptime PW_ROWS = PW_WAVES * 16 // PW_GQ
comptime PW_TK = 16
comptime PW_QS = HD + 8
comptime PW_VS = PW_TK + 8
comptime PW_PS = Float32(32768.0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(PW_THREADS)))
def amar_attn_prefill_wmma[
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, NAT: Int
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    pos: Int32,
    m: Int32,
    scale: Float32,
    att_i: Int32,
):
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2
    var kvh = Int(block_idx.x)
    var tile = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var M = Int(m)
    var P = Int(pos)
    var ai = Int(att_i)
    var row0 = tile * PW_ROWS
    var t_blk = P + min(row0 + PW_ROWS, M)
    var wrow = row0 + wave * (16 // PW_GQ)
    var r = wrow + h // PW_GQ
    var valid = r < M
    var qrow = r * NQH + kvh * PW_GQ + h % PW_GQ
    var wave_on = wrow < M
    var wlim = P + min(wrow + 16 // PW_GQ - 1, M - 1)
    var lim = P + r

    var qs = stack_allocation[f16, address_space = AddressSpace.SHARED](row_major[PW_WAVES * 16, PW_QS]())
    var ks = stack_allocation[f16, address_space = AddressSpace.SHARED](row_major[PW_TK, PW_QS]())
    var vt = stack_allocation[f16, address_space = AddressSpace.SHARED](row_major[HD, PW_VS]())


    var qsv = qs.vectorize[1, 8]()
    var ksv = ks.vectorize[1, 8]()
    var vtv = vt.vectorize[1, 8]()
    var Qv = Q.vectorize[1, 8]()
    var O_ = O

    comptime for s in range(PW_WAVES * 16 * HD // 8 // PW_THREADS):
        var v = tid + s * PW_THREADS
        var qi = v // (HD // 8)
        var c = v % (HD // 8)
        var rr = row0 + qi // PW_GQ
        var x = SIMD[f16, 8](0)
        if rr < M:
            x = rebind[SIMD[f32, 8]](Qv[rr * NQH + kvh * PW_GQ + qi % PW_GQ, c]).cast[f16]()
        qsv[qi, c] = rebind[qsv.ElementType](x)

    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var acc = Array[SIMD[f32, 8], HD // 16](fill=SIMD[f32, 8](0))
    var t0 = 0
    while t0 < t_blk:
        barrier()
        comptime for s in range(PW_TK * HD // 8 // PW_THREADS):
            var v = tid + s * PW_THREADS
            var key = v // (HD // 8)
            var c = v % (HD // 8)
            var p = t0 + key
            var x = SIMD[f16, 8](0)
            if p < t_blk:
                comptime if KVQ8:
                    var kb = kv_tab_off[NAT](tab, p, ai, kvh)
                    var Kr = TileTensor(Kc.ptr.unsafe_offset(kb), row_major[HD]()).vectorize[8]()
                    x = (rebind[SIMD[KVT, 8]](Kr[c]).cast[f32]() * dkv_sc[KVT, HD](Kc.ptr, kb, p)).cast[f16]()
                else:
                    var Kr = TileTensor(Kc.ptr.unsafe_offset(kv_tab_off[NAT](tab, p, ai, kvh)), row_major[HD]()).vectorize[8]()
                    x = rebind[SIMD[KVT, 8]](Kr[c]).cast[f16]()
            ksv[key, c] = rebind[ksv.ElementType](x)
        comptime for s in range(PW_TK * HD // 8 // PW_THREADS):
            var v = tid + s * PW_THREADS
            var key = v % PW_TK
            var c = v // PW_TK
            var p = t0 + key
            var x = SIMD[f16, 8](0)
            if p < t_blk:
                comptime if KVQ8:
                    var vb = kv_tab_off[NAT](tab, p, ai, kvh)
                    var Vr = TileTensor(Vc.ptr.unsafe_offset(vb), row_major[HD]()).vectorize[8]()
                    x = (rebind[SIMD[KVT, 8]](Vr[c]).cast[f32]() * dkv_sc[KVT, HD](Vc.ptr, vb, p)).cast[f16]()
                else:
                    var Vr = TileTensor(Vc.ptr.unsafe_offset(kv_tab_off[NAT](tab, p, ai, kvh)), row_major[HD]()).vectorize[8]()
                    x = rebind[SIMD[KVT, 8]](Vr[c]).cast[f16]()
            comptime for i in range(8):
                vt[c * 8 + i, key] = rebind[vt.ElementType](x[i])
        barrier()
        if wave_on and t0 <= wlim:
            var s0 = SIMD[f32, 8](0)
            var s1 = SIMD[f32, 8](0)
            comptime for c in range(HD // 16):
                var a = rebind[SIMD[f16, 8]](ksv[h, 2 * c]).join(rebind[SIMD[f16, 8]](ksv[h, 2 * c + 1]))
                var b = rebind[SIMD[f16, 8]](qsv[wave * 16 + h, 2 * c]).join(rebind[SIMD[f16, 8]](qsv[wave * 16 + h, 2 * c + 1]))
                var t = SIMD[f32, 8](0)
                comptime if c % 2 == 0:
                    mma(t, a, b, s0)
                    s0 = t
                else:
                    mma(t, a, b, s1)
                    s1 = t
            var sc = (s0 + s1) * scale
            var mx = Float32(-3.4e38)
            comptime for i in range(8):
                if valid and t0 + 2 * i + half <= lim:
                    mx = max(mx, sc[i])
            mx = max(mx, warp.shuffle_xor(mx, UInt32(16)))
            var m_new = max(m_run, mx)
            var alpha = exp(m_run - m_new)
            var pv = SIMD[f32, 8](0)
            comptime for i in range(8):
                if valid and t0 + 2 * i + half <= lim:
                    pv[i] = exp(sc[i] - m_new)
            var ps = pv.reduce_add()
            ps += warp.shuffle_xor(ps, UInt32(16))
            l_run = l_run * alpha + ps
            m_run = m_new
            var bf = SIMD[f16, 16](0)
            comptime for i in range(8):
                var own = (pv[i] * PW_PS).cast[f16]()
                var oth = (warp.shuffle_xor(pv[i], UInt32(16)) * PW_PS).cast[f16]()
                bf[2 * i] = own if half == 0 else oth
                bf[2 * i + 1] = oth if half == 0 else own
            comptime for dt in range(HD // 16):
                var a = rebind[SIMD[f16, 8]](vtv[dt * 16 + h, 0]).join(rebind[SIMD[f16, 8]](vtv[dt * 16 + h, 1]))
                var t = SIMD[f32, 8](0)
                mma(t, a, bf, acc[dt] * alpha)
                acc[dt] = t

        t0 += PW_TK
    if valid:
        var inv = 1 / (l_run * PW_PS)
        comptime for dt in range(HD // 16):
            comptime for i in range(8):
                O_[qrow, dt * 16 + 2 * i + half] = rebind[O_.ElementType](acc[dt][i] * inv)


def amar_attn_mla_decode[
    QL: TensorLayout, KL: TensorLayout, OL: TensorLayout, DKV: Int
](
    Q: TileTensor[f32, QL, MutAnyOrigin],
    Kc: TileTensor[bf16, KL, MutAnyOrigin],
    Vc: TileTensor[bf16, KL, MutAnyOrigin],
    O: TileTensor[f32, OL, MutAnyOrigin],
    pos: Int32,
    scale: Float32,
):
    """
    Multi-Head Latent Attention (MLA) compressed decoding kernel (512-dim low-rank latent).
    """
    var tid = Int(thread_idx.x)
    var P = Int(pos)
    var d = tid % DKV
    if d >= DKV or tid >= DKV:
        return

    var acc: Float32 = 0.0
    for t in range(P + 1):
        var qv = rebind[Scalar[f32]](Q[0, d])
        var kv = rebind[Scalar[bf16]](Kc[t, d]).cast[f32]()
        acc += qv * kv

    O[0, d] = rebind[O.ElementType](acc * scale)

