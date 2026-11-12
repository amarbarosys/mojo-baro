from arch import readlane
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.bit import log2_floor
from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.primitives import warp
from std.math import fma
from bexp import exp
from std.sys import llvm_intrinsic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from std.utils import StaticTuple

comptime f32 = DType.float32
comptime KVPAGE = 128
comptime KVPSH = 7
comptime DTHREADS = 256
comptime DWAVES = DTHREADS // WARP_SIZE
comptime NEG = Float32(-3.4e38)


@always_inline
def dkv_pad[KVT: DType]() -> Int:
    comptime if KVT == DType.int8:
        return KVPAGE * 4
    else:
        return 0


@always_inline
def dkv_off[HD: Int, NKVH: Int, NAT: Int, KVT: DType = DType.float32](t: Int, att_i: Int, kvh: Int) -> Int:
    return (((t >> KVPSH) * NAT + att_i) * NKVH + kvh) * (KVPAGE * HD + dkv_pad[KVT]()) + (t & (KVPAGE - 1)) * HD


@always_inline
def dkv_tab_off[HD: Int, NKVH: Int, NAT: Int, KVT: DType = DType.float32](tab: MutPointer[Scalar[DType.int32], MutAnyOrigin], t: Int, att_i: Int, kvh: Int) -> Int:
    return ((Int(tab[t >> KVPSH]) * NAT + att_i) * NKVH + kvh) * (KVPAGE * HD + dkv_pad[KVT]()) + (t & (KVPAGE - 1)) * HD


@always_inline
def dkv_sc_off[HD: Int](row: Int, t: Int) -> Int:
    return (row + KVPAGE * HD + (t & (KVPAGE - 1)) * (4 - HD)) >> 2


@always_inline
def dkv_sc[KVT: DType, HD: Int](p: MutPointer[Scalar[KVT], MutAnyOrigin], row: Int, t: Int) -> Float32:
    comptime if KVT == DType.int8:
        return p.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](row, t)]
    else:
        return Float32(1)


@always_inline
def dkv_dq[KVT: DType, W: Int](x: SIMD[KVT, W], sc: Float32) -> SIMD[f32, W]:
    comptime if KVT == DType.int8:
        return x.cast[f32]() * sc
    else:
        return x.cast[f32]()


@always_inline
def dkv_q(x: Float32, inv: Float32) -> Scalar[DType.int8]:
    var y = x * inv
    return (y + (Float32(0.5) if y >= 0 else Float32(-0.5))).cast[DType.int8]()


@always_inline
def dspan[HD: Int, NLD: Int]() -> Int:
    return NLD * (WARP_SIZE // (HD // 8))


@always_inline
def dattn_span[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int,
    QLayout: TensorLayout, KLayout: TensorLayout,
    QsL: TensorLayout, ScL: TensorLayout, RdL: TensorLayout,
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
    var wave = tid // WARP_SIZE
    if tid < HD:
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
        if tid < HD:
            var t = t0 + tid
            if t < t_hi:
                var kb = dkv_tab_off[HD, NKVH, NAT, KVT](tab, t, att_i, kvh)
                var Kr = TileTensor(kp.unsafe_offset(kb), row_major[HD]()).vectorize[8]()
                var ksc = dkv_sc[KVT, HD](kp, kb, t)
                var acc: Float32 = 0
                for d8 in range(HD // 8):
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
        comptime for w in range(HD // WARP_SIZE):
            var sc = rebind[Scalar[f32]](red[w])
            if sc > cmax:
                cmax = sc
        var m_new = max(m_run, cmax)
        var alpha = exp(m_run - m_new)
        barrier()
        if tid < HD:
            var e = exp(s - m_new)
            scores[tid] = rebind[scores.ElementType](e)
            var wsum = warp.sum(e)
            if lane == 0:
                red[wave] = rebind[red.ElementType](wsum)
        barrier()
        var csum = Float32(0)
        comptime for w in range(HD // WARP_SIZE):
            csum += rebind[Scalar[f32]](red[w])
        l_run = l_run * alpha + csum
        m_run = m_new
        o = o * alpha
        if tid < HD:
            var n = t_hi - t0
            if n > HD:
                n = HD
            var tt = 0
            while tt + 8 <= n:
                var v = Array[Float32, 8](uninitialized=True)
                var sc = Array[Float32, 8](uninitialized=True)
                comptime for j in range(8):
                    var vb = dkv_tab_off[HD, NKVH, NAT, KVT](tab, t0 + tt + j, att_i, kvh)
                    v[j] = dkv_dq[KVT, 1](vp[unsafe_offset=vb + tid], dkv_sc[KVT, HD](vp, vb, t0 + tt + j))
                    sc[j] = rebind[Scalar[f32]](scores[tt + j])
                comptime for j in range(8):
                    o += sc[j] * v[j]
                tt += 8
            while tt < n:
                var vb = dkv_tab_off[HD, NKVH, NAT, KVT](tab, t0 + tt, att_i, kvh)
                o += rebind[Scalar[f32]](scores[tt]) * dkv_dq[KVT, 1](vp[unsafe_offset=vb + tid], dkv_sc[KVT, HD](vp, vb, t0 + tt))
                tt += 1
        t0 += HD
    return SIMD[f32, 4](m_run, l_run, o, 0)


def amar_dattn_exact[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout,
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
    comptime assert Q.flat_rank == 2 and Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2
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
    var res = dattn_span[HD, NQH, NKVH, KVT, NAT](Q, Kc, Vc, tab, qs, scores, red, qrow, kvh, 0, T, tid, Int(lane_id()), scale, Int(att_i))
    if tid < HD:
        var inv = 1 / res[1]
        O_[qrow, tid] = rebind[O_.ElementType](res[2] * inv)


@always_inline
def dattn_load_span[
    HD: Int, NKVH: Int, KVT: DType, NAT: Int, NLD: Int, KLayout: TensorLayout
](
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    mut kr: Array[SIMD[f32, 8], NLD],
    mut vr: Array[SIMD[f32, 8], NLD],
    t0: Int, T: Int, lr: Int, lc: Int, ai: Int, kvh: Int,
):
    comptime RPL = WARP_SIZE // (HD // 8)
    comptime SPAN = NLD * RPL
    comptime LSTR = RPL * HD
    var row0 = dkv_tab_off[HD, NKVH, NAT, KVT](tab, t0, ai, kvh)
    var base = row0 + lc
    if t0 + SPAN <= T:
        var b = base + lr * HD
        comptime for i in range(NLD):
            comptime if KVT == DType.int8:
                var ti = lr + i * RPL
                kr[i] = dkv_dq[KVT, 8](Kc.ptr.unsafe_load[width=8](b + i * LSTR), dkv_sc[KVT, HD](Kc.ptr, row0 + ti * HD, t0 + ti))
                vr[i] = dkv_dq[KVT, 8](Vc.ptr.unsafe_load[width=8](b + i * LSTR), dkv_sc[KVT, HD](Vc.ptr, row0 + ti * HD, t0 + ti))
            else:
                kr[i] = Kc.ptr.unsafe_load[width=8](b + i * LSTR).cast[f32]()
                vr[i] = Vc.ptr.unsafe_load[width=8](b + i * LSTR).cast[f32]()
    else:
        var last = T - 1 - t0
        comptime for i in range(NLD):
            var ti = min(i * RPL + lr, last)
            var b = base + ti * HD
            comptime if KVT == DType.int8:
                kr[i] = dkv_dq[KVT, 8](Kc.ptr.unsafe_load[width=8](b), dkv_sc[KVT, HD](Kc.ptr, row0 + ti * HD, t0 + ti))
                vr[i] = dkv_dq[KVT, 8](Vc.ptr.unsafe_load[width=8](b), dkv_sc[KVT, HD](Vc.ptr, row0 + ti * HD, t0 + ti))
            else:
                kr[i] = Kc.ptr.unsafe_load[width=8](b).cast[f32]()
                vr[i] = Vc.ptr.unsafe_load[width=8](b).cast[f32]()


@always_inline
def dattn_bcast[RPL: Int, LPR: Int, DUP: Int, I: Int](p: Float32, lr: Int) -> Float32:
    comptime if RPL == 1:
        return readlane(p, Int32(I * DUP))
    else:
        return warp.shuffle_idx(p, UInt32(lr * LPR + I * DUP))


@always_inline
def dattn_step[
    HD: Int, G: Int, KVT: DType, NLD: Int, QsL: TensorLayout
](
    qs: TileTensor[f32, QsL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    kr: Array[SIMD[f32, 8], NLD],
    vr: Array[SIMD[f32, 8], NLD],
    mut o: Array[SIMD[f32, 8], G],
    mut m: Array[Float32, G],
    mut l: Array[Float32, G],
    t0: Int, T: Int, lane: Int, lr: Int, jtok: Int,
):
    comptime LPR = HD // 8
    comptime RPL = WARP_SIZE // LPR
    comptime DUP = LPR // NLD
    comptime LOG2_LPR = log2_floor(LPR)
    comptime assert qs.flat_rank == 1
    var qv = qs.vectorize[8]()
    var qi = lane % LPR
    var part = Array[Float32, NLD * G](uninitialized=True)
    comptime for i in range(NLD):
        var k8 = kr[i].cast[f32]()
        comptime for g in range(G):
            var q8 = rebind[SIMD[f32, 8]](qv[g * LPR + qi])
            var a0 = Float32(0)
            var a1 = Float32(0)
            comptime for e in range(4):
                a0 = fma(q8[2 * e], k8[2 * e], a0)
                a1 = fma(q8[2 * e + 1], k8[2 * e + 1], a1)
            part[g * NLD + i] = a0 + a1
    var t_me = t0 + jtok * RPL + lr
    var p = Array[Float32, G](uninitialized=True)
    comptime for g in range(G):
        comptime for r in range(LOG2_LPR):
            comptime off = LPR >> (r + 1)
            comptime n = NLD >> r
            comptime if n > 1:
                comptime half = n // 2
                var hi = (lane & off) != 0
                comptime for j in range(half):
                    var a = part[g * NLD + j]
                    var b = part[g * NLD + j + half]
                    var send = a if hi else b
                    var keep = b if hi else a
                    part[g * NLD + j] = keep + warp.shuffle_xor(send, UInt32(off))
            else:
                part[g * NLD] += warp.shuffle_xor(part[g * NLD], UInt32(off))
        var s = part[g * NLD] if t_me < T else NEG
        var m_new = max(m[g], warp.max(s))
        var alpha = exp(m[g] - m_new)
        var pg = exp(s - m_new)
        var psum = pg if (lane % DUP) == 0 else Float32(0)
        l[g] = l[g] * alpha + warp.sum(psum)
        o[g] = o[g] * alpha
        m[g] = m_new
        p[g] = pg
    comptime for i in range(NLD):
        var v8 = vr[i].cast[f32]()
        comptime for g in range(G):
            var pb = dattn_bcast[RPL, LPR, DUP, i](p[g], lr)
            o[g] = fma(v8, SIMD[f32, 8](pb), o[g])


@always_inline
def dattn_nsplit[HD: Int, NLD: Int, NKVH: Int](T: Int, m: Int, blocks: Int) -> Int:
    var NS = (T + dspan[HD, NLD]() - 1) // dspan[HD, NLD]()
    return max(min(max(blocks // (m * NKVH), 1), NS), 1)


@always_inline
def dattn_split_body[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int, NLD: Int, ROT: Bool,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    kvh: Int, sp: Int, r: Int, ns: Int, T: Int, scale: Float32, ai: Int, tid: Int,
):
    comptime assert Kc.flat_rank == 1 and Vc.flat_rank == 1 and O.flat_rank == 2
    comptime G = NQH // NKVH
    comptime LPR = HD // 8
    comptime RPL = WARP_SIZE // LPR
    comptime SPAN = NLD * RPL
    comptime DUP = LPR // NLD
    comptime PSTR = HD + 2
    comptime N8 = G * HD // 8
    comptime NW = DWAVES
    comptime assert NQH == G * NKVH and LPR * RPL == WARP_SIZE and DUP * NLD == LPR and N8 <= DTHREADS
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var lr = lane // LPR
    var lc = (lane % LPR) * 8
    var jtok = (lane % LPR) // DUP
    var NS = (T + SPAN - 1) // SPAN
    var s_lo = sp * NS // ns
    var s_hi = (sp + 1) * NS // ns
    var hb = r * NQH + kvh * G
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G * HD]())
    var qsv = qs.vectorize[8]()
    if tid < N8:
        qsv[tid] = rebind[qsv.ElementType](Q.ptr.unsafe_load[width=8](hb * HD + tid * 8) * scale)
    barrier()
    var o = Array[SIMD[f32, 8], G](fill=SIMD[f32, 8](0))
    var m = Array[Float32, G](fill=NEG)
    var l = Array[Float32, G](fill=Float32(0))
    var L = s_hi - s_lo
    var rot = 0
    comptime if ROT:
        if L > 0:
            rot = ((sp * NKVH + kvh) * 37) % L
    var kr = Array[SIMD[f32, 8], NLD](uninitialized=True)
    var vr = Array[SIMD[f32, 8], NLD](uninitialized=True)
    var k = wave
    while k < L:
        var kk = k + rot
        if kk >= L:
            kk -= L
        var t0 = (s_lo + kk) * SPAN
        dattn_load_span[HD, NKVH, KVT, NAT, NLD](Kc, Vc, tab, kr, vr, t0, T, lr, lc, ai, kvh)
        dattn_step[HD, G, KVT, NLD](qs, kr, vr, o, m, l, t0, T, lane, lr, jtok)
        k += NW
    comptime for g in range(G):
        comptime for rr in range(log2_floor(RPL)):
            comptime off = LPR << rr
            comptime for e in range(8):
                o[g][e] += warp.shuffle_xor(o[g][e], UInt32(off))
    var wm = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NW * G]())
    var wl = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NW * G]())
    var acc = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G * HD]())
    if lane == 0:
        comptime for g in range(G):
            wm[wave * G + g] = rebind[wm.ElementType](m[g])
            wl[wave * G + g] = rebind[wl.ElementType](l[g])
    barrier()
    var mb = Array[Float32, G](uninitialized=True)
    var lb = Array[Float32, G](uninitialized=True)
    comptime for g in range(G):
        var mm = NEG
        comptime for w in range(NW):
            mm = max(mm, rebind[Scalar[f32]](wm[w * G + g]))
        var ll = Float32(0)
        comptime for w in range(NW):
            ll += exp(rebind[Scalar[f32]](wm[w * G + g]) - mm) * rebind[Scalar[f32]](wl[w * G + g])
        mb[g] = mm
        lb[g] = ll
    var accv = acc.vectorize[8]()
    comptime for w in range(NW):
        if wave == w and lr == 0:
            comptime for g in range(G):
                var wgt = exp(m[g] - mb[g])
                var idx = (g * HD + lc) // 8
                comptime if w == 0:
                    accv[idx] = rebind[accv.ElementType](o[g] * wgt)
                else:
                    accv[idx] = rebind[accv.ElementType](rebind[SIMD[f32, 8]](accv[idx]) + o[g] * wgt)
        barrier()
    if tid < N8:
        var g = tid // (HD // 8)
        var c = (tid % (HD // 8)) * 8
        var v = rebind[SIMD[f32, 8]](accv[tid])
        var h = hb + g
        if ns == 1:
            var inv = 1 / lb[g]
            comptime for e in range(8):
                O[h, c + e] = rebind[O.ElementType](v[e] * inv)
        else:
            var pp = Pg.ptr.unsafe_offset((h * ns + sp) * PSTR)
            if c == 0:
                pp[unsafe_offset=0] = mb[g]
                pp[unsafe_offset=1] = lb[g]
            comptime for e in range(8):
                pp[unsafe_offset=2 + c + e] = v[e]


@always_inline
def dattn_combine_body[
    HD: Int, MAXS: Int, PLayout: TensorLayout, OLayout: TensorLayout,
](
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    mut O: TileTensor[f32, OLayout, MutAnyOrigin],
    hg: Int, ns: Int, tid: Int,
):
    comptime assert O.flat_rank == 2
    comptime PSTR = HD + 2
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var pp = Pg.ptr.unsafe_offset(hg * ns * PSTR)
    var wsh = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[MAXS]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[DWAVES]())
    var mloc = NEG
    var sp = tid
    while sp < ns:
        mloc = max(mloc, pp[unsafe_offset=sp * PSTR])
        sp += DTHREADS
    var wmax = warp.max(mloc)
    if lane == 0:
        red[wave] = rebind[red.ElementType](wmax)
    barrier()
    var mmax = NEG
    comptime for w in range(DWAVES):
        mmax = max(mmax, rebind[Scalar[f32]](red[w]))
    var lloc = Float32(0)
    sp = tid
    while sp < ns:
        var wgt = exp(pp[unsafe_offset=sp * PSTR] - mmax)
        wsh[sp] = rebind[wsh.ElementType](wgt)
        lloc += wgt * pp[unsafe_offset=sp * PSTR + 1]
        sp += DTHREADS
    var wsum = warp.sum(lloc)
    barrier()
    if lane == 0:
        red[wave] = rebind[red.ElementType](wsum)
    barrier()
    var lsum = Float32(0)
    comptime for w in range(DWAVES):
        lsum += rebind[Scalar[f32]](red[w])
    if tid < HD:
        var o0 = Float32(0)
        var o1 = Float32(0)
        var o2 = Float32(0)
        var o3 = Float32(0)
        var pd = pp.unsafe_offset(2 + tid)
        sp = 0
        while sp + 4 <= ns:
            o0 = fma(rebind[Scalar[f32]](wsh[sp]), pd[unsafe_offset=sp * PSTR], o0)
            o1 = fma(rebind[Scalar[f32]](wsh[sp + 1]), pd[unsafe_offset=(sp + 1) * PSTR], o1)
            o2 = fma(rebind[Scalar[f32]](wsh[sp + 2]), pd[unsafe_offset=(sp + 2) * PSTR], o2)
            o3 = fma(rebind[Scalar[f32]](wsh[sp + 3]), pd[unsafe_offset=(sp + 3) * PSTR], o3)
            sp += 4
        while sp < ns:
            o0 = fma(rebind[Scalar[f32]](wsh[sp]), pd[unsafe_offset=sp * PSTR], o0)
            sp += 1
        O[hg, tid] = rebind[O.ElementType](((o0 + o1) + (o2 + o3)) * (1 / lsum))
    barrier()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(HD)))
def amar_dattn_split[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NAT: Int, NLD: Int, ROT: Bool,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    nsplit: Int32,
    scale: Float32,
    att_i: Int32,
):
    var O_ = O
    var r = Int(block_idx.z)
    dattn_split_body[HD, NQH, NKVH, KVT, NAT, NLD, ROT](
        Q, Kc, Vc, O_, Pg, tab, Int(block_idx.x), Int(block_idx.y), r, Int(nsplit), Int(t_len) + r, scale, Int(att_i), Int(thread_idx.x)
    )


comptime DMAXS = 1024


def amar_dattn_combine[
    HD: Int, MAXS: Int, PLayout: TensorLayout, OLayout: TensorLayout,
](
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    nsplit: Int32,
):
    var O_ = O
    dattn_combine_body[HD, MAXS](Pg, O_, Int(block_idx.x), Int(nsplit), Int(thread_idx.x))
