"""Gemma4 decode attention: 8 query heads on 1 KV head (MQA), scale 1.0, f32 softmax, window or full.

KV pool contract: dkv_tab_off pages a pool with stride KVPAGE * HD_, so one pool holds one head dim.
Gemma4 needs two pools: S (HD 256, window 512) and F (HD 512, full), each with its own NAT; att_i is the
slot inside the pool of that kind. A shared layer passes the slot of its source layer in that pool.
t_len counts the new token: the kernel attends to positions max(0, t_len - WINDOW) .. t_len - 1.
Preconditions: t_len >= 1 (t_len 0 divides by a zero sum) and finite scores well above NEG (-3.4e38):
a non-finite q . k gives a non-finite output, and a live score equal to NEG is not told from a masked lane.
The launched block must be BT threads (whole waves); merge threads past HD_ // 8 return.
Launch: grid (NQH_ // GB, nsplit), block BT (g4_bt); nsplit > 1 writes partials to Pg (NQH_ * nsplit *
(HD_ + G4_PHDR) floats) and needs amar_attn_merge_gemma4, grid NQH_, block HD_ // 8.
g4_gb, g4_nld, g4_bt and g4_nsplit hold the measured picks (docs/lanes/2026-11-16-gemma4-attn-report.md).
"""
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.bit import log2_floor
from std.math import fma
from std.utils import StaticTuple
from bexp import exp
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import readlane
from dattn import dkv_tab_off, NEG

comptime f32 = DType.float32
comptime G4_SLAB = 32768
comptime G4_PHDR = 8
comptime G4_NSMAX = 64


def g4_gb(hd: Int, window: Int) -> Int:
    return 1 if window > 0 else (4 if hd <= 256 else 2)


def g4_bt(hd: Int, window: Int) -> Int:
    return 1024 if window > 0 else 256


def g4_nld(hd: Int, window: Int) -> Int:
    return 8 if window > 0 and hd <= 256 else 4


def g4_nsplit(window: Int, t_len: Int) -> Int:
    return 1 if window > 0 or t_len < 192 else min(8, t_len // 64)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BT)))
def amar_attn_decode_gemma4[
    HD_: Int, WINDOW: Int, NAT: Int, GB: Int, NLD: Int, BT: Int,
    QLayout: TensorLayout, KLayout: TensorLayout, OLayout: TensorLayout, PLayout: TensorLayout,
    NQH_: Int = 8, KVT: DType = DType.float32,
](
    Q: TileTensor[f32, QLayout, MutAnyOrigin],
    Kc: TileTensor[KVT, KLayout, MutAnyOrigin],
    Vc: TileTensor[KVT, KLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    t_len: Int32,
    nsplit: Int32,
    att_i: Int32,
):
    comptime W = HD_ // WARP_SIZE
    comptime NW = BT // WARP_SIZE
    comptime SLAB = NW * GB * HD_ * 4 <= G4_SLAB
    comptime DUP = WARP_SIZE // NLD
    comptime PSTR = HD_ + G4_PHDR
    comptime assert W * WARP_SIZE == HD_ and DUP * NLD == WARP_SIZE, "HD_ and NLD must fill a wave"
    comptime assert NQH_ % GB == 0 and GB <= NW, "GB heads per block: one output wave each"
    comptime assert BT % WARP_SIZE == 0, "BT must be whole waves"
    comptime assert KVT != DType.int8, "int8 KV needs its per-token scale"
    var hb = Int(block_idx.x) * GB
    var sp = Int(block_idx.y)
    var ns = Int(nsplit)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var lc = lane * W
    var T = Int(t_len)
    var t_lo = 0
    comptime if WINDOW > 0:
        t_lo = max(T - WINDOW, 0)
    var S0 = t_lo // NLD
    var NS = (T + NLD - 1) // NLD - S0
    var s_lo = S0 + sp * NS // ns
    var L = S0 + (sp + 1) * NS // ns - s_lo
    var kp = Kc.ptr
    var vp = Vc.ptr
    var q = Array[SIMD[f32, W], GB](uninitialized=True)
    comptime for g in range(GB):
        q[g] = Q.ptr.unsafe_load[width=W]((hb + g) * HD_ + lc)
    var o = Array[SIMD[f32, W], GB](fill=SIMD[f32, W](0))
    var m = Array[Float32, GB](fill=NEG)
    var l = Array[Float32, GB](fill=Float32(0))
    var k = wave
    while k < L:
        var t0 = (s_lo + k) * NLD
        var row0 = dkv_tab_off[HD_, 1, NAT, KVT](tab, t0, Int(att_i), 0) + lc
        var first = t_lo - t0
        var last = T - 1 - t0
        var kr = Array[SIMD[f32, W], NLD](uninitialized=True)
        var vr = Array[SIMD[f32, W], NLD](uninitialized=True)
        comptime for i in range(NLD):
            var b = row0 + min(max(i, first), last) * HD_
            kr[i] = kp.unsafe_load[width=W](b).cast[f32]()
            vr[i] = vp.unsafe_load[width=W](b).cast[f32]()
        var part = Array[Float32, GB * NLD](fill=Float32(0))
        comptime for e in range(W):
            comptime for i in range(NLD):
                comptime for g in range(GB):
                    part[g * NLD + i] = fma(q[g][e], kr[i][e], part[g * NLD + i])
        var t_me = t0 + lane // DUP
        var live = t_me >= t_lo and t_me < T
        var p = Array[Float32, GB](uninitialized=True)
        comptime for g in range(GB):
            comptime for r in range(log2_floor(WARP_SIZE)):
                comptime off = WARP_SIZE >> (r + 1)
                comptime n = NLD >> r
                comptime if n > 1:
                    var hi = (lane & off) != 0
                    comptime for j in range(n // 2):
                        var a = part[g * NLD + j]
                        var b = part[g * NLD + j + n // 2]
                        part[g * NLD + j] = (b if hi else a) + warp.shuffle_xor(a if hi else b, UInt32(off))
                else:
                    part[g * NLD] += warp.shuffle_xor(part[g * NLD], UInt32(off))
            var s = part[g * NLD] if live else NEG
            var m_new = max(m[g], warp.max(s))
            var alpha = exp(m[g] - m_new)
            var pg = exp(s - m_new)
            l[g] = l[g] * alpha + warp.sum(pg if lane % DUP == 0 else Float32(0))
            o[g] = o[g] * alpha
            m[g] = m_new
            p[g] = pg
        comptime for i in range(NLD):
            comptime for g in range(GB):
                o[g] = fma(vr[i], SIMD[f32, W](readlane(p[g], Int32(i * DUP))), o[g])
        k += NW
    var wm = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NW * GB]())
    var wl = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NW * GB]())
    var acc = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[(NW if SLAB else 1) * GB * HD_]())
    if lane == 0:
        comptime for g in range(GB):
            wm[wave * GB + g] = rebind[wm.ElementType](m[g])
            wl[wave * GB + g] = rebind[wl.ElementType](l[g])
    barrier()
    var mb = Array[Float32, GB](uninitialized=True)
    var lb = Array[Float32, GB](uninitialized=True)
    comptime for g in range(GB):
        var mm = NEG
        comptime for w in range(NW):
            mm = max(mm, rebind[Scalar[f32]](wm[w * GB + g]))
        var ll = Float32(0)
        comptime for w in range(NW):
            ll += exp(rebind[Scalar[f32]](wm[w * GB + g]) - mm) * rebind[Scalar[f32]](wl[w * GB + g])
        mb[g] = mm
        lb[g] = ll
        o[g] = o[g] * exp(m[g] - mm)
    var accv = acc.vectorize[W]()
    comptime if SLAB:
        comptime for g in range(GB):
            accv[(wave * GB + g) * WARP_SIZE + lane] = rebind[accv.ElementType](o[g])
        barrier()
    else:
        comptime for w in range(NW):
            if wave == w:
                comptime for g in range(GB):
                    comptime if w == 0:
                        accv[g * WARP_SIZE + lane] = rebind[accv.ElementType](o[g])
                    else:
                        accv[g * WARP_SIZE + lane] = rebind[accv.ElementType](rebind[SIMD[f32, W]](accv[g * WARP_SIZE + lane]) + o[g])
            barrier()
    comptime for g in range(GB):
        if wave == g:
            var v = rebind[SIMD[f32, W]](accv[tid])
            comptime if SLAB:
                for w in range(1, NW):
                    v += rebind[SIMD[f32, W]](accv[(w * GB + g) * WARP_SIZE + lane])
            if ns == 1:
                var op = O.ptr.unsafe_offset((hb + g) * HD_ + lc)
                var r = v * (1 / lb[g])
                comptime for e in range(W):
                    op[unsafe_offset=e] = r[e]
            else:
                var pp = Pg.ptr.unsafe_offset(((hb + g) * ns + sp) * PSTR)
                if lane == 0:
                    pp[unsafe_offset=0] = mb[g]
                    pp[unsafe_offset=1] = lb[g]
                comptime for e in range(W):
                    pp[unsafe_offset=G4_PHDR + lc + e] = v[e]


def amar_attn_merge_gemma4[
    HD_: Int, PLayout: TensorLayout, OLayout: TensorLayout,
](
    Pg: TileTensor[f32, PLayout, MutAnyOrigin],
    O: TileTensor[f32, OLayout, MutAnyOrigin],
    nsplit: Int32,
):
    comptime PSTR = HD_ + G4_PHDR
    var ns = Int(nsplit)
    var h = Int(block_idx.x)
    var c = Int(thread_idx.x) * 8
    if c >= HD_:
        return
    var pp = Pg.ptr.unsafe_offset(h * ns * PSTR)
    var mx = NEG
    for s in range(ns):
        mx = max(mx, pp[unsafe_offset=s * PSTR])
    var l = Float32(0)
    var o = SIMD[f32, 8](0)
    for s in range(ns):
        var w = exp(pp[unsafe_offset=s * PSTR] - mx)
        l = fma(w, pp[unsafe_offset=s * PSTR + 1], l)
        o = fma(pp.unsafe_load[width=8](s * PSTR + G4_PHDR + c), SIMD[f32, 8](w), o)
    o = o * (1 / l)
    var op = O.ptr.unsafe_offset(h * HD_ + c)
    comptime for e in range(8):
        op[unsafe_offset=e] = o[e]
