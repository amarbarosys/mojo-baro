# Persistent decode: one dispatch per token. MG blocks (default 192) x 256 threads walk every layer with bounded grid barriers between
# phases (skill persistent-kernel-gfx11). Every block recomputes the rms scale (same order as amar_rmsnorm_cast) and each
# GEMV wave applies x * scale * gamma while it streams its row, so no normed vector is shared; rows are wave-strided;
# attention is split over (q head, 32-token span) tasks plus a merge; gate and up of a row share a wave; the lm head ends in a per-block
# argmax that block 0 reduces into the token array.
from std.atomic import Atomic, Ordering
from std.math import fma
from std.sys import get_defined_int, llvm_intrinsic
from std.utils import StaticTuple
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from bexp import exp, rsqrt
from gemv_kq import kq_lane16, kq_bytes, kq_tree16, kq_wave_sum
from attn import attn_head_span, kv_tab_off, KVT
from attn_split import CH, NSMAX, PS
from moe import q6k_value, Q6K_BYTES
from q4x_kernels import UP
from rope import rot
from model import H, FFN, HD, NQH, NKVH, N_LAYERS, QF, KV, VOCAB

comptime f32 = DType.float32
comptime u32 = DType.uint32
comptime MG = get_defined_int["WG_MG", 192]() # blocks: 2 per CU at 132 VGPRs; 240 and 288 measured NOT-RESIDENT
comptime MT = 256
comptime NWAVE = MG * MT // 32
comptime SPIN_LIMIT = 1 << 22
comptime NW = 9 # weight-table slots per layer: q k v o gate up down attn_norm ffn_norm; then output_norm, token_embd, rope_freqs
comptime P32 = Pointer[Float32, MutAnyOrigin]
comptime PU = Pointer[Scalar[u32], MutAnyOrigin]

@always_inline
def grid_barrier(ctr: PU, gen: PU, fail: PU) -> Bool:
    """mojo-baro mega.mojo: bounded spin, a fail word every block honours."""
    barrier()
    if thread_idx.x == 0:
        var g = Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen)
        if Atomic[Scalar[u32], scope="agent"].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](ctr, 1) == UInt32(grid_dim.x) - 1:
            Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELAXED](ctr, 0)
            Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELEASE](gen, g + 1)
        else:
            var spins = 0
            while Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](gen) == g:
                llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
                spins += 1
                if spins > SPIN_LIMIT or Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.RELAXED](fail) != 0:
                    Atomic[Scalar[u32], scope="agent"].store[ordering=Ordering.RELEASE](fail, 1)
                    break
    barrier()
    return Atomic[Scalar[u32], scope="agent"].load[ordering=Ordering.ACQUIRE](fail) == 0

@always_inline
def rms_scale(x: P32, eps: Float32) -> Float32:
    """amar_rmsnorm_cast's sum, in its order, by every block."""
    var red = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[MT // WARP_SIZE]())
    var tid = Int(thread_idx.x)
    var partial: Float32 = 0
    var i = tid
    while i < H:
        partial += x[i] * x[i]
        i += MT
    var wsum = warp.sum(partial)
    if lane_id() == 0:
        red[tid // WARP_SIZE] = rebind[red.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(MT // WARP_SIZE):
        total += rebind[Float32](red[w])
    barrier()
    return rsqrt(total / Float32(H) + eps)

@always_inline
def wdot[GT: Int, K: Int, NORM: Bool](x: P32, g: P32, scale: Float32, w: Int, row: Int, lane: Int) -> Float32:
    """One wave: row of W (GT) times x, x taken as x * scale * g when NORM; amar_gemv_kq's order (S=1)."""
    var W = UP(unsafe_from_address=w)
    var rb = row * (K // 256) * kq_bytes[GT]()
    var acc = SIMD[f32, 16](0)
    var b = lane // 16
    while b < K // 256:
        var r = kq_lane16[GT](W, rb + b * kq_bytes[GT](), lane % 16)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        var at = b * 256 + r[3]
        var xv = x.unsafe_offset(at).unsafe_load[width=16]()
        comptime if NORM:
            xv = xv * scale * g.unsafe_offset(at).unsafe_load[width=16]()
        acc = fma(v, xv, acc)
        b += 2
    return kq_wave_sum(kq_tree16(acc))

@always_inline
def dot[K: Int, NORM: Bool](x: P32, g: P32, scale: Float32, w: Int, gt: Int32, row: Int, lane: Int) -> Float32:
    if gt == 14:
        return wdot[14, K, NORM](x, g, scale, w, row, lane)
    return wdot[12, K, NORM](x, g, scale, w, row, lane)

@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(MT)))
def wg_decode_token[CL: TensorLayout](
    wt: Pointer[Int64, MutAnyOrigin], wty: Pointer[Int32, MutAnyOrigin], x: P32, qkv: P32, att: P32, act: P32, pg: P32,
    hval: P32, hidx: Pointer[Int32, MutAnyOrigin], c: PU, toks: Pointer[Int32, MutAnyOrigin],
    Kc: TileTensor[KVT, CL, MutAnyOrigin], Vc: TileTensor[KVT, CL, MutAnyOrigin], tab: Pointer[Int32, MutAnyOrigin],
    pos32: Int32, n_prompt: Int32, eps: Float32,
):
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var gw = bid * (MT // WARP_SIZE) + tid // WARP_SIZE
    var pos = Int(pos32)
    var ctr = c
    var gen = c + 1
    var fail = c + 2
    comptime TAIL = N_LAYERS * NW

    # embed
    var i0 = bid * MT + tid
    if i0 < H:
        x[i0] = q6k_value(UP(unsafe_from_address=Int(wt[TAIL + 1])), Int(toks[pos]) * (H // 256) * Q6K_BYTES, i0)
    if not grid_barrier(ctr, gen, fail):
        return
    var Q = TileTensor(qkv, row_major[NQH, HD]())
    var Kt = TileTensor(qkv + QF, row_major[NKVH, HD]())
    var Vt = TileTensor(qkv + QF + KV, row_major[NKVH, HD]())
    var FF = TileTensor(P32(unsafe_from_address=Int(wt[TAIL + 2])), row_major[HD // 2]())
    var qs = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    for l in range(N_LAYERS):
        var t = l * NW
        # 1: attn norm + q, k, v
        var s1 = rms_scale(x, eps)
        var row = gw
        while row < QF + 2 * KV:
            var v: Float32
            if row < QF:
                v = dot[H, True](x, P32(unsafe_from_address=Int(wt[t + 7])), s1, Int(wt[t]), wty[t], row, lane)
            elif row < QF + KV:
                v = dot[H, True](x, P32(unsafe_from_address=Int(wt[t + 7])), s1, Int(wt[t + 1]), wty[t + 1], row - QF, lane)
            else:
                v = dot[H, True](x, P32(unsafe_from_address=Int(wt[t + 7])), s1, Int(wt[t + 2]), wty[t + 2], row - QF - KV, lane)
            if lane == 0:
                qkv[row] = v
            row += NWAVE
        if not grid_barrier(ctr, gen, fail):
            return
        # 2: rope q, k; k and v into the cache
        if bid < NQH + NKVH and tid < HD // 2:
            if bid < NQH:
                var y = rot(Q, FF, bid, pos, tid)
                Q[bid, 2 * tid] = rebind[Q.ElementType](y[0])
                Q[bid, 2 * tid + 1] = rebind[Q.ElementType](y[1])
            else:
                var kh = bid - NQH
                var y = rot(Kt, FF, kh, pos, tid)
                var off = kv_tab_off[N_LAYERS](rebind[MutPointer[Scalar[DType.int32], MutAnyOrigin]](tab), pos, l, kh) + 2 * tid
                Kc.ptr[off] = rebind[Scalar[KVT]](y[0])
                Kc.ptr[off + 1] = rebind[Scalar[KVT]](y[1])
                Vc.ptr[off] = rebind[Scalar[KVT]](rebind[Float32](Vt[kh, 2 * tid]))
                Vc.ptr[off + 1] = rebind[Scalar[KVT]](rebind[Float32](Vt[kh, 2 * tid + 1]))
        if not grid_barrier(ctr, gen, fail):
            return
        # 3: attention split over (q head, 32-token span) tasks across all blocks, then a merge per head
        var ns = (pos + CH) // CH
        var task = bid
        while task < NQH * ns:
            var h = task // ns
            var sp = task % ns
            var r = attn_head_span[NAT=N_LAYERS](Q, Kc, Vc, rebind[MutPointer[Scalar[DType.int32], MutAnyOrigin]](tab), qs, scores, red, h,
                h // (NQH // NKVH), sp * CH, min(pos + 1, sp * CH + CH), tid, lane, Float32(0.125), l)
            var pp = (h * NSMAX + sp) * PS
            if tid == 0:
                pg[pp] = r[0]
                pg[pp + 1] = r[1]
            if tid < HD:
                pg[pp + 2 + tid] = r[2]
            task += MG
        if not grid_barrier(ctr, gen, fail):
            return
        if bid < NQH and tid < HD:
            var m = Float32(-3.4e38)
            for sp in range(ns):
                m = max(m, pg[(bid * NSMAX + sp) * PS])
            var lsum: Float32 = 0
            var o: Float32 = 0
            for sp in range(ns):
                var pp = (bid * NSMAX + sp) * PS
                var a = exp(pg[pp] - m)
                lsum += a * pg[pp + 1]
                o += a * pg[pp + 2 + tid]
            att[bid * HD + tid] = o / lsum
        if not grid_barrier(ctr, gen, fail):
            return
        # 4: o, added into x
        row = gw
        while row < H:
            var v = dot[QF, False](att, att, 0, Int(wt[t + 3]), wty[t + 3], row, lane)
            if lane == 0:
                x[row] += v
            row += NWAVE
        if not grid_barrier(ctr, gen, fail):
            return
        # 5: ffn norm + gate and up of the same row, act = silu(gate) * up
        var s2 = rms_scale(x, eps)
        row = gw
        while row < FFN:
            var g = dot[H, True](x, P32(unsafe_from_address=Int(wt[t + 8])), s2, Int(wt[t + 4]), wty[t + 4], row, lane)
            var u = dot[H, True](x, P32(unsafe_from_address=Int(wt[t + 8])), s2, Int(wt[t + 5]), wty[t + 5], row, lane)
            if lane == 0:
                act[row] = g / (1 + exp(-g)) * u
            row += NWAVE
        if not grid_barrier(ctr, gen, fail):
            return
        # 6: down, added into x
        row = gw
        while row < H:
            var v = dot[FFN, False](act, act, 0, Int(wt[t + 6]), wty[t + 6], row, lane)
            if lane == 0:
                x[row] += v
            row += NWAVE
        if not grid_barrier(ctr, gen, fail):
            return
    if pos + 1 < Int(n_prompt):
        return
    # head: final norm + lm head (token_embd, q6_K), per-wave best, per-block best, block 0 reduces
    var s3 = rms_scale(x, eps)
    var best = Float32(-3.4e38)
    var bidx = 0
    var row = gw
    while row < VOCAB:
        var v = dot[H, True](x, P32(unsafe_from_address=Int(wt[TAIL])), s3, Int(wt[TAIL + 1]), 14, row, lane)
        if v > best:
            best = v
            bidx = row
        row += NWAVE
    var bv = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[MT // WARP_SIZE]())
    var bi = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](row_major[MT // WARP_SIZE]())
    if lane == 0:
        bv[tid // WARP_SIZE] = rebind[bv.ElementType](best)
        bi[tid // WARP_SIZE] = rebind[bi.ElementType](Int32(bidx))
    barrier()
    if tid == 0:
        var mv = rebind[Float32](bv[0])
        var mi = rebind[Int32](bi[0])
        comptime for w in range(1, MT // WARP_SIZE):
            var v = rebind[Float32](bv[w])
            var k = rebind[Int32](bi[w])
            if v > mv or (v == mv and k < mi):
                mv = v
                mi = k
        hval[bid] = mv
        hidx[bid] = mi
    if not grid_barrier(ctr, gen, fail):
        return
    if bid == 0 and tid == 0:
        var mv = hval[0]
        var mi = hidx[0]
        for k in range(1, MG):
            if hval[k] > mv or (hval[k] == mv and hidx[k] < mi):
                mv = hval[k]
                mi = hidx[k]
        toks[pos + 1] = mi
