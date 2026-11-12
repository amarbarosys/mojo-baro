from arch import realtime, sleep1
from arch import GPU_SCOPE
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.atomic import Atomic, Ordering
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import cos, fma, log1p, rsqrt, sin
from bexp import exp, log
from std.math import sqrt
from std.sys import get_defined_int, llvm_intrinsic
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from dattn import dattn_split_body, dattn_combine_body, dattn_nsplit, dkv_q, dkv_sc_off

from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS, q4_dot_blocks, bf16x16_to_f32
from gemv_kq import kq_lane_part_ld, kq_lane_part_ld2, kq_tree16, kq_wave_sum, kq_bytes, GT_Q4K, GT_Q6K
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS
from attn import HD, NQH, NKVH, KVQ8, KVT, TCAP, KVPAGE, KVPSH, KVHSTR, kv_off, kv_tab_off, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE, attn_head_body, attn_head_span
from model import H, FFN, QF, KV, N_LAYERS, MEGA_ALLOWED

# SSM inner width and attention output width; equal to H on Qwythos-9B only.
comptime INNER = NH_V * SSTATE
comptime ATT = NQH * HD

comptime u32 = DType.uint32
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime f16 = DType.float16
comptime u8 = DType.uint8
comptime i32 = DType.int32
comptime i64 = DType.int64
comptime MEGA_G = 96
comptime MEGA_G_WIN = 96
comptime DOT_UNROLL = get_defined_int["BARO_DOT_UNROLL", 1]()
comptime SPIN_LIMIT = 1 << 22
comptime QV = 16
comptime UNROLL = 4
comptime VOCAB = 248320
comptime ATT_SCALE = Float32(0.0625)
comptime DATT_NLD = 4
comptime RMS_EPS = Float32(1e-6)


@always_inline
def grid_barrier(
    ctr: MutPointer[Scalar[u32], MutAnyOrigin],
    gen: MutPointer[Scalar[u32], MutAnyOrigin],
    fail: MutPointer[Scalar[u32], MutAnyOrigin],
) -> Bool:
    barrier()
    if thread_idx.x == 0:
        var nb = UInt32(grid_dim.x)
        var g = Atomic[Scalar[u32], scope=GPU_SCOPE].load[ordering=Ordering.ACQUIRE](gen)
        if Atomic[Scalar[u32], scope=GPU_SCOPE].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](ctr, 1) == nb - 1:
            Atomic[Scalar[u32], scope=GPU_SCOPE].store[ordering=Ordering.RELAXED](ctr, 0)
            Atomic[Scalar[u32], scope=GPU_SCOPE].store[ordering=Ordering.RELEASE](gen, g + 1)
        else:
            var spins = 0
            while Atomic[Scalar[u32], scope=GPU_SCOPE].load[ordering=Ordering.ACQUIRE](gen) == g:
                sleep1()
                spins += 1
                if spins > SPIN_LIMIT or Atomic[Scalar[u32], scope=GPU_SCOPE].load[ordering=Ordering.RELAXED](fail) != 0:
                    Atomic[Scalar[u32], scope=GPU_SCOPE].store[ordering=Ordering.RELEASE](fail, 1)
                    break
    barrier()
    return Atomic[Scalar[u32], scope=GPU_SCOPE].load[ordering=Ordering.ACQUIRE](fail) == 0


@always_inline
def stamp(prof: MutPointer[Scalar[i64], MutAnyOrigin], idx: Int):
    if block_idx.x == 0 and thread_idx.x == 0:
        prof[unsafe_offset=idx] = realtime()


@always_inline
def q8_row_dot[
    MR: Int, Q4: Bool, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8 if Q4 else i8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, K: Int, M: Int,
) -> Array[Float32, MR]:
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[f32, QV], MR](fill=SIMD[f32, QV](0))
    comptime if Q4:
        return q4_dot_blocks[MR, 2](A, rebind[TileTensor[u8, QLayout, MutAnyOrigin]](Q), S, row, lane, 0, K // 32, M)
    else:
        comptime STEP = WARP_SIZE * QV
        var Qv = Q.vectorize[1, QV]()
        var kk = 0
        while kk + UNROLL * STEP <= K:
            var qs = Array[SIMD[i8, QV], UNROLL](uninitialized=True)
            var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
            comptime for u in range(UNROLL):
                var kb = kk + u * STEP
                qs[u] = rebind[SIMD[i8, QV]](Qv[row, kb // QV + lane])
                ds[u] = rebind[Scalar[f16]](S[row, (kb + lane * QV) // 32])
            comptime for u in range(UNROLL):
                var kb = kk + u * STEP
                var w = qs[u].cast[f32]() * ds[u].cast[f32]()
                comptime for r in range(MR):
                    if r < M:
                        var a = rebind[SIMD[bf16, QV]](Av[r, kb // QV + lane]).cast[f32]()
                        acc[r] += w * a
            kk += UNROLL * STEP
        while kk < K:
            var q = rebind[SIMD[i8, QV]](Qv[row, kk // QV + lane]).cast[f32]()
            var d = rebind[Scalar[f16]](S[row, (kk + lane * QV) // 32]).cast[f32]()
            var w = q * d
            comptime for r in range(MR):
                if r < M:
                    var a = rebind[SIMD[bf16, QV]](Av[r, kk // QV + lane]).cast[f32]()
                    acc[r] += w * a
            kk += STEP
    var out = Array[Float32, MR](fill=0)
    comptime for r in range(MR):
        if r < M:
            out[r] = warp.sum(acc[r].reduce_add())
    return out^


@always_inline
def stage_a[CBL: TensorLayout, AfL: TensorLayout, BF: Bool = False](
    A: TileTensor[bf16, CBL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    K: Int,
):
    # bench/mega-structural-protocol.md P1: the block's copy of the activation
    # row as f32 in LDS, chunk-major so a wave's per-lane block reads are
    # consecutive 16-byte ds_loads: chunk c (4 elements) of block blk lives at
    # vector index c * nb + blk.
    comptime assert A.flat_rank == 2 and Af.flat_rank == 1
    var nb = K // 32
    var Av8 = A.vectorize[1, 8]()
    var Afv = Af.vectorize[4]()
    comptime assert Afv.flat_rank == 1
    var t = Int(thread_idx.x)
    while t * 8 < K:
        var i0 = t * 8
        var blk = i0 // 32
        comptime if BF:
            # K=FFN rows stay bf16 in LDS (24 KB budget): chunk of 8 elements
            # c8 of block blk at vector index c8 * nb + blk, unpacked on read.
            var c8 = (i0 % 32) // 8
            Afv[c8 * nb + blk] = rebind[Afv.ElementType](bitcast[f32, 4](rebind[SIMD[bf16, 8]](Av8[0, t])))
        else:
            var c = (i0 % 32) // 4
            var v = rebind[SIMD[bf16, 8]](Av8[0, t]).cast[f32]()
            Afv[c * nb + blk] = rebind[Afv.ElementType](v.slice[4, offset=0]())
            Afv[(c + 1) * nb + blk] = rebind[Afv.ElementType](v.slice[4, offset=4]())
        t += ROW_THREADS
    barrier()

@always_inline
def stage_rms[XL: TensorLayout, GL: TensorLayout, AfL: TensorLayout](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
):
    comptime assert X.flat_rank == 2 and Gn.flat_rank == 1 and Af.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    var partial: Float32 = 0
    if tid < EW_THREADS:
        var i = tid
        while i < H:
            var v = rebind[Scalar[f32]](X[0, i])
            partial += v * v
            i += EW_THREADS
        var wsum = warp.sum(partial)
        if lane == 0:
            sums[wave] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(H) + RMS_EPS)
    var nb = H // 32
    var Xv8 = X.vectorize[1, 8]()
    var Gv8 = Gn.vectorize[8]()
    var Afv = Af.vectorize[4]()
    comptime assert Afv.flat_rank == 1
    var t = tid
    while t * 8 < H:
        var i0 = t * 8
        var blk = i0 // 32
        var c = (i0 % 32) // 4
        var v = (rebind[SIMD[f32, 8]](Xv8[0, t]) * scale * rebind[SIMD[f32, 8]](Gv8[t])).cast[bf16]().cast[f32]()
        Afv[c * nb + blk] = rebind[Afv.ElementType](v.slice[4, offset=0]())
        Afv[(c + 1) * nb + blk] = rebind[Afv.ElementType](v.slice[4, offset=4]())
        t += ROW_THREADS
    barrier()


@always_inline
def q4_dot_lds[
    UNROLL: Int, BF: Bool, AfL: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, nb: Int,
) -> Float32:
    # q4_dot_blocks with A read from the staged LDS copy: same nibble form,
    # same fma chain, so the result is the launch kernel's bit for bit.
    comptime QV = 16
    comptime STEP = WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Afv = Af.vectorize[4]()
    var acc = SIMD[f32, QV](0)
    var kk = 0
    while kk + UNROLL * STEP <= nb:
        var bytes_u = Array[SIMD[u8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            bytes_u[u] = rebind[SIMD[u8, QV]](Qb[row, blk])
            ds[u] = rebind[Scalar[f16]](S[row, blk])
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            var d = ds[u].cast[f32]()
            var w32 = bitcast[u32, 4](bytes_u[u])
            var lo = bitcast[u8, QV](w32 & 0x0F0F0F0F).cast[f32]()
            var hi = bitcast[u8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[f32]()
            var dv = SIMD[f32, QV](d)
            var m8d = SIMD[f32, QV](d * -8)
            var wlo = fma(lo, dv, m8d)
            var whi = fma(hi, dv, m8d)
            var a_lo: SIMD[f32, 16]
            var a_hi: SIMD[f32, 16]
            comptime if BF:
                a_lo = bf16x16_to_f32(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[0 * nb + blk])).join(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[1 * nb + blk]))))
                a_hi = bf16x16_to_f32(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[2 * nb + blk])).join(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[3 * nb + blk]))))
            else:
                a_lo = rebind[SIMD[f32, 4]](Afv[0 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[1 * nb + blk])).join(rebind[SIMD[f32, 4]](Afv[2 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[3 * nb + blk])))
                a_hi = rebind[SIMD[f32, 4]](Afv[4 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[5 * nb + blk])).join(rebind[SIMD[f32, 4]](Afv[6 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[7 * nb + blk])))
            acc = fma(wlo, a_lo, fma(whi, a_hi, acc))
        kk += UNROLL * STEP
    while kk < nb:
        var blk = kk + lane
        var bytes1 = rebind[SIMD[u8, QV]](Qb[row, blk])
        var d = rebind[Scalar[f16]](S[row, blk]).cast[f32]()
        var w32 = bitcast[u32, 4](bytes1)
        var lo = bitcast[u8, QV](w32 & 0x0F0F0F0F).cast[f32]()
        var hi = bitcast[u8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[f32]()
        var dv = SIMD[f32, QV](d)
        var m8d = SIMD[f32, QV](d * -8)
        var wlo = fma(lo, dv, m8d)
        var whi = fma(hi, dv, m8d)
        var a_lo: SIMD[f32, 16]
        var a_hi: SIMD[f32, 16]
        comptime if BF:
            a_lo = bf16x16_to_f32(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[0 * nb + blk])).join(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[1 * nb + blk]))))
            a_hi = bf16x16_to_f32(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[2 * nb + blk])).join(bitcast[bf16, 8](rebind[SIMD[f32, 4]](Afv[3 * nb + blk]))))
        else:
            a_lo = rebind[SIMD[f32, 4]](Afv[0 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[1 * nb + blk])).join(rebind[SIMD[f32, 4]](Afv[2 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[3 * nb + blk])))
            a_hi = rebind[SIMD[f32, 4]](Afv[4 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[5 * nb + blk])).join(rebind[SIMD[f32, 4]](Afv[6 * nb + blk]).join(rebind[SIMD[f32, 4]](Afv[7 * nb + blk])))
        acc = fma(wlo, a_lo, fma(whi, a_hi, acc))
        kk += STEP
    return warp.sum(acc.reduce_add())


@always_inline
def row_dot_a[
    MR: Int, Q4: Bool, LDSA: Bool, BF: Bool,
    ALayout: TensorLayout, AfL: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, KQ: Bool = False, XLDS: Bool = False,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    Q: TileTensor[u8 if Q4 else i8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, K: Int, M: Int, kind: Int = 0,
) -> Array[Float32, MR]:
    comptime if KQ:
        comptime assert MR == 1
        var out = Array[Float32, MR](fill=0)
        var w = Q.ptr.unsafe_bitcast[Scalar[u8]]()
        comptime if XLDS:
            out[0] = kq_row1(Af.ptr.unsafe_bitcast[Scalar[bf16]](), w, kind, row, lane, K)
        else:
            out[0] = kq_row1(A.ptr, w, kind, row, lane, K)
        return out^
    elif LDSA:
        var out = Array[Float32, MR](fill=0)
        out[0] = q4_dot_lds[DOT_UNROLL, BF](Af, rebind[TileTensor[u8, QLayout, MutAnyOrigin]](Q), S, row, lane, K // 32)
        return out^
    else:
        return q8_row_dot[MR, Q4](A, Q, S, row, lane, K, M)


@always_inline
def rmsc_phase[
    MR: Int, XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    M: Int,
):
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    for r in range(M):
        var partial: Float32 = 0
        if tid < EW_THREADS:
            var i = tid
            while i < H:
                var v = rebind[Scalar[f32]](X[r, i])
                partial += v * v
                i += EW_THREADS
            var wsum = warp.sum(partial)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](wsum)
        barrier()
        if tid < EW_THREADS:
            var total: Float32 = 0
            comptime for w in range(EW_THREADS // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            var scale = rsqrt(total / Float32(H) + RMS_EPS)
            var i = Int(block_idx.x) * EW_THREADS + tid
            while i < H:
                CurB[r, i] = rebind[CurB.ElementType](
                    (rebind[Scalar[f32]](X[r, i]) * scale * rebind[Scalar[f32]](Gn[i])).cast[bf16]()
                )
                i += Int(grid_dim.x) * EW_THREADS
        barrier()


@always_inline
def rms_f32_phase[
    XL: TensorLayout, GL: TensorLayout
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut O: TileTensor[f32, XL, MutAnyOrigin],
    M: Int,
):
    comptime assert X.flat_rank == 2 and O.flat_rank == 2 and Gn.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    for r in range(M):
        var partial: Float32 = 0
        if tid < EW_THREADS:
            var i = tid
            while i < H:
                var v = rebind[Scalar[f32]](X[r, i])
                partial += v * v
                i += EW_THREADS
            var wsum = warp.sum(partial)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](wsum)
        barrier()
        if tid < EW_THREADS:
            var total: Float32 = 0
            comptime for w in range(EW_THREADS // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            var scale = rsqrt(total / Float32(H) + RMS_EPS)
            var i = Int(block_idx.x) * EW_THREADS + tid
            while i < H:
                O[r, i] = rebind[O.ElementType](
                    rebind[Scalar[f32]](X[r, i]) * scale * rebind[Scalar[f32]](Gn[i])
                )
                i += Int(grid_dim.x) * EW_THREADS
        barrier()


@always_inline
def delta_col[SsL: TensorLayout, KqL: TensorLayout, OmL: TensorLayout](
    mut SAll: TileTensor[f32, SsL, MutAnyOrigin],
    kq: TileTensor[f32, KqL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    mut So: TileTensor[f32, OmL, MutAnyOrigin],
    eg: Float32, beta: Float32, vj: Float32,
    rs: Int, ws: Int, si: Int, h: Int, j: Int, r: Int,
):
    comptime assert SAll.flat_rank == 5 and kq.flat_rank == 2 and So.flat_rank == 3
    var col = SIMD[f32, SSTATE]()
    comptime for i in range(SSTATE):
        col[i] = rebind[Scalar[f32]](SAll[rs, si, h, i, j])
    var sk: Float32 = 0
    comptime for i in range(SSTATE):
        sk += col[i] * eg * rebind[Scalar[f32]](kq[1, i])
    var d = (vj - sk) * beta
    var o: Float32 = 0
    comptime for i in range(SSTATE):
        var s = col[i] * eg + rebind[Scalar[f32]](kq[1, i]) * d
        SAll[ws, si, h, i, j] = rebind[SAll.ElementType](s)
        o += s * rebind[Scalar[f32]](kq[0, i])
    So[r, h, j] = rebind[So.ElementType](o)


@always_inline
def ssm_phases[
    MR: Int, RELOAD: Bool, Q4: Bool,
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout, AfL: TensorLayout,
    QqL: TensorLayout, QsL: TensorLayout,
    HqL: TensorLayout, HsL: TensorLayout, OqL: TensorLayout, OsL: TensorLayout,
    ZL: TensorLayout, RBL: TensorLayout,
    AqL: TensorLayout, AsL: TensorLayout,
    CwL: TensorLayout, G32L: TensorLayout, NwL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout, CtrL: TensorLayout,
    KQ: Bool = False,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    Wqkvq: TileTensor[u8 if Q4 else i8, QqL, MutAnyOrigin], Wqkvs: TileTensor[f16, QsL, MutAnyOrigin],
    Wzq: TileTensor[u8 if Q4 else i8, HqL, MutAnyOrigin], Wzs: TileTensor[f16, HsL, MutAnyOrigin],
    Waq: TileTensor[u8 if Q4 else i8, AqL, MutAnyOrigin], Was: TileTensor[f16, AsL, MutAnyOrigin],
    Wbq: TileTensor[u8 if Q4 else i8, AqL, MutAnyOrigin], Wbs: TileTensor[f16, AsL, MutAnyOrigin],
    Cw: TileTensor[f32, CwL, MutAnyOrigin],
    SsmA: TileTensor[f32, G32L, MutAnyOrigin],
    DtB: TileTensor[f32, G32L, MutAnyOrigin],
    Nw: TileTensor[f32, NwL, MutAnyOrigin],
    Wsoutq: TileTensor[u8 if Q4 else i8, OqL, MutAnyOrigin], Wsouts: TileTensor[f16, OsL, MutAnyOrigin],
    mut Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    mut Zm: TileTensor[f32, ZL, MutAnyOrigin],
    mut Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    mut Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    mut So: TileTensor[f32, OmL, MutAnyOrigin],
    mut ResB: TileTensor[bf16, RBL, MutAnyOrigin],
    mut ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    mut SAll: TileTensor[f32, SsL, MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, M: Int,
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
    kk: SIMD[i32, 8] = SIMD[i32, 8](0),
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Qkvm.flat_rank == 2
    comptime assert Af.flat_rank == 1
    comptime LDSA = (MR == 1) and Q4
    comptime assert Conv.flat_rank == 2 and So.flat_rank == 3 and Eg.flat_rank == 2
    comptime assert ConvState.flat_rank == 4 and SAll.flat_rank == 5
    comptime assert Zm.flat_rank == 2 and Araw.flat_rank == 2 and Braw.flat_rank == 2 and Beta.flat_rank == 2
    comptime assert ResB.flat_rank == 2 and Cw.flat_rank == 2 and Nw.flat_rank == 1 and SsmA.flat_rank == 1 and DtB.flat_rank == 1
    comptime G_QKV = CONV // ROW_WAVES
    comptime G_Z = INNER // ROW_WAVES
    comptime G_AB = NH_V // ROW_WAVES
    comptime G_ALL = G_QKV + G_Z + 2 * G_AB
    comptime G_OUT = H // ROW_WAVES
    comptime GZ_IN = 0 if KQ else G_Z
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    var g = 0
    var si = Int(ssm_i)
    var sl = Int(slots)
    var rg = Int(ring)
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    var kq = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2, SSTATE]()
    )
    var gb = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[2]()
    )
    comptime assert 2 * SSTATE <= EW_THREADS

    stamp(prof, pbase + 12)
    comptime if LDSA:
        stage_rms(X, Gn, Af)
    elif KQ:
        stage_rms_kq(X, Gn, Af)
    else:
        rmsc_phase[MR](X, Gn, CurB, M)
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 1)

    g = bid
    while g < G_QKV + GZ_IN + G_AB + G_AB:
        if g < G_QKV:
            var row = g * ROW_WAVES + wave
            var paired = False
            comptime if KQ:
                if g + nblk < G_QKV:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wqkvq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[0]), row, row + nblk * ROW_WAVES, lane, H)
                    if lane == 0:
                        Qkvm[0, row] = rebind[Qkvm.ElementType](t2[0])
                        Qkvm[0, row + nblk * ROW_WAVES] = rebind[Qkvm.ElementType](t2[1])
                    g += nblk
                    paired = True
            if not paired:
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wqkvq, Wqkvs, row, lane, H, M, Int(kk[0]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Qkvm[r, row] = rebind[Qkvm.ElementType](t[r])
        elif g < G_QKV + GZ_IN:
            var row = (g - (G_QKV)) * ROW_WAVES + wave
            var paired = False
            comptime if KQ:
                if g + nblk < G_QKV + GZ_IN:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wzq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[1]), row, row + nblk * ROW_WAVES, lane, H)
                    if lane == 0:
                        Zm[0, row] = rebind[Zm.ElementType](t2[0])
                        Zm[0, row + nblk * ROW_WAVES] = rebind[Zm.ElementType](t2[1])
                    g += nblk
                    paired = True
            if not paired:
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wzq, Wzs, row, lane, H, M, Int(kk[1]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Zm[r, row] = rebind[Zm.ElementType](t[r])
        elif g < G_QKV + GZ_IN + G_AB:
            var row = (g - (G_QKV + GZ_IN)) * ROW_WAVES + wave
            var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Waq, Was, row, lane, H, M, Int(kk[2]))
            if lane == 0:
                comptime for r in range(MR):
                    if r < M:
                        Araw[r, row] = rebind[Araw.ElementType](t[r])
        else:
            var row = (g - (G_QKV + GZ_IN + G_AB)) * ROW_WAVES + wave
            var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wbq, Wbs, row, lane, H, M, Int(kk[3]))
            if lane == 0:
                comptime for r in range(MR):
                    if r < M:
                        Braw[r, row] = rebind[Braw.ElementType](t[r])
        g += nblk
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 2)

    comptime if not LDSA:
        if bid == 0 and tid < NH_V:
            var h = tid
            var sa = rebind[Scalar[f32]](SsmA[h])
            var db = rebind[Scalar[f32]](DtB[h])
            for r in range(M):
                var braw = rebind[Scalar[f32]](Braw[r, h])
                Beta[r, h] = rebind[Beta.ElementType](1 / (1 + exp(-braw)))
                var asum = rebind[Scalar[f32]](Araw[r, h]) + db
                var sp = log1p(exp(asum))
                Eg[r, h] = rebind[Eg.ElementType](exp(sp * sa))
        var c = bid * ROW_THREADS + tid
        while c < CONV:
            var cw0 = rebind[Scalar[f32]](Cw[c, 0])
            var cw1 = rebind[Scalar[f32]](Cw[c, 1])
            var cw2 = rebind[Scalar[f32]](Cw[c, 2])
            var cw3 = rebind[Scalar[f32]](Cw[c, 3])
            for r in range(M):
                var rs = (rg + r) % sl
                var ws = (rg + r + 1) % sl
                var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
                var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
                var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
                var w3 = rebind[Scalar[f32]](Qkvm[r, c])
                var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
                Conv[r, c] = rebind[Conv.ElementType](acc / (1 + exp(-acc)))
                ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
                ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
                ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)
            c += nblk * ROW_THREADS
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 3)

    comptime if not LDSA:
        if bid < NH_V:
            var head = bid
            var base = head * SSTATE
            for r in range(M):
                var v: Float32 = 0
                if tid < SSTATE:
                    v = rebind[Scalar[f32]](Conv[r, base + tid])
                    var ssq = warp.sum(v * v)
                    if lane == 0:
                        sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                if tid < SSTATE:
                    var total: Float32 = 0
                    comptime for w in range(SSTATE // WARP_SIZE):
                        total += rebind[Scalar[f32]](sums[w])
                    var inv = rsqrt(total + SSM_EPS)
                    if head < NH_K:
                        inv = inv / sqrt(Float32(SSTATE))
                    Conv[r, base + tid] = rebind[Conv.ElementType](v * inv)
                barrier()
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 4)

    if bid < NH_V:
        var h = bid
        var j = tid
        var kh = h % NH_K
        for r in range(M):
            var rs = (rg + r) % sl
            var ws = (rg + r + 1) % sl
            var vreg: Float32 = 0
            comptime if LDSA:
                if j == 0:
                    var sa = rebind[Scalar[f32]](SsmA[h])
                    var db = rebind[Scalar[f32]](DtB[h])
                    var braw = rebind[Scalar[f32]](Braw[r, h])
                    gb[1] = rebind[gb.ElementType](1 / (1 + exp(-braw)))
                    var asum = rebind[Scalar[f32]](Araw[r, h]) + db
                    var sp = log1p(exp(asum))
                    gb[0] = rebind[gb.ElementType](exp(sp * sa))
                var half = j // SSTATE
                var jj = j % SSTATE
                var xq: Float32 = 0
                if j < 2 * SSTATE:
                    var c = half * KDIM + kh * SSTATE + jj
                    var cw0 = rebind[Scalar[f32]](Cw[c, 0])
                    var cw1 = rebind[Scalar[f32]](Cw[c, 1])
                    var cw2 = rebind[Scalar[f32]](Cw[c, 2])
                    var cw3 = rebind[Scalar[f32]](Cw[c, 3])
                    var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
                    var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
                    var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
                    var w3 = rebind[Scalar[f32]](Qkvm[r, c])
                    var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
                    xq = acc / (1 + exp(-acc))
                    if h < NH_K:
                        ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
                        ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
                        ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)
                if j < SSTATE:
                    var c = 2 * KDIM + h * SSTATE + j
                    var cw0 = rebind[Scalar[f32]](Cw[c, 0])
                    var cw1 = rebind[Scalar[f32]](Cw[c, 1])
                    var cw2 = rebind[Scalar[f32]](Cw[c, 2])
                    var cw3 = rebind[Scalar[f32]](Cw[c, 3])
                    var w0 = rebind[Scalar[f32]](ConvState[rs, si, 0, c])
                    var w1 = rebind[Scalar[f32]](ConvState[rs, si, 1, c])
                    var w2 = rebind[Scalar[f32]](ConvState[rs, si, 2, c])
                    var w3 = rebind[Scalar[f32]](Qkvm[r, c])
                    var acc = w0 * cw0 + w1 * cw1 + w2 * cw2 + w3 * cw3
                    vreg = acc / (1 + exp(-acc))
                    ConvState[ws, si, 0, c] = rebind[ConvState.ElementType](w1)
                    ConvState[ws, si, 1, c] = rebind[ConvState.ElementType](w2)
                    ConvState[ws, si, 2, c] = rebind[ConvState.ElementType](w3)
                if j < 2 * SSTATE:
                    var ssq = warp.sum(xq * xq)
                    if lane == 0:
                        sums[wave] = rebind[sums.ElementType](ssq)
                barrier()
                if j < 2 * SSTATE:
                    var total: Float32 = 0
                    comptime for w in range(SSTATE // WARP_SIZE):
                        total += rebind[Scalar[f32]](sums[half * (SSTATE // WARP_SIZE) + w])
                    var inv = rsqrt(total + SSM_EPS)
                    if half == 0:
                        inv = inv / sqrt(Float32(SSTATE))
                    kq[half, jj] = rebind[kq.ElementType](xq * inv)
            else:
                if j < SSTATE:
                    kq[0, j] = rebind[kq.ElementType](Conv[r, kh * SSTATE + j])
                    kq[1, j] = rebind[kq.ElementType](Conv[r, KDIM + kh * SSTATE + j])
            barrier()
            if j < SSTATE:
                var eg: Float32
                var beta: Float32
                var vj: Float32
                comptime if LDSA:
                    eg = rebind[Scalar[f32]](gb[0])
                    beta = rebind[Scalar[f32]](gb[1])
                    vj = vreg
                else:
                    eg = rebind[Scalar[f32]](Eg[r, h])
                    beta = rebind[Scalar[f32]](Beta[r, h])
                    vj = rebind[Scalar[f32]](Conv[r, 2 * KDIM + h * SSTATE + j])
                comptime if not RELOAD:
                    delta_col(SAll, kq, So, eg, beta, vj, rs, ws, si, h, j, r)
                else:
                    comptime CHK = 32
                    var sk: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = rebind[Scalar[f32]](SAll[rs, si, h, c * CHK + ii, j])
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            sk = fma(t, rebind[Scalar[f32]](kq[1, c * CHK + ii]), sk)
                    var d = (vj - sk) * beta
                    var o: Float32 = 0
                    for c in range(SSTATE // CHK):
                        var col = Array[Float32, CHK](uninitialized=True)
                        comptime for ii in range(CHK):
                            col[ii] = rebind[Scalar[f32]](SAll[rs, si, h, c * CHK + ii, j])
                        comptime for ii in range(CHK):
                            var t = col[ii] * eg
                            var s = fma(rebind[Scalar[f32]](kq[1, c * CHK + ii]), d, t)
                            SAll[ws, si, h, c * CHK + ii, j] = rebind[SAll.ElementType](s)
                            o = fma(s, rebind[Scalar[f32]](kq[0, c * CHK + ii]), o)
                    So[r, h, j] = rebind[So.ElementType](o)
            barrier()
    comptime if KQ:
        var zlo = NH_V if nblk > NH_V else 0
        if bid >= zlo:
            var zs = nblk - zlo
            var gz = bid - zlo
            while gz < G_Z:
                var row = gz * ROW_WAVES + wave
                if gz + zs < G_Z:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wzq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[1]), row, row + zs * ROW_WAVES, lane, H)
                    if lane == 0:
                        Zm[0, row] = rebind[Zm.ElementType](t2[0])
                        Zm[0, row + zs * ROW_WAVES] = rebind[Zm.ElementType](t2[1])
                    gz += 2 * zs
                else:
                    var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wzq, Wzs, row, lane, H, M, Int(kk[1]))
                    if lane == 0:
                        Zm[0, row] = rebind[Zm.ElementType](t[0])
                    gz += zs
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 5)

    if bid < NH_V:
        var h = bid
        var j = tid
        var nwj: Float32 = 0
        if j < SSTATE:
            nwj = rebind[Scalar[f32]](Nw[j])
        for r in range(M):
            var v: Float32 = 0
            if j < SSTATE:
                v = rebind[Scalar[f32]](So[r, h, j])
                var ssq = warp.sum(v * v)
                if lane == 0:
                    sums[wave] = rebind[sums.ElementType](ssq)
            barrier()
            if j < SSTATE:
                var total: Float32 = 0
                comptime for w in range(SSTATE // WARP_SIZE):
                    total += rebind[Scalar[f32]](sums[w])
                var scale = rsqrt(total / Float32(SSTATE) + SSM_EPS)
                var z = rebind[Scalar[f32]](Zm[r, h * SSTATE + j])
                ResB[r, h * SSTATE + j] = rebind[ResB.ElementType](
                    (v * scale * nwj * (z / (1 + exp(-z)))).cast[bf16]()
                )
            barrier()
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 6)

    comptime if LDSA:
        stage_a(ResB, Af, INNER)
    stamp(prof, pbase + 13)
    g = bid
    while g < G_OUT:
        var row = g * ROW_WAVES + wave
        var paired = False
        comptime if KQ:
            if g + nblk < G_OUT:
                var t2 = kq_row2(ResB.ptr, Wsoutq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[4]), row, row + nblk * ROW_WAVES, lane, INNER)
                if lane == 0:
                    X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t2[0])
                    X[0, row + nblk * ROW_WAVES] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row + nblk * ROW_WAVES]) + t2[1])
                g += nblk
                paired = True
        if not paired:
            var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ](ResB, Af, Wsoutq, Wsouts, row, lane, INNER, M, Int(kk[4]))
            if lane == 0:
                comptime for r in range(MR):
                    if r < M:
                        X[r, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, row]) + t[r])
        g += nblk
    stamp(prof, pbase + 14)
    return True


@always_inline
def ffn_phases[
    MR: Int, Q4: Bool,
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout, AfL: TensorLayout,
    GqL: TensorLayout, GsL: TensorLayout, DqL: TensorLayout, DsL: TensorLayout,
    PL: TensorLayout, FBL: TensorLayout, CtrL: TensorLayout,
    KQ: Bool = False,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    Wgq: TileTensor[u8 if Q4 else i8, GqL, MutAnyOrigin], Wgs: TileTensor[f16, GsL, MutAnyOrigin],
    Wuq: TileTensor[u8 if Q4 else i8, GqL, MutAnyOrigin], Wus: TileTensor[f16, GsL, MutAnyOrigin],
    Wdq: TileTensor[u8 if Q4 else i8, DqL, MutAnyOrigin], Wds: TileTensor[f16, DsL, MutAnyOrigin],
    mut Pg: TileTensor[f32, PL, MutAnyOrigin],
    mut Pu: TileTensor[f32, PL, MutAnyOrigin],
    mut FgB: TileTensor[bf16, FBL, MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    M: Int,
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
    sv: MutPointer[Scalar[f32], MutAnyOrigin], sv_on: Int32,
    kk: SIMD[i32, 8] = SIMD[i32, 8](0),
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    comptime assert Af.flat_rank == 1
    comptime LDSA = (MR == 1) and Q4
    comptime assert Pg.flat_rank == 2 and Pu.flat_rank == 2 and FgB.flat_rank == 2
    comptime G_F = FFN // ROW_WAVES
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    var g = 0

    comptime if LDSA:
        stage_rms(X, Gn, Af)
    elif KQ:
        stage_rms_kq(X, Gn, Af)
    else:
        rmsc_phase[MR](X, Gn, CurB, M)
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 1)

    g = bid
    comptime if LDSA:
        while g < G_F:
            var row = g * ROW_WAVES + wave
            var tg = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wgq, Wgs, row, lane, H, M, Int(kk[0]))
            var tu = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wuq, Wus, row, lane, H, M, Int(kk[1]))
            if lane == 0:
                var gg = tg[0]
                var silu = gg / (1 + exp(-gg))
                FgB[0, row] = rebind[FgB.ElementType]((silu * tu[0]).cast[bf16]())
            g += nblk
    elif KQ:
        while g < G_F:
            var row = g * ROW_WAVES + wave
            var t = kq_gate_up(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wgq.ptr.unsafe_bitcast[Scalar[u8]](), Wuq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[0]), Int(kk[1]), row, lane, H)
            if lane == 0:
                var gg = t[0]
                var silu = gg / (1 + exp(-gg))
                FgB[0, row] = rebind[FgB.ElementType]((silu * t[1]).cast[bf16]())
            g += nblk
    else:
        while g < G_F + G_F:
            if g < G_F:
                var row = g * ROW_WAVES + wave
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wgq, Wgs, row, lane, H, M, Int(kk[0]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Pg[r, row] = rebind[Pg.ElementType](t[r])
            else:
                var row = (g - (G_F)) * ROW_WAVES + wave
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wuq, Wus, row, lane, H, M, Int(kk[1]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Pu[r, row] = rebind[Pu.ElementType](t[r])
            g += nblk
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 2)

    comptime if not LDSA and not KQ:
        var c = bid * ROW_THREADS + tid
        while c < M * FFN:
            var r = c // FFN
            var cc = c % FFN
            var gg = rebind[Scalar[f32]](Pg[r, cc])
            var u = rebind[Scalar[f32]](Pu[r, cc])
            var silu = gg / (1 + exp(-gg))
            FgB[r, cc] = rebind[FgB.ElementType]((silu * u).cast[bf16]())
            c += nblk * ROW_THREADS
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 3)

    comptime if LDSA:
        stage_a[BF=True](FgB, Af, FFN)
    stamp(prof, pbase + 8)
    g = bid
    while g < H // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var paired = False
        comptime if KQ:
            if g + nblk < H // ROW_WAVES:
                var t2 = kq_row2(FgB.ptr, Wdq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[2]), row, row + nblk * ROW_WAVES, lane, FFN)
                if lane == 0:
                    # steering (sv_on): the vector for this layer is added right after the residual add, in the same order as the launch path
                    var s0: Float32 = 0
                    var s1: Float32 = 0
                    if sv_on != 0:
                        s0 = sv[unsafe_offset=row]
                        s1 = sv[unsafe_offset=row + nblk * ROW_WAVES]
                    X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t2[0] + s0)
                    X[0, row + nblk * ROW_WAVES] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row + nblk * ROW_WAVES]) + t2[1] + s1)
                g += nblk
                paired = True
        if not paired:
            var t = row_dot_a[MR, Q4, LDSA, True, KQ=KQ](FgB, Af, Wdq, Wds, row, lane, FFN, M, Int(kk[2]))
            if lane == 0:
                comptime for r in range(MR):
                    if r < M:
                        var sx: Float32 = 0
                        if sv_on != 0:
                            sx = sv[unsafe_offset=row]
                        X[r, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, row]) + t[r] + sx)
        g += nblk
    return True


@always_inline
def rope_cs(j: Int, pos: Int) -> SIMD[f32, 2]:
    var theta_ex = Float32(pos) * exp(Float32(-2 * j) / Float32(NROT) * log(FREQ_BASE))
    var theta_in = FREQ_SCALE * theta_ex
    var ramp = (Float32(j) - YARN_LOW) / max(YARN_HIGH - YARN_LOW, 0.001)
    ramp = min(max(ramp, 0), 1)
    var theta = theta_in * (1 - ramp) + theta_ex * ramp
    return SIMD[f32, 2](cos(theta) * MSCALE, sin(theta) * MSCALE)


@always_inline
def attn_phases[
    MR: Int, Q4: Bool, NAT: Int,
    XL: TensorLayout, GL: TensorLayout, CBL: TensorLayout, AfL: TensorLayout, RBL: TensorLayout,
    QqL: TensorLayout, QsL: TensorLayout, KqL: TensorLayout, KsL: TensorLayout,
    HqL: TensorLayout, HsL: TensorLayout, HdL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout,
    GfL: TensorLayout, CacheL: TensorLayout, CtrL: TensorLayout, PL: TensorLayout,
    KQ: Bool = False,
](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
    Wqq: TileTensor[u8 if Q4 else i8, QqL, MutAnyOrigin], Wqs: TileTensor[f16, QsL, MutAnyOrigin],
    Wkq: TileTensor[u8 if Q4 else i8, KqL, MutAnyOrigin], Wks: TileTensor[f16, KsL, MutAnyOrigin],
    Wvq: TileTensor[u8 if Q4 else i8, KqL, MutAnyOrigin], Wvs: TileTensor[f16, KsL, MutAnyOrigin],
    Qn: TileTensor[f32, HdL, MutAnyOrigin],
    Kn: TileTensor[f32, HdL, MutAnyOrigin],
    Woq: TileTensor[u8 if Q4 else i8, HqL, MutAnyOrigin], Wos: TileTensor[f16, HsL, MutAnyOrigin],
    mut Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    mut Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    mut Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    mut Q: TileTensor[f32, QmL, MutAnyOrigin],
    mut Gate: TileTensor[f32, GfL, MutAnyOrigin],
    mut Ao: TileTensor[f32, QmL, MutAnyOrigin],
    mut AoB: TileTensor[bf16, RBL, MutAnyOrigin],
    mut Kc: TileTensor[KVT, CacheL, MutAnyOrigin],
    mut Vc: TileTensor[KVT, CacheL, MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    mut Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    mut Pg: TileTensor[f32, PL, MutAnyOrigin],
    pos: Int, M: Int, att_i: Int, att_split: Int,
    prof: MutPointer[Scalar[i64], MutAnyOrigin], pbase: Int,
    kk: SIMD[i32, 8] = SIMD[i32, 8](0),
) -> Bool:
    comptime assert X.flat_rank == 2 and CurB.flat_rank == 2 and Gn.flat_rank == 1
    comptime assert Af.flat_rank == 1
    comptime LDSA = (MR == 1) and Q4
    comptime assert Qfm.flat_rank == 2 and Kflat.flat_rank == 2 and Vflat.flat_rank == 2
    comptime assert Q.flat_rank == 2 and Ao.flat_rank == 2 and Gate.flat_rank == 1 and AoB.flat_rank == 2
    comptime assert Kc.flat_rank == 1 and Vc.flat_rank == 1 and Qn.flat_rank == 1 and Kn.flat_rank == 1
    comptime assert MR * (NQH + NKVH) <= MEGA_G
    comptime G_Q = QF // ROW_WAVES
    comptime G_KV = KV // ROW_WAVES
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var ctr = Ctr.ptr
    var gen = Ctr.ptr.unsafe_offset(1)
    var fail = Ctr.ptr.unsafe_offset(2)
    var g = 0
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD // WARP_SIZE]()
    )
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD]()
    )
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[HD]()
    )

    comptime if LDSA:
        stage_rms(X, Gn, Af)
    elif KQ:
        stage_rms_kq(X, Gn, Af)
    else:
        rmsc_phase[MR](X, Gn, CurB, M)
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 1)

    g = bid
    while g < G_Q + G_KV + G_KV:
        if g < G_Q:
            var row = g * ROW_WAVES + wave
            var paired = False
            comptime if KQ:
                if g + nblk < G_Q:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wqq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[0]), row, row + nblk * ROW_WAVES, lane, H)
                    if lane == 0:
                        Qfm[0, row] = rebind[Qfm.ElementType](t2[0])
                        Qfm[0, row + nblk * ROW_WAVES] = rebind[Qfm.ElementType](t2[1])
                    g += nblk
                    paired = True
            if not paired:
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wqq, Wqs, row, lane, H, M, Int(kk[0]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Qfm[r, row] = rebind[Qfm.ElementType](t[r])
        elif g < G_Q + G_KV:
            var row = (g - (G_Q)) * ROW_WAVES + wave
            var paired = False
            comptime if KQ:
                if g + nblk < G_Q + G_KV:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wkq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[1]), row, row + nblk * ROW_WAVES, lane, H)
                    if lane == 0:
                        Kflat[0, row] = rebind[Kflat.ElementType](t2[0])
                        Kflat[0, row + nblk * ROW_WAVES] = rebind[Kflat.ElementType](t2[1])
                    g += nblk
                    paired = True
            if not paired:
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wkq, Wks, row, lane, H, M, Int(kk[1]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Kflat[r, row] = rebind[Kflat.ElementType](t[r])
        else:
            var row = (g - (G_Q + G_KV)) * ROW_WAVES + wave
            var paired = False
            comptime if KQ:
                if g + nblk < G_Q + G_KV + G_KV:
                    var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Wvq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[2]), row, row + nblk * ROW_WAVES, lane, H)
                    if lane == 0:
                        Vflat[0, row] = rebind[Vflat.ElementType](t2[0])
                        Vflat[0, row + nblk * ROW_WAVES] = rebind[Vflat.ElementType](t2[1])
                    g += nblk
                    paired = True
            if not paired:
                var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB, Af, Wvq, Wvs, row, lane, H, M, Int(kk[2]))
                if lane == 0:
                    comptime for r in range(MR):
                        if r < M:
                            Vflat[r, row] = rebind[Vflat.ElementType](t[r])
        g += nblk
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 2)

    if bid < M * NQH:
        var r = bid // NQH
        var h = bid % NQH
        var qrow = r * NQH + h
        var v: Float32 = 0
        if tid < HD:
            v = rebind[Scalar[f32]](Qfm[r, h * 2 * HD + tid])
            Gate[r * NQH * HD + h * HD + tid] = rebind[Gate.ElementType](Qfm[r, h * 2 * HD + HD + tid])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if tid < HD:
            var total: Float32 = 0
            comptime for w in range(HD // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            Q[qrow, tid] = rebind[Q.ElementType](
                v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Qn[tid])
            )
        barrier()
        if tid < NROT // 2:
            var cs = rope_cs(tid, pos + r)
            var x0 = rebind[Scalar[f32]](Q[qrow, tid])
            var x1 = rebind[Scalar[f32]](Q[qrow, tid + NROT // 2])
            Q[qrow, tid] = rebind[Q.ElementType](x0 * cs[0] - x1 * cs[1])
            Q[qrow, tid + NROT // 2] = rebind[Q.ElementType](x0 * cs[1] + x1 * cs[0])
    elif bid < M * NQH + M * NKVH:
        var b = bid - M * NQH
        var r = b // NKVH
        var h = b % NKVH
        var v: Float32 = 0
        if tid < HD:
            v = rebind[Scalar[f32]](Kflat[r, h * HD + tid])
            var ssq = warp.sum(v * v)
            if lane == 0:
                sums[wave] = rebind[sums.ElementType](ssq)
        barrier()
        if tid < HD:
            var total: Float32 = 0
            comptime for w in range(HD // WARP_SIZE):
                total += rebind[Scalar[f32]](sums[w])
            Kflat[r, h * HD + tid] = rebind[Kflat.ElementType](
                v * rsqrt(total / Float32(HD) + RMS_EPS) * rebind[Scalar[f32]](Kn[tid])
            )
        barrier()
        if tid < NROT // 2:
            var cs = rope_cs(tid, pos + r)
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
                var rb = kv_tab_off[NAT](tab, pos + r, att_i, Int(h))
                Kc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(kx, Float32(127) / km if km > 0 else Float32(0)))
                Vc.ptr[unsafe_offset=rb + tid] = rebind[Scalar[KVT]](dkv_q(vx, Float32(127) / vm if vm > 0 else Float32(0)))
                if tid == 0:
                    Kc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, pos + r)] = km / 127
                    Vc.ptr.unsafe_bitcast[Scalar[f32]]()[unsafe_offset=dkv_sc_off[HD](rb, pos + r)] = vm / 127
        else:
            if tid < HD:
                var kb = kv_tab_off[NAT](tab, pos + r, att_i, Int(h)) + tid
                Kc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](
                    rebind[Scalar[f32]](Kflat[r, h * HD + tid]).cast[KVT]()
                )
                Vc.ptr[unsafe_offset=kb] = rebind[Scalar[KVT]](
                    rebind[Scalar[f32]](Vflat[r, h * HD + tid]).cast[KVT]()
                )
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 3)

    var do_split = pos + 1 > att_split
    if not do_split:
        if bid < M * NQH:
            var r = bid // NQH
            var h = bid % NQH
            var qrow = r * NQH + h
            var kvh = h // (NQH // NKVH)
            var T = pos + 1 + r
            var res = attn_head_span[NAT=NAT](Q, Kc, Vc, tab, qs, scores, sums, qrow, kvh, 0, T, tid, lane, ATT_SCALE, att_i)
            if tid < HD:
                var inv = 1 / res[1]
                comptime if LDSA:
                    var gg = rebind[Scalar[f32]](Gate[qrow * HD + tid])
                    AoB[0, qrow * HD + tid] = rebind[AoB.ElementType](((res[2] * inv) * (1 / (1 + exp(-gg)))).cast[bf16]())
                else:
                    Ao[qrow, tid] = rebind[Ao.ElementType](res[2] * inv)
    else:
        var ns = dattn_nsplit[HD, DATT_NLD, NKVH](pos + 1, M, MEGA_G)
        if bid < M * NKVH * ns:
            var r = bid // (NKVH * ns)
            var rem = bid % (NKVH * ns)
            dattn_split_body[HD, NQH, NKVH, KVT, NAT, DATT_NLD, False](
                Q, Kc, Vc, Ao, Pg, tab, rem // ns, rem % ns, r, ns, pos + 1 + r, ATT_SCALE, att_i, tid
            )
        if ns > 1:
            if not grid_barrier(ctr, gen, fail):
                return False
            if bid < M * NQH:
                dattn_combine_body[HD, MEGA_G](Pg, Ao, bid, ns, tid)
    if not grid_barrier(ctr, gen, fail):
        return False
    stamp(prof, pbase + 4)

    if do_split or not LDSA:
        var i = bid * ROW_THREADS + tid
        while i < M * ATT:
            var gg = rebind[Scalar[f32]](Gate[i])
            AoB[i // ATT, i % ATT] = rebind[AoB.ElementType](
                (rebind[Scalar[f32]](Ao[i // HD, i % HD]) * (1 / (1 + exp(-gg)))).cast[bf16]()
            )
            i += nblk * ROW_THREADS
        if not grid_barrier(ctr, gen, fail):
            return False
    stamp(prof, pbase + 5)

    comptime if LDSA:
        stage_a(AoB, Af, ATT)
    g = bid
    while g < H // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var paired = False
        comptime if KQ:
            if g + nblk < H // ROW_WAVES:
                var t2 = kq_row2(AoB.ptr, Woq.ptr.unsafe_bitcast[Scalar[u8]](), Int(kk[3]), row, row + nblk * ROW_WAVES, lane, ATT)
                if lane == 0:
                    X[0, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row]) + t2[0])
                    X[0, row + nblk * ROW_WAVES] = rebind[X.ElementType](rebind[Scalar[f32]](X[0, row + nblk * ROW_WAVES]) + t2[1])
                g += nblk
                paired = True
        if not paired:
            var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ](AoB, Af, Woq, Wos, row, lane, ATT, M, Int(kk[3]))
            if lane == 0:
                comptime for r in range(MR):
                    if r < M:
                        X[r, row] = rebind[X.ElementType](rebind[Scalar[f32]](X[r, row]) + t[r])
        g += nblk
    return True


@always_inline
def wq[N: Int, K: Int, Q4: Bool](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[u8 if Q4 else i8, type_of(row_major[N, (K // 2) if Q4 else K]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o).unsafe_bitcast[Scalar[u8 if Q4 else i8]](), row_major[N, (K // 2) if Q4 else K]())


@always_inline
def ws[N: Int, K: Int, Q4: Bool](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f16, type_of(row_major[N, K // 32]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o + ((N * K // 2) if Q4 else (N * K))).unsafe_bitcast[Scalar[f16]](), row_major[N, K // 32]())


@always_inline
def wf[N: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f32, type_of(row_major[N]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o).unsafe_bitcast[Scalar[f32]](), row_major[N]())


@always_inline
def wf2[N: Int, M: Int](wbuf: MutPointer[Scalar[u8], MutAnyOrigin], o: Int) -> TileTensor[f32, type_of(row_major[N, M]()), MutAnyOrigin]:
    return TileTensor(wbuf.unsafe_offset(o).unsafe_bitcast[Scalar[f32]](), row_major[N, M]())


@always_inline
def kq_gate_up[XO: Origin, XS: AddressSpace, //](xp: Pointer[Scalar[bf16], XO, address_space=XS], wg: MutPointer[Scalar[u8], MutAnyOrigin], wu: MutPointer[Scalar[u8], MutAnyOrigin], kg: Int, ku: Int, row: Int, lane: Int, K: Int) -> Tuple[Float32, Float32]:
    if kg == GT_Q4K and ku == GT_Q4K:
        var t = kq_lane_part_ld2[GT_Q4K, bf16](xp, wg, wu, row * (K // 256) * kq_bytes[GT_Q4K](), row * (K // 256) * kq_bytes[GT_Q4K](), K, lane)
        return (kq_wave_sum(kq_tree16(t[0])), kq_wave_sum(kq_tree16(t[1])))
    return (kq_row1(xp, wg, kg, row, lane, K), kq_row1(xp, wu, ku, row, lane, K))


@always_inline
def kq_row1[XO: Origin, XS: AddressSpace, //](xp: Pointer[Scalar[bf16], XO, address_space=XS], w: MutPointer[Scalar[u8], MutAnyOrigin], kind: Int, row: Int, lane: Int, K: Int) -> Float32:
    if kind == GT_Q6K:
        return kq_wave_sum(kq_tree16(kq_lane_part_ld[GT_Q6K, bf16](xp, w, row * (K // 256) * kq_bytes[GT_Q6K](), K, lane)))
    return kq_wave_sum(kq_tree16(kq_lane_part_ld[GT_Q4K, bf16](xp, w, row * (K // 256) * kq_bytes[GT_Q4K](), K, lane)))


@always_inline
def kq_row2[XO: Origin, XS: AddressSpace, //](xp: Pointer[Scalar[bf16], XO, address_space=XS], w: MutPointer[Scalar[u8], MutAnyOrigin], kind: Int, row0: Int, row1: Int, lane: Int, K: Int) -> Tuple[Float32, Float32]:
    if kind == GT_Q6K:
        var t = kq_lane_part_ld2[GT_Q6K, bf16](xp, w, w, row0 * (K // 256) * kq_bytes[GT_Q6K](), row1 * (K // 256) * kq_bytes[GT_Q6K](), K, lane)
        return (kq_wave_sum(kq_tree16(t[0])), kq_wave_sum(kq_tree16(t[1])))
    var t = kq_lane_part_ld2[GT_Q4K, bf16](xp, w, w, row0 * (K // 256) * kq_bytes[GT_Q4K](), row1 * (K // 256) * kq_bytes[GT_Q4K](), K, lane)
    return (kq_wave_sum(kq_tree16(t[0])), kq_wave_sum(kq_tree16(t[1])))


@always_inline
def stage_rms_kq[XL: TensorLayout, GL: TensorLayout, AfL: TensorLayout](
    mut X: TileTensor[f32, XL, MutAnyOrigin],
    Gn: TileTensor[f32, GL, MutAnyOrigin],
    mut Af: TileTensor[f32, AfL, MutUntrackedOrigin, address_space = AddressSpace.SHARED],
):
    # rmsc_phase's bf16 row, computed by every block into its own LDS (no grid barrier); the
    # element arithmetic is rmsc_phase's, so the kq dots read the same bytes.
    comptime assert X.flat_rank == 2 and Gn.flat_rank == 1 and Af.flat_rank == 1
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](
        row_major[EW_THREADS // WARP_SIZE]()
    )
    var partial: Float32 = 0
    if tid < EW_THREADS:
        var i = tid
        while i < H:
            var v = rebind[Scalar[f32]](X[0, i])
            partial += v * v
            i += EW_THREADS
        var wsum = warp.sum(partial)
        if lane == 0:
            sums[wave] = rebind[sums.ElementType](wsum)
    barrier()
    var total: Float32 = 0
    comptime for w in range(EW_THREADS // WARP_SIZE):
        total += rebind[Scalar[f32]](sums[w])
    var scale = rsqrt(total / Float32(H) + RMS_EPS)
    comptime assert H % 16 == 0
    var xb = Af.ptr.unsafe_bitcast[Scalar[bf16]]()
    var Xv = X.vectorize[1, 16]()
    var Gv = Gn.vectorize[16]()
    comptime assert Xv.flat_rank == 2 and Gv.flat_rank == 1
    var t = tid
    while t * 16 < H:
        var v = rebind[SIMD[f32, 16]](Xv[0, t]) * scale * rebind[SIMD[f32, 16]](Gv[t])
        xb.unsafe_offset(t * 16).store(v.cast[bf16]())
        t += ROW_THREADS
    barrier()


@always_inline
def kq_split(mut o: Int) -> Int32:
    var k = Int32(o >> 56)
    o = o & ((1 << 56) - 1)
    return k


@always_inline
def mega_body[
    MR: Int, RELOAD: Bool, Q4: Bool,
    XL: TensorLayout, CBL: TensorLayout, ZL: TensorLayout, RBL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout, GfL: TensorLayout,
    PfL: TensorLayout, FbL: TensorLayout, OffL: TensorLayout, CtrL: TensorLayout, TkL: TensorLayout, DkL: TensorLayout,
    NL: Int, NAT: Int, KQ: Bool = False,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: TileTensor[i64, OffL, MutAnyOrigin],
    X: TileTensor[f32, XL, MutAnyOrigin],
    CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    ResB: TileTensor[bf16, RBL, MutAnyOrigin],
    Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    Zm: TileTensor[f32, ZL, MutAnyOrigin],
    Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    So: TileTensor[f32, OmL, MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Q: TileTensor[f32, QmL, MutAnyOrigin],
    Gate: TileTensor[f32, GfL, MutAnyOrigin],
    Ao: TileTensor[f32, QmL, MutAnyOrigin],
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    Pg: TileTensor[f32, PfL, MutAnyOrigin],
    Pu: TileTensor[f32, PfL, MutAnyOrigin],
    FgB: TileTensor[bf16, FbL, MutAnyOrigin],
    Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    Toks: TileTensor[i32, TkL, MutAnyOrigin],
    Dtok: TileTensor[i32, DkL, MutAnyOrigin],
    Hn: TileTensor[f32, XL, MutAnyOrigin],
    hmax: MutPointer[Scalar[f32], MutAnyOrigin],
    hidx: MutPointer[Scalar[i32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, m: Int32, dump: Int32, fold_head: Int32, att_split: Int32,
    sv: MutPointer[Scalar[f32], MutAnyOrigin], sv_on: Int32,
):
    comptime assert off.flat_rank == 1 and Toks.flat_rank == 1 and Dtok.flat_rank == 1 and Hn.flat_rank == 2
    comptime if KQ:
        comptime assert MR == 1, "the kq row pairs and stage_rms_kq read row 0 only"
    comptime cache_layout = row_major[TCAP]()
    var X_ = X
    var CurB_ = CurB
    var ResB_ = ResB
    var Qkvm_ = Qkvm
    var Zm_ = Zm
    var Araw_ = Araw
    var Braw_ = Braw
    var Eg_ = Eg
    var Beta_ = Beta
    var Conv_ = Conv
    var So_ = So
    var ConvState_ = ConvState
    var SAll_ = SAll
    var Qfm_ = Qfm
    var Kflat_ = Kflat
    var Vflat_ = Vflat
    var Q_ = Q
    var Gate_ = Gate
    var Ao_ = Ao
    var Pg_ = Pg
    var Pu_ = Pu
    var FgB_ = FgB
    var Ctr_ = Ctr
    var Toks_ = Toks
    var Dtok_ = Dtok
    var Hn_ = Hn
    comptime LDSA = (MR == 1) and Q4
    # f32 LDS staging holds one K-row: every staged K (H, INNER, ATT) must fit.
    comptime assert max(H, max(INNER, ATT)) <= FFN // 2
    var Af = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[(FFN // 2) if ((MR == 1) and Q4) else ((H // 2) if KQ else 4)]())
    var M = Int(m)
    var fail = Ctr_.ptr.unsafe_offset(2)
    if Atomic[Scalar[u32], scope=GPU_SCOPE].load[ordering=Ordering.ACQUIRE](fail) != 0:
        return
    var w = 1
    var ssm_i = 0
    var att_i = 0
    var p = Int(pos)
    for layer in range(NL):
        stamp(prof, 16 * layer)
        var o0 = Int(rebind[Scalar[i64]](off[w]))
        if (layer + 1) % 4 == 0:
            var o1 = Int(rebind[Scalar[i64]](off[w + 1]))
            var o2 = Int(rebind[Scalar[i64]](off[w + 2]))
            var o3 = Int(rebind[Scalar[i64]](off[w + 3]))
            var o4 = Int(rebind[Scalar[i64]](off[w + 4]))
            var o5 = Int(rebind[Scalar[i64]](off[w + 5]))
            var o6 = Int(rebind[Scalar[i64]](off[w + 6]))
            var kk = SIMD[i32, 8](0)
            comptime if KQ:
                kk[0] = kq_split(o1)
                kk[1] = kq_split(o2)
                kk[2] = kq_split(o3)
                kk[3] = kq_split(o6)
            var Kc = TileTensor(kc, cache_layout)
            var Vc = TileTensor(vc, cache_layout)
            if not attn_phases[MR, Q4, NAT, KQ=KQ](
                X_, wf[H](wbuf, o0), CurB_, Af,
                wq[QF, H, Q4](wbuf, o1), ws[QF, H, Q4](wbuf, o1),
                wq[KV, H, Q4](wbuf, o2), ws[KV, H, Q4](wbuf, o2),
                wq[KV, H, Q4](wbuf, o3), ws[KV, H, Q4](wbuf, o3),
                wf[HD](wbuf, o4), wf[HD](wbuf, o5),
                wq[H, ATT, Q4](wbuf, o6), ws[H, ATT, Q4](wbuf, o6),
                Qfm_, Kflat_, Vflat_, Q_, Gate_, Ao_, ResB_, Kc, Vc, tab, Ctr_, Pg_, p, M, att_i, Int(att_split), prof, 16 * layer, kk,
            ):
                return
            att_i += 1
            w += 7
        else:
            var o1 = Int(rebind[Scalar[i64]](off[w + 1]))
            var o2 = Int(rebind[Scalar[i64]](off[w + 2]))
            var o3 = Int(rebind[Scalar[i64]](off[w + 3]))
            var o4 = Int(rebind[Scalar[i64]](off[w + 4]))
            var o5 = Int(rebind[Scalar[i64]](off[w + 5]))
            var o6 = Int(rebind[Scalar[i64]](off[w + 6]))
            var o7 = Int(rebind[Scalar[i64]](off[w + 7]))
            var o8 = Int(rebind[Scalar[i64]](off[w + 8]))
            var o9 = Int(rebind[Scalar[i64]](off[w + 9]))
            var kk = SIMD[i32, 8](0)
            comptime if KQ:
                kk[0] = kq_split(o1)
                kk[1] = kq_split(o2)
                kk[2] = kq_split(o3)
                kk[3] = kq_split(o4)
                kk[4] = kq_split(o9)
            if not ssm_phases[MR, RELOAD, Q4, KQ=KQ](
                X_, wf[H](wbuf, o0), CurB_, Af,
                wq[CONV, H, Q4](wbuf, o1), ws[CONV, H, Q4](wbuf, o1),
                wq[INNER, H, Q4](wbuf, o2), ws[INNER, H, Q4](wbuf, o2),
                wq[NH_V, H, Q4](wbuf, o3), ws[NH_V, H, Q4](wbuf, o3),
                wq[NH_V, H, Q4](wbuf, o4), ws[NH_V, H, Q4](wbuf, o4),
                wf2[CONV, 4](wbuf, o5), wf[NH_V](wbuf, o6), wf[NH_V](wbuf, o7), wf[SSTATE](wbuf, o8),
                wq[H, INNER, Q4](wbuf, o9), ws[H, INNER, Q4](wbuf, o9),
                Qkvm_, Zm_, Araw_, Braw_, Eg_, Beta_, Conv_, So_, ResB_, ConvState_, SAll_, Ctr_,
                ring, Int32(ssm_i), slots, M, prof, 16 * layer, kk,
            ):
                return
            ssm_i += 1
            w += 10
        var ctr = Ctr_.ptr
        var gen = Ctr_.ptr.unsafe_offset(1)
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, 16 * layer + 7)
        if dump != 0 and block_idx.x == 0:
            var i = Int(thread_idx.x)
            while i < H:
                dbg[unsafe_offset=(2 * layer) * H + i] = rebind[Scalar[f32]](X_[0, i])
                i += ROW_THREADS
        var f0 = Int(rebind[Scalar[i64]](off[w]))
        var f1 = Int(rebind[Scalar[i64]](off[w + 1]))
        var f2 = Int(rebind[Scalar[i64]](off[w + 2]))
        var f3 = Int(rebind[Scalar[i64]](off[w + 3]))
        var kf = SIMD[i32, 8](0)
        comptime if KQ:
            kf[0] = kq_split(f1)
            kf[1] = kq_split(f2)
            kf[2] = kq_split(f3)
        if not ffn_phases[MR, Q4, KQ=KQ](
            X_, wf[H](wbuf, f0), CurB_, Af,
            wq[FFN, H, Q4](wbuf, f1), ws[FFN, H, Q4](wbuf, f1),
            wq[FFN, H, Q4](wbuf, f2), ws[FFN, H, Q4](wbuf, f2),
            wq[H, FFN, Q4](wbuf, f3), ws[H, FFN, Q4](wbuf, f3),
            Pg_, Pu_, FgB_, Ctr_, M, prof, 16 * layer + 7, sv + layer * H, sv_on, kf,
        ):
            return
        w += 4
        if not grid_barrier(ctr, gen, fail):
            return
        stamp(prof, 16 * layer + 11)
        if dump != 0 and block_idx.x == 0:
            var i = Int(thread_idx.x)
            while i < H:
                dbg[unsafe_offset=(2 * layer + 1) * H + i] = rebind[Scalar[f32]](X_[0, i])
                i += ROW_THREADS
    if fold_head == 0:
        return
    stamp(prof, 16 * NL)
    var ctrh = Ctr_.ptr
    var genh = Ctr_.ptr.unsafe_offset(1)
    var ho0 = Int(rebind[Scalar[i64]](off[w]))
    var ho1 = Int(rebind[Scalar[i64]](off[w + 1]))
    var kh = 0
    comptime if KQ:
        kh = Int(kq_split(ho1))
    if fold_head == 2:
        rms_f32_phase(X_, wf[H](wbuf, ho0), Hn_, M)
    comptime if LDSA:
        stage_rms(X_, wf[H](wbuf, ho0), Af)
    elif KQ:
        stage_rms_kq(X_, wf[H](wbuf, ho0), Af)
    else:
        rmsc_phase[MR](X_, wf[H](wbuf, ho0), CurB_, M)
        if not grid_barrier(ctrh, genh, fail):
            return
    stamp(prof, 16 * NL + 1)
    var Whq = wq[VOCAB, H, Q4](wbuf, ho1)
    var Whs = ws[VOCAB, H, Q4](wbuf, ho1)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nblk = Int(grid_dim.x)
    var bv = Array[Float32, MR](fill=Float32(-3.4e38))
    var bi = Array[Int32, MR](fill=Int32(0))
    var g = bid
    while g < VOCAB // ROW_WAVES:
        var row = g * ROW_WAVES + wave
        var paired = False
        comptime if KQ:
            if g + nblk < VOCAB // ROW_WAVES:
                var row1 = row + nblk * ROW_WAVES
                var t2 = kq_row2(Af.ptr.unsafe_bitcast[Scalar[bf16]](), Whq.ptr.unsafe_bitcast[Scalar[u8]](), kh, row, row1, lane, H)
                if t2[0] > bv[0] or (t2[0] == bv[0] and Int32(row) < bi[0]):
                    bv[0] = t2[0]
                    bi[0] = Int32(row)
                if t2[1] > bv[0] or (t2[1] == bv[0] and Int32(row1) < bi[0]):
                    bv[0] = t2[1]
                    bi[0] = Int32(row1)
                g += nblk
                paired = True
        if not paired:
            var t = row_dot_a[MR, Q4, LDSA, False, KQ=KQ, XLDS=True](CurB_, Af, Whq, Whs, row, lane, H, M, kh)
            comptime for r in range(MR):
                if r < M:
                    if t[r] > bv[r] or (t[r] == bv[r] and Int32(row) < bi[r]):
                        bv[r] = t[r]
                        bi[r] = Int32(row)
        g += nblk
    var wv = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[MR, ROW_WAVES]())
    var wi = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[MR, ROW_WAVES]())
    if lane == 0:
        comptime for r in range(MR):
            wv[r, wave] = rebind[wv.ElementType](bv[r])
            wi[r, wave] = rebind[wi.ElementType](bi[r])
    barrier()
    if tid == 0:
        comptime for r in range(MR):
            var pv = Float32(-3.4e38)
            var pi: Int32 = 0
            comptime for k in range(ROW_WAVES):
                var v = rebind[Scalar[f32]](wv[r, k])
                var ix = rebind[Scalar[i32]](wi[r, k])
                if v > pv or (v == pv and ix < pi):
                    pv = v
                    pi = ix
            hmax[unsafe_offset=r * nblk + bid] = pv
            hidx[unsafe_offset=r * nblk + bid] = pi
    if not grid_barrier(ctrh, genh, fail):
        return
    stamp(prof, 16 * NL + 2)
    if bid == 0 and tid == 0:
        comptime for r in range(MR):
            if r < M:
                var fv = Float32(-3.4e38)
                var fi: Int32 = 0
                for k in range(nblk):
                    var v = hmax[unsafe_offset=r * nblk + k]
                    var ix = hidx[unsafe_offset=r * nblk + k]
                    if v > fv or (v == fv and ix < fi):
                        fv = v
                        fi = ix
                if fold_head == 2:
                    Dtok_[r] = rebind[Dtok_.ElementType](fi)
                else:
                    Toks_[p + 1] = rebind[Toks_.ElementType](fi)
    stamp(prof, 16 * NL + 3)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_mega_token[
    MR: Int, RELOAD: Bool, Q4: Bool,
    XL: TensorLayout, CBL: TensorLayout, ZL: TensorLayout, RBL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout, GfL: TensorLayout,
    PfL: TensorLayout, FbL: TensorLayout, OffL: TensorLayout, CtrL: TensorLayout, TkL: TensorLayout, DkL: TensorLayout,
    NL: Int, NAT: Int, KQ: Bool = False,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: TileTensor[i64, OffL, MutAnyOrigin],
    X: TileTensor[f32, XL, MutAnyOrigin],
    CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    ResB: TileTensor[bf16, RBL, MutAnyOrigin],
    Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    Zm: TileTensor[f32, ZL, MutAnyOrigin],
    Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    So: TileTensor[f32, OmL, MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Q: TileTensor[f32, QmL, MutAnyOrigin],
    Gate: TileTensor[f32, GfL, MutAnyOrigin],
    Ao: TileTensor[f32, QmL, MutAnyOrigin],
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    Pg: TileTensor[f32, PfL, MutAnyOrigin],
    Pu: TileTensor[f32, PfL, MutAnyOrigin],
    FgB: TileTensor[bf16, FbL, MutAnyOrigin],
    Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    Toks: TileTensor[i32, TkL, MutAnyOrigin],
    Dtok: TileTensor[i32, DkL, MutAnyOrigin],
    Hn: TileTensor[f32, XL, MutAnyOrigin],
    hmax: MutPointer[Scalar[f32], MutAnyOrigin],
    hidx: MutPointer[Scalar[i32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, m: Int32, dump: Int32, fold_head: Int32, att_split: Int32,
    sv: MutPointer[Scalar[f32], MutAnyOrigin], sv_on: Int32,
):
    mega_body[MR, RELOAD, Q4, XL, CBL, ZL, RBL, QkvL, G32mL, ConvL, OmL, CsL, SsL, QfL, KvfL, QmL, GfL, PfL, FbL, OffL, CtrL, TkL, DkL, NL, NAT, KQ](wbuf, off, X, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, ConvState, SAll, Qfm, Kflat, Vflat, Q, Gate, Ao, kc, vc, tab, Pg, Pu, FgB, Ctr, prof, dbg, Toks, Dtok, Hn, hmax, hidx, ring, slots, pos, m, dump, fold_head, att_split, sv, sv_on)


def amar_mega_window[
    MR: Int, RELOAD: Bool, Q4: Bool,
    XL: TensorLayout, CBL: TensorLayout, ZL: TensorLayout, RBL: TensorLayout,
    QkvL: TensorLayout, G32mL: TensorLayout, ConvL: TensorLayout, OmL: TensorLayout,
    CsL: TensorLayout, SsL: TensorLayout,
    QfL: TensorLayout, KvfL: TensorLayout, QmL: TensorLayout, GfL: TensorLayout,
    PfL: TensorLayout, FbL: TensorLayout, OffL: TensorLayout, CtrL: TensorLayout, TkL: TensorLayout, DkL: TensorLayout,
    NL: Int, NAT: Int, KQ: Bool = False,
](
    wbuf: MutPointer[Scalar[u8], MutAnyOrigin],
    off: TileTensor[i64, OffL, MutAnyOrigin],
    X: TileTensor[f32, XL, MutAnyOrigin],
    CurB: TileTensor[bf16, CBL, MutAnyOrigin],
    ResB: TileTensor[bf16, RBL, MutAnyOrigin],
    Qkvm: TileTensor[f32, QkvL, MutAnyOrigin],
    Zm: TileTensor[f32, ZL, MutAnyOrigin],
    Araw: TileTensor[f32, G32mL, MutAnyOrigin],
    Braw: TileTensor[f32, G32mL, MutAnyOrigin],
    Eg: TileTensor[f32, G32mL, MutAnyOrigin],
    Beta: TileTensor[f32, G32mL, MutAnyOrigin],
    Conv: TileTensor[f32, ConvL, MutAnyOrigin],
    So: TileTensor[f32, OmL, MutAnyOrigin],
    ConvState: TileTensor[f32, CsL, MutAnyOrigin],
    SAll: TileTensor[f32, SsL, MutAnyOrigin],
    Qfm: TileTensor[f32, QfL, MutAnyOrigin],
    Kflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Vflat: TileTensor[f32, KvfL, MutAnyOrigin],
    Q: TileTensor[f32, QmL, MutAnyOrigin],
    Gate: TileTensor[f32, GfL, MutAnyOrigin],
    Ao: TileTensor[f32, QmL, MutAnyOrigin],
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    Pg: TileTensor[f32, PfL, MutAnyOrigin],
    Pu: TileTensor[f32, PfL, MutAnyOrigin],
    FgB: TileTensor[bf16, FbL, MutAnyOrigin],
    Ctr: TileTensor[u32, CtrL, MutAnyOrigin],
    prof: MutPointer[Scalar[i64], MutAnyOrigin],
    dbg: MutPointer[Scalar[f32], MutAnyOrigin],
    Toks: TileTensor[i32, TkL, MutAnyOrigin],
    Dtok: TileTensor[i32, DkL, MutAnyOrigin],
    Hn: TileTensor[f32, XL, MutAnyOrigin],
    hmax: MutPointer[Scalar[f32], MutAnyOrigin],
    hidx: MutPointer[Scalar[i32], MutAnyOrigin],
    ring: Int32, slots: Int32, pos: Int32, m: Int32, dump: Int32, fold_head: Int32, att_split: Int32,
    sv: MutPointer[Scalar[f32], MutAnyOrigin], sv_on: Int32,
):
    mega_body[MR, RELOAD, Q4, XL, CBL, ZL, RBL, QkvL, G32mL, ConvL, OmL, CsL, SsL, QfL, KvfL, QmL, GfL, PfL, FbL, OffL, CtrL, TkL, DkL, NL, NAT, KQ](wbuf, off, X, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, ConvState, SAll, Qfm, Kflat, Vflat, Q, Gate, Ao, kc, vc, tab, Pg, Pu, FgB, Ctr, prof, dbg, Toks, Dtok, Hn, hmax, hidx, ring, slots, pos, m, dump, fold_head, att_split, sv, sv_on)
