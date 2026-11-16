from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.math import fma
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from matmul import dtype
from matmul_skinny import bf16x16_to_f32, ROW_WAVES, ROW_THREADS
from arch import sdot4, sudot4
from std.sys import llvm_intrinsic

comptime QV = 16
comptime STR = 20
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16


@always_inline
def turbo_lds_dwords[MR: Int, UNROLL: Int]() -> Int:
    return MR * (WARP_SIZE * UNROLL) * STR


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_turbo_rows[
    N: Int, K: Int, MR: Int, UNROLL: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime assert N % ROW_WAVES == 0 and K % (32 * WARP_SIZE) == 0
    comptime NB = K // 32
    comptime KCB = WARP_SIZE * UNROLL
    comptime NCH = (NB + KCB - 1) // KCB
    comptime ITEMS = MR * KCB * 2
    var lds = stack_allocation[u32, address_space = AddressSpace.SHARED](row_major[turbo_lds_dwords[MR, UNROLL]()]())
    var bp = lds.ptr.unsafe_bitcast[Scalar[bf16]]()
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + tid // WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))
    for c in range(NCH):
        var cb = c * KCB
        barrier()
        var it = tid
        while it < ITEMS:
            var r = it // (2 * KCB)
            var rem = it % (2 * KCB)
            var b = rem // 2
            if r < M and cb + b < NB:
                bp.unsafe_offset((r * KCB + b) * STR * 2 + (rem % 2) * QV).store(rebind[SIMD[bf16, QV]](Av[r, (cb + b) * 2 + rem % 2]))
            it += ROW_THREADS
        barrier()
        var bytes_u = Array[SIMD[u8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var blk = cb + u * WARP_SIZE + lane
                bytes_u[u] = rebind[SIMD[u8, QV]](Qb[row, blk])
                ds[u] = rebind[Scalar[f16]](S[row, blk])
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var d = ds[u].cast[dtype]()
                var w32 = bitcast[u32, 4](bytes_u[u])
                var lo = bitcast[u8, QV](w32 & 0x0F0F0F0F).cast[dtype]()
                var hi = bitcast[u8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[dtype]()
                var dv = SIMD[dtype, QV](d)
                var m8d = SIMD[dtype, QV](d * -8)
                var wlo = fma(lo, dv, m8d)
                var whi = fma(hi, dv, m8d)
                var lb = u * WARP_SIZE + lane
                comptime for r in range(MR):
                    if r < M:
                        var o = (r * KCB + lb) * STR * 2
                        var a_lo = bf16x16_to_f32(bp.unsafe_offset(o).load[width=QV]())
                        var a_hi = bf16x16_to_f32(bp.unsafe_offset(o + QV).load[width=QV]())
                        acc[r] = fma(wlo, a_lo, fma(whi, a_hi, acc[r]))
    comptime for r in range(MR):
        if r < M:
            var t = warp.sum(acc[r].reduce_add())
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](t)


@always_inline
def dot2(a: SIMD[bf16, 2], b: SIMD[bf16, 2], c: Float32) -> Float32:
    return llvm_intrinsic["llvm.amdgcn.fdot2.f32.bf16", Float32](a, b, c, False)


@always_inline
def nib_pair(w: UInt32, s: UInt32) -> SIMD[bf16, 2]:
    # nibbles at bits s..s+3 and s+16..s+19 as bf16(128 + n): 0x4300 | n is exactly 128 + n
    return bitcast[bf16, 2](((w >> s) & 0x000F000F) | 0x43004300)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_turbo_rows_dot2[
    N: Int, K: Int, MR: Int, UNROLL: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
):
    # amar_turbo_rows with v_dot2_f32_bf16: weights become bf16(128 + n) straight from the nibbles (pairs j, j+2 of
    # one dword), A is staged in the matching pair order with its per-half-block sums, and per row and block
    # acc = fma(d, dot(128 + n, a) - 136 * sum(a), acc). About 2 VALU per weight at MR = 3 instead of ~8.
    # Not bit-exact with amar_turbo_rows (f32 sum order and the 136 * sum(a) fold).
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime assert N % ROW_WAVES == 0 and K % (32 * WARP_SIZE) == 0
    comptime NB = K // 32
    comptime KCB = WARP_SIZE * UNROLL
    comptime NCH = (NB + KCB - 1) // KCB
    comptime ITEMS = MR * KCB * 2
    var lds = stack_allocation[u32, address_space = AddressSpace.SHARED](row_major[turbo_lds_dwords[MR, UNROLL]()]())
    var asum = stack_allocation[DType.float32, address_space = AddressSpace.SHARED](row_major[ITEMS]())
    var bp = lds.ptr.unsafe_bitcast[Scalar[bf16]]()
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + tid // WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[Float32, MR](fill=0)
    for c in range(NCH):
        var cb = c * KCB
        barrier()
        var it = tid
        while it < ITEMS:
            var r = it // (2 * KCB)
            var rem = it % (2 * KCB)
            var b = rem // 2
            if r < M and cb + b < NB:
                var v = rebind[SIMD[bf16, QV]](Av[r, (cb + b) * 2 + rem % 2])
                bp.unsafe_offset((r * KCB + b) * STR * 2 + (rem % 2) * QV).store(v.shuffle[0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15]())
                asum[it] = v.cast[DType.float32]().reduce_add()
            it += ROW_THREADS
        barrier()
        var bytes_u = Array[SIMD[u8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var blk = cb + u * WARP_SIZE + lane
                bytes_u[u] = rebind[SIMD[u8, QV]](Qb[row, blk])
                ds[u] = rebind[Scalar[f16]](S[row, blk])
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var d = ds[u].cast[DType.float32]()
                var w = bitcast[u32, 4](bytes_u[u])
                var wl = Array[SIMD[bf16, 2], 8](uninitialized=True)
                var wh = Array[SIMD[bf16, 2], 8](uninitialized=True)
                comptime for i in range(4):
                    wl[2 * i] = nib_pair(w[i], 0)
                    wl[2 * i + 1] = nib_pair(w[i], 8)
                    wh[2 * i] = nib_pair(w[i], 4)
                    wh[2 * i + 1] = nib_pair(w[i], 12)
                var lb = u * WARP_SIZE + lane
                comptime for r in range(MR):
                    if r < M:
                        var o = (r * KCB + lb) * STR * 2
                        var al = bitcast[bf16, 16](bp.unsafe_offset(o).load[width=QV]())
                        var ah = bitcast[bf16, 16](bp.unsafe_offset(o + QV).load[width=QV]())
                        var tl = Float32(0)
                        var th = Float32(0)
                        comptime for k in range(8):
                            tl = dot2(wl[k], al.slice[2, offset=2 * k](), tl)
                            th = dot2(wh[k], ah.slice[2, offset=2 * k](), th)
                        var s = rebind[Float32](asum[(r * KCB + lb) * 2]) + rebind[Float32](asum[(r * KCB + lb) * 2 + 1])
                        acc[r] = fma(d, (tl + th) - 136 * s, acc[r])
    comptime for r in range(MR):
        if r < M:
            var t = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](t)


comptime QSTR = 12  # dwords per (row, block) in the int8 stage: 8 q words, f32 scale, i32 sum, 2 pad (48 B, b128 aligned)


@always_inline
def turbo_q8_lds_dwords[MR: Int, UNROLL: Int]() -> Int:
    return MR * (WARP_SIZE * UNROLL) * QSTR


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_turbo_rows_q8[
    N: Int, K: Int, MR: Int, UNROLL: Int, PRE: Bool,
    ALayout: TensorLayout, AQLayout: TensorLayout, ASLayout: TensorLayout,
    QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Aq: TileTensor[DType.int8, AQLayout, MutAnyOrigin],
    As: TileTensor[f16, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
):
    """Round A: A rows as int8 + f16 scale per 32-block (amar_quantize_q8_rows rounding), sudot4 against the Q4_0 nibbles.
    PRE=False quantizes the bf16 A rows while staging; PRE=True stages Aq/As written by amar_quantize_q8_rows.
    Both arms stage identical words, so their outputs are bit-identical."""
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime assert N % ROW_WAVES == 0 and K % (32 * WARP_SIZE) == 0
    comptime NB = K // 32
    comptime KCB = WARP_SIZE * UNROLL
    comptime NCH = (NB + KCB - 1) // KCB
    comptime ITEMS = MR * KCB
    comptime i32 = DType.int32
    comptime f32 = DType.float32
    var lds = stack_allocation[u32, address_space = AddressSpace.SHARED](row_major[turbo_q8_lds_dwords[MR, UNROLL]()]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + tid // WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var Aqv = Aq.vectorize[1, QV]()
    var ones = SIMD[i32, 8](0x01010101)
    var acc = Array[Scalar[f32], MR](fill=0)
    for c in range(NCH):
        var cb = c * KCB
        barrier()
        var it = tid
        while it < ITEMS:
            var r = it // KCB
            var b = it % KCB
            if r < M and cb + b < NB:
                var blk = cb + b
                var q = SIMD[DType.int8, 32]()
                var d: Scalar[f32]
                comptime if PRE:
                    q = rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2]).join(rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2 + 1]))
                    d = rebind[Scalar[f16]](As[r, blk]).cast[f32]()
                else:
                    var x = bf16x16_to_f32(rebind[SIMD[bf16, QV]](Av[r, blk * 2])).join(bf16x16_to_f32(rebind[SIMD[bf16, QV]](Av[r, blk * 2 + 1])))
                    var wmax = abs(x).reduce_max()
                    var scale = wmax / Float32(127) if wmax > 0 else Float32(1)
                    var v = x / SIMD[f32, 32](scale)
                    var t = (abs(v) + 0.5).cast[i32]()
                    var sg = bitcast[i32, 32](v) >> 31  # -1 for negative, 0 otherwise: round half away from zero
                    var qi = (t ^ sg) - sg
                    q = qi.clamp(-127, 127).cast[DType.int8]()
                    d = scale.cast[f16]().cast[f32]()
                var qw = bitcast[i32, 8](q)
                var s = Int32(0)
                comptime for i in range(8):
                    s = sdot4(ones[i], qw[i], s, False)
                var o = (r * KCB + b) * QSTR
                lds.ptr.unsafe_offset(o).store(bitcast[u32, 8](qw))
                lds.ptr.unsafe_offset(o + 8).store(SIMD[u32, 2](bitcast[u32](d), bitcast[u32](s)))
            it += ROW_THREADS
        barrier()
        var bytes_u = Array[SIMD[u8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var blk = cb + u * WARP_SIZE + lane
                bytes_u[u] = rebind[SIMD[u8, QV]](Qb[row, blk])
                ds[u] = rebind[Scalar[f16]](S[row, blk])
        comptime for u in range(UNROLL):
            if cb + u * WARP_SIZE < NB:
                var dw = ds[u].cast[f32]()
                var w32 = bitcast[i32, 4](bytes_u[u])
                var lo = w32 & 0x0F0F0F0F
                var hi = w32 & Int32(-252645136)  # 0xF0F0F0F0: high nibbles in place, the sum is a multiple of 16
                var lb = u * WARP_SIZE + lane
                comptime for r in range(MR):
                    if r < M:
                        var o = (r * KCB + lb) * QSTR
                        var aw = bitcast[i32, 8](lds.ptr.unsafe_offset(o).load[width=8]())
                        var ex = lds.ptr.unsafe_offset(o + 8).load[width=2]()
                        var dlo = Int32(0)
                        var dhi = Int32(0)
                        comptime for i in range(4):
                            dlo = sudot4(False, lo[i], True, aw[i], dlo, False)
                            dhi = sudot4(False, hi[i], True, aw[4 + i], dhi, False)
                        var tot = dlo + (dhi >> 4) - 8 * bitcast[i32](ex[1])
                        acc[r] = fma(Float32(tot), dw * bitcast[f32](ex[0]), acc[r])
    comptime for r in range(MR):
        if r < M:
            var t = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](t)
