
from arch import sdot4
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.memory import bitcast
from std.sys.intrinsics import llvm_intrinsic
from std.math import fma
from bexp import exp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype
from std.utils import StaticTuple

comptime SM = 8
comptime SBN = 256
comptime SPLITK = 32
comptime SK_THREADS = SBN
comptime KCHUNK = 128
comptime ROW_WAVES = 8
comptime ROW_THREADS = ROW_WAVES * WARP_SIZE
comptime ROW_VEC = 8


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny[
    in_dtype: DType,
    ALayout: TensorLayout, BLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    B: TileTensor[in_dtype, BLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var col = block_idx.x * SBN + tid

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    comptime assert sav.flat_rank == 2

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            for k in range(clen):
                var b_val = rebind[Scalar[in_dtype]](B[kk + k, col]).cast[
                    dtype
                ]()
                var a_vec = rebind[SIMD[dtype, SM]](sav[k, 0])
                acc += a_vec * b_val
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_v2[
    in_dtype: DType, CPT: Int,
    ALayout: TensorLayout, BLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    B: TileTensor[in_dtype, BLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var c0 = (block_idx.x * SK_THREADS + tid) * CPT

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    var Bv = B.vectorize[1, CPT]()

    var acc = SIMD[dtype, SM * CPT](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if c0 < N:
            for k in range(clen):
                var b_vec = rebind[SIMD[in_dtype, CPT]](
                    Bv[kk + k, c0 // CPT]
                ).cast[dtype]()
                var a_vec = rebind[SIMD[dtype, SM]](sav[k, 0])
                comptime for i in range(SM):
                    comptime for j in range(CPT):
                        acc[i * CPT + j] += a_vec[i] * b_vec[j]
        barrier()
        kk += KCHUNK

    if c0 < N:
        comptime for i in range(SM):
            comptime for j in range(CPT):
                Cp[block_idx.y, i, c0 + j] = rebind[Cp.ElementType](
                    acc[i * CPT + j]
                )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_m1[
    in_dtype: DType, CPT: Int,
    ALayout: TensorLayout, BLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    B: TileTensor[in_dtype, BLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and Cp.flat_rank == 3

    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var c0 = (block_idx.x * SK_THREADS + tid) * CPT

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK]()
    )
    var Bv = B.vectorize[1, CPT]()

    var acc = SIMD[dtype, CPT](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        if tid < KCHUNK:
            sa[tid] = rebind[sa.ElementType](
                rebind[Scalar[in_dtype]](A[0, kk + tid]).cast[
                    dtype
                ]() if tid < clen else 0
            )
        barrier()

        if c0 < N:
            for k in range(clen):
                var b_vec = rebind[SIMD[in_dtype, CPT]](
                    Bv[kk + k, c0 // CPT]
                ).cast[dtype]()
                var a_val = rebind[Scalar[dtype]](sa[k])
                acc += b_vec * a_val
        barrier()
        kk += KCHUNK

    if c0 < N:
        comptime for j in range(CPT):
            Cp[block_idx.y, 0, c0 + j] = rebind[Cp.ElementType](acc[j])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_m1_row[
    in_dtype: DType, UNROLL: Int,
    ALayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    W: TileTensor[in_dtype, WLayout, MutAnyOrigin],
    O: TileTensor[dtype, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and W.flat_rank == 2 and O.flat_rank == 1

    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Wv = W.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = SIMD[dtype, ROW_VEC](0)

    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = Array[SIMD[in_dtype, ROW_VEC], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])

        comptime for u in range(UNROLL):
            var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, (kk + u * STEP) // ROW_VEC + lane]).cast[dtype]()
            acc += ws[u].cast[dtype]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[dtype]()
        var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, kk // ROW_VEC + lane]).cast[dtype]()
        acc += w * a
        kk += STEP

    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        O[row] = rebind[O.ElementType](total)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_m1_row2[
    in_dtype: DType, UNROLL: Int,
    ALayout: TensorLayout, WLayout: TensorLayout, OLayout: TensorLayout
](
    A: TileTensor[in_dtype, ALayout, MutAnyOrigin],
    W1: TileTensor[in_dtype, WLayout, MutAnyOrigin],
    W2: TileTensor[in_dtype, WLayout, MutAnyOrigin],
    O1: TileTensor[dtype, OLayout, MutAnyOrigin],
    O2: TileTensor[dtype, OLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and W1.flat_rank == 2 and O1.flat_rank == 1

    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= 2 * N:
        return
    var second = row >= N
    if second:
        row -= N

    var Wv = W2.vectorize[1, ROW_VEC]() if second else W1.vectorize[1, ROW_VEC]()
    var Av = A.vectorize[1, ROW_VEC]()
    comptime STEP = WARP_SIZE * ROW_VEC
    var acc = SIMD[dtype, ROW_VEC](0)

    var kk = 0
    while kk + UNROLL * STEP <= K:
        var ws = Array[SIMD[in_dtype, ROW_VEC], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            ws[u] = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, (kk + u * STEP) // ROW_VEC + lane])

        comptime for u in range(UNROLL):
            var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, (kk + u * STEP) // ROW_VEC + lane]).cast[dtype]()
            acc += ws[u].cast[dtype]() * a
        kk += UNROLL * STEP
    while kk < K:
        var w = rebind[SIMD[in_dtype, ROW_VEC]](Wv[row, kk // ROW_VEC + lane]).cast[dtype]()
        var a = rebind[SIMD[in_dtype, ROW_VEC]](Av[0, kk // ROW_VEC + lane]).cast[dtype]()
        acc += w * a
        kk += STEP

    var total = warp.sum(acc.reduce_add())
    if lane == 0:
        if second:
            O2[row] = rebind[O2.ElementType](total)
        else:
            O1[row] = rebind[O1.ElementType](total)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q8row[
    UNROLL: Int, MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    comptime QV = 16 if MR <= 5 else 8
    comptime STEP = WARP_SIZE * QV
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))

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
            var w = qs[u].cast[dtype]() * ds[u].cast[dtype]()
            comptime for r in range(MR):
                if r < M:
                    var a = rebind[SIMD[DType.bfloat16, QV]](Av[r, kb // QV + lane]).cast[dtype]()
                    acc[r] += w * a
        kk += UNROLL * STEP
    while kk < K:
        var q = rebind[SIMD[DType.int8, QV]](Qv[row, kk // QV + lane]).cast[dtype]()
        var d = rebind[Scalar[DType.float16]](S[row, (kk + lane * QV) // 32]).cast[dtype]()
        var w = q * d
        comptime for r in range(MR):
            if r < M:
                var a = rebind[SIMD[DType.bfloat16, QV]](Av[r, kk // QV + lane]).cast[dtype]()
                acc[r] += w * a
        kk += STEP

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r].reduce_add())
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q4row[
    UNROLL: Int, MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    # Q is [N, K/2] bytes, ggml Q4_0 nibble order per 32-element block: byte j
    # (j in 0..15) packs element j in its low nibble, element j+16 in its
    # high nibble; both decode as nibble - 8 scaled by the block's fp16 d.
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    comptime QV = 16
    comptime STEP = WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))
    var eight = SIMD[dtype, QV](8)

    var nb = K // 32
    var kk = 0
    while kk + UNROLL * STEP <= nb:
        var bytes_u = Array[SIMD[DType.uint8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[DType.float16], UNROLL](uninitialized=True)

        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            bytes_u[u] = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
            ds[u] = rebind[Scalar[DType.float16]](S[row, blk])

        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            var d = ds[u].cast[dtype]()
            var wlo = ((bytes_u[u] & UInt8(0xF)).cast[dtype]() - eight) * d
            var whi = ((bytes_u[u] >> UInt8(4)).cast[dtype]() - eight) * d
            comptime for r in range(MR):
                if r < M:
                    var a_lo = rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2]).cast[dtype]()
                    var a_hi = rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2 + 1]).cast[dtype]()
                    acc[r] = fma(wlo, a_lo, fma(whi, a_hi, acc[r]))
        kk += UNROLL * STEP
    while kk < nb:
        var blk = kk + lane
        var bytes1 = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
        var d = rebind[Scalar[DType.float16]](S[row, blk]).cast[dtype]()
        var wlo = ((bytes1 & UInt8(0xF)).cast[dtype]() - eight) * d
        var whi = ((bytes1 >> UInt8(4)).cast[dtype]() - eight) * d
        comptime for r in range(MR):
            if r < M:
                var a_lo = rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2]).cast[dtype]()
                var a_hi = rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2 + 1]).cast[dtype]()
                acc[r] = fma(wlo, a_lo, fma(whi, a_hi, acc[r]))
        kk += STEP

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r].reduce_add())
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


@always_inline
def bf16x16_to_f32(a: SIMD[DType.bfloat16, 16]) -> SIMD[dtype, 16]:
    var au = bitcast[DType.uint32, 8](a)
    var lo = bitcast[dtype, 8](au << 16)
    var hi = bitcast[dtype, 8](au & 0xFFFF0000)
    return lo.interleave(hi)


@always_inline
def q4_dot_blocks[
    MR: Int, UNROLL: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, kb0: Int, kb1: Int, M: Int,
) -> Array[Float32, MR]:
    # One Q4_0 row-dot body over the 32-element blocks [kb0, kb1), shared by
    # amar_matmul_skinny_q4rowb and the megakernel so both paths contract
    # identically: nibbles masked as u32 (one v_cvt_f32_ubyteN per byte), -8
    # folded into the scale as fma(nib, d, -8d) (one rounding of the same real
    # number as (nib-8)*d), bf16 A unpacked by shift/mask, and the explicit
    # fma chain acc = fma(wlo, a_lo, fma(whi, a_hi, acc)).
    comptime QV = 16
    comptime STEP = WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))
    var kk = kb0
    while kk + UNROLL * STEP <= kb1:
        var bytes_u = Array[SIMD[DType.uint8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[DType.float16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            bytes_u[u] = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
            ds[u] = rebind[Scalar[DType.float16]](S[row, blk])
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            var d = ds[u].cast[dtype]()
            var w32 = bitcast[DType.uint32, 4](bytes_u[u])
            var lo = bitcast[DType.uint8, QV](w32 & 0x0F0F0F0F).cast[dtype]()
            var hi = bitcast[DType.uint8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[dtype]()
            var dv = SIMD[dtype, QV](d)
            var m8d = SIMD[dtype, QV](d * -8)
            var wlo = fma(lo, dv, m8d)
            var whi = fma(hi, dv, m8d)
            comptime for r in range(MR):
                if r < M:
                    var a_lo = bf16x16_to_f32(rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2]))
                    var a_hi = bf16x16_to_f32(rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2 + 1]))
                    acc[r] = fma(wlo, a_lo, fma(whi, a_hi, acc[r]))
        kk += UNROLL * STEP
    while kk < kb1:
        var blk = kk + lane
        var bytes1 = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
        var d = rebind[Scalar[DType.float16]](S[row, blk]).cast[dtype]()
        var w32 = bitcast[DType.uint32, 4](bytes1)
        var lo = bitcast[DType.uint8, QV](w32 & 0x0F0F0F0F).cast[dtype]()
        var hi = bitcast[DType.uint8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[dtype]()
        var dv = SIMD[dtype, QV](d)
        var m8d = SIMD[dtype, QV](d * -8)
        var wlo = fma(lo, dv, m8d)
        var whi = fma(hi, dv, m8d)
        comptime for r in range(MR):
            if r < M:
                var a_lo = bf16x16_to_f32(rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2]))
                var a_hi = bf16x16_to_f32(rebind[SIMD[DType.bfloat16, QV]](Av[r, blk * 2 + 1]))
                acc[r] = fma(wlo, a_lo, fma(whi, a_hi, acc[r]))
        kk += STEP
    var out = Array[Float32, MR](fill=0)
    comptime for r in range(MR):
        if r < M:
            out[r] = warp.sum(acc[r].reduce_add())
    return out^


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q4rowb[
    UNROLL: Int, MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout, KSPLIT: Int = 1
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    # Same contract as amar_matmul_skinny_q4row (bench/q4-alu-protocol.md A1),
    # body in q4_dot_blocks. KSPLIT > 1 (bench/q4-splitk-protocol.md): work
    # item = (row, part), K split into KSPLIT ranges of 32-blocks, partial
    # written to Cp[part, r, row]; amar_skinny_reduce_add[.., KSPLIT] sums
    # them as (0 + p0) + p1 + ... before the residual add.
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var item = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if item >= N * KSPLIT:
        return
    var row = item // KSPLIT
    var part = item % KSPLIT
    var nb = K // 32
    var t = q4_dot_blocks[MR, UNROLL](A, Q, S, row, lane, part * (nb // KSPLIT), (part + 1) * (nb // KSPLIT), M)
    comptime for r in range(MR):
        if r < M:
            if lane == 0:
                Cp[part, r, row] = rebind[Cp.ElementType](t[r])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q8dot[
    UNROLL: Int, MR: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout,
    QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    Aq: TileTensor[DType.int8, AQLayout, MutAnyOrigin],
    As: TileTensor[DType.float16, ASLayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert Aq.flat_rank == 2 and As.flat_rank == 2
    comptime assert Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    comptime QV = 16
    comptime STEP = WARP_SIZE * QV
    comptime DOTN = QV // 4
    var Qv = Q.vectorize[1, QV]()
    var Aqv = Aq.vectorize[1, QV]()
    var acc = Array[Scalar[dtype], MR](fill=Scalar[dtype](0))

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
            var ws = ds[u].cast[dtype]()
            var wv4 = bitcast[DType.int32, DOTN](qs[u])
            comptime for r in range(MR):
                if r < M:
                    var aq = rebind[SIMD[DType.int8, QV]](Aqv[r, kb // QV + lane])
                    var as_r = rebind[Scalar[DType.float16]](As[r, (kb + lane * QV) // 32]).cast[dtype]()
                    var av4 = bitcast[DType.int32, DOTN](aq)
                    var dot = Int32(0)
                    comptime for d in range(DOTN):
                        dot = sdot4(wv4[d], av4[d], dot, False)
                    acc[r] += Float32(dot) * ws * as_r
        kk += UNROLL * STEP
    while kk < K:
        var q = rebind[SIMD[DType.int8, QV]](Qv[row, kk // QV + lane])
        var ws = rebind[Scalar[DType.float16]](S[row, (kk + lane * QV) // 32]).cast[dtype]()
        var wv4 = bitcast[DType.int32, DOTN](q)
        comptime for r in range(MR):
            if r < M:
                var aq = rebind[SIMD[DType.int8, QV]](Aqv[r, kk // QV + lane])
                var as_r = rebind[Scalar[DType.float16]](As[r, (kk + lane * QV) // 32]).cast[dtype]()
                var av4 = bitcast[DType.int32, DOTN](aq)
                var dot = Int32(0)
                comptime for d in range(DOTN):
                    dot = sdot4(wv4[d], av4[d], dot, False)
                acc[r] += Float32(dot) * ws * as_r
        kk += STEP

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


comptime Q8_BLOCK = 32


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q8b[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Bq: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    Bs: TileTensor[dtype, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Bq.flat_rank == 2
    comptime assert Bs.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var col = block_idx.x * SBN + tid

    var kslice = (K + SPLITK - 1) // SPLITK
    var k0 = block_idx.y * kslice
    var k1 = min(k0 + kslice, K)

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[KCHUNK, SM]()
    )
    var sav = sa.vectorize[1, SM]()
    comptime assert sav.flat_rank == 2

    var acc = SIMD[dtype, SM](0)

    var kk = k0
    while kk < k1:
        var clen = min(KCHUNK, k1 - kk)

        comptime for i in range(SM * KCHUNK // SK_THREADS):
            var e = tid + i * SK_THREADS
            var r = e // KCHUNK
            var kc = e % KCHUNK
            var g = kk + kc
            sa[kc, r] = rebind[sa.ElementType](
                rebind[Scalar[DType.bfloat16]](A[r, g]).cast[
                    dtype
                ]() if r < M and kc < clen else 0
            )
        barrier()

        if col < N:
            for blk in range(clen // Q8_BLOCK):
                var kb = kk + blk * Q8_BLOCK
                var scale = rebind[Scalar[dtype]](Bs[kb // Q8_BLOCK, col])
                comptime for j in range(Q8_BLOCK):
                    var q = rebind[Scalar[DType.int8]](Bq[kb + j, col])
                    var a_vec = rebind[SIMD[dtype, SM]](
                        sav[blk * Q8_BLOCK + j, 0]
                    )
                    acc += a_vec * (q.cast[dtype]() * scale)
        barrier()
        kk += KCHUNK

    if col < N:
        comptime for i in range(SM):
            Cp[block_idx.y, i, col] = rebind[Cp.ElementType](acc[i])


def amar_skinny_reduce_swiglu[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Gp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Up: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Gp.flat_rank == 3 and Up.flat_rank == 3 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var g: Scalar[dtype] = 0
    var u: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var silu = g / (1 + exp(-g))
    C[r, c] = rebind[C.ElementType](silu * u)


def amar_skinny_reduce_swiglu_bf16[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Gp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Up: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[DType.bfloat16, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Gp.flat_rank == 3 and Up.flat_rank == 3 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var g: Scalar[dtype] = 0
    var u: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        g += rebind[Scalar[dtype]](Gp[s, r, c])
        u += rebind[Scalar[dtype]](Up[s, r, c])
    var silu = g / (1 + exp(-g))
    C[r, c] = rebind[C.ElementType]((silu * u).cast[DType.bfloat16]())


def amar_skinny_reduce_add[
    PLayout: TensorLayout, YLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    Y: TileTensor[dtype, YLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Cp.flat_rank == 3 and Y.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var acc: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        acc += rebind[Scalar[dtype]](Cp[s, r, c])
    Y[r, c] = rebind[Y.ElementType](rebind[Scalar[dtype]](Y[r, c]) + acc)


def amar_skinny_reduce[
    PLayout: TensorLayout, CLayout: TensorLayout, NSPLIT: Int = SPLITK
](
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Cp.flat_rank == 3 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var gid = global_idx.x
    if gid >= M * N:
        return
    var r = gid // N
    var c = gid % N

    var acc: Scalar[dtype] = 0
    comptime for s in range(NSPLIT):
        acc += rebind[Scalar[dtype]](Cp[s, r, c])
    C[r, c] = rebind[C.ElementType](acc)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def amar_matmul_skinny_q4dot[
    UNROLL: Int, MR: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout,
    QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    Aq: TileTensor[DType.int8, AQLayout, MutAnyOrigin],
    As: TileTensor[DType.float16, ASLayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert Aq.flat_rank == 2 and As.flat_rank == 2
    comptime assert Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    var M = Int(m)
    var N = Int(n)
    var nb = Int(k_dim) // 32
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return
    comptime QV = 16
    comptime STEP = WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Aqv = Aq.vectorize[1, QV]()
    var acc = Array[Scalar[dtype], MR](fill=Scalar[dtype](0))
    var ones = Int32(0x01010101)
    var kk = 0
    while kk + UNROLL * STEP <= nb:
        var qs = Array[SIMD[DType.uint8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[DType.float16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            qs[u] = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
            ds[u] = rebind[Scalar[DType.float16]](S[row, blk])
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            var d = ds[u].cast[dtype]()
            var w32 = bitcast[DType.int32, 4](qs[u])
            var lo4 = w32 & Int32(0x0F0F0F0F)
            var hi4 = (w32 >> 4) & Int32(0x0F0F0F0F)
            comptime for r in range(MR):
                if r < M:
                    var alo = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2]))
                    var ahi = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2 + 1]))
                    var d8 = rebind[Scalar[DType.float16]](As[r, blk]).cast[dtype]()
                    var dot = Int32(0)
                    var sa = Int32(0)
                    comptime for i in range(4):
                        dot = sdot4(lo4[i], alo[i], dot, False)
                        dot = sdot4(hi4[i], ahi[i], dot, False)
                        sa = sdot4(ones, alo[i], sa, False)
                        sa = sdot4(ones, ahi[i], sa, False)
                    acc[r] += Float32(dot - 8 * sa) * d * d8
        kk += UNROLL * STEP
    while kk < nb:
        var blk = kk + lane
        if blk < nb:
            var q = rebind[SIMD[DType.uint8, QV]](Qb[row, blk])
            var d = rebind[Scalar[DType.float16]](S[row, blk]).cast[dtype]()
            var w32 = bitcast[DType.int32, 4](q)
            var lo4 = w32 & Int32(0x0F0F0F0F)
            var hi4 = (w32 >> 4) & Int32(0x0F0F0F0F)
            comptime for r in range(MR):
                if r < M:
                    var alo = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2]))
                    var ahi = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, QV]](Aqv[r, blk * 2 + 1]))
                    var d8 = rebind[Scalar[DType.float16]](As[r, blk]).cast[dtype]()
                    var dot = Int32(0)
                    var sa = Int32(0)
                    comptime for i in range(4):
                        dot = sdot4(lo4[i], alo[i], dot, False)
                        dot = sdot4(hi4[i], ahi[i], dot, False)
                        sa = sdot4(ones, alo[i], sa, False)
                        sa = sdot4(ones, ahi[i], sa, False)
                    acc[r] += Float32(dot - 8 * sa) * d * d8
        kk += STEP
    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)
