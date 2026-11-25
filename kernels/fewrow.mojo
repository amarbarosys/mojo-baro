from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.math import fma
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from std.sys import llvm_intrinsic
from arch import mma, acc_row, GFX12_ROWS

comptime FW_WAVES = 8
comptime FW_THREADS = FW_WAVES * WARP_SIZE
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16


@always_inline
def nib_bf16[MAGIC: Bool = False](v: SIMD[u32, 4]) -> SIMD[bf16, 16]:
    comptime if MAGIC:
        # Exact 16+q: bf16 ULP at 16 is 1/8. Adjust the offset correction to 24.
        return bitcast[bf16, 16]((bitcast[u8, 16](v).cast[DType.uint16]() << 3) | 0x4180)
    var f = bitcast[u8, 16](v).cast[f32]()
    return bitcast[bf16, 16]((bitcast[u32, 16](f) >> 16).cast[DType.uint16]())


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_wmma[
    N: Int, K: Int, D: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
    MODE: Int = 0,
    WAVES: Int = FW_WAVES,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 16 == 0 and NB % (WAVES * D) == 0
    comptime NBW = NB // WAVES
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[WAVES * 256]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * 16 + h
    var Qb = Q.vectorize[1, 16]()
    var Av = A.vectorize[1, 16]()
    var live = h < M
    var tot = SIMD[f32, 8](0)
    var ones = SIMD[bf16, 16](1)
    var b0 = wave * NBW
    var qb = Array[SIMD[u8, 16], D](uninitialized=True)
    var sd = Array[Scalar[f16], D](uninitialized=True)
    var alo = Array[SIMD[bf16, 16], D](uninitialized=True)
    var ahi = Array[SIMD[bf16, 16], D](uninitialized=True)
    for c in range(0, NBW, D):
        if MODE != 4 or c == 0:
            comptime for u in range(D):
                var blk = b0 + c + u
                comptime if MODE >= 2:
                    qb[u] = rebind[SIMD[u8, 16]](Qb[blk, row])
                    sd[u] = rebind[Scalar[f16]](S[blk, row])
                else:
                    qb[u] = rebind[SIMD[u8, 16]](Qb[row, blk])
                    sd[u] = rebind[Scalar[f16]](S[row, blk])
                if live:
                    alo[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2])
                    ahi[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2 + 1])
                else:
                    alo[u] = SIMD[bf16, 16](0)
                    ahi[u] = SIMD[bf16, 16](0)
        comptime for u in range(D):
            var w32 = bitcast[u32, 4](qb[u])
            var lo = nib_bf16[MODE == 3](w32 & 0x0F0F0F0F)
            var hi = nib_bf16[MODE == 3]((w32 >> 4) & 0x0F0F0F0F)
            comptime if MODE == 1:
                lo -= SIMD[bf16, 16](8)
                hi -= SIMD[bf16, 16](8)
            var t = SIMD[f32, 8](0)
            mma(t, alo[u], lo, SIMD[f32, 8](0))
            var t2 = SIMD[f32, 8](0)
            mma(t2, ahi[u], hi, t)
            var dl = SIMD[f32, 8](sd[u].cast[f32]())
            comptime if MODE == 1:
                tot = fma(dl, t2, tot)
            else:
                var c1 = SIMD[f32, 8](0)
                mma(c1, alo[u], ones, SIMD[f32, 8](0))
                var c2 = SIMD[f32, 8](0)
                mma(c2, ahi[u], ones, c1)
                tot = fma(dl, fma(c2, SIMD[f32, 8](-24.0 if MODE == 3 else -8.0), t2), tot)
            comptime if MODE == 4:
                if c + D < NBW:
                    var blk = b0 + c + D + u
                    qb[u] = rebind[SIMD[u8, 16]](Qb[blk, row])
                    sd[u] = rebind[Scalar[f16]](S[blk, row])
                    if live:
                        alo[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2])
                        ahi[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2 + 1])
                    else:
                        alo[u] = SIMD[bf16, 16](0)
                        ahi[u] = SIMD[bf16, 16](0)
    comptime for i in range(8):
        red[wave * 256 + acc_row(0, i, half) * 16 + h] = tot[i]
    barrier()
    for output in range(tid, 256, WAVES * WARP_SIZE):
        var col = output // 16
        var r = output % 16
        if col < M:
            var s = Float32(0)
            comptime for w in range(WAVES):
                s += rebind[Float32](red[w * 256 + col * 16 + r])
            Cp[0, col, Int(block_idx.x) * 16 + r] = rebind[Cp.ElementType](s)


def amar_q8_cols[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Aq: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    Ast: TileTensor[f32, SLayout, MutAnyOrigin],
    m: Int32,
):
    # int8 activation rows for amar_fewrow_iu8: Aq[16, K] (rows >= m are zero), scales transposed Ast[K/32, 16].
    comptime assert A.flat_rank == 2 and Aq.flat_rank == 2 and Ast.flat_rank == 2
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var k = blk * 32 + lane
    var x = rebind[Scalar[bf16]](A[row, k]).cast[f32]() if row < Int(m) else Float32(0)
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(127) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    q = max(min(q, 127), -127)
    Aq[row, k] = rebind[Aq.ElementType](Int8(q))
    if lane == 0:
        Ast[blk, row] = rebind[Ast.ElementType](scale if row < Int(m) else Float32(0))


@always_inline
def s8_nib(v: SIMD[u32, 4]) -> SIMD[DType.int32, 4]:
    # packed nibbles n (0..15) -> signed bytes n - 8: (n + 0x78) ^ 0x80, no carry between bytes
    return bitcast[DType.int32, 4]((v + 0x78787878) ^ 0x80808080)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu8[
    N: Int, K: Int, D: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    Aq: TileTensor[DType.int8, AQLayout, MutAnyOrigin],
    Ast: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    # Geometry of amar_fewrow_wmma (16 output rows per block, 8 waves split K). The q4 nibbles enter
    # v_wmma_i32_16x16x16_iu8 as signed bytes against int8 activations: two WMMAs and ~7 VALU per weight
    # dword, no bf16 widening and no activation-sum WMMAs.
    comptime assert Aq.flat_rank == 2 and Ast.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 16 == 0 and NB % (FW_WAVES * D) == 0
    comptime NBW = NB // FW_WAVES
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 256]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * 16 + h
    var Qb = Q.vectorize[1, 16]()
    var Av = Aq.vectorize[1, 16]()
    var Sv = Ast.vectorize[1, 16]()
    var tot = SIMD[f32, 8](0)
    var b0 = wave * NBW
    for c in range(0, NBW, D):
        var qb = Array[SIMD[u32, 4], D](uninitialized=True)
        var sd = Array[Scalar[f16], D](uninitialized=True)
        var alo = Array[SIMD[DType.int32, 4], D](uninitialized=True)
        var ahi = Array[SIMD[DType.int32, 4], D](uninitialized=True)
        var asc = Array[SIMD[f32, 16], D](uninitialized=True)
        comptime for u in range(D):
            var blk = b0 + c + u
            qb[u] = bitcast[u32, 4](rebind[SIMD[u8, 16]](Qb[row, blk]))
            sd[u] = rebind[Scalar[f16]](S[row, blk])
            alo[u] = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, 16]](Av[h, blk * 2]))
            ahi[u] = bitcast[DType.int32, 4](rebind[SIMD[DType.int8, 16]](Av[h, blk * 2 + 1]))
            asc[u] = rebind[SIMD[f32, 16]](Sv[blk, 0])
        comptime for u in range(D):
            var lo = s8_nib(qb[u] & 0x0F0F0F0F)
            var hi = s8_nib((qb[u] >> 4) & 0x0F0F0F0F)
            var acc = llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu8", SIMD[DType.int32, 8]](
                True, alo[u], True, lo, SIMD[DType.int32, 8](0), False)
            acc = llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu8", SIMD[DType.int32, 8]](
                True, ahi[u], True, hi, acc, False)
            var sc = SIMD[f32, 8](0)
            comptime for i in range(8):
                sc[i] = asc[u][acc_row(0, i, 1)] if half == 1 else asc[u][acc_row(0, i, 0)]
            tot = fma(acc.cast[f32](), sc * sd[u].cast[f32](), tot)
    comptime for i in range(8):
        red[wave * 256 + acc_row(0, i, half) * 16 + h] = tot[i]
    barrier()
    if tid < 256:
        var col = tid // 16
        var r = tid % 16
        if col < M:
            var s = Float32(0)
            comptime for w in range(FW_WAVES):
                s += rebind[Float32](red[w * 256 + col * 16 + r])
            Cp[0, col, Int(block_idx.x) * 16 + r] = rebind[Cp.ElementType](s)


def amar_fw_prep[
    ALayout: TensorLayout, PLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Ap: TileTensor[bf16, PLayout, MutAnyOrigin],
    Asum: TileTensor[f32, SLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Ap.flat_rank == 2 and Asum.flat_rank == 2
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var e = lane % 16
    var src = (lane // 16) * 16 + 4 * ((e % 8) // 2) + 2 * (e % 2) + e // 8
    var x = rebind[Scalar[bf16]](A[row, blk * 32 + src]) if row < Int(m) else Scalar[bf16](0)
    Ap[row, blk * 32 + lane] = rebind[Ap.ElementType](x)
    var s = warp.sum(x.cast[f32]())
    if lane == 0 and row < 8:
        Asum[blk, (row % 2) * 4 + row // 2] = rebind[Asum.ElementType](s)


@always_inline
def nib_frag[SH: Int](w: SIMD[u32, 4]) -> SIMD[bf16, 16]:
    var ev = ((w >> SIMD[u32, 4](SH)) & 0x000F000F) | 0x43004300
    var od = ((w >> SIMD[u32, 4](SH + 8)) & 0x000F000F) | 0x43004300
    return bitcast[bf16, 16](ev.join(od))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_nib[
    N: Int, K: Int, D: Int,
    ALayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    Ap: TileTensor[bf16, ALayout, MutAnyOrigin],
    Asum: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Ap.flat_rank == 2 and Asum.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 16 == 0 and NB % (FW_WAVES * D) == 0 and not GFX12_ROWS
    comptime NBW = NB // FW_WAVES
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 128]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * 16 + h
    var Qb = Q.vectorize[1, 16]()
    var Av = Ap.vectorize[1, 16]()
    var Sv = Asum.vectorize[1, 4]()
    var tot = SIMD[f32, 4](0)
    var b0 = wave * NBW
    for c in range(0, NBW, D):
        var qb = Array[SIMD[u32, 4], D](uninitialized=True)
        var sd = Array[Scalar[f16], D](uninitialized=True)
        var alo = Array[SIMD[bf16, 16], D](uninitialized=True)
        var ahi = Array[SIMD[bf16, 16], D](uninitialized=True)
        var asm = Array[SIMD[f32, 4], D](uninitialized=True)
        comptime for u in range(D):
            var blk = b0 + c + u
            qb[u] = bitcast[u32, 4](rebind[SIMD[u8, 16]](Qb[row, blk]))
            sd[u] = rebind[Scalar[f16]](S[row, blk])
            alo[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2])
            ahi[u] = rebind[SIMD[bf16, 16]](Av[h, blk * 2 + 1])
            asm[u] = rebind[SIMD[f32, 4]](Sv[blk, half])
        comptime for u in range(D):
            var t = SIMD[f32, 8](0)
            mma(t, alo[u], nib_frag[0](qb[u]), SIMD[f32, 8](0))
            var t2 = SIMD[f32, 8](0)
            mma(t2, ahi[u], nib_frag[4](qb[u]), t)
            tot = fma(SIMD[f32, 4](sd[u].cast[f32]()), fma(asm[u], SIMD[f32, 4](-136.0), t2.slice[4, offset=0]()), tot)
    comptime for i in range(4):
        red[wave * 128 + (2 * i + half) * 16 + h] = tot[i]
    barrier()
    if tid < 128:
        var col = tid // 16
        var r = tid % 16
        if col < M:
            var s = Float32(0)
            comptime for w in range(FW_WAVES):
                s += rebind[Float32](red[w * 128 + col * 16 + r])
            Cp[0, col, Int(block_idx.x) * 16 + r] = rebind[Cp.ElementType](s)


def amar_fw_prep4[
    P: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Aq: TileTensor[u8, QLayout, MutAnyOrigin],
    Asc: TileTensor[f32, SLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Aq.flat_rank == 2 and Asc.flat_rank == 2
    comptime QM = (1 << (4 * P - 1)) - 1
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var x = rebind[Scalar[bf16]](A[row, blk * 32 + lane]).cast[f32]() if row < Int(m) else Float32(0)
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(QM) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    q = max(min(q, Int32(QM)), Int32(-QM))
    var qs = warp.sum(q)
    var o = warp.shuffle_xor(q, UInt32(16))
    if lane < 16:
        comptime for p in range(P):
            var b = ((q >> Int32(4 * p)) & 15) | (((o >> Int32(4 * p)) & 15) << 4)
            Aq[row, (blk * P + p) * 16 + lane] = rebind[Aq.ElementType](UInt8(b))
    if lane == 0:
        Asc[blk, (row % 2) * 8 + row // 2] = rebind[Asc.ElementType](scale)
        Asc[blk, (row % 2) * 8 + 4 + row // 2] = rebind[Asc.ElementType](Float32(-8) * scale * Float32(qs))


@always_inline
def wmma_iu4(sa: Bool, a: SIMD[DType.int32, 2], b: SIMD[DType.int32, 2], c: SIMD[DType.int32, 8]) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu4", SIMD[DType.int32, 8]](sa, a, False, b, c, False)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu4[
    N: Int, K: Int, D: Int, P: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
    ABLA: Int = 0,
](
    Aq: TileTensor[u8, AQLayout, MutAnyOrigin],
    Asc: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Aq.flat_rank == 2 and Asc.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 16 == 0 and NB % (FW_WAVES * D) == 0 and not GFX12_ROWS
    comptime NBW = NB // FW_WAVES
    comptime i32 = DType.int32
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 128]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * 16 + h
    var Qb = Q.vectorize[1, 16]()
    var Av = Aq.vectorize[1, 16]()
    var Sv = Asc.vectorize[1, 8]()
    var tot = SIMD[f32, 4](0)
    var b0 = wave * NBW
    for c in range(0, NBW, D):
        var qb = Array[SIMD[i32, 4], D](uninitialized=True)
        var sd = Array[Scalar[f16], D](uninitialized=True)
        var ap = Array[SIMD[i32, 4], D * P](uninitialized=True)
        var sc = Array[SIMD[f32, 8], D](uninitialized=True)
        comptime for u in range(D):
            var blk = b0 + c + u
            qb[u] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Qb[row, b0 + u if ABLA == 2 else blk]))
            sd[u] = rebind[Scalar[f16]](S[row, blk])
            comptime for p in range(P):
                ap[u * P + p] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Av[h % 8, (u if ABLA >= 1 else blk) * P + p]))
            sc[u] = rebind[SIMD[f32, 8]](Sv[u if ABLA >= 1 else blk, half])
        comptime for u in range(D):
            var w1 = qb[u].slice[2, offset=0]()
            var w2 = qb[u].slice[2, offset=2]()
            var acc = SIMD[i32, 8](0)
            comptime for pp in range(P):
                comptime p = P - 1 - pp
                comptime if pp > 0:
                    acc = (acc.slice[4, offset=0]() << SIMD[i32, 4](4)).join(SIMD[i32, 4](0))
                acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=0](), w1, acc)
                acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=2](), w2, acc)
            var v = fma(acc.slice[4, offset=0]().cast[f32](), sc[u].slice[4, offset=0](), sc[u].slice[4, offset=4]())
            tot = fma(SIMD[f32, 4](sd[u].cast[f32]()), v, tot)
    comptime for i in range(4):
        red[wave * 128 + (2 * i + half) * 16 + h] = tot[i]
    barrier()
    if tid < 128:
        var col = tid // 16
        var r = tid % 16
        if col < M:
            var s = Float32(0)
            comptime for w in range(FW_WAVES):
                s += rebind[Float32](red[w * 128 + col * 16 + r])
            Cp[0, col, Int(block_idx.x) * 16 + r] = rebind[Cp.ElementType](s)


@always_inline
def fw4_rec[P: Int]() -> Int:
    return 8 * P * 16 + 64


def amar_fw_prep4s[
    P: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Ab: TileTensor[u8, QLayout, MutAnyOrigin],
    Af: TileTensor[f32, SLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Ab.flat_rank == 2 and Af.flat_rank == 2
    comptime QM = (1 << (4 * P - 1)) - 1
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var x = rebind[Scalar[bf16]](A[row, blk * 32 + lane]).cast[f32]() if row < Int(m) else Float32(0)
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(QM) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    q = max(min(q, Int32(QM)), Int32(-QM))
    var qs = warp.sum(q)
    var o = warp.shuffle_xor(q, UInt32(16))
    if lane < 16:
        comptime for p in range(P):
            var b = ((q >> Int32(4 * p)) & 15) | (((o >> Int32(4 * p)) & 15) << 4)
            Ab[blk, (row * P + p) * 16 + lane] = rebind[Ab.ElementType](UInt8(b))
    if lane == 0:
        var so = 8 * P * 4 + (row % 2) * 8 + row // 2
        Af[blk, so] = rebind[Af.ElementType](scale)
        Af[blk, so + 4] = rebind[Af.ElementType](Float32(-8) * scale * Float32(qs))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu4s[
    N: Int, K: Int, D: Int, P: Int, KCB: Int, XM: Int,
    ABLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    Ab: TileTensor[u8, ABLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Ab.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime REC = fw4_rec[P]()
    comptime assert N % (FW_WAVES * 32) == 0 and KCB % D == 0 and NB % D == 0 and not GFX12_ROWS
    comptime NCH = (NB + KCB - 1) // KCB
    comptime ITEMS = KCB * REC // 16
    comptime i32 = DType.int32
    var lds = stack_allocation[u32, address_space = AddressSpace.SHARED](row_major[KCB * REC // 4]())
    var lp = lds.ptr.unsafe_bitcast[Scalar[i32]]()
    var fp = lds.ptr.unsafe_bitcast[Scalar[f32]]()
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var lo = half == 0
    var row0 = (Int(block_idx.x) * FW_WAVES + tid // WARP_SIZE) * 32
    var Qb = Q.vectorize[1, 16]()
    var Av = Ab.vectorize[1, 16]()
    var tot0 = SIMD[f32, 4](0)
    var tot1 = SIMD[f32, 4](0)
    for ch in range(NCH):
        var cb = ch * KCB
        barrier()
        var it = tid
        while it < ITEMS:
            var b = it // (REC // 16)
            if cb + b < NB:
                lp.unsafe_offset(it * 4).store(bitcast[i32, 4](rebind[SIMD[u8, 16]](Av[cb + b, it % (REC // 16)])))
            it += FW_THREADS
        barrier()
        for c in range(0, KCB, D):
            if cb + c < NB:
                var qb = Array[SIMD[i32, 4], D](uninitialized=True)
                var sd = Array[Float32, D](uninitialized=True)
                var ap = Array[SIMD[i32, 4], D * P](uninitialized=True)
                var sc = Array[SIMD[f32, 8], D](uninitialized=True)
                comptime for u in range(D):
                    var blk = cb + c + u
                    qb[u] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Qb[row0 + lane, blk]))
                    sd[u] = rebind[Scalar[f16]](S[row0 + lane, blk]).cast[f32]()
                    var o = (c + u) * (REC // 4)
                    comptime for p in range(P):
                        ap[u * P + p] = lp.unsafe_offset(o + ((h % 8) * P + p) * 4).load[width=4]()
                    sc[u] = fp.unsafe_offset(o + 8 * P * 4 + half * 8).load[width=8]()
                comptime for u in range(D):
                    var qx = SIMD[i32, 4](0)
                    comptime for i in range(4):
                        qx[i] = warp.shuffle_xor(qb[u][i], UInt32(16))
                    var dx = warp.shuffle_xor(sd[u], UInt32(16))
                    comptime for g in range(2):
                        var qg = qb[u] if g == 0 else qx
                        comptime if XM == 1:
                            if g == 0:
                                qg = qb[u] if lo else qx
                            else:
                                qg = qx if lo else qb[u]
                        var w1 = qg.slice[2, offset=0]()
                        var w2 = qg.slice[2, offset=2]()
                        var acc = SIMD[i32, 8](0)
                        comptime for pp in range(P):
                            comptime p = P - 1 - pp
                            comptime if pp > 0:
                                acc = (acc.slice[4, offset=0]() << SIMD[i32, 4](4)).join(SIMD[i32, 4](0))
                            acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=0](), w1, acc)
                            acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=2](), w2, acc)
                        var v = fma(acc.slice[4, offset=0]().cast[f32](), sc[u].slice[4, offset=0](), sc[u].slice[4, offset=4]())
                        comptime if g == 0:
                            tot0 = fma(SIMD[f32, 4](sd[u] if lo or XM == 2 else dx), v, tot0)
                        else:
                            tot1 = fma(SIMD[f32, 4](dx if lo or XM == 2 else sd[u]), v, tot1)
    comptime for i in range(4):
        if 2 * i + half < M:
            Cp[0, 2 * i + half, row0 + h + (half * 16 if XM == 2 else 0)] = rebind[Cp.ElementType](tot0[i])
            Cp[0, 2 * i + half, row0 + h + ((1 - half) * 16 if XM == 2 else 16)] = rebind[Cp.ElementType](tot1[i])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu4x[
    N: Int, K: Int, D: Int, P: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    Aq: TileTensor[u8, AQLayout, MutAnyOrigin],
    Asc: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Aq.flat_rank == 2 and Asc.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 32 == 0 and NB % (FW_WAVES * D) == 0 and not GFX12_ROWS
    comptime NBW = NB // FW_WAVES
    comptime i32 = DType.int32
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 256]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row0 = Int(block_idx.x) * 32
    var Qb = Q.vectorize[1, 16]()
    var Av = Aq.vectorize[1, 16]()
    var Sv = Asc.vectorize[1, 8]()
    var tot0 = SIMD[f32, 4](0)
    var tot1 = SIMD[f32, 4](0)
    var b0 = wave * NBW
    for c in range(0, NBW, D):
        var qb = Array[SIMD[i32, 4], D](uninitialized=True)
        var sd = Array[Float32, D](uninitialized=True)
        var ap = Array[SIMD[i32, 4], D * P](uninitialized=True)
        var sc = Array[SIMD[f32, 8], D](uninitialized=True)
        comptime for u in range(D):
            var blk = b0 + c + u
            qb[u] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Qb[row0 + lane, blk]))
            sd[u] = rebind[Scalar[f16]](S[row0 + lane, blk]).cast[f32]()
            comptime for p in range(P):
                ap[u * P + p] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Av[h % 8, blk * P + p]))
            sc[u] = rebind[SIMD[f32, 8]](Sv[blk, half])
        comptime for u in range(D):
            var qx = SIMD[i32, 4](0)
            comptime for i in range(4):
                qx[i] = warp.shuffle_xor(qb[u][i], UInt32(16))
            var dx = warp.shuffle_xor(sd[u], UInt32(16))
            comptime for g in range(2):
                var qg = qb[u] if g == 0 else qx
                var w1 = qg.slice[2, offset=0]()
                var w2 = qg.slice[2, offset=2]()
                var acc = SIMD[i32, 8](0)
                comptime for pp in range(P):
                    comptime p = P - 1 - pp
                    comptime if pp > 0:
                        acc = (acc.slice[4, offset=0]() << SIMD[i32, 4](4)).join(SIMD[i32, 4](0))
                    acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=0](), w1, acc)
                    acc = wmma_iu4(p == P - 1, ap[u * P + p].slice[2, offset=2](), w2, acc)
                var v = fma(acc.slice[4, offset=0]().cast[f32](), sc[u].slice[4, offset=0](), sc[u].slice[4, offset=4]())
                comptime if g == 0:
                    tot0 = fma(SIMD[f32, 4](sd[u]), v, tot0)
                else:
                    tot1 = fma(SIMD[f32, 4](dx), v, tot1)
    comptime for i in range(4):
        red[wave * 256 + (2 * i + half) * 32 + half * 16 + h] = tot0[i]
        red[wave * 256 + (2 * i + half) * 32 + (1 - half) * 16 + h] = tot1[i]
    barrier()
    var col = tid // 32
    var r = tid % 32
    if col < M:
        var s = Float32(0)
        comptime for w in range(FW_WAVES):
            s += rebind[Float32](red[w * 256 + col * 32 + r])
        Cp[0, col, row0 + r] = rebind[Cp.ElementType](s)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu4d[
    N: Int, K: Int, DQ: Int, DA: Int, P: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    Aq: TileTensor[u8, AQLayout, MutAnyOrigin],
    Asc: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Aq.flat_rank == 2 and Asc.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 32 == 0 and NB % FW_WAVES == 0 and DQ % DA == 0 and not GFX12_ROWS
    comptime NBW = NB // FW_WAVES
    comptime i32 = DType.int32
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 256]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row0 = Int(block_idx.x) * 32
    var Qb = Q.vectorize[1, 16]()
    var Av = Aq.vectorize[1, 16]()
    var Sv = Asc.vectorize[1, 8]()
    var tot0 = SIMD[f32, 4](0)
    var tot1 = SIMD[f32, 4](0)
    var b0 = wave * NBW
    for c in range(0, NBW, DQ):
        var qb = Array[SIMD[i32, 4], DQ](uninitialized=True)
        var sd = Array[Float32, DQ](uninitialized=True)
        comptime for u in range(DQ):
            if c + u < NBW:
                qb[u] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Qb[row0 + lane, b0 + c + u]))
                sd[u] = rebind[Scalar[f16]](S[row0 + lane, b0 + c + u]).cast[f32]()
        comptime for u0 in range(0, DQ, DA):
            if c + u0 < NBW:
                var ap = Array[SIMD[i32, 4], DA * P](uninitialized=True)
                var sc = Array[SIMD[f32, 8], DA](uninitialized=True)
                comptime for v in range(DA):
                    var blk = min(b0 + c + u0 + v, NB - 1)
                    comptime for p in range(P):
                        ap[v * P + p] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Av[h % 8, blk * P + p]))
                    sc[v] = rebind[SIMD[f32, 8]](Sv[blk, half])
                comptime for v in range(DA):
                    if c + u0 + v < NBW:
                        var qx = SIMD[i32, 4](0)
                        comptime for i in range(4):
                            qx[i] = warp.shuffle_xor(qb[u0 + v][i], UInt32(16))
                        var dx = warp.shuffle_xor(sd[u0 + v], UInt32(16))
                        comptime for g in range(2):
                            var qg = qb[u0 + v] if g == 0 else qx
                            var w1 = qg.slice[2, offset=0]()
                            var w2 = qg.slice[2, offset=2]()
                            var acc = SIMD[i32, 8](0)
                            comptime for pp in range(P):
                                comptime p = P - 1 - pp
                                comptime if pp > 0:
                                    acc = (acc.slice[4, offset=0]() << SIMD[i32, 4](4)).join(SIMD[i32, 4](0))
                                acc = wmma_iu4(p == P - 1, ap[v * P + p].slice[2, offset=0](), w1, acc)
                                acc = wmma_iu4(p == P - 1, ap[v * P + p].slice[2, offset=2](), w2, acc)
                            var x = fma(acc.slice[4, offset=0]().cast[f32](), sc[v].slice[4, offset=0](), sc[v].slice[4, offset=4]())
                            comptime if g == 0:
                                tot0 = fma(SIMD[f32, 4](sd[u0 + v]), x, tot0)
                            else:
                                tot1 = fma(SIMD[f32, 4](dx), x, tot1)
    comptime for i in range(4):
        red[wave * 256 + (2 * i + half) * 32 + half * 16 + h] = tot0[i]
        red[wave * 256 + (2 * i + half) * 32 + (1 - half) * 16 + h] = tot1[i]
    barrier()
    var col = tid // 32
    var r = tid % 32
    if col < M:
        var s = Float32(0)
        comptime for w in range(FW_WAVES):
            s += rebind[Float32](red[w * 256 + col * 32 + r])
        Cp[0, col, row0 + r] = rebind[Cp.ElementType](s)


def amar_fw_prep4r[
    P: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Aq: TileTensor[u8, QLayout, MutAnyOrigin],
    Asc: TileTensor[f32, SLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert A.flat_rank == 2 and Aq.flat_rank == 2 and Asc.flat_rank == 2
    comptime QM = (1 << (4 * P - 1)) - 1
    comptime NS = (P + 2) // 2
    var row = Int(block_idx.x)
    var blk = Int(block_idx.y)
    var lane = Int(lane_id())
    var x = rebind[Scalar[bf16]](A[row, blk * 32 + lane]).cast[f32]() if row < Int(m) else Float32(0)
    var wmax = warp.max(abs(x))
    var scale = wmax / Float32(QM) if wmax > 0 else Float32(1)
    var v = x / scale
    var q = Int32(v + Float32(0.5)) if v >= 0 else -Int32(-v + Float32(0.5))
    q = max(min(q, Int32(QM)), Int32(-QM))
    var o = warp.shuffle_xor(q, UInt32(16))
    if lane < 16:
        comptime for j in range(P + 1):
            var nq = (q >> Int32(4 * j)) & 15 if j < P else (q >> 31) & 1
            var no = (o >> Int32(4 * j)) & 15 if j < P else (o >> 31) & 1
            Aq[blk * NS + j // 2, (row + 8 * (j % 2)) * 16 + lane] = rebind[Aq.ElementType](UInt8(nq | (no << 4)))
    if lane == 0:
        Asc[blk, (row % 2) * 4 + row // 2] = rebind[Asc.ElementType](scale)


@always_inline
def wmma_iu4b(a: SIMD[DType.int32, 2], b: SIMD[DType.int32, 2], c: SIMD[DType.int32, 8]) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic["llvm.amdgcn.wmma.i32.16x16x16.iu4", SIMD[DType.int32, 8]](False, a, True, b, c, False)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FW_THREADS)))
def amar_fewrow_iu4r[
    N: Int, K: Int, D: Int, P: Int,
    AQLayout: TensorLayout, ASLayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
    ABLA: Int = 0,
](
    Aq: TileTensor[u8, AQLayout, MutAnyOrigin],
    Asc: TileTensor[f32, ASLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    m: Int32,
):
    comptime assert Aq.flat_rank == 2 and Asc.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    comptime NB = K // 32
    comptime assert N % 16 == 0 and NB % FW_WAVES == 0 and not GFX12_ROWS
    comptime NBW = NB // FW_WAVES
    comptime NS = (P + 2) // 2
    comptime i32 = DType.int32
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[FW_WAVES * 128]())
    var M = Int(m)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = Int(lane_id())
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * 16 + h
    var Qb = Q.vectorize[1, 16]()
    var Av = Aq.vectorize[1, 16]()
    var Sv = Asc.vectorize[1, 4]()
    var tot = SIMD[f32, 4](0)
    var b0 = wave * NBW
    for c in range(0, NBW, D):
        var cc = c if NBW % D == 0 else min(c, NBW - D)
        var qb = Array[SIMD[i32, 4], D](uninitialized=True)
        var sd = Array[Scalar[f16], D](uninitialized=True)
        var ap = Array[SIMD[i32, 4], D * NS](uninitialized=True)
        var sc = Array[SIMD[f32, 4], D](uninitialized=True)
        comptime for u in range(D):
            var blk = b0 + cc + u
            qb[u] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Qb[row, b0 + u if ABLA == 2 else blk]))
            sd[u] = rebind[Scalar[f16]](S[row, blk])
            comptime for s in range(NS):
                ap[u * NS + s] = bitcast[i32, 4](rebind[SIMD[u8, 16]](Av[(u if ABLA >= 1 else blk) * NS + s, h]))
            sc[u] = rebind[SIMD[f32, 4]](Sv[u if ABLA >= 1 else blk, half])
        comptime for u in range(D):
            var qs = qb[u] ^ SIMD[i32, 4](Int32(-2004318072))
            var w1 = qs.slice[2, offset=0]()
            var w2 = qs.slice[2, offset=2]()
            var r = SIMD[i32, 4](0)
            comptime for s in range(NS):
                var acc = wmma_iu4b(ap[u * NS + s].slice[2, offset=0](), w1, SIMD[i32, 8](0))
                acc = wmma_iu4b(ap[u * NS + s].slice[2, offset=2](), w2, acc)
                comptime if s == 0:
                    r = acc.slice[4, offset=0]() + (acc.slice[4, offset=4]() << SIMD[i32, 4](4))
                else:
                    comptime if 2 * s == P:
                        r -= acc.slice[4, offset=0]() << SIMD[i32, 4](8 * s)
                    else:
                        r += acc.slice[4, offset=0]() << SIMD[i32, 4](8 * s)
                        comptime if 2 * s + 1 == P:
                            r -= acc.slice[4, offset=4]() << SIMD[i32, 4](8 * s + 4)
                        else:
                            r += acc.slice[4, offset=4]() << SIMD[i32, 4](8 * s + 4)
            tot = fma(sc[u] * SIMD[f32, 4](sd[u].cast[f32]() if NBW % D == 0 or cc + u >= c else Float32(0)), r.cast[f32](), tot)
    comptime for i in range(4):
        red[wave * 128 + (2 * i + half) * 16 + h] = tot[i]
    barrier()
    if tid < 128:
        var col = tid // 16
        var r = tid % 16
        if col < M:
            var s = Float32(0)
            comptime for w in range(FW_WAVES):
                s += rebind[Float32](red[w * 128 + col * 16 + r])
            Cp[0, col, Int(block_idx.x) * 16 + r] = rebind[Cp.ElementType](s)
