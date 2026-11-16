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
from arch import mma, acc_row

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
