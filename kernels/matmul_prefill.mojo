from arch import perm
from max.gpu import block_idx, global_idx, thread_idx, WARP_SIZE
from std.math import fma
from bexp import exp
from std.memory import bitcast
from std.sys.intrinsics import llvm_intrinsic
from layout import TileTensor, TensorLayout, row_major
from arch import mma

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime PF_WAVES = 8
comptime PF_THREADS = PF_WAVES * WARP_SIZE


@always_inline
def f32x16_to_bf16_trunc(x: SIMD[f32, 16]) -> SIMD[bf16, 16]:
    var u = bitcast[DType.uint32, 16](x)
    var packed = SIMD[DType.uint32, 8](0)
    comptime for i in range(8):
        packed[i] = perm(u[2 * i + 1], u[2 * i], UInt32(0x07060302))
    return bitcast[bf16, 16](packed)


def amar_matmul_prefill_q4[
    WTM: Int, WTN: Int, WAVES_M: Int, ACC: Bool,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    CLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and C.flat_rank == 2
    comptime WAVES_N = PF_WAVES // WAVES_M
    comptime BM = WAVES_M * WTM * 16
    comptime BN = WAVES_N * WTN * 16

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var wm = wave // WAVES_N
    var wn = wave % WAVES_N
    var row0 = Int(block_idx.y) * BM + wm * WTM * 16
    var col0 = Int(block_idx.x) * BN + wn * WTN * 16
    if row0 >= M or col0 >= N:
        return

    var Av = A.vectorize[1, 16]()
    var Qv = Q.vectorize[1, 16]()
    var acc = Array[SIMD[f32, 8], WTM * WTN](fill=SIMD[f32, 8](0))
    var eight = SIMD[f32, 16](8)
    var nb = K // 32

    for kb in range(nb):
        var afr = Array[SIMD[bf16, 16], WTM * 2](uninitialized=True)
        comptime for tm in range(WTM):
            var r = row0 + tm * 16 + h
            if r < M:
                afr[2 * tm] = rebind[SIMD[bf16, 16]](Av[r, kb * 2])
                afr[2 * tm + 1] = rebind[SIMD[bf16, 16]](Av[r, kb * 2 + 1])
            else:
                afr[2 * tm] = SIMD[bf16, 16](0)
                afr[2 * tm + 1] = SIMD[bf16, 16](0)
        comptime for tn in range(WTN):
            var c = min(col0 + tn * 16 + h, N - 1)
            var bytes16 = rebind[SIMD[DType.uint8, 16]](Qv[c, kb])
            var d = rebind[Scalar[DType.float16]](S[c, kb]).cast[f32]()
            var lo = f32x16_to_bf16_trunc((bytes16 & UInt8(0xF)).cast[f32]() - eight)
            var hi = f32x16_to_bf16_trunc((bytes16 >> UInt8(4)).cast[f32]() - eight)
            comptime for tm in range(WTM):
                var t = SIMD[f32, 8](0)
                var t2 = SIMD[f32, 8](0)
                mma(t, afr[2 * tm], lo, SIMD[f32, 8](0))
                mma(t2, afr[2 * tm + 1], hi, t)
                acc[tm * WTN + tn] = fma(t2, SIMD[f32, 8](d), acc[tm * WTN + tn])

    comptime for tm in range(WTM):
        comptime for tn in range(WTN):
            var c = col0 + tn * 16 + h
            if c < N:
                comptime for i in range(8):
                    var r = row0 + tm * 16 + 2 * i + half
                    if r < M:
                        comptime if ACC:
                            C[r, c] = rebind[C.ElementType](
                                rebind[Scalar[f32]](C[r, c]) + acc[tm * WTN + tn][i]
                            )
                        else:
                            C[r, c] = rebind[C.ElementType](acc[tm * WTN + tn][i])


def amar_matmul_prefill_q8[
    WTM: Int, WTN: Int, WAVES_M: Int, ACC: Bool,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    CLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.int8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and C.flat_rank == 2
    comptime WAVES_N = PF_WAVES // WAVES_M
    comptime BM = WAVES_M * WTM * 16
    comptime BN = WAVES_N * WTN * 16

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var wm = wave // WAVES_N
    var wn = wave % WAVES_N
    var row0 = Int(block_idx.y) * BM + wm * WTM * 16
    var col0 = Int(block_idx.x) * BN + wn * WTN * 16
    if row0 >= M or col0 >= N:
        return

    var Av = A.vectorize[1, 16]()
    var Qv = Q.vectorize[1, 16]()
    var acc = Array[SIMD[f32, 8], WTM * WTN](fill=SIMD[f32, 8](0))
    var nb = K // 32

    for kb in range(nb):
        var afr = Array[SIMD[bf16, 16], WTM * 2](uninitialized=True)
        comptime for tm in range(WTM):
            var r = row0 + tm * 16 + h
            if r < M:
                afr[2 * tm] = rebind[SIMD[bf16, 16]](Av[r, kb * 2])
                afr[2 * tm + 1] = rebind[SIMD[bf16, 16]](Av[r, kb * 2 + 1])
            else:
                afr[2 * tm] = SIMD[bf16, 16](0)
                afr[2 * tm + 1] = SIMD[bf16, 16](0)
        comptime for tn in range(WTN):
            var c = min(col0 + tn * 16 + h, N - 1)
            var d = rebind[Scalar[DType.float16]](S[c, kb]).cast[f32]()
            var lo = f32x16_to_bf16_trunc(rebind[SIMD[DType.int8, 16]](Qv[c, kb * 2]).cast[f32]())
            var hi = f32x16_to_bf16_trunc(rebind[SIMD[DType.int8, 16]](Qv[c, kb * 2 + 1]).cast[f32]())
            comptime for tm in range(WTM):
                var t = SIMD[f32, 8](0)
                var t2 = SIMD[f32, 8](0)
                mma(t, afr[2 * tm], lo, SIMD[f32, 8](0))
                mma(t2, afr[2 * tm + 1], hi, t)
                acc[tm * WTN + tn] = fma(t2, SIMD[f32, 8](d), acc[tm * WTN + tn])

    comptime for tm in range(WTM):
        comptime for tn in range(WTN):
            var c = col0 + tn * 16 + h
            if c < N:
                comptime for i in range(8):
                    var r = row0 + tm * 16 + 2 * i + half
                    if r < M:
                        comptime if ACC:
                            C[r, c] = rebind[C.ElementType](
                                rebind[Scalar[f32]](C[r, c]) + acc[tm * WTN + tn][i]
                            )
                        else:
                            C[r, c] = rebind[C.ElementType](acc[tm * WTN + tn][i])


def amar_prefill_swiglu_bf16[
    GLayout: TensorLayout, OLayout: TensorLayout
](
    Gate: TileTensor[f32, GLayout, MutAnyOrigin],
    Up: TileTensor[f32, GLayout, MutAnyOrigin],
    O: TileTensor[bf16, OLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
):
    comptime assert Gate.flat_rank == 2 and Up.flat_rank == 2 and O.flat_rank == 2
    var N = Int(n)
    var gid = Int(global_idx.x)
    if gid >= Int(m) * N:
        return
    var r = gid // N
    var c = gid % N
    var g = rebind[Scalar[f32]](Gate[r, c])
    var u = rebind[Scalar[f32]](Up[r, c])
    var silu = g / (1 + exp(-g))
    O[r, c] = rebind[O.ElementType]((silu * u).cast[bf16]())
