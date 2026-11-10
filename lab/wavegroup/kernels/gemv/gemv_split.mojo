# split-K k-quant GEMV over up to three weight segments (e.g. q, k, v: separate tensors, own quant types) in one launch.
# Rows [0, n0) read W0 as GT0, [n0, n0+n1) W1 as GT1, the rest W2 as GT2. S waves share one row (each takes every S-th
# superblock pair); partial sums meet in LDS in wave order, so the result is deterministic. A block is 8 waves = 8/S rows.
from std.math import fma
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from gemv_kq import kq_lane16, kq_bytes, kq_tree16, kq_wave_sum, gemv_epi
from q4x_kernels import UP

comptime f32 = DType.float32

@always_inline
def seg_dot[GT: Int, S: Int, ADT: DType, XL: TensorLayout](X: TileTensor[ADT, XL, MutAnyOrigin], W: UP, row: Int, K: Int, wave: Int, lane: Int) -> SIMD[f32, 16]:
    var Xv = X.vectorize[1, 16]()
    var rb = row * (K // 256) * kq_bytes[GT]()
    var acc = SIMD[f32, 16](0)
    var b = (wave % S) * 2 + lane // 16
    while b < K // 256:
        var r = kq_lane16[GT](W, rb + b * kq_bytes[GT](), lane % 16)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        acc = fma(v, rebind[SIMD[ADT, 16]](Xv[0, (b * 256 + r[3]) // 16]).cast[f32](), acc)
        b += 2 * S
    return acc

def wg_gemv_kq[GT0: Int, GT1: Int, GT2: Int, S: Int, EPI: Int, ADT: DType, XL: TensorLayout, CL: TensorLayout, OL: TensorLayout](
    X: TileTensor[ADT, XL, MutAnyOrigin], W0: UP, W1: UP, W2: UP, C: TileTensor[f32, CL, MutAnyOrigin], O: TileTensor[ADT, OL, MutAnyOrigin],
    n0: Int32, n1: Int32, n: Int32, k_dim: Int32,
):
    comptime assert X.flat_rank == 2 and 8 % S == 0
    var wave = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * (8 // S) + wave // S
    var K = Int(k_dim)
    var acc = SIMD[f32, 16](0)
    if row < Int(n0):
        acc = seg_dot[GT0, S](X, W0, row, K, wave, lane)
    elif row < Int(n0) + Int(n1):
        acc = seg_dot[GT1, S](X, W1, row - Int(n0), K, wave, lane)
    elif row < Int(n):
        acc = seg_dot[GT2, S](X, W2, row - Int(n0) - Int(n1), K, wave, lane)
    var part = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[8]())
    var total = kq_wave_sum(kq_tree16(acc))
    if lane == 0:
        part[wave] = rebind[part.ElementType](total)
    barrier()
    if wave % S == 0 and lane == 0 and row < Int(n):
        var sum: Float32 = 0
        comptime for i in range(S):
            sum += rebind[Scalar[f32]](part[wave + i])
        gemv_epi[EPI, ODT=ADT](C, O, row, sum)
