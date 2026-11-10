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

comptime QV = 16
comptime STR = 20
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16


@always_inline
def turbo_lds_dwords[MR: Int, UNROLL: Int]() -> Int:
    return MR * (WARP_SIZE * UNROLL) * STR


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
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
