from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from matmul import dtype
from matmul_ldst import BM, BN, BK, TM, TN, NTHREADS, LDS_PAD

comptime VEC = 4
comptime A_LOADERS = BM * BK // VEC
comptime B_LOADERS = BK * BN // VEC


def amar_matmul_vec4[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    comptime assert BK % VEC == 0 and BN % VEC == 0
    comptime assert A_LOADERS + B_LOADERS <= NTHREADS

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tx = thread_idx.x
    var ty = thread_idx.y
    var tid = ty * (BN // TN) + tx

    var block_row = block_idx.y * BM
    var block_col = block_idx.x * BN

    var sa = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BM + LDS_PAD]()
    )
    var sb = stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[BK, BN]()
    )
    var sav = sa.vectorize[1, TM]()
    var sbv = sb.vectorize[1, TN]()
    var sbw = sb.vectorize[1, VEC]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2 and sbw.flat_rank == 2

    var Av = A.vectorize[1, VEC]()
    var Bv = B.vectorize[1, VEC]()
    comptime assert Av.flat_rank == 2 and Bv.flat_rank == 2

    var acc = SIMD[dtype, TM * TN](0)

    var k_tile = 0
    while k_tile < K:
        if tid < A_LOADERS:
            var r = tid * VEC // BK
            var c = tid * VEC % BK
            var gr = block_row + r
            var gc = k_tile + c
            var v = SIMD[dtype, VEC](0)
            if gr < M and gc + VEC <= K:
                v = rebind[SIMD[dtype, VEC]](Av[gr, gc // VEC])
            comptime for j in range(VEC):
                sa[c + j, r] = rebind[sa.ElementType](v[j])
        elif tid < A_LOADERS + B_LOADERS:
            var e = (tid - A_LOADERS) * VEC
            var r = e // BN
            var c = e % BN
            var gr = k_tile + r
            var gc = block_col + c
            var v = SIMD[dtype, VEC](0)
            if gr < K and gc + VEC <= N:
                v = rebind[SIMD[dtype, VEC]](Bv[gr, gc // VEC])
            sbw[r, c // VEC] = rebind[sbw.ElementType](v)
        barrier()

        for k in range(BK):
            var a_reg = rebind[SIMD[dtype, TM]](sav[k, ty])
            var b_reg = rebind[SIMD[dtype, TN]](sbv[k, tx])
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += a_reg[i] * b_reg[j]
        barrier()
        k_tile += BK

    comptime for i in range(TM):
        comptime for j in range(TN):
            var r = block_row + ty * TM + i
            var c = block_col + tx * TN + j
            if r < M and c < N:
                C[r, c] = rebind[C.ElementType](acc[i * TN + j])
