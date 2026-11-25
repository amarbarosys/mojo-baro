from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import mma, acc_row

comptime WMMA_M = 16
comptime WMMA_N = 16
comptime WMMA_K = 16

comptime WARPS_M = 4
comptime WARPS_N = 2
comptime WTILE_M = 2
comptime WTILE_N = 4

comptime BLK_M = WARPS_M * WTILE_M * WMMA_M
comptime BLK_N = WARPS_N * WTILE_N * WMMA_N
comptime BLK_K = WMMA_K
comptime NTHREADS = WARPS_M * WARPS_N * 32
comptime VEC = 8
comptime A_LOADERS = BLK_M * BLK_K // VEC
comptime B_LOADERS = BLK_K * BLK_N // VEC


def amar_matmul_wmma_lds[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[DType.float16, ALayout, MutAnyOrigin],
    B: TileTensor[DType.float16, BLayout, MutAnyOrigin],
    C: TileTensor[DType.float32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)

    var tid = thread_idx.x
    var lane = tid % 32
    var warp = tid // 32
    var warp_m = warp // WARPS_N
    var warp_n = warp % WARPS_N
    var h = lane % 16
    var half = lane // 16

    var block_row = block_idx.y * BLK_M
    var block_col = block_idx.x * BLK_N

    var sa = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
        row_major[BLK_M, BLK_K]()
    )
    var sb = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
        row_major[BLK_K, BLK_N]()
    )

    var acc = SIMD[DType.float32, WTILE_M * WTILE_N * 8](0)

    var sav = sa.vectorize[1, VEC]()
    var sbv = sb.vectorize[1, VEC]()
    var Av = A.vectorize[1, VEC]()
    var Bv = B.vectorize[1, VEC]()
    comptime assert sav.flat_rank == 2 and sbv.flat_rank == 2
    comptime assert Av.flat_rank == 2 and Bv.flat_rank == 2

    var kt = 0
    while kt < K:
        comptime for s in range(ceildiv(A_LOADERS, NTHREADS)):
            var v = tid + s * NTHREADS
            if v < A_LOADERS:
                var r = v // (BLK_K // VEC)
                var c = (v % (BLK_K // VEC)) * VEC
                var gr = block_row + r
                var gc = kt + c
                if gr < M and gc + VEC <= K:
                    sav[r, c // VEC] = rebind[sav.ElementType](Av[gr, gc // VEC])
                else:
                    comptime for i in range(VEC):
                        sa[r, c + i] = rebind[sa.ElementType](
                            A[gr, gc + i] if gr < M and gc + i < K
                            else Scalar[DType.float16](0)
                        )
        comptime for s in range(ceildiv(B_LOADERS, NTHREADS)):
            var v = (tid + NTHREADS - A_LOADERS % NTHREADS) % NTHREADS + s * NTHREADS
            if v < B_LOADERS:
                var r = v // (BLK_N // VEC)
                var c = (v % (BLK_N // VEC)) * VEC
                var gr = kt + r
                var gc = block_col + c
                if gr < K and gc + VEC <= N:
                    sbv[r, c // VEC] = rebind[sbv.ElementType](Bv[gr, gc // VEC])
                else:
                    comptime for i in range(VEC):
                        sb[r, c + i] = rebind[sb.ElementType](
                            B[gr, gc + i] if gr < K and gc + i < N
                            else Scalar[DType.float16](0)
                        )
        barrier()

        comptime for ks in range(BLK_K // WMMA_K):
            comptime for tm in range(WTILE_M):
                var a = SIMD[DType.float16, 16](0)
                var arow = (warp_m * WTILE_M + tm) * WMMA_M + h
                comptime for i in range(16):
                    a[i] = rebind[Scalar[DType.float16]](sa[arow, ks * WMMA_K + i])

                comptime for tn in range(WTILE_N):
                    var b = SIMD[DType.float16, 16](0)
                    var bcol = (warp_n * WTILE_N + tn) * WMMA_N + h
                    comptime for i in range(16):
                        b[i] = rebind[Scalar[DType.float16]](
                            sb[ks * WMMA_K + i, bcol]
                        )

                    var c = SIMD[DType.float32, 8](0)
                    comptime for i in range(8):
                        c[i] = acc[(tm * WTILE_N + tn) * 8 + i]
                    var d = SIMD[DType.float32, 8](0)
                    mma(d, a, b, c)
                    comptime for i in range(8):
                        acc[(tm * WTILE_N + tn) * 8 + i] = d[i]
        barrier()
        kt += BLK_K

    comptime for tm in range(WTILE_M):
        comptime for tn in range(WTILE_N):
            comptime for i in range(8):
                var row = acc_row(block_row + (warp_m * WTILE_M + tm) * WMMA_M, i, half)
                var col = block_col + (warp_n * WTILE_N + tn) * WMMA_N + h
                if row < M and col < N:
                    C[row, col] = rebind[C.ElementType](
                        acc[(tm * WTILE_N + tn) * 8 + i]
                    )
