# entry: three plain pointers (no hidden kernargs) around the pipelined fp16 WMMA GEMM
# C[M,N] = A[M,K] @ B, B row-major [K,N] or, with -D WMMA_TB=1, stored [N,K] (x @ W.T). Sizes fixed at build time
# (-D TG_M/TG_N/TG_K, default 4096), multiples of 128. Tile rule from tools/sweep_tiles.sh (docs/m1.md); -D TG_WM/WN/TM/TN override it.
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import get_defined_int
from layout import TileTensor, row_major
from matmul_wmma_pipe import amar_matmul_wmma_pipe, C_DTYPE, TB

comptime M = get_defined_int["TG_M", 4096]()
comptime N = get_defined_int["TG_N", 4096]()
comptime K = get_defined_int["TG_K", 4096]()
comptime a_layout = row_major[M, K]()
comptime B_ROWS = N if TB == 1 else K
comptime B_COLS = K if TB == 1 else N
comptime b_layout = row_major[B_ROWS, B_COLS]()
comptime c_layout = row_major[M, N]()
comptime B128 = ceildiv(N, 128) * ceildiv(M, 128)
comptime WM = get_defined_int["TG_WM", 4 if B128 >= 64 else 2]()
comptime WN = get_defined_int["TG_WN", 4 if 64 <= B128 < 512 else 2]()
comptime TM = get_defined_int["TG_TM", 2]()
comptime TN = get_defined_int["TG_TN", 4 if B128 >= 512 else 2]()

def tg_wmma_gemm(a: UnsafePointer[Float16, MutAnyOrigin], b: UnsafePointer[Float16, MutAnyOrigin], c: UnsafePointer[Scalar[C_DTYPE], MutAnyOrigin]):
    amar_matmul_wmma_pipe[type_of(a_layout), type_of(b_layout), type_of(c_layout), WM, WN, TM, TN](
        TileTensor(a, a_layout), TileTensor(b, b_layout), TileTensor(c, c_layout), Int32(M), Int32(N), Int32(K))

def main() raises:
    var ctx = DeviceContext()
    _ = ctx.compile_function[tg_wmma_gemm]()
