"""Emit-only instantiation of the engine's m=1 q4 GEMV pair (G1 scouting, not part of run.sh).

`amar_matmul_skinny_q4rowb[2, 1]` + `amar_skinny_reduce[.., 1]`, the dispatch gemm_q4 uses at
m == 1 (serve/registry.mojo). Shape from `-D N=<rows> -D K=<cols>` (K a multiple of 1024, N of 8). Built with `--target-accelerator apple-m1 --emit asm`, never run.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from matmul_skinny import amar_matmul_skinny_q4rowb, amar_skinny_reduce, ROW_WAVES, ROW_THREADS

comptime N = get_defined_int["N", 1024]()
comptime K = get_defined_int["K", 4096]()


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU target"
    var ctx = DeviceContext()
    comptime a_l = row_major[1, K]()
    comptime q_l = row_major[N, K // 2]()
    comptime s_l = row_major[N, K // 32]()
    comptime p_l = row_major[1, 1, N]()
    comptime c_l = row_major[1, N]()

    var a_buf = ctx.enqueue_create_buffer[DType.bfloat16](K)
    var A = TileTensor(a_buf, a_l)
    var q_buf = ctx.enqueue_create_buffer[DType.uint8](N * (K // 2))
    var Q = TileTensor(q_buf, q_l)
    var s_buf = ctx.enqueue_create_buffer[DType.float16](N * (K // 32))
    var S = TileTensor(s_buf, s_l)
    var p_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var P = TileTensor(p_buf, p_l)
    var c_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var C = TileTensor(c_buf, c_l)

    ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 1, type_of(a_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        A, Q, S, P, Int32(1), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[amar_skinny_reduce[type_of(p_l), type_of(c_l), 1]](
        P, C, Int32(1), Int32(N), grid_dim=ceildiv(N, 256), block_dim=256)
    ctx.synchronize()
