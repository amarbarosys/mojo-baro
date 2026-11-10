"""The amar_gemv_kq kernel at a model shape for a wavewait SQTT capture: one Q4_K GEMV of vibethinker's FFN up projection
(8960 rows x 1536, bf16 activations, epilogue 0), launched RUNS times so the last dispatch runs with a warm
instruction cache. Timing only: the weight bytes are a fixed pattern (finite f16 scales), the output is not checked
(kernels/test_gemv_kq holds the bytes). Capture recipe: ~/Brain/Skills/own/wavewait/SKILL.md.
"""
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from gemv_kq import amar_gemv_kq, kq_bytes, GT_Q4K
from matmul_skinny import ROW_WAVES, ROW_THREADS

comptime bf16 = DType.bfloat16
comptime ROWS = 8960
comptime K = 1536
comptime RUNS = 3


def main() raises:
    var ctx = DeviceContext()
    var nbytes = ROWS * (K // 256) * kq_bytes[GT_Q4K]()
    var w_d = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    w_d.enqueue_fill(0x11)
    var x_d = ctx.enqueue_create_buffer[bf16](K)
    x_d.enqueue_fill(1)
    var c_d = ctx.enqueue_create_buffer[DType.float32](ROWS)
    c_d.enqueue_fill(0)
    var o_d = ctx.enqueue_create_buffer[bf16](ROWS)
    o_d.enqueue_fill(0)
    var X = TileTensor(x_d, row_major[1, K]())
    var C = TileTensor(c_d, row_major[ROWS]())
    var O = TileTensor(o_d, row_major[ROWS]())
    for _ in range(RUNS):
        ctx.enqueue_function[amar_gemv_kq[GT_Q4K, GT_Q4K, 0, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType, bf16]](
            X, w_d.unsafe_ptr(), C, O, Int32(ROWS), Int32(K), Int32(ROWS), Int32(nbytes),
            grid_dim=(ROWS + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS,
        )
    ctx.synchronize()
    print("ok gemv-kq-att:", RUNS, "launches,", ROWS, "x", K, "q4k,", nbytes, "weight bytes")
