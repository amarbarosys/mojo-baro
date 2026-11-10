"""amar_gemv_kq_rows (kernels/gemv_kq.mojo) byte test: the multi-session decode GEMV (serve/rows.mojo) must give
every row the bytes the m = 1 decode kernel amar_gemv_kq gives that row alone, so a session's greedy ids do not
depend on how many sessions share its step (docs/design/multi-session-serve.md). Random Q4_K and Q6_K tensors
(test_gemv_kq's generator), even and odd superblock counts, bf16 activations, m = 1..8 rows in one launch of R = 8.
Fails on the first mismatching element."""
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from gemv_kq import amar_gemv_kq, amar_gemv_kq_rows, GT_Q4K, GT_Q6K
from matmul_skinny import ROW_WAVES, ROW_THREADS
from test_gemv_kq import Lcg, gen_rows

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime N = 40
comptime R = 8
comptime KMAX = 1536


def run_case[GT: Int](ctx: DeviceContext, name: String, k: Int, mut rng: Lcg) raises:
    var wb = List[UInt8]()
    gen_rows(rng, GT, N, k, wb)
    var w_d = ctx.enqueue_create_buffer[u8](len(wb))
    var w_h = ctx.enqueue_create_host_buffer[u8](len(wb))
    var x_d = ctx.enqueue_create_buffer[bf16](R * KMAX)
    var x_h = ctx.enqueue_create_host_buffer[bf16](R * KMAX)
    var c_d = ctx.enqueue_create_buffer[f32](R * N)
    var c_h = ctx.enqueue_create_host_buffer[f32](R * N)
    var one_d = ctx.enqueue_create_buffer[f32](N)
    var one_h = ctx.enqueue_create_host_buffer[f32](N)
    var o_d = ctx.enqueue_create_buffer[bf16](N)
    ctx.synchronize()
    for i in range(len(wb)):
        w_h[i] = wb[i]
    for i in range(R * KMAX):
        x_h[i] = rng.unit().cast[bf16]()
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.synchronize()
    var X = TileTensor(x_d, row_major[R, KMAX]())
    var C = TileTensor(c_d, row_major[R * N]())
    var C1 = TileTensor(one_d, row_major[N]())
    var O = TileTensor(o_d, row_major[N]())
    var bad = 0
    for m in range(1, R + 1):
        ctx.enqueue_memset(c_d, 0)
        ctx.enqueue_function[amar_gemv_kq_rows[GT, R, type_of(X).LayoutType, type_of(C).LayoutType]](
            X, w_d.unsafe_ptr(), C, Int32(N), Int32(k), Int32(m), grid_dim=(N + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
        ctx.synchronize()
        for j in range(m):
            var Xj = TileTensor(DeviceBuffer[bf16](ctx, x_d.unsafe_ptr().unsafe_offset(j * KMAX), KMAX, owning=False), row_major[1, KMAX]())
            ctx.enqueue_function[amar_gemv_kq[GT, GT, 0, type_of(Xj).LayoutType, type_of(C1).LayoutType, type_of(O).LayoutType]](
                Xj, w_d.unsafe_ptr(), C1, O, Int32(N), Int32(k), Int32(N), Int32(0), grid_dim=(N + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS)
            ctx.enqueue_copy(dst_buf=one_h, src_buf=one_d)
            ctx.synchronize()
            for row in range(N):
                if bitcast[DType.uint32](c_h[j * N + row]) != bitcast[DType.uint32](one_h[row]):
                    if bad < 10:
                        print("  rows mismatch", name, "m", m, "row j", j, "out", row, c_h[j * N + row], "vs m=1", one_h[row])
                    bad += 1
    print(name, "k", k, "m 1..", R, "mismatches", bad)
    if bad != 0:
        raise Error("FAIL " + name)


def main() raises:
    var ctx = DeviceContext()
    var rng = Lcg(11)
    run_case[GT_Q4K](ctx, "q4k", 1536, rng)
    run_case[GT_Q6K](ctx, "q6k", 1536, rng)
    run_case[GT_Q4K](ctx, "q4k-odd", 768, rng)
    run_case[GT_Q6K](ctx, "q6k-odd", 1280, rng)
    print("PASS test_gemv_kq_rows")
