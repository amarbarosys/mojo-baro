# M2 gate: k-quant GEMV (f32 activations) at a decode shape, effective GB/s against 960, total and kernel-only.
# Weights rotate over >= 256 MB of copies so the 96 MB Infinity Cache cannot serve them (decode reads each layer once).
# -D GV_N/GV_K: rows and reduction length (K multiple of 256). -D GV_GT: 12 q4_K, 14 q6_K.
# -D GV_S: 0 = mojo-baro's amar_gemv_kq, S in {1,2,4,8} = wg_gemv_kq split S ways, checked against amar_gemv_kq.
from std.sys import get_defined_int
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from gemv_kq import amar_gemv_kq, kq_bytes
from matmul_skinny import ROW_WAVES, ROW_THREADS
from gemv_split import wg_gemv_kq

comptime N = get_defined_int["GV_N", 2048]()
comptime K = get_defined_int["GV_K", 2048]()
comptime GT = get_defined_int["GV_GT", 12]()
comptime S = get_defined_int["GV_S", 0]()
comptime WB = N * (K // 256) * kq_bytes[GT]()
comptime COPIES = max(1, (256 << 20) // WB)

def empty():
    pass

def main() raises:
    var ctx = DeviceContext()
    var wh = ctx.enqueue_create_host_buffer[DType.uint8](WB)
    var xh = ctx.enqueue_create_host_buffer[DType.float32](K)
    var ch = ctx.enqueue_create_host_buffer[DType.float32](N)
    var rh = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(WB):
        wh[i] = UInt8((i * 7) % 101) # bytes <= 0x64: every f16 scale finite
    for i in range(K):
        xh[i] = Float32((i % 9) - 4) * 0.25
    var w = ctx.enqueue_create_buffer[DType.uint8](WB * COPIES)
    for i in range(COPIES):
        ctx.enqueue_copy(dst_buf=w.create_sub_buffer[DType.uint8](i * WB, WB), src_buf=wh)
    var x = ctx.enqueue_create_buffer[DType.float32](K)
    var c = ctx.enqueue_create_buffer[DType.float32](N)
    var r = ctx.enqueue_create_buffer[DType.float32](N)
    var o = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=x, src_buf=xh)
    var X = TileTensor(x, row_major[1, K]())
    var C = TileTensor(c, row_major[N]())
    var R = TileTensor(r, row_major[N]())
    var O = TileTensor(o, row_major[N]())
    comptime ref_k = amar_gemv_kq[GT, GT, 0, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType, DType.float32]
    comptime wg_k = wg_gemv_kq[GT, GT, GT, max(S, 1), 0, DType.float32, type_of(X).LayoutType, type_of(C).LayoutType, type_of(O).LayoutType]

    def launch(i: Int) raises {ctx, w, X, C, O}:
        var wp = w.unsafe_ptr() + (i % COPIES) * WB
        comptime if S == 0:
            ctx.enqueue_function[ref_k](X, wp, C, O, Int32(N), Int32(K), Int32(N), Int32(0), grid_dim=(N + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS)
        else:
            ctx.enqueue_function[wg_k](X, wp, wp, wp, C, O, Int32(N), Int32(0), Int32(N), Int32(K), grid_dim=(N * S + 7) // 8, block_dim=256)

    ctx.enqueue_function[ref_k](X, w.unsafe_ptr(), R, O, Int32(N), Int32(K), Int32(N), Int32(0), grid_dim=(N + ROW_WAVES - 1) // ROW_WAVES, block_dim=ROW_THREADS)
    launch(0)
    ctx.enqueue_copy(dst_buf=ch, src_buf=c)
    ctx.enqueue_copy(dst_buf=rh, src_buf=r)
    ctx.synchronize()
    var worst = 0.0
    for i in range(N):
        worst = max(worst, abs(Float64(ch[i]) - Float64(rh[i])) / max(abs(Float64(rh[i])), 1.0))

    for i in range(2000): # clock ramp
        launch(i)
    ctx.synchronize()

    def timed() raises {launch}:
        for i in range(COPIES * 4):
            launch(i)

    def empties() raises {ctx}:
        for _ in range(2000):
            ctx.enqueue_function[empty](grid_dim=1, block_dim=32)

    var us = Float64(ctx.execution_time(timed, 1)) / 1e3 / Float64(COPIES * 4)
    var e = Float64(ctx.execution_time(empties, 1)) / 1e3 / 2000 # device time per launch, paid by every kernel
    print("n", N, "k", K, "gt", GT, "s", S, "us", us, "of960", Float64(WB) / us / 1e3 / 960, "launch_us", e, "kernel_of960", Float64(WB) / (us - e) / 1e3 / 960, "rel_err_vs_ref", worst)
    if worst > 5e-4: # S=1 is bit-exact; S>1 reorders the sum (S=2 measured 1.1e-4)
        raise Error("FAIL gemv: rel_err_vs_ref " + String(worst))
