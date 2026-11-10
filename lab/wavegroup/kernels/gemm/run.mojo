# M1 gate: our WMMA GEMM on DeviceContext. Exact check (small-integer inputs, fp64 CPU oracle on sampled cells),
# warm TFLOPS over 300 back-to-back launches, and empty-kernel launches/s (the M5 decision).
# Sizes at build time: -D TG_M/TG_N/TG_K, multiples of 128; -D WMMA_TB=1 reads B as [N,K].
from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from tg_wmma_gemm import tg_wmma_gemm, M, N, K, WM, WN, TM, TN
from matmul_wmma_pipe import TB, C_DTYPE

def empty():
    pass

def main() raises:
    var ctx = DeviceContext()
    var ah = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var bh = ctx.enqueue_create_host_buffer[DType.float16](K * N)
    var ch = ctx.enqueue_create_host_buffer[C_DTYPE](M * N)
    ctx.synchronize()
    for i in range(M * K):
        ah[i] = Float16((i % 7) - 3) # small integers: fp16 products and fp32 sums stay exact
    for i in range(K * N):
        bh[i] = Float16((i % 5) - 2)
    var ad = ctx.enqueue_create_buffer[DType.float16](M * K)
    var bd = ctx.enqueue_create_buffer[DType.float16](K * N)
    var cd = ctx.enqueue_create_buffer[C_DTYPE](M * N)
    ctx.enqueue_copy(dst_buf=ad, src_buf=ah)
    ctx.enqueue_copy(dst_buf=bd, src_buf=bh)
    comptime GRID = (ceildiv(N, WN * TN * 16), ceildiv(M, WM * TM * 16))
    comptime NT = WM * WN * 32

    def run(n: Int) raises {ctx, ad, bd, cd}:
        for _ in range(n):
            ctx.enqueue_function[tg_wmma_gemm](ad.unsafe_ptr(), bd.unsafe_ptr(), cd.unsafe_ptr(), grid_dim=GRID, block_dim=NT)
        ctx.synchronize()

    var w0 = perf_counter_ns()
    while perf_counter_ns() - w0 < 3_000_000_000: # clocks idle low and ramp only under load
        run(20)
    var t0 = perf_counter_ns()
    run(300)
    var ms = Float64(perf_counter_ns() - t0) / 1e6 / 300

    def one() raises {ctx, ad, bd, cd}:
        ctx.enqueue_function[tg_wmma_gemm](ad.unsafe_ptr(), bd.unsafe_ptr(), cd.unsafe_ptr(), grid_dim=GRID, block_dim=NT)

    var dev_ms = Float64(ctx.execution_time(one, 300)) / 1e6 / 300 # device timer: no host enqueue time

    ctx.enqueue_copy(dst_buf=ch, src_buf=cd)
    ctx.synchronize()
    var worst = 0.0
    for r in range(0, M, 37):
        for c in range(0, N, 41):
            var want = 0.0
            for p in range(K):
                want += Float64(ah[r * K + p]) * Float64(bh[c * K + p] if TB == 1 else bh[p * N + c])
            worst = max(worst, abs(Float64(ch[r * N + c]) - want) / max(abs(want), 1.0))

    var l0 = perf_counter_ns()
    for _ in range(10000):
        ctx.enqueue_function[empty](grid_dim=1, block_dim=32)
    ctx.synchronize()
    var lps = 1e13 / Float64(perf_counter_ns() - l0)
    print("m", M, "n", N, "k", K, "tb", TB, "ms", ms, "tflops", Float64(2 * M * N * K) / ms / 1e9, "dev_ms", dev_ms, "dev_tflops", Float64(2 * M * N * K) / dev_ms / 1e9, "max_rel_err", worst, "launches_per_s", lps)
    if worst >= 1e-2:
        raise Error("FAIL gemm: max_rel_err " + String(worst))
