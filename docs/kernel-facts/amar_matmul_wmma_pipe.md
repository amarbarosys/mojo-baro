# Kernel amar_matmul_wmma_pipe

Source kernels/matmul_wmma_pipe.mojo. Registry roles: serve/spark.mojo, bench/bench_fp16_pipe.mojo, bench/bench_kq_prefill.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
674:(`kq_gemm`, `amar_matmul_wmma_pipe[TB_=1, CF16=0]`); the NH_V = 32 projections take a one-warp-wide tile. Scratch
Recent commits on the file:
75eabc9 kernels(wmma): TB_ and CF16 comptime params on amar_matmul_wmma_pipe
7f2740a mojo-baro: initial public snapshot
