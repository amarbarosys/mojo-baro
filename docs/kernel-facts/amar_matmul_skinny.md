# Kernel amar_matmul_skinny

Source kernels/matmul_skinny.mojo. Registry roles: bench/bench_coldcache.mojo, bench/bench_decode.mojo, kernels/test_attn_block.mojo, kernels/test_gguf_gemm.mojo, kernels/test_q8_gemm.mojo, kernels/test_ssm_block.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
71:weight GEMMs are `amar_matmul_skinny_q8row`: one wave per weight row over the
391:window is row-scaling-bound: `amar_matmul_skinny_q4rowb[2,3]` costs 1.41x / 1.29x its m = 1 self on the FFN shapes (P5 receipt
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
