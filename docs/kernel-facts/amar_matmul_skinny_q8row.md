# Kernel amar_matmul_skinny_q8row

Source kernels/matmul_skinny.mojo. Registry roles: gemm_q8 dispatch. Default path: in registry.
Callers: serve/registry.mojo, bench/bench_coldcache_mrow.mojo, bench/bench_coldcache_q8row.mojo, bench/bench_coldcache_ternary.mojo, bench/bench_fixedm_kernel_invariance.mojo, bench/bench_mega_gemm.mojo, kernels/test_mega_block.mojo. Tests naming it: 1.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
71:weight GEMMs are `amar_matmul_skinny_q8row`: one wave per weight row over the
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
