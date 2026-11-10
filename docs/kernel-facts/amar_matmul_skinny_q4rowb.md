# Kernel amar_matmul_skinny_q4rowb

Source kernels/matmul_skinny.mojo. Registry roles: serve/registry.mojo, bench/bench_coldcache_mrow.mojo, bench/bench_fixedm_kernel_invariance.mojo, bench/bench_prefill.mojo, bench/bench_prefill_abl.mojo, bench/bench_q4dot.mojo, bench/bench_turbo_rows.mojo, kernels/test_cpu_dot.mojo, kernels/test_cpu_layer.mojo, kernels/test_prefill.mojo, kernels/test_turbo_rows.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Wavewait capture: spill 0.0% of cycles, barrier 0.0%, total 2077044 cycles.
Measured in docs/BASELINE.md:
391:window is row-scaling-bound: `amar_matmul_skinny_q4rowb[2,3]` costs 1.41x / 1.29x its m = 1 self on the FFN shapes (P5 receipt
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
