# Kernel amar_matmul_skinny_q4row

Source kernels/matmul_skinny.mojo. Registry roles: bench/bench_coldcache_mrow.mojo, kernels/test_mega_block.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
391:window is row-scaling-bound: `amar_matmul_skinny_q4rowb[2,3]` costs 1.41x / 1.29x its m = 1 self on the FFN shapes (P5 receipt
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
