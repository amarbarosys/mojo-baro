# Kernel amar_matmul_prefill_kq

Source kernels/matmul_prefill_kq.mojo. Registry roles: bench/bench_kq_prefill.mojo, kernels/test_kq_prefill_gemm.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
693:Run 9's f16 scratch round trip is gone: `amar_matmul_prefill_kq` (`kernels/matmul_prefill_kq.mojo`) decodes Q4_K / Q6_K
794:- **kq dense prefill** (Ornith Q4_K_M, 1023 rows): prefill_s 0.252 s, `amar_matmul_prefill_kq` 139 of 165 ms of prefill kernel
Recent commits on the file:
21f76fa perf(kq-prefill): hoist the K-quant header loads per superblock, unroll K steps per kind
8d60892 feat(kq-prefill): K-quant decode inside the LDS GEMM's B staging, behind BARO_KQ_MMQ
