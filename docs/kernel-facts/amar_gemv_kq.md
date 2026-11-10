# Kernel amar_gemv_kq

Source kernels/gemv_kq.mojo. Registry roles: serve/spark.mojo, bench/gemv-kq-att.mojo, bench/gemv-kq-rows.mojo, kernels/test_gemv_kq.mojo, kernels/test_kq_prefill_gemm.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA, max over instances in 5 receipts: vgpr 80, sgpr 18, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Measured in docs/BASELINE.md:
623:runs them through `amar_gemv_kq` (`kernels/gemv_kq.mojo`, byte test `kernels/test_gemv_kq.mojo`); every GEMV input
Recent commits on the file:
ba3a3d4 fix(mega): kq rms stage H assert, MR guard, mixed-kind kq pack falls back to launches
4661cb3 perf(mega): K-quant rows with 2 superblocks and 2 rows in flight, rms staged in LDS
a931225 perf(spark): split-K QKV GEMV (4 waves per row), Llama 1B decode +2.5%
