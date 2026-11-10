# Kernel amar_gemv_kq_splitk

Source kernels/gemv_kq.mojo. Registry roles: serve/spark.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Recent commits on the file:
ba3a3d4 fix(mega): kq rms stage H assert, MR guard, mixed-kind kq pack falls back to launches
4661cb3 perf(mega): K-quant rows with 2 superblocks and 2 rows in flight, rms staged in LDS
a931225 perf(spark): split-K QKV GEMV (4 waves per row), Llama 1B decode +2.5%
