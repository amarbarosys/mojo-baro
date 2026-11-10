# Kernel amar_gemv_q8

Source kernels/spark_kernels.mojo. Registry roles: serve/spark.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA: not in any receipt, unknown.
Recent commits on the file:
2cf86ed feat(spark): fused decode rope behind BARO_ROPE_FUSE (default off), byte test, prereg frozen
36950cc feat(spark): split decode attention behind BARO_ATT_SPLIT (default off), prereg frozen
04e5131 fix(spark): apply rope_freqs (Llama 3 rope scaling), Llama 1B ids now match llama.cpp 13/21
