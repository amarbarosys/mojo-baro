# Kernel amar_sample_row

Source kernels/sample.mojo. Registry roles: sample_row_k, sample_row_1. Default path: in registry.
Callers: serve/registry.mojo, serve/engine.mojo, serve/spark.mojo, bench/sample-spark-device.mojo, kernels/test_sample.mojo, kernels/test_sample_device.mojo, kernels/test_sample_mask.mojo, kernels/test_sample_pen.mojo. Tests naming it: 4.
ISA, max over instances in 4 receipts: vgpr 101, sgpr 76, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Measured in docs/BASELINE.md:
417:the launch-path head plus `amar_sample_row` after it; T<=0 is byte-identical
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
