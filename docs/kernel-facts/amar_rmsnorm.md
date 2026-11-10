# Kernel amar_rmsnorm

Source kernels/elementwise.mojo. Registry roles: rms_m, rms_h2. Default path: in registry.
Callers: serve/registry.mojo, serve/spark.mojo, kernels/test_attn_block.mojo, kernels/test_elementwise.mojo, kernels/test_mega_block.mojo, kernels/test_ssm_block.mojo. Tests naming it: 4.
ISA, max over instances in 4 receipts: vgpr 12, sgpr 18, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
