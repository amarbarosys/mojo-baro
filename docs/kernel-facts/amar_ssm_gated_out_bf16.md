# Kernel amar_ssm_gated_out_bf16

Source kernels/ssm.mojo. Registry roles: gated_k. Default path: in registry.
Callers: serve/registry.mojo, bench/bench_launch_floor.mojo, kernels/test_mega_block.mojo, kernels/test_prefill.mojo. Tests naming it: 2.
ISA, max over instances in 4 receipts: vgpr 15, sgpr 18, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
