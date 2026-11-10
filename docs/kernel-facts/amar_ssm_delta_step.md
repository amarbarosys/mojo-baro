# Kernel amar_ssm_delta_step

Source kernels/ssm.mojo. Registry roles: serve/registry.mojo, bench/bench_launch_floor.mojo, kernels/test_mega_block.mojo, kernels/test_prefill.mojo, kernels/test_ssm_block.mojo, kernels/test_ssm_rows.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA, max over instances in 4 receipts: vgpr 256, sgpr 45, vgpr spills 62, sgpr spills 0, LDS bytes 0, scratch bytes 88.
Spills: yes.
Measured in docs/BASELINE.md:
561:  v_cndmask select trees, spills and flat loads. Open: `amar_ssm_delta_step` spills 1082-1241
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
