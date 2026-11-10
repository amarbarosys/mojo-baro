# Kernel amar_argmax_pos

Source kernels/elementwise.mojo. Registry roles: argmax_k, argmax_d. Default path: in registry.
Callers: serve/registry.mojo, bench/mspec_gaps.mojo, kernels/test_mega_block.mojo. Tests naming it: 1.
ISA, max over instances in 4 receipts: vgpr 19, sgpr 18, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
