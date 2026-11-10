# Kernel amar_attn_prefill_wmma

Source kernels/attn.mojo. Registry roles: attpw_k. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_prefill.mojo. Tests naming it: 1.
ISA, max over instances in 4 receipts: vgpr 219, sgpr 46, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
