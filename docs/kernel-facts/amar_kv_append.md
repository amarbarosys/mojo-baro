# Kernel amar_kv_append

Source kernels/attn.mojo. Registry roles: append_k, append_1, append_p. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_attn_block.mojo, kernels/test_certificate.mojo, kernels/test_cpu_layer.mojo, kernels/test_mega_block.mojo. Tests naming it: 4.
ISA, max over instances in 4 receipts: vgpr 3, sgpr 20, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
