# Kernel amar_attn_decode

Source kernels/attn.mojo. Registry roles: att_k, att_1. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_attn_block.mojo, kernels/test_attn_gqa.mojo, kernels/test_certificate.mojo, kernels/test_cpu_layer.mojo, kernels/test_mega_block.mojo, kernels/test_prefill.mojo. Tests naming it: 6.
ISA, max over instances in 4 receipts: vgpr 38, sgpr 29, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
