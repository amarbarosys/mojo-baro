# Kernel amar_rope_yarn

Source kernels/attn.mojo. Registry roles: rope_q, rope_k, rope_qp, rope_kp. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_attn_block.mojo, kernels/test_certificate.mojo, kernels/test_cpu_layer.mojo, kernels/test_mega_block.mojo. Tests naming it: 4.
ISA, max over instances in 4 receipts: vgpr 11, sgpr 18, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
