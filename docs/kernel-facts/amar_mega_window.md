# Kernel amar_mega_window

Source kernels/mega.mojo. Registry roles: mega_win_k, mega_win_q4_k. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_mega_block.mojo. Tests naming it: 1.
ISA, max over instances in 4 receipts: vgpr 192, sgpr 107, vgpr spills 166, sgpr spills 150, LDS bytes 0, scratch bytes 604.
Spills: yes.
Recent commits on the file:
500a9fc perf(mega): kq z projection runs in the delta phase on idle blocks
ba3a3d4 fix(mega): kq rms stage H assert, MR guard, mixed-kind kq pack falls back to launches
4661cb3 perf(mega): K-quant rows with 2 superblocks and 2 rows in flight, rms staged in LDS
