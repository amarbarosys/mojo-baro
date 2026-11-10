# Kernel amar_mega_token

Source kernels/mega.mojo. Registry roles: mega_token_k, mega_token_q4_k, mega_token_kq_k. Default path: in registry.
Callers: serve/registry.mojo, serve/engine.mojo, kernels/test_mega_block.mojo. Tests naming it: 1.
ISA, max over instances in 4 receipts: vgpr 256, sgpr 107, vgpr spills 288, sgpr spills 152, LDS bytes 0, scratch bytes 1140.
Spills: yes.
Wavewait capture: spill 0.1% of cycles, barrier 11.3%, total 74553400 cycles.
Measured in docs/BASELINE.md:
401:(`kernels/mega.mojo::amar_mega_token`, G=96 x 512 threads, bounded grid
646:`amar_mega_token[KQ=True]` reads Q4_K / Q6_K rows of an `engine-pack --kq` trunk with `gemv_kq.kq_lane_part`'s
Recent commits on the file:
500a9fc perf(mega): kq z projection runs in the delta phase on idle blocks
ba3a3d4 fix(mega): kq rms stage H assert, MR guard, mixed-kind kq pack falls back to launches
4661cb3 perf(mega): K-quant rows with 2 superblocks and 2 rows in flight, rms staged in LDS
