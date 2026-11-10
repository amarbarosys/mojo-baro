# Kernel amar_turbo_rows

Source kernels/turbo_rows.mojo. Registry roles: bench/bench_turbo_rows.mojo, kernels/test_turbo_rows.mojo. Default path: in registry.
Callers: . Tests naming it: 0.
ISA, max over instances in 1 receipts: vgpr 147, sgpr 34, vgpr spills 0, sgpr spills 0, LDS bytes 0, scratch bytes 0.
Spills: none measured.
Wavewait capture: spill 0.0% of cycles, barrier 21.8%, total 1990137 cycles.
Measured in docs/BASELINE.md:
392:`bench/bench_turbo_rows.mojo`); `kernels/turbo_rows.mojo` (`amar_turbo_rows`, the A rows staged once per block in LDS, same fma chain,
Recent commits on the file:
f9fa87a kernels(turbo_rows): bf16 LDS stage only, f32 variant and the pipelined v2 dropped after the
3cf8774 kernels(turbo_rows): m-row Q4_0 GEMV with the A rows staged once per block in LDS, bit-ident
