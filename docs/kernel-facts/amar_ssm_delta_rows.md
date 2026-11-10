# Kernel amar_ssm_delta_rows

Source kernels/ssm.mojo. Registry roles: deltar_p. Default path: in registry.
Callers: serve/registry.mojo, kernels/test_ssm_rows.mojo. Tests naming it: 1.
ISA: not in any receipt, unknown.
Measured in docs/BASELINE.md:
791:  896 rows: `amar_ssm_delta_rows` 65.0 ms (30 launches), `moe_down_q6k_grouped` 38.5 ms (3 launches), trunk `matmul_prefill_lds`
Recent commits on the file:
7f2740a mojo-baro: initial public snapshot
