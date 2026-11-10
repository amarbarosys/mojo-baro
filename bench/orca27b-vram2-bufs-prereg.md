# OrcaSAQ-27B buffer round 2, preregistration

Question: can smaller prefill staging capacity lower the decode resident footprint below 20.93 GB without reducing generation speed or changing tokens?

Candidate: expose `BARO_CP` as a compile-time override, default 1024; build the same c4b1484 source with `BARO_SEQ_CAP=1`, `BARO_CP=512`, and other runtime env unchanged. Control: the same source and runtime env with default CP=1024.

Predictions from the measured 94-buffer table and source formulas:

- CP-dependent allocations total 500,012,032 B at CP=1024.
- At CP=512, expected reduction is 249,989,120 B (0.250 decimal GB, 238 MiB), from all linear CP buffers plus `pf_ge_d` and `pf_gp_d`.
- Peak VRAM: baseline 20.93 GB -> about 20.68 GB. `img_d` (83,886,080 B), split-K `p_v_d` (254,279,680 B), and MAX's 2% pool (~0.24 GB) remain unchanged.
- Identity: all 20 MTP prompts equal except the two established nondeterministic prompts; their outputs must match branches already reproduced by same-binary controls.
- Performance: median `tok/s_gen` candidate/control ratio >= 1.000; any lower ratio fails the candidate. Compare in one `bench/ab-resident.sh` job with identical runtime env, 64 generated tokens, all 20 prompts, spec+mega on, TMAX=1088, SEQ_CAP=1, pool=2%.

Measurement: use the existing resident A/B harness VRAM sampler, which discovers the DRM card from `mem_info_vram_total`. Log arm definitions and full output under `.work/orca27b-vram2-bufs/`. No additional VRAM logger or decode-path sync.
