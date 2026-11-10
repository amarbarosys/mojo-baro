# OrcaSAQ buffer round 2 follow-up preregistration, 2026-10-17

The first matched run used the harness default `SEQ=0`: 20 control prompts ran before 20 candidate prompts. It measured median speed ratio 0.997, below the original 1.000 gate, and p17 identity differed. Before deciding, test the two confounds the first receipt cannot resolve: fixed arm order and p17 same-binary variance.

One GPU-wait job will run:

1. Full 20-prompt CP1024-vs-CP512 A/B with `SEQ=1`, alternating arm order in five-prompt blocks. Keep original runtime flags, 64 generated tokens, TMAX=1088, and pool=2%. Report this result beside the first run, never replace or hide it. The balanced run's paired median must be >=1.000 to clear the speed gate.
2. Same-binary CP1024-vs-CP1024 and CP512-vs-CP512 controls on p08, p09, and p17. Only difference in arm env strings: inert `ORCA_CTRL_COPY=left|right`, which must have no reader in source. These controls establish same-job identity variance; they do not waive a candidate-only branch.

Identity interpretation: p08/p09 have existing control variability; p17 remains unresolved until these controls. VRAM is sampled by the resident harness. No further run after this one to seek a better number.
