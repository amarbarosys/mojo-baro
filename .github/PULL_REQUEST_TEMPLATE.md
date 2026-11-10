<!-- One PR, one claim. CONTRIBUTING.md, "What a change has to bring", explains every field. -->

**Claim** (one sentence):

**Kind:** correctness | speed | feature | harness | docs

**Check I ran** (command, then its last lines):

```
```

**Identity gate / byte test** (required when a kernel or the engine changes; write `n/a, docs` otherwise):

**Speed only: commit of the frozen prediction** (hash, protocol file, predicted range, falsifier):

**Arm receipt** (card, ROCm, power cap, the engine's echo of each knob):

**UNVERIFIED** (what I did not check, or "nothing"):

- [ ] `tools/ci-checks.sh` exits 0
- [ ] `./run-tests.sh` exits 0 on a gfx1100 card, or I have no such card and said so above
- [ ] no comments or docstrings added to `kernels/matmul*.mojo` or `kernels/elementwise.mojo`
