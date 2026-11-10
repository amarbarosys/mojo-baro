# Carry-over from the tinygrad fork

Knowledge, by Brain path:
- `~/Brain/tinygrad/2026-10-23-amd-gemm-rating.md`: GEMM rating, hang hunt, prefill hang log comparison.
- `~/Brain/tinygrad/verdicts.md`: checked diagnoses (HDP flush, ruled-out causes).
- `~/Brain/OS/2026-10-26-uv-100-hang-ab.md`: undervolt is not the hang cause.

Facts a KFD backend must honour from day one:
- Host writes to VRAM over the BAR need an HDP flush before the CP reads them, else the CP reads stale words (no flush 12/13 hung, flush 0/13).
- Compute ring rptr/wptr count dwords; SDMA rptr counts bytes. KFD rings are uncached host memory.
- A hang report must print every queue's put/wptr/rptr and decode the waits at rptr (compare func and mask from the packet; skip register polls, they are not host addresses).

Code: the Mojo GEMM (`kernels/gemm/`), the hang tools (`ref/tinygrad-hang/`). Local tinygrad checkout `~/Projects/AMD/tinygrad` stays as reference; our branches are kept as `refs/keep/fork-*` there.
