# OrcaSAQ-27B VRAM

At `TMAX=1088`, the 17.205 GB packed model measured 25.65 GB VRAM after first token with default `SEQ_CAP=4`. The excess was primarily four reserved sequence slots. Setting `BARO_SEQ_CAP=1` reduced measured decode VRAM to 20.93 GB, saving 4.72 GB. The 20-prompt identity gate passed: 18 prompts were bit-exact, and the two differing prompts (`p08-sql`, `p09-explain-gpu`) reproduced both output branches on same-binary controls, confirming known run-to-run nondeterminism and zero true regressions.

Allocation instrumentation and the capacity change are committed in `76c5598`. Full measured allocation table, run stages, and identity receipts: [agy report](../exchange/orca27b-vram-report.md). Codex's independent breakdown and stage accounting: [Codex report](../exchange/orca27b-vram-report-codex.md).
