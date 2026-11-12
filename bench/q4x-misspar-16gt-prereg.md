# Q4X_MISSPAR=3 A/B on the 16 GT/s link: prereg (2026-10-21, written before the run)
Source table: rows-trace `.work/trace-16gt` (32 tokens, gen 19.22 ms/token). GPU idle 6.63 ms/token (35%). Largest bucket:
after router>50us, 2.35 ms/token, mean gap 124 us, growing ~25 us per SDMA copy inside (3 copies 99 us, 12 copies 313 us):
copies are serial, so spreading them on 3 streams can hide part of it.
- Arms: A `Q4X_INERT=0`, B `Q4X_MISSPAR=3`, engine `.work/q4x-devarg` (4ad5a01, no serve/kernels change since), q4x-ab.sh NG 64, 20 prompts, ABBA.
- Prediction: B/A 0.95 (saves ~1 of the 2.35 ms). Band 0.92 to 0.98. Kill line: B/A > 0.99 or identity not 20/20.
- Validity: report per-run spread; a post hoc rule may annotate, never flip.
