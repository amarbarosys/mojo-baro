# A3 precondition: MAX VRAM reservation, receipt link (2026-09-17)

The first A3 measurement is already answered by the COMFY lane. Do not repeat
that fixed-reservation arm. The dense q4 result and its source receipt are
recorded in `exchange/lane-A3-report.md`.

## Imported receipt

- Source: `exchange/lane-COMFY-report.md`, item 0.
- Spark-X2.5-4B q8, 5.32 GB pack: 21.75 GB held, 0.77 GB baseline.
- Dense Q4, 6.72 GB pack: same approximately 22 GB reservation class.
- Verdict: reservation is fixed per MAX engine process, not per model pack.
- A3 N=4 budget therefore uses one engine's existing reservation minus the
  trunk; it does not multiply the process reservation by tracked sequences.

## Optional follow-up only

If A3(b) needs a per-sequence KV growth number, run a separate arm with one
engine and 1, 2, and 4 live sequences. Read back `gpu-wait gpu --json` and
`rocm-smi --showmeminfo vram` at each count, then write raw logs under
`.work/a3-vram-kv/`. This does not re-measure item 0.
