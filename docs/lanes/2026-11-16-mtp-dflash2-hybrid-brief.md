# Brief: MTP + DFlash2 hybrid speculation, investigation only (2026-11-16)

Boss pane: a lane pane. Output: `docs/research/2026-11-16-mtp-dflash2-hybrid.md`. When done:
`~/iTools/bin/herd tell a lane pane "DONE mtp-dflash2-hybrid -> docs/research/2026-11-16-mtp-dflash2-hybrid.md | best=<idea> est_tok_s=<n>"`.
No GPU, no code edits, read-only except the output file. Toolscript is mandatory.

## Question (the maintainer)
Can MTP (the target's own 1-2 token heads) and the DFlash2 block-diffusion drafter (8-16 tokens per step) be combined for a larger speedup than either alone, on the nerkyor Qwen3.8-27B, RX 7900 XTX?

## Facts (measured today)
- Ours: plain 44.4 tok/s, MTP k=2 68.7. D1: 9-row verify window = 76.72 tok/s equivalent; break-even accepted tau = 3.2-3.8.
- llama.cpp: MTP 56.7-74.7; DFlash2 12.7-26.5 (ngl52, CPU offload, unfair); DFlash2 accept 32-82% of drafted tokens.
- Decode is bandwidth-bound: 14.7 GB of weights read per target step; one verify of m rows costs about the same as m=1 up to m of about 8.

## Investigate (cite papers and code; arxiv-net / paperbrain-ask, llama.cpp `common/speculative.cpp`)
1. Candidate schemes: (a) MTP extends DFlash2's accepted prefix (bonus token plus MTP heads for 1-2 more); (b) DFlash2 block as a tree, MTP picks branches; (c) MTP as a cheap first-pass filter that truncates the DFlash2 block before verify; (d) whether hidden-tap sharing lets both drafts come from one target pass.
2. For each: expected tokens per target step, extra cost, VRAM (we have ~1-2 GB headroom), the change to the engine (LOC: XS/S/M/L), and the risk.
3. A model: tok/s = (accepted+1) / (T_verify(m) + T_draft). Plug in the numbers above and give the best scheme with its estimate, plus the cheapest falsifier experiment.
