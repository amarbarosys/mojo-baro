# Innovation ideas for mojo-baro (2026-11-16)

Grounded in what exists: megakernel, latent hidden-state handoff (E8), self-describing GGUF, KV int8, teacher-forced
gates, gpu-wait, HQ proxy. "No one did" = belief, not yet a prior-art check.

## Performance
1. Per-GPU kernel autotune stored in the GGUF: first run on a new card tunes once, writes the result back.
2. Self-speculative decode: draft from layer N/4 of the same model (hidden states already read out of the megakernel).
3. VRAM-elastic KV: demote old pages q8 -> q4 -> host by measured attention mass, not age; sliding layers exempt.
4. Prefix-cache sharing across agents: hash shared system-prompt KV pages once behind the HQ proxy.

## Quality
5. Live confidence per token: stream entropy + top-2 margin; proxy flags guessed spans for targeted re-checks.
6. Self-check on load: GGUF carries golden teacher-forced logits (64 tokens); refuse or warn below the agreement bar.
7. Hidden-state steering vectors applied at a chosen layer per request, no fine-tune.

## Agent experience
8. Latent handoff between agents (E8 productized): pass hidden state, not text, within a model family.
9. Fork and rewind: copy-on-write KV pages, branch at token K, keep the better plan.
10. Grammar-constrained tool calls compiled to a GPU mask kernel, no CPU round trip per token.
11. Budget-aware serve: request states ms or joules; engine picks quant, draft depth, context trim.

## User experience and creative
12. `baro why`: attention-mass trace from a token back to the context spans that drove it.
13. `baro info model.gguf`: architecture, kernels, measured speed on this card, golden gate, provenance hash.
14. Live hot-swap of LoRA or steering vectors mid-stream, KV kept.

## Pick
Start #6 (S/M, reuses gates + self-describing GGUF) and #4 (largest win for the agent fleet). Moat: #8, #9.
Plan for #8: `docs/plans/2026-11-16-latent-handoff-product.md`.
