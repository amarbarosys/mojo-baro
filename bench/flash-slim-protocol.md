# Flash slim, round 1: can a research+skills expert subset keep Flash's output? (prereg)

FROZEN by the commit that adds this file, before any trace or eval run. Changing a bar, a corpus or the split after
a run is a new prereg, not an amendment.

## Question

Qwen3.8-Flash-Next (GSQ-RCO Q2_0, 66.4 GB (two shards, 66,423,878,624 bytes), 48 layers x 512 routed experts, top-10) does not fit in RAM next to the
desktop, so q4x reads expert misses from the shard mmap. If text from one domain (research papers + our skills)
routes to a small subset of experts, a model restricted to that subset could stay resident and keep its output on
that domain. Two numbers decide it: how many experts the domain needs, and how far the restricted model's
predictions move.

## Method

Restriction is a router mask, not a new GGUF: `Q4X_KEEP=path` (lines `il id`) sets the router logit of every other
expert to -inf before softmax and top-10, so the router picks the best 10 kept experts and renormalizes over them.
That is exactly what a pruned file with the same router would compute. No weights are rewritten or requantized.

All runs are batched prefill (`Q4X_PREFILL=1`, `Q4X_SEQ`, 8 sequences, one per category), teacher-forced on the text
itself: every position's prediction is scored against the unmasked model's prediction at the same position. Scoring
runs on the CPU from the final-norm hidden rows (`Q4X_HIDDEN_F32=1`) times `output.weight` (Q5_K, dequantized by
`tools/ggml_quant.mojo`, bit-equal to gguf-py), checked against a numpy oracle on sample rows.

## Data (8 categories, disjoint trace and eval documents)

| category | source | trace docs | eval docs |
|---|---|---|---|
| skills (in-domain) | `~/Brain/Skills/**/SKILL.md`, unique realpaths, sorted | index % 7 == 0 | index % 7 == 3 |
| papers (in-domain) | AiTrainer `research-paper-corpus/chunks-capped.jsonl`, `text` field, papers numbered by first appearance of their `page_id` | paper % 50 == 0 | paper % 50 == 25 |
| notes | `~/Brain/mojo/mojo-baro/2026-*.md`, sorted | index % 7 == 0 | index % 7 == 3 |
| mojo | `git ls-files 'serve/*.mojo'`, sorted | index % 2 == 0 | index % 2 == 1 |
| python | `git ls-files 'tools/*.py'`, sorted | index % 7 == 0 | index % 7 == 3 |
| math | `~/Models/datasets/gsm8k/main/test.jsonl`, question + answer | line % 2 == 0 | line % 2 == 1 |
| prose | `~/Models/quant-lab/wikitext-2-raw/wiki.test.raw`, articles split at ` = ` headings | index % 2 == 0 | index % 2 == 1 |
| docs | AiTrainer `datasets/external/python-3.14-docs-text/`, files sorted | index % 7 == 0 | index % 7 == 3 |

Amendment before any run (2026-10-14): papers were split by chunk line, which puts chunks of one paper on both
sides (about 40 chunks per paper); the split is by paper.

Selected documents are concatenated in order, tokenized with `tools/baro-tokenize` from the Flash GGUF and cut to
1024 tokens per category (trace) and 256 tokens per category (eval). Trace batch 8 x 1024 = 8192 tokens, eval batch
8 x 256 = 2048 tokens. `bench/flash-slim-prompts.sh` builds both and records the sha256 of every id file.

## Arms (one gpu-wait job, one engine process per arm, same binary)

| arm | batch | knobs | output |
|---|---|---|---|
| T trace | trace | `Q4X_EXPERTS` | expert ids per layer per token |
| R reference | eval | none | hidden rows |
| I identity | eval | `Q4X_KEEP` = all 512 experts | hidden rows, must equal R byte for byte |
| M masked | eval | `Q4X_KEEP` = keep set below | hidden rows |

Keep set: per layer, the smallest set of experts (by hit count in the skills + papers trace, ties to the lower id)
that covers 95% of those hits, but at least 10. Computed on the CPU inside the job between arms R and M.

## Bars

1. Identity: I equals R byte for byte. Otherwise the run is void.
2. Kill line (slimming): mean over 48 layers of |keep set| / 512 >= 0.60 means the domain does not concentrate;
   the slim route is DEAD for this domain (arm M still runs, its numbers are recorded).
3. In-domain quality: top-1 agreement of M with R over the 512 skills + papers eval positions >= 97.34%. That is the
   agreement one llama.cpp build of Flash (llama-server -ngl 0) reaches against another (-ngl 99 --cpu-moe) in the
   20x32 gate (`5ffd49a`): the masked model must differ from Flash no more than two builds of the same Flash differ
   from each other. the maintainer's target of 99% is reported beside it.
4. Reported, no bar: mean KL(R || M) per category, out-of-domain agreement per category, expert misses of R vs M
   (`expert misses` line), per-category 95% keep fractions from the trace.

No minimum gain: a pass at any keep fraction under the kill line counts.

## Prediction (frozen)

- Mean keep fraction for 95% in-domain coverage: 0.45 (range 0.35 to 0.60).
- In-domain agreement of M: 95% (range 92% to 97.5%), so bar 3 most likely FAILS at the 95% coverage level; a miss
  means a larger keep level next (97%, 99%), not a drop.
- Out-of-domain agreement lower than in-domain in every category, lowest for mojo and python (85% to 92%).
- Expert misses of M below R.

## Budget

GPU: one job, 4 processes, about 8192 + 3 x 2048 tokens at about 5 ms/token batched prefill plus load, predicted
under 3 minutes. CPU: tokenization, the keep set (seconds) and scoring (two 2048-row arms against a 248,320 x 2560
head, chunked, under 3 GB RAM).
