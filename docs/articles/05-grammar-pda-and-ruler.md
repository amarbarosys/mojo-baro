# Tool calls that always parse in 11 microseconds — and "effective context" turns out to be a per-task number, not a per-model one

Two unrelated pieces of work landed the same day, and they share one
lesson: a single headline number hides the case that breaks it. A
byte-level grammar engine that constrains decoding to always-valid JSON
has a median mask-fill cost of **11.48 microseconds** — 43x under its own
500us budget. Its worst case is **74.2 milliseconds**, 2400x slower than
the median, and that's not noise, it's a specific and explainable case.
Separately, this model's "effective context length" isn't one number
either: three of four RULER task families hold to 32k tokens, and the
fourth collapses to 46% accuracy at that length while still being
completely fine at 16k. Both numbers are correct. Neither is the whole
answer, and reporting only the median or only the aggregate would have
been a lie by omission.

The grammar engine is CPU-only (7800X3D), no GPU involved. The RULER run
is on the same box as the rest of this series: AMD RX 7900 XTX, gfx1100,
ROCm 7.2, via llama.cpp as noted below.

## Constrained decoding: a byte-level PDA, not a general parser

Tool-call JSON is unambiguous by construction — no left recursion, no
recursive `$ref`, no grammar where the same prefix could mean two
different things. That means the constrained-decoding engine doesn't need
the general Earley-style parser some grammar libraries carry; a JSON
schema compiles directly into a set of byte-level DFAs linked by
call/return transitions over an explicit stack — the same structural
shape XGrammar and llguidance use, minus the layer neither of them needs
for plain JSON either. At runtime, `Matcher.fill_mask` walks the current
automaton state against a trie built once over the model's real
248,320-token vocabulary (not the ~151k I'd guessed going in — an
extended Qwen3 vocab) and returns a bitmask: every token that would still
be valid JSON if sampled next.

Before writing a line of the mask-fill path I wrote down a prediction —
partly to make sure I was allowed to claim victory only if the number
actually beat it: llguidance's own published numbers put a comparable
cache-hit mask fill around 50 microseconds on their hardware and vocab
size. Measured median here: **11.48 us**. Beating a prediction you wrote
down before running the code is a different, stronger claim than beating
one you found convenient after seeing the number — and I want to be
honest that this is the rare case where the number came in better than I
expected, not worse.

## Correctness came with a real bug, not a clean pass

Before any timing number matters, the thing has to produce valid JSON,
always. The test harness generates 100 samples per schema across 32
schemas — 3,200 total — by sampling uniformly from the live token mask
until the automaton terminates, then validates every sample against the
schema with Python's own `jsonschema` library as an independent oracle. A
prior run of this exact suite, inherited mid-lane, showed **241 of 3,200
samples failing** — one schema (`const` combined with `enum`) was
silently dropping a required field, always emitting `{"op":"add"}` no
matter what the sampled value should have been. The bug was in how the
compiler handled a suffix of all-optional properties; fixed, the full
corpus re-ran clean: **3,200 / 3,200, zero failures.** I'm including the
broken number here because a grammar engine that produces syntactically
valid-but-semantically-wrong JSON on one schema in 32 is a worse failure
mode than one that crashes — it looks like it's working.

## The tail is real, and it's explainable

The 500-microsecond budget is stated as a median target, and the median
clears it by 43x. The worst observed sample took **74.2 milliseconds** —
p90 sits at 27.85ms, p99 at 46.56ms. That's not the same claim as "the
median is fine," and I don't want the 11.48us number to imply a
constant-time guarantee it doesn't have.

The cause is specific: the engine caches a token-validity bitset per
`(rule, state)` pair the first time it's visited, because computing that
bitset means walking the ~500,000-node token trie once. Every subsequent
visit to the same state is just a bitset OR — cheap, which is where the
11.48us median comes from. But the *first* visit to any given state pays
the full trie walk, and on a long enough generation with a rich enough
schema, a state you haven't touched yet can show up late. The tail isn't
a bug or an unbounded blowup; it's the one-time cost of populating a
cache, visible because the benchmark measures 1000 individual token
steps rather than reporting only the amortized total. Whether that
one-time cost is acceptable depends entirely on where in a real
generation it lands — which this measurement doesn't answer, and I'm not
going to pretend it does.

## RULER: what "effective context" measures, and what it doesn't yet

Separately, before shipping a quantized KV-cache format for long context,
the plan is to require its effective context length to match the
unquantized baseline's — not just on average, but per task family. Before
that comparison can mean anything, the baseline itself has to be measured
on this specific model, because published effective-length numbers are
for other models entirely.

**What this measurement is, precisely:** RULER's task generators (needle-
in-a-haystack retrieval, multi-key retrieval, variable tracking, common-
word extraction), run against this model through llama.cpp's own Q4_0
build as the reference harness, at 4k/8k/16k/32k tokens, 25 examples per
task. This is *not* a measurement of this engine's own long-context path
— that path is a separate, and currently much weaker, story (this
series' first article already said prefill here is 3-5x behind llama.cpp
at long context). This run establishes the anchor: what does bf16 KV
actually get, on this model, on this hardware, before the quantized
format gets graded against it. That second half — the actual comparison
— hasn't run yet; it's flagged as not-yet-done below, not implied.

The results:

| task | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| niah_single (needle retrieval) | 88.0 | 68.0 | 96.0 | 100.0 |
| niah_multikey | 96.0 | 96.0 | 100.0 | 88.0 |
| variable tracking | 100.0 | 100.0 | 100.0 | 100.0 |
| common-word extraction | 93.6 | 100.0 | 88.4 | **46.0** |

Three of four tasks hold at 32k — by the effective-length convention
here (largest size scoring at least 85% of that task's own 4k score),
that's an effective length of 32,768 tokens for retrieval and tracking.
Common-word extraction, an aggregation task rather than a single-needle
lookup, is a completely different story: 88.4 at 16k, then a collapse to
46.0 at 32k. Its effective length is **16,384** — half the other three.
A single "this model's context is effective to 32k" headline would have
been true for 75% of the tasks measured and wrong for the one that
actually stresses aggregation over the full window. "Effective context"
isn't a property of the model alone; it's a property of the model and
the task together, and averaging across tasks would have hidden exactly
the number that matters for anyone using this model for a task that
looks like counting or aggregating across a long document.

One more honest note: running this at N=25 through 32k already took
long enough on this hardware that continuing to 64k and 128k, or running
the quantized-KV arm this whole exercise exists to gate, was deferred —
not run, not estimated to have run, genuinely not done yet. The
predicted cost for the full sweep this protocol originally scoped was
itself a finding: north of five hours before any 64k data point, on this
hardware, at this sample count. That's a real constraint on how often
this kind of gate can run, and it's the reason the actual KV-format
comparison — the thing this measurement exists to enable — is still
ahead of this article, not in it.

## The falsifier

`grammar/test_timing.mojo` reproduces the mask-fill distribution exactly
(same corpus schema, same 1000-sample-after-50-warmup protocol);
`grammar/test_corpus.mojo` reproduces the 3,200-sample correctness run.
`bench/ruler-protocol.md` has the frozen prediction, the exact task
generator sources, and the commands to reproduce the arm-A table above —
and to run the arm-B comparison this piece deliberately doesn't claim.
