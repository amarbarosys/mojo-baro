# Served-path timing of the MoE model (A0.1)

Preregistered 2026-09-15, before the first run, at commit `aa9f263`
(`docs/NEXT-PLAN.md` A0, "Served-path timing of the MoE model after R1 to R3;
the 93.46 is the one-shot engine"). Binds `bench/PROTOCOL-RULES.md`.

## Question

`bench/moe-perf-protocol.md` R3 landed the MoE decode at a 20-prompt median of
**93.46 tok/s_gen**, measured one process per prompt (`BARO_PROMPT`, arm B of
`bench/ab-prompts.sh`). Every user meets the model through `baro-serve`, not
through that path. What does the served path measure, and does the resident
engine decode at the same rate?

Two numbers, because they answer different questions:

- **served tok/s_gen**: the response's `timings.tok_s`, which the engine
  computes as `(n-1)/decode_s` inside the same decode loop the one-shot line
  reports. Comparable to 93.46 by construction.
- **served wall-clock tok/s**: `(n-1)/` the client's own round-trip time for
  the whole HTTP request. Includes prefill, queueing, the stdin/stdout line
  protocol, JSON and HTTP. This is what a user actually waits for, and it has
  never been measured on this model.

## Arms

One binary, two paths, one stint. `.work/moe-served/engine`, built at
`aa9f263` from a clean `git archive HEAD` tree (the working tree carries the
coordinator's in-flight C3 sampler edits in `kernels/sample.mojo` and
`serve/sample_ref.mojo`, which must not enter a timed binary), with
`-D BARO_MODEL=qwen35moe -I <tree> -I <tree>/kernels`.

| arm | path | env |
|---|---|---|
| A | one process per prompt, `BARO_PROMPT=<tokens>` | `BARO_SPEC=0 BARO_MEGA=0 BARO_PACK=.work/moe-w1/pack` |
| B | resident behind `baro-serve`, `POST /v1/completions` with the prompt as token ids, `max_tokens` 64, `spec` false | same, inherited by the spawned engine |

The arms are not interleaved per prompt the way `bench/ab-prompts.sh`
interleaves two binaries: the pack is 21 GB, so a one-shot process cannot
allocate while the server holds the card. Arm A runs to completion, then the
server starts and arm B runs. `bench/clock-probe.sh` wraps both.

`bench/ab-prompts.sh`'s same-sha refusal (champion against itself) does not
apply: one binary on two paths is the experiment. The distinguishing parameter
is the path, and it is read back from `/health` and from the server's own
stderr, not from the command line.

## P1 read-back, before the numbers

- Arm A, per run, from the engine's own stdout: `BARO_MEGA:`, `BARO_SPEC:`,
  `spec k:`, `prompt tokens:`, `TMAX:`, `pack loaded in`, `tokens: 64`.
- Arm B, from `GET /health` on the running server: `pack`, `limits.tmax`,
  `limits.spec_k`, `limits.mrows`, `tokenizer`. Plus the same engine parameter
  echo, forwarded to the server's stderr as log lines (`serve/PROTOCOL.md`).
- Both: engine sha256, `baro-serve` sha256, power cap and `OD_VDDGFX_OFFSET`
  read from sysfs, sclk from `bench/clock-probe.sh`, commit and the working
  tree's dirty list.
- Arm B's per-request settings are read back from the response itself:
  `timings.n`, `timings.finish`, and the absence of `drafted`/`accepted`
  (which the engine prints only under spec).

## Frozen predictions

1. **Identity 20/20.** The served `choices[0].tokens` equal the one-shot
   `GENERATED:` ids on every prompt. Decode is greedy on both paths and the
   sampler fields are not acted on by the engine yet
   (`serve/src/protocol.rs:11`). A single FAIL is a defect in the served path,
   reported as such, not a timing footnote.
2. **Arm A reproduces R3.** Arm A's median is within 2% of 93.46. Outside that
   band the stint may not be compared to the R3 receipt, and only the internal
   A versus B ratio stands. (The binary is not byte-identical to
   `.work/moe-perf/engine-r3`: `8184f7d` vendored latentos and touched
   `serve/`, so this is a real re-measurement, not a copy.)
3. **Served tok/s_gen within 2% of arm A**, i.e. ratio in 0.98 to 1.02. Both
   are `(n-1)/decode_s` from the same loop. Below 0.95 means the resident path
   costs decode time (per-token JSON lines on stdout are the first suspect) and
   is a finding to trace, not a number to publish.
4. **Served wall-clock median between 65 and 85 tok/s**, from
   `prefill_s` about 0.15 s plus 0.68 s of decode plus a few ms of HTTP, on a
   13 to 59 token prompt generating 64 tokens. Under 60 means the server adds
   more than the arithmetic allows and the gap is traced before it is reported.

## Budget

Under 10 GPU minutes (GPU rule, `docs/NEXT-PLAN.md`): arm A is 20 runs of
about 2.5 s including a 1.5 s pack load, arm B is one 10 s load plus 20
requests of about 0.9 s. Actual minutes are reported with the result.

## Result (2026-09-15): all four predictions held

`gpu-wait run --vram 23 -- bench/clock-probe.sh bench/served-prompts.sh
.work/moe-served/engine .work/moe-w1/pack .work/moe-served/run
"BARO_SPEC=0 BARO_MEGA=0"`, receipts in `.work/moe-served/run/`.
**GPU held 2.73 minutes** (163.6 s, 528 clock samples), inside the 10-minute
rule.

Read-back, before the numbers (P1):

- engine sha256 `68bc4eb42e6793d2`, `baro-serve` sha256 `84f30413e60f806c`,
  commit `4209cc4`, tree carrying only untracked files (`git status` line in
  `.work/moe-served/run/arm.txt`). The binary was built from a clean
  `git archive` of `aa9f263`; `4209cc4` (the C3 sampler round) landed while
  arm A was running and does not enter this binary. It would not change it
  either: the MoE profile decodes greedily and never calls
  `amar_sample_row`.
- `GET /health`: `{"limits":{"kmax":8,"mrows":8,"spec_k":2,"tmax":1088},
  "pack":".work/moe-w1/pack","pool":[0],"queue":0,"status":"ok",
  "tokenizer":false}`. `tokenizer:false` is why the prompts go in as token
  ids on `/v1/completions`: the MoE pack carries no `tokenizer.json`, and
  posting ids is also what makes the two arms the same computation.
- per one-shot run: `BARO_MEGA: False`, `BARO_SPEC: False`, `spec k: 2`,
  `prompt tokens: N`, `TMAX: 1088`, `tokens: 64`.
- per served response: `timings.finish = "length"`, `drafted`/`accepted`
  null (no spec), `prefill_rows`, `cached`, `restore_s`.
- power cap 290 W, `OD_VDDGFX_OFFSET` -100 mV, sclk min/med/max
  141/3270/3318 MHz, junction max 75 C.

| | one-shot | served (`timings.tok_s_gen`) | served wall clock |
|---|---|---|---|
| 20-prompt median tok/s_gen | **93.52** | **92.89** | **74.74** |
| spread | 7.6% | 2.7% | |
| ratio to one-shot | 1.000 | 0.993 | 0.799 |

Identity 20/20 PASS: the served `choices[0].tokens` equal the one-shot
`GENERATED:` ids on every prompt (prediction 1).

Arm A reproduces R3 to 0.06% (93.52 against 93.46, prediction 2), so the
stint is comparable to the R3 receipt. Its 7.6% spread is wider than R3's
1.8% and comes from two prompts only (p11 88.32, p13 86.97); every other
one-shot row is inside 93.14 to 94.11, and the resident arm, which pays no
per-run process start, holds 2.7%. Cold-process variance, not a decode
effect: p11 and p13 are the two prompts whose served rows are 92.84 and
93.02, in line with the rest.

Served decode is 0.7% below one-shot (prediction 3, band 0.98 to 1.02), so
the resident path costs nothing measurable per token: the per-token
`{"id":N,"tok":T}` line on the engine's stdout does not show up against a
10.7 ms token.

The wall-clock number is the new one, and it is what a user waits for:
**74.74 tok/s over the whole HTTP round trip** (prediction 4, band 65 to
85), 0.80x of the decode-only figure. Served medians per request: prefill
159.3 ms, decode 0.678 s, wall 0.843 s, so **5.5 ms** is everything else
(HTTP, JSON, the stdin/stdout line protocol, queueing, `restore_s` 1.3 ms).
The gap between 92.89 and 74.74 is prefill, not serving overhead: a 64-token
completion pays a 159 ms prefill on a 13 to 59 token prompt, which is 19% of
the request. p17 (59 prompt tokens) is the slowest wall-clock row at 49.81
and the cheapest prefill is p13's, exactly as the arithmetic predicts.

Harness note: the first summary pass voided all 20 rows on two parser bugs of
mine (`tok/s_gen:` sits mid-line in the engine's output, not at line start;
and the HTTP layer renames the engine's `tok_s` to `tok_s_gen` on the way out,
`serve/src/main.rs:373`, so it matches the one-shot line's name). Both were
fixed against the run's own artifacts with no second GPU stint, which is why
the summary is `bench/served-summary.py`, a separate program from the harness
that produces the artifacts. P10 did its job here, loudly: 20 VOID rows
reported FAIL and exited non-zero instead of averaging over survivors.
