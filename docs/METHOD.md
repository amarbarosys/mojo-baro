# Method: how a number in this repo becomes reproducible or refutable

Written for a stranger with an RDNA3 card and no access to this box. If you
read only one section, read "What you get and what you do not" at the end.

## 1. What this is for

Every tok/s, speedup, or identity claim in `docs/BASELINE.md` is meant to
survive a third party rerunning it on their own hardware, without trusting
the people who wrote it. That is the whole claim of this document: not
"trust our numbers" but "check our numbers", with the exact commands to do
that, the rules the numbers were produced under, and two places where the
rules caught the team's own mistakes rather than someone else's.

This is not a promise that our numbers are the fastest possible, or that
they will match on your card. It is a promise that the method producing
them is written down, checkable, and has already been used to find and
retract wrong claims from this same repository (section 3).

## 2. The rules, in short

The full rule set lives in `bench/PROTOCOL-RULES.md`, P1 through P14, cited
by every protocol file in `bench/`. This section names what they are for;
read the file itself for the exact text and the incident each rule is
named after.

- **P1, parameter read-back.** A flag being passed is not evidence it took
  effect. Every value that defines an arm (dtype, KV type, spec width,
  thread and block dims, clock state) is read back from the running
  system's own reporting before a timed run counts. Two of this repo's
  worst numbers were silently-inert parameters that produced clean,
  tight-spread results anyway; spread does not catch this class of bug,
  only read-back does.
- **P4, twenty prompts or it is not a claim.** Any tok/s or acceptance
  number reported as "ours" is the median over the 20-prompt set
  (`bench/mtp-prompts/`), with min-max beside it. A single prompt is an
  instrument receipt or a bug probe; it never appears in a verdict line.
  One race prompt once reported 1.33x llama.cpp; the same code measured
  0.78x on 20 real prompts.
- **P7, the arm file is written by the run, never assembled beside it.**
  Every A/B harness prints the sha256 of the binary that actually executed,
  from inside that execution path, and refuses to run if both arms hash
  equal. An arm file written by a separate step can name a binary that
  never ran.
- **P10 and P11, a gate proves it can fail before its pass counts.** A gate
  that skips, voids, or errors reports FAIL, never an average over the
  arms that survived. And a gate is not trusted until it has been fed a
  known-bad input and shown to catch it; a pass with no demonstrated
  failure mode means the gate ran, not that the artifact is correct.

P2, P3, P5, P6, P8, P9, P12, P13, and P14 cover prediction freezing,
receipt completeness, row-scaling checks at m>1, harness-before-kernel
profiling, proving a code change was actually reached, element-wise
tensor comparison, hypothesis-versus-oracle discipline, rebuilding a
lane's own claim from its committed tree, and setting a gate's pass bar
from a number a known-good path has actually hit. Read
`bench/PROTOCOL-RULES.md` for all fourteen; this document does not restate
them.

## 3. Two negative results

Stated as results, not apologies: both are cases where the method caught
something a less careful measurement would have shipped.

### 3.1 Per-kernel host-synced timing cannot judge a geometry change

What was believed: a kernel-level A/B, timed with a host sync
(`ctx.synchronize()`) immediately before and after the kernel under test,
is a valid way to judge whether a change to that kernel's geometry (a
split factor, `SSM_JSPLIT`) made it faster or slower.

What the measurement showed
(`exchange/2026-09-15-carryover-probe.md`): a gate-2-style host-synced
measurement found the changed (split) kernel 2.3% *faster* in isolation.
An in-stream measurement of the same two arms, with no host sync and no
external tracer (`bench/carryover-stamp.py`, section 4 below), found the
full decode loop 2.8% *slower* in the split arm, at equal shader clock in
both arms (ratio 1.003 to 1.006, measured inside the kernels themselves),
with the entire cost landing in kernels the split never touches: the gate
GEMV immediately after it +3.8%, the down GEMV six kernels later +6.2%,
the rest of the layer +2.2%. The kernel the change touched was faster; the
kernels it did not touch paid for it. A host sync isolates the timed
kernel from the stream its real cost is paid in.

What rule came out of it: no per-kernel host-synced timing may be used to
judge a geometry change on this card. In-stream, no-sync device timing
(section 4.1) is the only instrument this repo trusts for that question.

### 3.2 Small-vocab sampler gates that passed while a real-vocab defect shipped

What was believed: the device sampler's top-p implementation and its host
reference implementation compute the same nucleus, because both port the
same formula, and the gate comparing them (a 64-token synthetic vocabulary,
chi-square distribution test) passing meant the two agreed.

What the measurement showed
(`exchange/2026-09-15-m5-sampler-diagnosis.md`): at the real 248,320-token
vocabulary, the two disagreed. The host reference's top-p cutoff used
`ceil()` on a float64 probability mass computed with unit 1, where the
device's fixed-point formula the host was porting used unit `2^-40` (where
`ceil` is exact). On a peaked real-vocab row, the host's `ceil` rounded the
target up to the next whole unit of mass, most or all of the top-k set,
so the host's own reference was the wrong side, not the device. Every gate
that had exercised top-p ran only at the 64-token synthetic vocabulary,
never at real vocabulary size, so the defect was invisible to the gates
that existed.

What rule came out of it: a distribution gate run only at a small
synthetic vocabulary does not cover the real vocabulary's shape, and a
component's status as "the reference" is not evidence it is correct;
diffing against an independent oracle (here, a third, from-scratch numpy
implementation) is what actually settled which side was wrong. The
affected sampling shape (`temperature > 0`, `top_p < 1`, `min_p <= 0`) is
refused by the server with an explicit error today, rather than served
from a gate that cannot yet confirm it.

## 4. The instruments

### 4.1 `bench/carryover-stamp.py`: in-stream attribution, no host sync, no external tracer

Reads timing from inside the kernel stream itself, using
`llvm.readsteadycounter` (a 100 MHz wall clock) and `llvm.readcyclecounter`
(the 20-bit `SHADER_CYCLES` register) stamped at kernel entry and exit,
accumulated per launch with a native atomic add, with no
`ctx.synchronize()` anywhere in the timed path and no external profiler
attached. It is the instrument that caught the negative result in 3.1; a
host-synced timer could not have.

It works by deriving stamped copies of the engine's sources: for every
kernel named in its `KERNELS` registry and every call site named in its
`SITES` registry, it clones the kernel into a `_st` variant carrying the
timer and rewrites the call site to route through it, then builds that
copy under `.work/carryover/src-<arm>`. Tracked files under `kernels/` and
`serve/` are never touched; every text replacement it makes asserts an
exact match count, so a stale anchor fails loudly instead of silently
patching nothing.

Command:

```
bench/carryover-stamp.py
```

With no arguments this builds both default arms (`C`, the current
checkout unmodified, and `D2`, the current checkout with
`kernels/ssm.mojo` and `serve/registry.mojo` swapped in from a named git
ref and `SSM_JSPLIT` forced to 2) into `.work/carryover/src-C` and
`.work/carryover/src-D2`, ready to build and run under `gpu-wait`. To
build only one arm: `bench/carryover-stamp.py --arm C`. Extending it to a
different kernel or a different call site is a new entry in the
`KERNELS` or `SITES` registry (the module's own docstring explains the
shape of each); the harness, window, and engine plumbing underneath is
already site-count-agnostic.

### 4.2 `tools/gguf-verify.sh`: rebuild from the file, gate identity, print the hardware receipt

The entry point a stranger runs. It rebuilds the inference engine
entirely from source text embedded in the gguf's own `baro.kernel.src.*`
metadata keys (via `tools/gguf-closure.sh`), gates the rebuilt engine's
output tokens against the file's own reference tokens, and prints this
card's identity next to the `baro.hw.*` values the file carries (card
name, driver, ROCm version, power cap, the embedded 20-prompt tok/s
median, config, and protocol reference).

Command (run under `gpu-wait`, see section 6):

```
tools/gguf-verify.sh MODEL-BARO-<sha>.gguf [OUTDIR]
```

Exit 0 means the sources embedded in the file were complete and its
tokens are identical on this card. The tok/s line it prints is always a
receipt for the hardware it ran on, never a pass/fail threshold: a
different card produces a different number, and that is expected, not a
failure.

**Finding, this document (2026-09-15):** running this exact command against
`RegesCore-1.0-35B-UD-Q4_K_S-BARO-8184f7d.gguf` (the qwen35moe bake) fails
on a clean checkout. The closure builds and the engine loads its 21 GB
pack, then exits with `Failed to open file '.work/moe-w3/one.tokens': No
such file or directory`. `tools/gguf-closure.sh`'s qwen35moe branch
defaults `BARO_PROMPT` to that path, which is under gitignored `.work/`
and was never written by the closure itself or carried in the file's
`baro.kernel.src.*` keys; it is leftover local state from whoever last
built the baseline number, not part of the closure. This contradicts
`docs/BASELINE.md`'s claim that the qwen35moe bake's "closure rebuilds the
harness with no path outside the file": for this model class it currently
does not. The dense (Qwythos) bake's closure branch does not default
`BARO_PROMPT` to a `.work/`-relative path the same way; whether it has an
equivalent gap was not tested here. Fixing `tools/gguf-closure.sh` is out
of scope for this lane (no new tooling); this is reported as a finding for
whoever picks up B1 or the closure script next.

## 5. What a third party gets and what they do not

**Identity gates are teacher-forced agreement, never greedy 64-token
equality past roughly 256 token ids.** Two independently-trained
implementations of the same model, run in float32/bf16/int8 at
40-plus layers, diverge under floating-point rounding well before 256
generated tokens even when both are correct; llama.cpp's own f16-KV
configuration fails its own f32 reference at 5 of 7 tested lengths for the
same reason (`bench/PROTOCOL-RULES.md` P14). A gate that demands longer
greedy equality is not a stricter gate, it is a gate that fails correct
implementations, and this repo has already measured the resulting false
failure once (P14's own account). What `tools/gguf-verify.sh` checks is
therefore teacher-forced token agreement against the file's own reference,
not unbounded greedy generation matching some other engine byte for byte.

**A tok/s number on a different card is a receipt for that card, never a
pass or fail.** The embedded `baro.hw.tok_s_gen_20p` value describes the
hardware that produced it. A verification run on a second card that
completes with exit 0 and a different tok/s number has succeeded at what
`gguf-verify.sh` checks (source completeness, identity); it has not
reproduced the original throughput claim, because throughput is a
hardware receipt, not a property of the file. Anyone comparing two cards'
numbers is comparing two separate receipts, not auditing one claim against
itself.

## 6. Running the file itself, and the two modes that keep it honest

`tools/baro` is the entry point, and it has two modes because verification and
convenience pull in opposite directions.

```
tools/baro run      MODEL.gguf [--port N] [--pack DIR] [--no-serve]
tools/baro verify   MODEL.gguf [--append]
tools/baro receipts MODEL.gguf
```

**run** extracts the sources from the file, builds the engine using the
harness the file carries (`baro.run.src.engine.mojo` plus the serve modules it
imports), builds the weight pack from the file's own tensors with the
`engine-pack.py` the file carries, applies the env the file says the model
needs (`baro.run.env`), and serves it over HTTP. Every number it prints is
labelled **self-reported**, because the file supplied the stopwatch. That
label is the whole reason run mode is a separate mode.

**verify** does the opposite and needs the checkout. It takes
`serve/engine.mojo` from git at the file's own `baro.kernel.commit`, compares
it byte for byte with the copy inside the file, and refuses to go on if they
differ: that is how a tampered clock is caught. Then it rebuilds from the
embedded kernel sources, decodes the file's own prompt
(`baro.run.prompt.tokens`) and gates the result against the file's own
reference ids (`baro.run.ref.tokens`), so nothing outside the file is
consulted except the harness, which is exactly the thing that must come from
outside.

**receipts** prints the ledger. Only verify mode may write to it, every record
carries `mode: verified` and the commit its harness was checked against, and
`tools/gguf-receipt.py` refuses a record that carries neither. The ledger
lives in a sidecar JSONL beside the model by default; `--in-file` puts it in
the file's own `baro.hw.receipts` key, which means rewriting the whole
container (21 GB for the MoE bake), so it is opt-in.

What this does not do yet: a pack has no `tokenizer.json` unless one was built
beside it, so a served model whose pack lacks one accepts token ids on
`/v1/completions` and refuses the text and chat endpoints. Converting the
gguf's own tokenizer into that file is not built.
