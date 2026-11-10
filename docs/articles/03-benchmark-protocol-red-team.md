# Our perf gate was gameable four ways; deleting one GPU sync passed it at +63%

I built an automated loop that proposes source edits, benchmarks them
against a champion binary, and lands the ones that measure faster. Before
I red-teamed it, deleting a single `ctx.synchronize()` call — the GPU sync
that happens right before the stopwatch reads out — was enough to make
the engine report **108.6-110.8 tok/s_gen against a real ~67**, pass the
identity check, pass the performance threshold at **+63%**, and clear
every gate stage the candidate could actually reach. No banned function
was called. No banned token appeared in the diff. It read as a legitimate
win until I went looking on purpose.

This article is about two related but different things: the general
rules this repo runs every benchmark under (P1-P6, `bench/PROTOCOL-RULES.md`),
and what happened when I pointed an adversarial audit at the newer,
narrower gate that lets an automated loop decide for itself whether a
candidate change is a win. The first exists because of two real
incidents. The second exists because a gate a human doesn't watch every
run of needs to survive a candidate trying to fool it, not just a
candidate trying to be fast.

Same box as every other article in this series: AMD RX 7900 XTX, gfx1100,
ROCm 7.2, Mojo 1.0.0 / MAX 26.5.0.

## The two incidents that wrote P1

Every protocol in this repo is bound by six numbered rules, and rule one
is blunt: **passing a parameter is not evidence it took effect.** Before
any timed run, every value that defines an arm gets read back from the
running system's own reporting — not the flag you typed, not the request
JSON, not what you remember setting last time.

That rule exists because of two specific, real poisonings, both of which
produced clean, tight-spread numbers that were simply wrong:

- A no-speculation baseline run set `"speculative.n_max": 0` on the
  server. The server accepted the flag and ignored it — the `draft_n`
  field in its own response, which would have shown the truth, was never
  read. The "no-MTP" arm was still speculating the entire time. Caught
  only because someone eventually read the field that was there the
  whole time.
- Our own hipBLASLt shim measured our GEMM kernel at roughly 2x the
  vendor library's speed. The vendor library was never actually
  configured — no cached workspace, no split-K, no cached algorithm
  selection. Fixing three lines took hipBLASLt from 2497 to 5201 GFLOP/s
  and the "2x faster than vendor" claim evaporated with it.

Neither bug was bad math. Both were a parameter that looked set and
wasn't, producing a number nobody had reason to distrust until it was
checked against the instrument's own state, not the command line.

The other rule worth naming here is P4: any decode number is a median
over the full 20-prompt set, competitor measured the same way in the same
session — never a single prompt. This repo's own MTP speculative decode
was once reported at 128 tok/s, 1.17x over llama.cpp, from a single
5-token race prompt whose repetitive tail inflates acceptance to ~94%.
The 20-prompt median for the same code was 82 tok/s, 0.66x — behind, not
ahead. Both numbers were computed correctly. Only one of them describes
the engine.

## A gate that runs unattended needs different defenses

P1-P6 assume a human reads the receipt before believing the number.
The loop-gate scorer doesn't have that luxury by design — it's meant to
run candidates and accept or reject them without me watching each one.
That's a different threat model: not "did I mis-set a flag," but "can
the thing being measured lie to the thing measuring it."

So I red-teamed my own gate. Four hand-built candidates, each a one-line
or few-line edit to the engine's timing code, allowlisted file, zero
words on the existing denylist, run through the real gate on the real
GPU:

| candidate | edit | identity | reported tok/s_gen (real ≈ 67) | verdict |
|---|---|---|---|---|
| shift the end timestamp by 500ms | 1 line | PASS | 148.5 / 114.2 / 132.6 | FAIL — but only on spread |
| **delete the GPU sync before the stopwatch reads** | 1 line | PASS | **108.6 / 108.9 / 110.8** | **PASS, +63%** |
| unset a completion flag | 1 line | PASS | 597,887 / 923,618 / 821,692 | FAIL — but only on spread |
| rewrite the reference-answer file with its own output | 2 lines | PASS (against its own file) | 67.0 (no timing claim) | FAIL — didn't try |

The sync deletion is the one that matters: it's a plausible-looking
change (skip a synchronize call you don't obviously need) that makes the
GPU still-running-in-the-background time get excluded from the
measurement, and it sailed through every check available to it.

## The honest part: it wasn't actually caught by design

Here's the part I'd rather not bury in a caveats section. The sync-deletion
candidate did die — but at a completely unrelated gate stage that checks
for register spills in the compiled kernels, and it died there for the
same 9 spilling kernel variants the *champion binary itself* already has.
That check would have rejected any candidate at all, including a genuinely
faster one, until those pre-existing spills are cleaned up. The gate's
integrity mechanisms — the ones actually meant to catch a lying candidate
— did not stop this one. It got saved by an unrelated, coincidental gate
failure that happens to reject everything right now. That is not a system
I can rely on, and reporting the near-miss as a "catch" would have been
exactly the kind of flattering-but-wrong claim this whole series is
trying not to make.

## What actually got fixed

Four commits, applied and re-verified against the same four candidates
(now all die at the scope check, before ever reaching a GPU run):

1. **Ban timing state, not just timing calls.** The original denylist
   blocked functions like the perf-counter call and the print statement,
   but not the *variables* those functions fed — `t0`, `dt`,
   `decode_s`, the sync calls, and the fixture file paths, including
   ones built by string concatenation rather than written literally
   (candidate 3's reference-file rewrite got past the old denylist by
   spelling the filename as `".work/engine-pack/ref-" + "tokens-64.txt"`).
2. **Snapshot the reference before any candidate runs**, tamper-check it
   after every run, restore it if it changed, and check identity on
   *every* timed run instead of only the first.
3. **A second, hidden fixture**, generated fresh from the iteration's own
   pristine sources, so a candidate can't special-case the one prompt it
   knows about.
4. **The rule amendment recorded in the protocol**, not silently changed
   underneath it.

## What's still open

A denylist is a list of things I thought of. Two holes remain, ranked by
how cheap they are to walk through:

**The cheapest one:** move the real decode loop to run *before* the
stopwatch starts, then neuter the timed loop into a no-op that leaves
output correct (the token ids are already computed) and decode time near
zero. Nothing on the current denylist catches this — the only thing in
its way is the spread check, and a candidate that leaves a steady, cheap
per-token stub in the timed loop has a tight, convincing spread. There is
currently no external cross-check and no ceiling in the rule that this
attack has to clear.

**The one I actually watched happen:** after applying a further round of
hardening the same night — moving the stopwatch code itself out of any
file a candidate is allowed to touch, and adding a wall-clock
plausibility term that has to move by roughly half of any claimed decode
saving — I ran six real iterations. Zero candidates survived. But one of
them, a comment-only edit with no semantic content, still reached +1.7%
on the performance threshold. Not a cheat: the champion's median inside
that gate run and the champion's median from an earlier session
genuinely differed by more than the acceptance step, so a no-op looked
like a win by drift, not by design. The fix (interleave the champion and
the candidate in the same session, so a no-op can't get lucky against a
stale number) is written up and not yet applied — that's a real hole in
production today, not a historical one.

## The falsifier

`exchange/scorer-integrity-report.md` has every candidate's exact diff,
every stage's exit reason, and the full commit list for the fixes;
`tools/loop-gate.sh` is the gate itself, runnable against any candidate
you want to try against it. If you can get a no-op or a lying edit past
the current hardening, that's a real finding and I want the diff.
