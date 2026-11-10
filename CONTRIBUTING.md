# Contributing

The most useful contribution to this repo is a benchmark receipt from a card we
do not have.

Every performance number here was measured on one AMD RX 7900 XTX. The kernels
target RDNA3 (`gfx1100`) and are tuned against that card's 96 compute units,
96 MB Infinity Cache and 960 GB/s of bandwidth. Whether any of it holds on an
RX 7900 XT, a 7800 XT, a W7900, or an RDNA4 card is unknown, not "probably
fine", genuinely unmeasured.

## Send a hardware report

Step-by-step guide per AMD family, with what to expect on your card:
[docs/amd-family.md](docs/amd-family.md). The short version:

Needs an AMD GPU, ROCm, and about fifteen minutes. **No model weights are
required**: the fp16 GEMM benchmark is self-contained.

```sh
git clone https://github.com/amarbaro/mojo-baro && cd mojo-baro
uv sync                    # repo-local .venv, pinned Mojo/MAX; nothing machine-wide
./bench/report.sh --check  # preflight: GPU, ROCm, hipBLASLt, toolchain
./bench/report.sh          # the real run
```

It prints a markdown block and writes `results/report-<gfx>-<card>-<commit>.json`.
Open a [hardware report issue](../../issues/new?template=hardware-report.yml),
paste the block, attach the JSON.

`./bench/report.sh --quick` runs three sizes instead of ten if you are short on
time. Prefer the full run when you can: the interesting disagreements have so
far been at the small end, where filling the GPU is the constraint.

**File the issue even if it fails.** A build error on gfx1201, a card that
throttles, a hipBLASLt version that behaves differently: those are results. The
script writes a receipt marked `"valid": false` rather than nothing, and says
why.

## What the script protects

Two ways to produce a number that looks fine and is not, both of which have
bitten this repo:

- **Short warm-up.** The same binary reads 66k GFLOP/s at 4096³ after one second
  and 91k after ten, purely because clocks have not settled. The benches warm for
  10 s and record `warmup_s`; the receipt is marked invalid below that floor.
- **An untuned baseline.** An early version of `docs/BASELINE.md` claimed ~2x
  over hipBLASLt. It was measuring a badly-configured vendor call. The shim now
  sets per-call splitK/wgm, and the receipt records the algorithm hipBLASLt
  actually chose.

Numbers are not quoted anywhere in this repo unless they can be traced to a
commit, a warm-up and a correctness check. That is why the script refuses to
emit a clean-looking receipt it cannot stand behind.

## Other benches on unfamiliar hardware

`bench/wmma-peak.sh` measures the WMMA issue-rate roofline. The grid is sized to
fill the card, so the compute-unit count is arm-defining; the script reads it
from the running GPU rather than trusting the 96 hardcoded in the source for
this box's XTX. Pass a count explicitly to override.

The rest of the benches need model weights, which are not distributed here, so
`bench/report.sh` and `bench/wmma-peak.sh` are the two that will run on a fresh
clone.

## Where help is wanted

Ordered by how much a second pair of hands would move things. Each item names the check that
decides it, so you know when you are done.

| area | what is open | decided by |
|---|---|---|
| Other AMD cards | every number is from one RX 7900 XTX | a hardware report (above) |
| Long-context MoE prefill | chunk attention uses the exact decode kernel: 32k takes 133 s, the faster WMMA kernel reorders sums and fails identity (97.92% agreement against a 99% bar) | `bench/moe-prefill-identity.sh` 23/23 and a byte test against `amar_attn_decode` |
| MoE persistent kernel, round R6.3 | kernel landed, the preregistered A/B has no numbers | `bench/moe-persist-protocol.md`, R6.3 procedure |
| Quality roster | Ornith, Qwythos-v2 and the Spark perplexity rows were never run | `tools/baro eval-all` |
| Spark JSON schema | implemented, no live gate | a gate in the shape of `bench/moe-prefill-identity.sh` |
| Models | a GGUF that fails `tools/baro` import | the import receipt the tool prints |

## What a change has to bring

One pull request, one claim. The template in `.github/PULL_REQUEST_TEMPLATE.md` asks for exactly
this, and a PR that leaves a field empty gets asked for it before anyone reads the diff:

1. **Claim.** One sentence: what is true after this change that was not before.
2. **Kind.** `correctness` (output changes or a bug), `speed` (same output, faster), `feature`,
   `harness` (gates, benches, tools) or `docs`.
3. **Check.** The command you ran, and its last lines. For kernels: a byte test against the kernel
   you replace or extend (`kernels/test_moe_rows.mojo` is the model: same inputs, compare bytes,
   fail when the reference output is all zero). For anything the engine runs: the identity gate.
   No GPU? Say so; `tools/ci-checks.sh` and `bench/preflight.sh` run on a CPU and we run the rest.
   Short paths that keep the full bar: `bench/gate-scope.sh` names the gates a change needs (none,
   cpu, full; frozen map `bench/gate-scope.tsv`); preflight rebuilds only the builds whose import
   closure changed and leaves the engines at `.work/preflight/engine-{dense,moe}`; `bench/a2-gate.sh`
   stops at the first prompt under its bar (`FAILFAST=0` for the full table); `bench/ab-resident.sh
   SEQ=1` stops an A/B once the paired CI excludes 1.00 (P4 amendment).
   `./run-tests.sh` skips a test whose binary sha256 and inputs already passed (`CACHED PASS`);
   run `RUN_ALL=1 ./run-tests.sh` before a merge. `tools/ci-checks.sh` caches its slow latent steps
   unless a non-docs file changed (`CI_CACHE=0` reruns them).
   `bench/a2-gate.sh` is resumable: prompts are independent (each loads its set's saved doc state)
   and each result is recorded as it finishes; kill the job any time, rerun the same command.
4. **For `speed` only: the prediction, committed before the run.** Question, instrument, predicted
   range with the mechanism, the result that would falsify it, the adoption rule. Put it in the
   protocol file of the area (`bench/*-protocol.md`) as its own commit, then run, then record the
   verdict in the same file, misses as loudly as hits. A speed PR without the earlier commit is
   reviewed as an experiment, not as a result.
5. **Arm receipt.** Card, ROCm version, power cap, the engine's own echo of every knob that
   defines the arm (`BARO_PREFILL:`, `tmax` in the ready line, ...). A flag on the command line is
   not evidence that it took effect.
6. **What you did not check.** Write `UNVERIFIED: <what>` rather than leaving it out.

Rules a reviewer will hold you to: `bench/PROTOCOL-RULES.md` P1 to P20. The short version: read
parameters back, rotate buffers past the 96 MB cache, the reference arm is an arm too, a gate the
candidate can write is not a gate, failures are loud.

## Code changes

Read `bench/PROTOCOL-RULES.md` first, and `docs/BASELINE.md` for current truth.
Performance claims follow the preregistration flow: the question, the
instrument, the predicted range and the condition that would falsify it are
committed **before** the run, and the result is recorded whether or not it
agreed. Missed predictions stay in the file: three of the four cold-cache
rounds falsified their own predictions, and that history is the point.

Kernel files (`kernels/matmul*.mojo`, `kernels/elementwise.mojo`) carry no
comments or docstrings by convention; rationale goes in commit messages and
`docs/`.

Correctness gates before anything else: `./run-tests.sh` builds the shim and
runs the parity tests, and `tools/kernel-census.mojo` fails if a kernel is
unreachable from the engine, a bench or a test.

`tools/ci-checks.sh` runs every invariant that does not need a GPU (the Mojo
compiler is needed only to build the census) -- the kernel census, whether `docs/KERNELS.md` is current, Python and
shell syntax, the issue templates, that `PROTOCOL-RULES.md` still carries P1-P6 (the file runs to P20),
that every repo path the docs cite exists, and that the receipts in `results/`
are internally consistent. GitHub Actions runs it on every push and pull
request. It is not a substitute for `./run-tests.sh`: no hosted runner has an
RDNA3 card, so the kernels themselves are only verified on hardware.

## Sign off what you send

Every commit in a pull request carries a `Signed-off-by:` line, which `git
commit -s` adds for you:

```
Signed-off-by: Your Name <your@email>
```

That line is the Developer Certificate of Origin (DCO 1.1, at
[developercertificate.org](https://developercertificate.org/)): you are stating
that you wrote the change, or that you have the right to submit it, and that it
may be distributed under this project's Apache-2.0 license. Nothing more is
asked of you. You keep the copyright in your own work.

There is no contributor licence agreement here, and there will not be one. A
CLA would ask you to hand over rights so the project could be relicensed later
without asking you; a project that takes patches from strangers and reserves
the option to close the source is asking for a favour it has not earned. The
consequence is deliberate: once a contribution lands, this repository can only
change its licence with the agreement of everyone whose work is in it, which is
the guarantee that Apache-2.0 here means Apache-2.0 for good.
