# P4 multi-GPU protocol

Status: AMENDED TWICE 2026-09-17/18 (round 2, lane P4B, then the P4B follow-up), frozen by commit
before any identity run each time. The iGPU arm stays as the pinning-and-wiring receipt, reported,
never gated. The identity gate runs as two engine processes on the XTX, sending the real 20-prompt,
64-token-per-prompt fixture as token ids and comparing ids, not text (`bench/p4-router-identity.sh`,
added 2026-09-18: the earlier amendment cited team A's `bench/p0b-gate1-placement.sh` as this
receipt, which is wrong; see "Gate 1" below for why and what it demotes to). The state-move gate
does not exist. This supersedes the "wiring preflight, timed gate pending" status below it
replaces; the effective-device preflight, fixture manifest, and receipt discipline are unchanged.

## Scope

P4 tests process-level data parallel wiring, not cross-model quality or throughput. The discrete
arm uses the dense q4 fixture and the iGPU arm uses self-describing Qwen2.5-7B-Instruct BARO
`b4271cd`. Each arm is compared only with its own solo reference. The iGPU is a functional harness
device, not a performance arm, and (see "The iGPU finding" below) not an identity-gated one.

The fixture manifest is `bench/p4-fixtures/manifest.tsv`. It records model and pack paths and
hashes, but does not copy model bytes. The 20 prompts are the existing
`bench/mtp-prompts/p*.tokens` set, whose sorted-file aggregate hash is recorded there.
The Qwen7 pack is extracted from the GGUF's embedded `engine-pack.py` and `--dense` metadata,
matching the `tools/baro serve` cache builder. The import receipt reports min 96.9%, mean 99.4%
on its 20-prompt gate and pack index hash `e2ef587f...`; the generated local pack must reproduce
that index hash before a GPU run.

## What each arm proves

- **iGPU arm** (Qwen2.5-7B via `serve/spark.mojo`, gfx1030 code objects loaded under
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`): proves the harness can pin, launch and read back a second,
  physically distinct GPU process (`ROCR_VISIBLE_DEVICES` set, `HIP_VISIBLE_DEVICES` unset), and
  that its kernels are deterministic when nothing goes wrong (768 logits rows bit-identical across
  12 fresh processes; 60 of 60, then 5 of 5, per-request checksum tables identical). It is
  REPORTED, not gated: see "The iGPU finding" below for why.
- **Identity gate**: two real engine processes on the same XTX at `BARO_TMAX=4096`. The 20
  `bench/mtp-prompts/p*.tokens` prompts are sent as token ids, `max_tokens` 64, `temperature` 0,
  `spec` false, through the router, and each response's token ids are compared (`cmp`, never on
  text) against the same prompt sent directly to one engine (the solo baseline); placement must
  spread (both engines serve at least one of the 20) or the gate fails. This is the only gated
  claim P4 makes, and does not exercise a second physical device.

## The iGPU finding (why it moved out of the gate)

Full report: `exchange/lane-P4-report-round2.md`; summary: `[[2026-09-17-p4-igpu-transient-corruption]]`
(Brain, mojo-baro). The unchanged gate on identical iGPU binaries went FAIL, FAIL, PASS over three
runs: 3 one-token deviations in about 11,000 iGPU tokens, each one wrong logit off by more than 26
followed by the stream continuing from it. Ruled out with receipts: request-state bleed (a fresh
single-request process still deviated; history-bearing requests did not), `f9048f6`/save_state
(not in the failing binary, which is `spark.mojo`, not `engine.mojo`), systematic
gfx1030-on-gfx1036 miscompute (768 logits rows bit-identical across 12 fresh processes when
nothing went wrong), and KFD queue eviction as a sufficient cause (about 100 evict/restore cycles
produced no corruption). The same kernels, same source, same pack, built for gfx1100 and pinned to
the XTX: 51,200 tokens, 0 deviations. Cause not placed; open candidates (untested or only
partially tested) are power/clock state transitions under bursty desktop use and transient DDR5
read faults, both outside the engine, not a per-execution race in the kernels. A rig that lies
about 1 in 3,700 tokens hides a regression; never gate identity on this device.

## Repeat rule

The identity gate has passed by luck before: FAIL, FAIL, PASS on the unchanged iGPU gate, same
binaries, 2026-09-17 (a single PASS is luck, not evidence, per the finding above). Any PASS claim
for P4's identity gate (`bench/p4-router-identity.sh`, the 20-prompt/64-token/token-id gate, not
the placement gate) is run 3 times on identical binaries; all 3 must pass. A miss in any of the 3
keeps P4 FAILED; it is not retried until it passes.

## State movement: no longer a P4 gate

The state-move gate (moving state XTX to iGPU and back through P1, comparing ids with the
single-node arm) does not exist. LatentOS (P1) is an exploration, ungated, blocking nothing, since
`50f9d34` (2026-09-17, "LatentOS rules removed, P1 becomes an exploration"); this is the same rule
dropped for LatentOS everywhere else (`docs/PLATFORM-PLAN.md`'s P1 section: "EXPLORATION, ungated,
blocks nothing"; whiteboard card, same wording). P4's kill line is the identity gate alone.

## Effective-device preflight

Before any prompt is sent, the receipt must contain all of:

1. `igpu-env --probe` exit 0 and its printed `ROCR_VISIBLE_DEVICES`, unset
   `HIP_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION=10.3.0` (iGPU wiring receipt only).
2. `rocminfo` supplies the gfx1100 `Uuid: GPU-*` selector for the XTX; the bare hex
   `rocm-smi --showuniqueid` value is retained as a suffix cross-check. `rocm-smi --showpids`
   covers every running engine process, with PID-to-device mapping (both engine PIDs on the XTX,
   for the identity gate).
3. For the iGPU wiring receipt: the process's engine log showing the expected HIP API path, plus
   external `rocminfo`/`rocm-smi` output naming the gfx1030 object and mapping the running PID to
   it. MAX's `DeviceContext.name()` is not accepted as the device identity: under this override it
   reports the host CPU name even while `ctx.api()` reports HIP. `ROCR_VISIBLE_DEVICES=1` alone
   is insufficient because MAX lists the Raphael device as gfx1030 objects under this override
   and does not list gfx1036. The misleading engine self-name is retained in the receipt.
4. The per-process environment read-back from the Rust spawn seam, including the server port,
   selector, and pack path, and (for the identity gate) `BARO_TMAX` as printed in each engine's
   own ready line.

Any missing or contradictory read-back stops the gate before prompts.

## Gate 0 (demoted): team A's placement gate is not the identity receipt

`bench/p0b-gate1-placement.sh`'s 3 runs from the round-2 amendment (`exchange/lane-P0B-report.md`,
`.work/p4b/gate1-run{1,2,3}/`) are PLACEMENT receipts only: `pair-dispatch OK: 20/20 ok through the
router` and `placement OK: a=10 b=10` still stand as P0b's own claim. Their identity check was 5
prompts x 4 repeats of short text answers (`\n\namber`, `\n\n4`, ...), about 70 tokens compared per
run, roughly 35x fewer than P4's own 20-prompt/64-token fixture. Three passes of that check do not
put the repeat-rule receipt behind P4's identity claim; that requires the gate below.

## Gate 1: identity, two engine processes on the XTX, the real 20-prompt fixture

`bench/p4-router-identity.sh`. Forks the launch half (`start_engine`, router bring-up, health
wait) from `bench/p0b-gate1-placement.sh` rather than sourcing it (that script is linear, not
decomposed into functions a caller can pull in without either re-running its own weaker identity
check or restructuring team A's file); the payload/token-extraction/request functions are lifted
unchanged from `.work/p4/run-two-engines.sh` in the main checkout (team B). For each of the 20
`bench/mtp-prompts/p*.tokens` prompts: one request direct to engine a (the solo baseline) and one
through the router, both with `max_tokens` 64, `temperature` 0, `spec` false, token ids in and out;
`cmp` the two token-id files, never text. Placement must spread: `GET /v1/workloads` after all 20,
grouped by engine, and the gate fails if either engine served zero (this gate's reverse arm, see
`reverse-arm-gates`: an identity pass where one engine did all 20 proves nothing about the split).
Device read-back per run, written to files, not asserted in prose: `rocm-smi --showpids` to a log
file, both engines' actual GPU-touching child PIDs (found via `pgrep -P` on the tracked
`baro-serve` PID, matched by `/proc/PID/comm` == `engine`) grepped present in it, and
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` plus `BARO_TMAX` read from each engine child's
own `/proc/PID/environ`. CPU-only checks before any GPU run: `--selftest` (a comparator negative
control: identical token files PASS, a one-id difference FAILs with a non-zero exit),
`P4_CPU_PREFLIGHT=1` (binaries, pack hash, tools, prompt-file count), and `gate-dryrun` (stops at
`GATE_DRYRUN=1` before the first real engine launch, arm file read back).

The timed invocation is from the lane root, using an absolute script path and an explicit
runtime `PATH`. `gpu-wait` admits jobs with a clean environment and may not preserve the caller's
cwd, so the short relative form is not a valid submission receipt:

```text
cd ~/Projects/mojo/mojo-baro-lanes/p4b
~/.local/bin/gpu-wait run --priority 20 --timeout 600 -- env PATH=/opt/rocm/bin:~/iTools/bin:/usr/bin:/bin BARO_ENGINE=... BARO_PACK=... BARO_SERVE_BIN=... ROUTER_BIN=... ~/Projects/mojo/mojo-baro-lanes/p4b/bench/p4-router-identity.sh .work/p4b/router-identity-runN
```

The full harness is one serialized GPU job per run. No nested `gpu-wait` is allowed inside the
admitted job. The iGPU wiring receipt, when it is (re-)collected, is included in the queue like
any other GPU job; it is never exempt.

## Kill line and receipts

Kill the item on any identity miss (in any of the 3 repeat runs), wrong device pin, missing
effective-parameter read-back, or stale/missing fixture hash. A gate that lacks its dependency
remains explicitly blocked and cannot be replaced by a weaker claim.

The command, exit code, preflight, device mapping, and all 3 verdicts land in
`exchange/lane-P4B-report.md`. The CPU dry-run must stop before the first GPU step and is recorded
before each queued invocation.
