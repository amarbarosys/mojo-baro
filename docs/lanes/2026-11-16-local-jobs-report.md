# Local models on real mojo-baro jobs: report (2026-11-16)

Two local models, run through our own engines and `baro-serve`, found 0 true defects in 111 review claims, wrote 1 of 3 small Mojo tasks that compile and pass their gate,
and produced a digest with six confirmed defects in 11 sections. Neither model is usable as a reviewer; Qwythos-9B is usable for one narrow edit task.
Brief: `docs/lanes/2026-11-16-local-jobs-brief.md`. Receipts: `.work/local-jobs/` (ignored by git; the numbers are copied here).

## Setup

| | Qwythos-9B q4 | Forge-gemma4 E2B f16 |
|---|---|---|
| engine, pack | `.work/engine`, `.work/engine-pack-q4` | `.work/gemma4/engine`, `.work/gemma4/pack` (lane-gemma4-port 88a9c66) |
| window | TMAX auto 32768 (`/health`) | 16384, baked at build (`/health` read-back) |
| answer cap | 16000, finish reasons recorded | 16384 minus prompt minus 400 (4248 to 15435): deviation from the 16K rule, chosen by the maintainer 2026-11-16 |
| thinking | off, read back: `reasoning` length 0 on every row | off, same read-back |

Every engine ran inside one `gpu-wait run --timeout --vram` job under `systemd-run -p MemoryMax=10G`, one server at a time, stopped by trap at the end.
Thinking on at the 16000 cap on Qwythos-9B ended `finish=length`, 16000 completion tokens, empty content, 33,250 characters of reasoning, on the first request
(`fewrow.mojo`), so the batch moved to thinking off. That receipt was deleted by my own `rm -rf` while restarting; the numbers are from the printed response.
The thinking-on rerun at 28000 tokens never ran (dropped under the smallest-run rule).

## J1 reviewers: 0 of 111 claims true

Same prompt for both models: one function or file per request, real line numbers, ask = bugs, dead code, slow paths as `L<line> | kind | claim | quote`.
Inputs (8): `kernels/fewrow.mojo`, `kernels/turbo_rows.mojo`, `kernels/mega.mojo` `ssm_phases` and `ffn_phases`, `serve/window.mojo` `gemm_w` and `step_window` in three 360-line chunks.
I read the source for every claim (`findings-qwythos9b-nothink.md`, `findings-gemma4e2b-nothink.md`).

| model | requests | finished | claims | VERIFIED | FALSE | unsupported |
|---|---|---|---|---|---|---|
| Qwythos-9B q4 | 8 | 8 stop | 71 | 0 | 63 | 8 (one not shown removable) |
| Forge-gemma4 E2B | 8 | 4 stop, 4 length | 40 on the finished rows | 0 | most | the rest name no evidence |

What the false claims look like: `comptime for` called dead code; `comptime` constants called runtime divisions; names called unused that are used 6 to 23 times;
"undefined name" for names defined elsewhere in the file; `print` and `synchronize` called per-token slow paths when they sit under `if cfg.prof`, `if lp_trace`, `if b.tier.bell_on`;
a "bug" whose fix is the identical text. Gemma4 looped on 4 of 8 inputs (176 lines with 3 distinct, 255 with 2, one claim repeated on 246 lines) and filled the cap; those rows are void.
This is a statement about these 111 claims, not a proof that the targets are clean.

## J2 digest: not usable as written

`.work/local-jobs/out/qwythos9b-nothink/digest.md` (Qwythos-9B, one page from 12 per-file digests). I checked the numbers and wording of seven sections against source,
and Haiku (read-only pane, quotes required) checked 109 claims: 91 supported, 5 contradicted, 13 not found.
Confirmed defects:
1. Briefs written as finished work in §1, §2, §6, §7 ("Ported", "Implemented", "Benchmarked").
2. §3 cites `gemma4_attn.mojo:67 and :137` as the emitted-code-receipt finding; the source has `:87 and :137` (`:67` is the launch-assumption NOTE).
3. §4 blurs a missed prediction ("split pays from T 1024 to 2048", MISS) with the measured T about 192.
4. §8 states "QA-PRE lost due to latency-bound dependent sudot4 chains" as a finding; the prereg says "Candidate (not yet measured)".
5. §2 carries "max rel err 5.8e-6", which is in the report's digest and not in the brief's source: the merge step moved it.
6. §10 and §11 say "no numbers" next to 18.46, 4 and two job ids, and pad with absence claims no line states.
Supported: the numbers I grepped (400 of 400, 733 GB/s, 91.7 and 13.6 and 1653 us, 99.30%, 89.2 / 119.8 / 106.0 us, 866 tensors, sha 7fcc5a5921b813ea, load 18.46).
Haiku's NOT_FOUND column is unreliable: "SSSSF x7" and "1 KV head" are in `gemma4-attn-brief.txt:21-22`. Its contradictions matched my checks.

## J3 gated Mojo writing (Qwythos-9B only)

Three tasks, one repair round fed the gate output; a proposal counts only if it compiles and its gate passes. Each gate was run with a reference solution first: PASS t1, t2, t3.
- T1 (add K = 3072 to `kernels/test_turbo_rows.mojo` `main`): **survived on attempt 1**, byte-identical to my reference (gate: 13 `K 3072` lines and PASS on GPU). A mechanical edit of given text.
- T2 (`tools/finish-count.mojo`): failed to build twice: `std.string`, `std.io`, `std.sort`, `tokenizer` imports that do not exist, `std.math.Float64`, and `args[0]` (the program itself).
- T3 (`tools/rules-check.mojo`): failed to build twice: `expected construct name to import`, then `package 'io' does not contain 'open'`, `unable to locate module 'str'`.
Survivors: 1 of 3. The 2026-10-12 verdict (local 27B, 0 of 19 Mojo proposals) is not overturned: the one survivor is a copy with one number changed.

## Not run

- Gemma4 J3: queued (`gemma4e2b-j3`, position 0 for 30 minutes behind the nerkyor lane's higher-priority `q-A-ours` and `q-A-llama` jobs). It stays queued; results go in `.work/local-jobs/out/gemma4e2b-j3-nothink/j3`.
- nerkyor J1 and J3: skipped under the maintainer's condition (GPU idle and no dflash2 queued); the GPU was held by the nerkyor lane. Its S3 had not landed.
- Nothing was committed to main from the findings: no claim was VERIFIED, so no fix.

## Disclosures and corrections

- Window deviation on Gemma4 (above), chosen by the maintainer. No other cap below 16000 on Qwythos.
- My errors, each with a rule in `docs/rules.md` and a SINDEX entry: a no-op `gpu-wait run -- true` probe (cancelled); the thinking-on batch without a one-request smoke test; `rm -rf` of a receipts directory;
  `herd tell` verb and a masked exit code; `set -e` aborting a job at a loop's last status; duplicate presets; a T3 fixture written before reading the real input; a zero-row parse read as NONE;
  a digest section doubted from a file name; "9 of 9" for 8 inputs (corrected in the Qwythos findings file).
- Gates: `gpu_bare_gate.py` probe() and the `receipts` rule in `protocol_gates.py` (iTools e65a0cf, 6c902b0). The receipts gate first matched `rm -r` text inside a commit message and blocked its own commit; fixed to command position.
- Unasked finding: `bench/PROTOCOL-RULES.md` has two P21 headings (lines 323 and 362); the second is `## P21 Wavewait capture on every kernel change` with no dot, so a strict heading parse misses it. Not edited.
