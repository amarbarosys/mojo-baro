# Plan: finish lane rdna4-epi (remaining steps, cold start)

The code is done and committed (`9a300f0`); what is missing is every receipt from the tree itself, the engine check and the report.

Worktree `$HOME/Projects/mojo/mojo-baro-lanes/rdna4-epi`, branch `lane-rdna4-epi`.
Brief: `docs/lanes/2026-11-12-rdna4-epilogue-brief.md`. What was learned: `~/Brain/mojo/mojo-baro/2026-11-12-rdna4-epi-handoff.md` (read it first).
Skills to load and list in the report: `rdna4-port`, `rdna4-isa`, `gpu-kernels` (rules #7 to #9), `mojo-baro`, `gate-authoring`.

State: all kernel sites are edited and the gate script exists. Every receipt so far came from a scratch copy
(`.work/rdna4-epi/scratch`, same file bytes as the commit). Nothing was rebuilt from the tree after the commit, the engine check never ran,
and no report is written. No kernel edit is expected in the steps below; if one is needed, the 3-repair rule of the brief applies.

Rules that stay: kernel files hold zero comments; commits by pathspec, no attribution trailer, no em dashes; never edit a test to pass;
GPU only through `gpu-wait` (the gate script queues itself); run `git status --porcelain` before each commit (a test run rewrites the tracked
`exchange/receipts/front/text-oracle.jsonl`: restore it with `git checkout -- exchange/receipts/front/text-oracle.jsonl`, never commit it).

Set once per shell:

    cd $HOME/Projects/mojo/mojo-baro-lanes/rdna4-epi
    export BARO_KQ_PACK=$HOME/Projects/mojo/mojo-baro/.work/ailab/pack-kq
    G=bench/rdna4-epi-gate.sh

## 1. Build the three variants from the tree (CPU, about 8 minutes, no GPU)

    JOBS=3 $G build . def gfx1100
    JOBS=3 $G build . emu gfx1100 -D BARO_EMU_GFX12_ROWS=true
    JOBS=3 $G build . r4  gfx1201

Pass bar: each prints `OK build-many rdna4-epi-<tag>: 8 builds`. The `r4` line is half of brief gate 4.
Baseline binaries of `91c933f` are in `.work/rdna4-epi/bin/pre` (8 files). If they are gone:
`mkdir -p .work/rdna4-epi/pre-src && git archive 91c933f | tar -x -C .work/rdna4-epi/pre-src && JOBS=3 $G build .work/rdna4-epi/pre-src pre gfx1100`
(the build path does not change the code objects: measured).

## 2. Gate 1a: gfx11 ISA identity (CPU, 1 minute)

    $G isa pre def

Pass bar: 8 `PASS isa` lines with these code-object counts: test_wmma_rdna34 1, test_prefill 32, test_mmq 7, test_moe_wmma 6,
test_kq_prefill_gemm 29, bench_fp16 1, bench_fp16_pipe 1, engine 288. Any `FAIL isa` = a gfx11 byte moved: stop, find the site
(`.work/rdna4-epi/isa/<tag>/<target>/coN.s`), do not continue. Identical ISA means no perf A/B and no wavewait capture (rdna4-port s10): write that
sentence in the report.

## 3. Gates 1b and 2: parity, default then emulation (GPU, about 1.5 minutes each, `--shared --vram 1`)

    $G run def
    $G run emu

Pass bar: last line `run <tag>: fails=0 skips=0` for both, 7 `PASS` lines each. A `SKIP` is not a pass (kq pack path wrong).
Also compare `grep -o '"max_err": [0-9.e-]*' .work/rdna4-epi/run-{def,emu}-bench_fp16*.log`: default and emulation must be equal to the last
digit (the emulation is a permutation). Receipts: `.work/rdna4-epi/arm-<tag>.txt`, `.work/rdna4-epi/run-<tag>-<test>.log`.

## 4. Gate 3: mutants (CPU builds + 13 small GPU jobs, about 10 minutes wall)

    JOBS=2 $G mutants

Pass bar: last line `mutants: alive_or_broken=0 skipped=0`, 13 lines `mutant <name>: PASS mutant-check: the gate caught it`.
Receipt `.work/rdna4-epi/mutants.txt`. Scratch result to reproduce: 13 of 13 killed. Record each mutant's failure line in the report
(`gate said:` lines; `rdna34_store` printed 250/256 when run by hand).

## 5. Gate 4b: probe table unchanged (CPU, 20 seconds)

    tools/rdna4-probe/rdna4-probe.sh > .work/rdna4-epi/probe.log 2>&1
    diff <(sed -n '/^probe /,$p' .work/rdna4-epi/probe.log | head -13) <(head -13 docs/rdna4/probe-2026-11-12.txt) && echo PROBE-SAME

Pass bar: `PROBE-SAME`.

## 6. Gate 5: real-engine agreement (GPU, estimate 5 minutes, `--shared --vram 12`)

The `engine` subcommand has NEVER run (only `bash -n`). Do these in order:

a. Mutant engine (CPU, 3 minutes). The dense prefill path calls the WMMA attention (`serve/window.mojo:581`), so mutate its output row:

       M=.work/rdna4-epi/mut/engine_attn; rm -rf $M; $G srccopy . $M
       sed -i 's/acc_row(dt \* 16, i, half)\]/dt * 16 + 2 * i + half]/' $M/kernels/attn.mojo
       cmp -s kernels/attn.mojo $M/kernels/attn.mojo && echo "FAIL sed changed nothing"
       ONLY='^engine$' $G build $M mut-engine gfx1100 -D BARO_EMU_GFX12_ROWS=true

b. Check the script before the full run (gate-authoring "Before the queue"). Not yet proven against a real one-shot log: the read-back
   regexes (`TMAX: 16384 `, `prefill rows: N`, `mega fail word: 0`, `forced agreement: A / N`, `^GENERATED:`) and that a mutant engine
   exits without hanging (`|| true` covers a non-zero exit, the job timeout of 1800 s covers a hang). Read `engine()` once, run it, and
   look at the first result row and `.work/rdna4-epi/engine/p01-*.log` as soon as they exist; cancel and fix the script if a read-back is
   empty, then commit the script.
c. Whiteboard GPU holds: read `~/Brain/mojo/mojo-baro/whiteboard.md` for an open hold (none on 2026-11-12). State the GPU-minutes (5) before queueing.
d. Run:

       $G engine def emu mut-engine

Bar (preregistered, the script prints it into `.work/rdna4-epi/engine/arm.txt` before the first run): 20 prompts (first 1920 ids of
`bench/prefill-prompts/p32768.tokens` + each `bench/mtp-prompts` question), `BARO_TMAX=16384`, `BARO_SPEC=0`, q4 pack; control arm (default
build forced on its own greedy ids) 100% on all 20, else VOID; emulation build 100% on all 20; `prefill rows >= 16` on every run; fail word 0;
mutant below 100% on every prompt. Pass line: `PASS engine: emu 100% on 20/20 prompts, ctl 100%, mutant below 100% on every prompt (mut=yes)`.
The 64 forced ids are the fixed-length comparison that `bench/min16k.sh` exempts from the 16000 max_tokens floor; say so in the report.
If the control arm is below 100% on a prompt, the instrument is void for that prompt: report it, do not lower the bar.

## 7. Full suite, the lane brief's gate (GPU, about 10 minutes)

    gpu-wait run --shared --vram 14 --timeout 2400 -- bash -c "cd $PWD && env STATE_ENGINE=$PWD/.work/rdna4-epi/bin/def/engine BARO_KQ_PACK=$BARO_KQ_PACK ./run-tests.sh > .work/rdna4-epi-gate.txt 2>&1; echo \"suite exit \$?\" >> .work/rdna4-epi-gate.txt"

Baseline on the unchanged tree `91c933f`: `.work/rdna4-epi/suite-pre.txt`, `suite exit 132`: `test_text` dies with SIGILL after 15 PASS lines.
That failure is in the baseline, not in this lane. It appears because `.work/ailab` is linked into the worktree (the preflight cargo tests need
that link; without it they fail 4 tests). Do not fix it here. Pass bar for this lane: the after-run shows the same PASS lines as the baseline
up to the same point (`diff <(grep -E 'PASS' .work/rdna4-epi/suite-pre.txt | sed 's/CACHED //') <(grep -E 'PASS' .work/rdna4-epi-gate.txt | sed 's/CACHED //')`
is empty) and no new FAIL. Report the count before and after (before = 15) and report the `test_text` SIGILL to the boss as a baseline defect.

## 8. Preflight on the final tree (CPU, about 6 minutes)

    JOBS=2 bench/preflight.sh > .work/rdna4-epi/preflight-post.log 2>&1; echo "exit $?"; tail -3 .work/rdna4-epi/preflight-post.log

Pass bar: exit 0. It runs ci-checks, the probe diff, every test build, the gfx1201 engine build and cargo. Baseline preflight failed at cargo
only because `.work/ailab` was not linked yet; it is linked now (`ls .work/ailab/pack-kq/tokenizer.json`).

## 9. Report and close

- `docs/lanes/2026-11-12-rdna4-epilogue-report.md` (what the plan brief asks): skills loaded, commits, each gate with receipt path and numbers,
  mutants with their failure lines, the site table below, the three design calls of the handoff note, the out-of-scope findings.
- `$HOME/Projects/mojo/mojo-baro/exchange/lane-rdna4-epi-report.md` (what the lane brief asks): gate command, exit code, test count
  before/after, evidence path, commits.
- Site table for the report (the status column is the scratch result; confirm each from steps 3, 4 and 6):

  | site | change | status |
  |---|---|---|
  | `matmul_prefill.mojo:93` (q4), `:169` (q8) | `acc_row(row0 + tm * 16, i, half)` | VERIFIED-EMU (test_prefill, mutants prefill_q4, prefill_q8) |
  | `matmul_prefill_lds.mojo:158` | same | VERIFIED-EMU (test_prefill 1b, mutant prefill_lds) |
  | `matmul_prefill_kq.mojo:202` | same | VERIFIED-EMU (test_kq_prefill_gemm, mutant prefill_kq) |
  | `matmul_mmq.mojo:277` rows | same | VERIFIED-EMU (test_mmq, mutant mmq_rows) |
  | `matmul_mmq.mojo:157` scale staging | LDS slot tid from global slot `mmq_scale_slot(tid % BM)` under GFX12_ROWS | VERIFIED-EMU (test_mmq, mutant mmq_scale) |
  | `matmul_mmq.mojo:24` `mmq_scale_slot`, `:199` `half * 8` | unchanged on purpose (buffer format, see handoff) | VERIFIED-EMU through :157 |
  | `moe_wmma.mojo:156` | `acc_row(wm * TM * 16 + tm * 16, i, half)` | VERIFIED-EMU (test_moe_wmma, mutant moe_wmma) |
  | `matmul_wmma_lds.mojo:161` | `acc_row(block_row + ..., i, half)` | VERIFIED-EMU (bench_fp16 correct flag, mutant wmma_lds); no parity test in the suite |
  | `matmul_wmma_pipe.mojo:879` | same | VERIFIED-EMU (bench_fp16_pipe correct flag, mutant wmma_pipe); no parity test in the suite |
  | `attn.mojo:747,754` mask | `acc_row(t0, i, half) <= lim` | VERIFIED-EMU (test_prefill 2b, mutant attn_mask) |
  | `attn.mojo:764,765` P-to-B repack | `bf[acc_row(0, i, 0)]`, `bf[acc_row(0, i, 1)]` | VERIFIED-EMU for the layout; stdlib half-slice not modelled (handoff, attention verdict) |
  | `attn.mojo:777` output row | `acc_row(dt * 16, i, half)` | VERIFIED-EMU (mutant attn_out) |
  | `kernels/test_freivalds.mojo:106`, `miniagi/latent/lops.mojo:98` | not touched, outside the file list | UNPORTED: wrong rows on gfx12 |

  Every row: gfx12 silicon UNVERIFIED until a gfx12 owner's run.
- Findings to hand to the boss (not fixed here): the two UNPORTED sites; `tools/merge-gate.sh:9` builds test_prefill without `-I serve`;
  `.lane-prep` lacks `ailab`; no q8 pack on this box (the gate runs `test_prefill` on a q8 fixture requantised from the q4 pack);
  `test_prefill` and `test_mmq` are not in `run-tests.sh`; `test_text` SIGILL on the baseline.
- Skill update: `~/Brain/Skills/own/rdna4-port/SKILL.md` s2 and s3: sites ported in `9a300f0`, `acc_row(base, i, half)`, the scale-slot call,
  the attention verdict; add a changelog line; then `skilldex`.
- Board: tick the rdna4 card in `~/Brain/mojo/mojo-baro/whiteboard.md` with the Brain path.
- Commit the report (pathspec), then `herd tell a lane pane 'DONE -> $HOME/Projects/mojo/mojo-baro/exchange/lane-rdna4-epi-report.md'`.
- Scratch can go after the report: `rm -rf .work/rdna4-epi/scratch .work/rdna4-epi/mut .work/rdna4-epi/bin/s-*`.
