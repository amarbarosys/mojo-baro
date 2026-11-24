# Plan: four invented training methods for the 63M trainer (build only, no GPU runs)

Thesis: our best lever was zinit (zero-start capacity, -0.053 bpb); these four methods grow, teach and anneal capacity
from that zero start. Each ships behind one flag, off by default, with a written but unrun GPU gate.

Date 2026-11-19. Owner: the maintainer. Builder + reviewer: Sonnet workflow, one method at a time, in order M4, M3, M1, M2
(cheapest first; M1 and M2 reuse the Matryoshka machinery from b24f308).

## Ground rules (every method)

- Repo `~/Projects/mojo/mojo-baro`, files `miniagi/latent/{run,lmodel,lops}.mojo` (+ new helpers if needed).
- **NO GPU RUNS.** The card is reserved by another pane. Compile only:
  `.venv/bin/mojo build miniagi/latent/run.mojo -D LAT_D=1024 -D LAT_H=16 -D LAT_L=5 -D LAT_FF=2816 -D LAT_V=8192 -D LAT_EC=2048 -D LAT_OPT=7864319 -D LAT_S=512 -D LAT_QKN=1 -I miniagi/latent -I miniagi/train -I kernels -o .work/latent63m/inv512`
  (timeout 900; must print no `error:`). GPU gates are WRITTEN as scripts (`miniagi/latent/<name>-gate.sh`, `set -euo pipefail`,
  `FAIL <step>: <reason>`, inside ONE `gpu-wait run --vram 22 --mem 16 --timeout S -- ...`) but NOT executed.
- **Off by default:** with the flag off, the code path runs the same kernels as before (guard new work with `if flag`;
  no reordering of existing kernels).
- **One flag per method**, parsed in run.mojo like the existing `elif key == "...":` chain, printed in the `params` line,
  documented in the run.mojo docstring k=v list (lines ~7-30).
- Recipe the methods sit on: `lr=3e-4 warmup=30 cool=0.2 zinit=1`, toks arm, S512 b32, QK-norm build (LAT_QKN=1).
- Matryoshka exists (b24f308): `m.ffm` runtime active FFN width (prefix of w1/w3 columns, w2 rows), `mat=1` schedule, `ffm=M`.
  Only the bf16 AB16/BT/DWG/BF2 path supports `ffm < FF`.
- Commit each method separately: conventional subject, why body, no AI or Co-Authored-By line. Name paths, never `git add -A`.
- Before code: `brain-ask --brief "<method>"`, codebase-memory graph (`search_graph`/`get_code_snippet` on mojo-baro; index with
  `index_repository` if missing), toolscript `mcp__hub__run_js` for batched reads/greps (plain repeated reads are blocked).
- Skills to load before writing: `memory-optimization`, `training`, `codebase-memory`, `toolscript`; for any GPU kernel edit
  read `~/Brain/Skills/own/gpu-kernels/SKILL.md` and `references/isa-rdna3.md` first.
- After each method: one row in `~/Brain/LocalTraining/methods-63m.md` (status BUILT, flag, gate script, "not run") and commit Brain.

## M4 Error-annealed cooldown (`ecool=F`, default 0)

During the WSD cooldown (it >= steps * (1 - cool)), replay the hardest windows: after each update, read the per-row loss
(`m.lrow`, rows r * b + i), take the per-sequence mean loss of the b sequences, keep the window start offsets of the top
round(F * b) sequences; the next batch reuses those offsets for its first round(F * b) sequences, fresh windows for the rest.
Outside the cooldown nothing changes. Optional `ecorp=PATH emix=G`: during the cooldown a fraction G of the fresh sequences
come from a second u16 corpus (CoT data, e.g. `.work/cot/metamath.u16`); default off.
- Where: the batch fill for the toks arm (find it via the graph: the function that fills bf.ids/bf.tgt for corpus windows;
  `fill_bytes` is the eval twin) and the train loop after `m.step` (~run.mojo:1168).
- Gate script `ecool-gate.sh`: (1) ecool=0 params after 20 updates byte-identical to the binary without the change
  (cmp params.f32); (2) with ecool=0.25 and cool=0.5 over 20 updates the log line `ECOOL it N replay K` appears from the
  cooldown start only, K = round(0.25 * 32) = 8; (3) planted: replay offsets must equal the previous update's top-8 (print both).
- Falsifier A/B script (`.work/ab-ecool.sh`, not run): recipe vs recipe + ecool=0.25, 600 updates, 3 seeds; win = lower val bpb.

## M3 Self-taught Matryoshka (`mdist=L`, default 0; needs mat=1)

On a small-width update of the mat schedule (FF,768,FF,1408,FF,2048) train on the SAME batch as the previous full-width update
(do not advance the data stream) and add in-place distillation: loss = (1 - L) CE + L KL(p_full || p_small) per row.
Teacher = the full-width logits of that batch, saved after the full update's forward: keep a bf16 copy of `m.z` rows
[R, V] (R = S * b; 16384 x 8192 x 2 B = 256 MB). dz for the student row = (1 - L)(p - onehot) + L (p - q), q = softmax(teacher row).
- Where: a ce kernel variant next to `ce_w_k` in lops.mojo (one wave per row, same layout, reads the teacher row too), the
  small-step data reuse in run.mojo, the teacher copy after the full step.
- Gate script `mdist-gate.sh`: (1) mdist=0 with mat=1 byte-identical params to mat=1 before the change after 12 updates;
  (2) finite-difference check of the KL term on S48 (build with LAT_S=48): directional derivative of the loss along the
  gradient matches the dz-implied one within 5 % at L = 1, using `lr=0` and `gdump=`; (3) with L=1 and teacher =
  student (same width), the KL gradient is ~0 (rel < 1e-2).
- A/B script (`.work/ab-mdist.sh`): mat=1 vs mat=1 mdist=0.5; members scored at ffm=768/1408 with `ffm=`.

## M1 Mistake-seeded growth (`grow=STEP:FROM:TO`, `gseed=E`, defaults off / 0)

Train at `m.ffm = FROM` until update STEP, then switch to TO for the rest. Function-preserving: at the switch, w2 rows [FROM, TO)
must be exactly 0 (true under zinit; force them to 0 at the switch otherwise) so the loss before and after the switch is equal.
With gseed=E > 0, seed the new neurons from the model's mistakes (the mojo-neuron implant recipe): on the switch batch, run one
forward at width FROM, take the TO - FROM rows with the highest CE (m.lrow), and per layer set new neuron j's w1 and w3 column
to the layer's normalised FFN input of that row (a.n2, the rms output) scaled to the mean column norm of w1[:, :FROM]; set w2 row j
to E times the tied-embedding row of that row's target token. E = 0 means zero growth (only w2 = 0; w1/w3 keep their init).
Write into params p and the bf16 copy w16 (refresh rebuilds t16), and zero the Adam moments of the new slices.
- Where: run.mojo train loop (switch), a host-side helper in lmodel.mojo for the seeding (copy rows to host, compute, upload).
- Gate script `grow-gate.sh`: (1) grow off = byte-identical params; (2) gseed=0: loss of the switch batch at width FROM vs TO
  equal within 1e-4 (print both); (3) gseed=0.01: new w2 rows nonzero and equal to 0.01 * E[target]; (4) planted: a switch
  that leaves one w2 row nonzero must change the loss (the check must catch it).
- A/B script (`.work/ab-grow.sh`): full width vs grow=300:1408:2816 gseed=0 vs gseed=0.01, 600 updates, 3 seeds, equal steps.

## M2 Seedling depth growth (`dgrow=STEP:L0`, default off)

Train with only the first L0 layers active until STEP, then all L. Inactive layers are skipped entirely (no kernels: that is
the speed-up); the residual stream after layer L0 - 1 feeds the final norm. At the switch the inactive layers must be the
identity: wo = 0 and w2 = 0 (true under zinit and no gradients reach them while inactive; weight decay keeps 0 at 0;
force-zero them at the switch otherwise) so the loss is unchanged at the switch.
- Where: lmodel.mojo forward/backward layer loops (find `for l in range(L)` and `for l in reversed(range(L))` in fwd_seg /
  bwd_seg), the final-norm input (acts[L - 1]), and the optimizer (inactive layers get zero gradients: make sure the grad
  buffer slices are zero; AdamW then leaves them unchanged apart from weight decay on zero).
- Gate script `dgrow-gate.sh`: (1) dgrow off = byte-identical; (2) dgrow=STEP:5 (L0 = L) = byte-identical; (3) loss at the
  switch batch with L0 layers vs all L equal within 1e-4; (4) STEADY ms with L0 = 2 vs 5 printed (expect ~0.45x of the layer
  part); (5) planted: one inactive layer with a nonzero wo must change the switch loss.
- A/B script (`.work/ab-dgrow.sh`): equal WALL TIME: fixed 5 layers N updates vs dgrow=K:2 with as many updates as fit.

## Reviewer checklist (second Sonnet, per method)

1. Diff matches the spec (flags, defaults, where). 2. Flag off = no new kernels on the default path (read the guards).
3. Compiles (command above). 4. Gate script exists, uses gpu-wait, has the planted check, is NOT run. 5. Registry row + commits.
6. No unrelated edits, no reformatting. Fix small issues directly (Edit), report anything bigger.
