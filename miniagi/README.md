# miniagi

A 1.65M-parameter byte-LM trains in Mojo at torch parity (gate `tools/train-gate.sh`
PASS 6/6, bit-identical rerun), at 33.0k bytes/s fp32 and 36.4k with the fp16 WMMA
forward. That is M0+M1 of a Mojo rebuild of mini-AGI (Volotat), a byte-level model
meant to keep learning from one stream without forgetting. Source `~/Models/mini-AGI`,
analysis `~/Brain/research/mini-AGI/2026-10-09-analysis.md`.

| milestone | what | where | status |
|---|---|---|---|
| M0 | gemm[ta, tb], elementwise + attention bwd, AdamW | `train/` | done |
| M1 | dense byte-LM trainer at torch parity | `train/` | done |
| M2 | expert pool, per-expert bf16 Adam moments, disk paging, batch > 1 | | next |
| M3 | grow/prune + LR controller on a VAL split, TEST untouched | | |
| M4 | CL harness: matched-LR control, replay, EWC, 3 seeds | | |
| M5 | serve via baro-serve, live learning with measured self-drift | | |

## bridge/: latent-bridge trainer

Translator block + readers over frozen-model dumps (`miniagi/bridge/README.md`); data from `tools/latent-data.mojo`, dumps from `bench/latent-bridge-dump.sh` (needs `Q4X_STDIN`, `Q4X_DUMP_ATT`, `@VEC` in `serve/q4x.mojo`), checks `miniagi/bridge/checks.sh`.

## latent/: continuous-thought training

A 2-layer, 8-head model (1.59M params) trained as text chain of thought solves 8-hop pointer chasing at
>= 99.3% in 6-9 s on one RX 7900 XTX, and scores the same when its thoughts are fed back as soft vectors.
Phase C entry 3 (fresh seeds 110-112, card reserved) PASSED the frozen bar: 99.73 / 99.56 / 99.54% on 4096
untouched cases at 8.01 / 8.51 / 5.90 s (exchange/miniagi-latent-phasec-prereg-2026-10-09.md). Entry 1 failed on
wall (10.30 s, shared card); entry 2's wall half was voided by its load probe. The greedy text-chain control on all
nine checkpoints is within 0.08 pp of the soft eval: the soft thought costs nothing and has shown no gain at this scale. Weights: `~/Models/miniagi/latent-8hop-h8/seed*/`.

The thoughts are the model's own output distributions embedded (softmax(z/0.5) E fed back as the next
row's input), trained only as text chain of thought. Raw-hidden Coconut (feeding the final-norm hidden
state) is implemented and oracle-checked; its one control run reached 93.5% in 1500 updates, but at H=4 and
lr 1e-3 against the winner's H=8 and lr 3e-3 (`.work`-era `b/b1-even.log`), so it is not a matched comparison and
does not show a ceiling. The matched control (H=8, short+nobot, 750 updates per stage, seeds 110-112) collapses
to 32-35% at lr 3e-3 and reaches 99.71/96.63/99.88% at lr 1e-3: raw ties text CoT (one seed above greedy, one
below) and is lr-fragile where soft is not (`exchange/miniagi-rawctl-lr-prereg-2026-10-10.md`).

Self-training smoke (`star=1`, prereg `exchange/miniagi-star-smoke-prereg-2026-10-10.md`): from an 81.4% checkpoint,
300 updates on the model's own answer-filtered chains collapsed it to 5.9% (chance 6.25%), while ground-truth chains
reached 91.6%. Online self-labeling finds a self-consistent degenerate chain the answer filter cannot reject; any
retry needs STaR's frozen generator per round and a validation gate with rollback.

Reading hidden states: Anthropic's Jacobian lens (`tools/jlens_multihop.py`, receipts `exchange/receipts/jlens-qwen05/`)
on Qwen2.5-0.5B shows the intermediate hop of a two-hop question in 27.4% of 84 items against 3.6% chance; the
Jacobian transport adds about 3.6 pp over the plain logit lens at this size. A first 44.1% headline was a first-digit
artifact and is retracted.

```sh
./.venv/bin/mojo build miniagi/latent/run.mojo -D LAT_H=8 -I miniagi/latent -I miniagi/train -I kernels -o .work/miniagi/run
mkdir -p OUTDIR  # run.mojo does not create it
gpu-wait run --vram 20 --timeout 300 -- ./.work/miniagi/run text 1000 256 SEED OUTDIR 4096 \
  short=1 nobot=1 lr=3e-3 warmup=50 decay=1 horizon=1000 every=50 evalfrom=700 evaln=4096 \
  target=0.995 stop=1 hit=soft temp=0.5
```

| file | what |
|---|---|
| `latent/lmodel.mojo` | position-major batch, latent/soft/pause/generated rows as segments, K/V and dK/dV kept full length, backward in reverse segment order (no recompute) |
| `latent/lops.mojo` | bf16 WMMA split-K GEMM (dW), LDS attention fwd/bwd, warp rmsnorm, soft-feed fwd/bwd, AdamW, clip |
| `latent/run.mojo` | task, arms (nocot/text/latent/pause), curricula, eval streams, stop rule, all knobs on one parameter line |
| `latent/test_latent.mojo` | loss and every grad vs `tools/miniagi_latent_oracle.py` (torch recompute) |

`--vram 20` reserves the card: a small-VRAM job is co-scheduled with others, which is how entry 1 missed its wall bar.

Mining while training (Pearl cuPOW, int8 GEMM) is not possible at this shape: consensus needs reduction
dim k >= 2048 and every forward GEMM here has k = 256 or 688; under 10% overhead needs k >= 4096 and
m, n >= 2048 (`exchange/2026-10-12-from-miners-int8-contract.md`). Mining and miniagi runs share the card by time slot.

The oracle reads env `LAT_H` (default 4); set `LAT_H=8` to match a `-D LAT_H=8` build.

Checks: `test_latent CASE_DIR` (row weights bit-equal to the oracle's, fp32 1e-4, bf16 3e-2), `test_gemm`, `latent/dcheck.sh`.
`latent/dcheck.sh prep` (CPU) writes the oracle fixtures and builds `test_latent_d256/d512` into `.work/miniagi`; `latent/dcheck.sh`
(GPU, under `gpu-wait`) runs them, plus run identity when `.work/miniagi/run_base` (pre-change runner) and `run_new` exist.
`DCHECK_CASES="small cert"` adds the certified shape (S=41, SOFT rows, b=12): fp32 passes at D=256, bf16 sits 1.05-1.09x over
the torch floor, D=512 is ill-conditioned (torch fp32 vs fp64 4e-4); receipts `exchange/receipts/miniagi-latent/r4-r6/`. Width is `-D LAT_D=256|512|...` (multiple of 256, head dim <= 64, FF = D*43/16); at D=512 the bf16 grads miss 3e-2 because the bf16 rounding floor itself is 7-11% (torch with bf16-rounded GEMMs, `tools/miniagi_bf16_floor.py`), so that width is gated on fp32 exact plus bf16 at or under the floor. Forward and dX GEMMs run on MAX
`linalg.matmul` (bf16, transpose_b); text step 8.03 ms at b=256, S=41, H=8 (7.66 at H=4, same job; `m34/`). The design round and every
measurement: `exchange/think/latent-instant/`, receipts `exchange/receipts/miniagi-latent/`.

Builds take `-I miniagi/train -I kernels`. Oracle `tools/train_oracle.py` runs on CPU
torch in `.work/oracle-venv` with `python -P`.

Not shown yet: backward is naive fp32 GEMMs, fp16 backward needs loss scaling, and
attention stores P (256 MB per layer at 4k context), so nothing past T=1k.
