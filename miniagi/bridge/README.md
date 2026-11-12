# miniagi/bridge: the latent-bridge trainer (P4)

Trains the translator block + readers of `the latent-bridge plan` on P3 dumps of frozen
Qwen3.8-Flash. Three files: `bkern.mojo` (kernels, no comments by rule), `bblock.mojo` (config, parameter layout,
block + reader forward/backward on explicit buffers, no trainer state: P6 compiles it into `serve/q4x.mojo`),
`bridge.mojo` (data index, pinned pread loader, AdamW loop, logging, checkpoints). Oracle `tools/bridge_oracle.py`,
checks `checks.sh`, receipts `exchange/receipts/latent-bridge/p4/`.

Build: `mojo build miniagi/bridge/bridge.mojo -I miniagi/bridge -I miniagi/latent -I miniagi/train -o .work/bridge/bridge`
(lops `Gemm` + `ops` kernels are reused). Run: `bridge RUN [k=v ...]`, always under `gpu-wait run`.

## Model (d 2560, 20 heads x 128, no mining shape)

```
sources  Flash rows (as is) · Qwythos-9B rows → wp1 [4096→2560] · Qwen3.5-4B rows → wp2 [2560→2560]
memory   m = rmsnorm(src + esrc[s]) · gm      rows item-major, source-minor (item i: src0 Lp_i rows, src1 Lp_i, src2 Lp_i)
block    Z = z0 [32, 2560] per item;  R=4 tied rounds:  Z += wo·attn(wq·rms(Z)·g1 → K=wk·m, V=wv·m)
                                                        Z += w2·swiglu(w1·rms(Z)·g2, w3·rms(Z)·g2)      (ff 6912)
         Zn = rms(Z)·gz
reader A h' = h + ga ⊙ woa·attn(wqa·rms(h)·gan → wka·Zn, wva·Zn) at answer rows;  loss CE(head·h', ans) + lam·Σ(h'−hT)²/ΣhT²
reader B q = dumped layer-47 q, dims qlo..255 of each of 24 heads (192 unroped dims); K = wkb·Zn [32, 2 x 192],
         V = wvb·Zn [32, 2 x 256]; GQA head h reads kv h//12; delta [24 x 256] ⊙ gb[h]; out = aoS + delta·wo47ᵀ;
         loss Σ(out − aoT)²/ΣaoT²   (softmax scale 1/sqrt(192), separate from the engine's own attention)
reader C c = wc·Zn;  loss Σ(c − ct)²/Σct² + mean_j(1 − cos(c_j, ct_j))
```

Every reader is a gated residual; `ga` (per channel) and `gb` (per q head) init 0, so with gates at 0 the logits
equal `head·h` and the attention output equals `aoS` bit for bit (`gate0=1` checks it). Losses are relative MSEs
(sum of squares over the micro-batch divided by the target's sum of squares), CE is the mean over answer rows.
Attention, norms and losses run in f32; GEMMs round both operands to bf16 and accumulate in f32 (lops `tgemm_k`),
forward and backward alike, and the oracle mirrors that (`MM` autograd function).

## Runs (one binary)

| run | sources | readers | note |
|---|---|---|---|
| `A` `B` `C` | Flash | that reader | |
| `D` | 3 | A | |
| `ALL` | 3 | A+B+C | each reader loss divided by its gate-0 value measured on batch 0 before training |
| `single` | 3 | A | memory zeroed after the norm (block input-independent); = `D single=1` |
| `identical` | 3 | A | Flash rows feed all three sources (wp1/wp2 become [2560→2560]); = `D ident=1` |
| `noxattn` | 3 | A | same parameters, the block's attention reads rms(Z) itself instead of the memory; = `D own=1` |

Flags: `data= steps= b=32 seed= lr=3e-4 warmup=100 wd=0.01 clip=1 lam=1 every=50 ckpt=500 out= items= evalsplit= patience=500
ginit= single= ident= own= check= gate0=`. The batch of 32 runs as two micro-batches of 16 (micro-batch loss weight 1/2);
their two gradient norms give the gradient noise scale `Bsimple` (McCandlish: |G|² = 2·G32² − G16², trΣ = 32(G16² − G32²)).
AdamW (0.9, 0.999, eps 1e-8), fp32 master weights and moments, clip 1.0 on the global norm, no decay on `z0 esrc gm g1 g2
gz gan ga gb`, linear warmup then cosine to 0 at `steps`. Stops early when the eval loss has not improved for `patience` steps.

Log line every `every` steps: train/eval loss per reader (`A B C`), teacher-forced answer-token accuracy (`acc`, reader A),
separation (`sep`: eval loss with thoughts of the next item in the batch minus matched, per reader), gate means, grad norm +
clip factor, `Bsimple` (instantaneous and EMA), ms/step. `RESULT` line: ms/step, samples/s, dump GB read and GB/s of the
pread stream, peak VRAM (`mem_info_vram_used`) and peak RSS.

## Data (all under `~/.cache/baro/latent-bridge/`)

`data/items.tsv` (`id task split rank answer pre rat post ans`), `data/sel-{train,val}.txt` (ids in dump order),
`weights/{head,wo47,ctarget-*}.f32`, `dumps/{flash-student,flash-teacher,qwythos9b,qwen35-4b}-{split}.{hid,att,idx}`.
Geometry per item: Lp = |pre|, Lq = |post|, La = |ans| ≤ 11; answer rows of the student sequence are
`Lp + Lq − 1 + k`; in every 12-row tail (teacher hid, both att files) they are rows `12 − La + k`.
The loader preads exactly those rows (pre rows of each source, answer rows of hid/att, 32 ctarget rows) into a pinned
host buffer, two batches alternating, so the fill of batch i+1 overlaps the GPU work of batch i; one upload per batch.
`cfg.txt` in the data dir overrides the dims (`d nh ff nz vocab s1 s2 nhb nkvb hdb qlo R`); the tiny synthetic set
written by `bridge_oracle.py gen` uses it (d 64, 4 heads, vocab 300, R 4).

## Checkpoint format

`<out>/step-N.ckpt`: one ASCII header line, then the flat f32 parameters:

```
baro-bridge v1 run=ALL step=200 seed=1 np=NNN d=2560 nh=20 ff=6912 nz=32 vocab=248320 nsrc=3 s1=4096 s2=2560 nhb=24 nkvb=2 hdb=256 qlo=64 R=4 order=wq wk wv wo w1 w3 w2 wp1 wp2 wqa wka wva woa wkb wvb wc z0 esrc gm g1 g2 gz gan ga gb
```

Tensor order and shapes (row-major, `[in, out]` for matrices so `y = x · W`): `wq wk wv wo` [d,d] · `w1 w3` [d,ff] ·
`w2` [ff,d] · `wp1` [s1,d] · `wp2` [s2,d] · `wqa wka wva woa` [d,d] · `wkb` [d, nkvb·(hdb−qlo)] · `wvb` [d, nkvb·hdb] ·
`wc` [d,d] · `z0` [nz,d] · `esrc` [3,d] · `gm g1 g2 gz gan ga` [d] · `gb` [nhb]. The first 16 tensors take weight decay.
Unused tensors of a run (e.g. `wp*` in `A`) stay at their init. Checkpoints are written every `ckpt` steps and at the
end, before any buffer is freed.

## Checks (`checks.sh`, receipts in `exchange/receipts/latent-bridge/p4/`)

1. gradient oracle: tiny config, 7 run variants, and one real micro-batch, rel L2 per tensor ≤ 1e-4
2. gate-0 identity (`gate0=1`): reader A logits == `head·h`, reader B output == `aoS`, byte for byte
3. overfit: `A items=32 evalsplit=train steps=300` reaches ≥ 95% teacher-forced answer-token accuracy
4. rerun: two runs with the same seed give byte-identical `step-200.ckpt`
5. speed + bytes: `RESULT` lines of 60-step `A` and `ALL` runs

## Oracle bar

`checks.sh tiny-oracle` holds every tensor to rel L2 1e-4 against a torch recompute. 7 of 9 variants pass (worst 8.5e-5);
`A` at gate 0.3 (8.6e-4) and `D` (1.3e-4 on `wp2`) sit above the bar because both implementations round GEMM operands to
bf16 and one element on a rounding tie flips the next GEMM; the stage prints the per-tensor bf16 flip counts and exits
non-zero for those two.
