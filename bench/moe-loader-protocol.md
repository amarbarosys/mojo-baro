# MoE loader: pack index resolution and profile-sized host state

Lane MOE-LOADER's slice of W3 (`briefs/2026-09-11-moe-engine-wiring.md`):
make the qwen35moe pack loadable and its layout resolvable, without any MoE
kernel and without wiring decode. Deterministic census gate, not a
statistical lap (no noise-floor sizing: every check below is an exact
arithmetic match or it is not).

## Question

Does `serve/moe_pack.mojo`'s name-keyed index parser resolve all 733 tensors
of the real `.work/moe-w1/pack` (built by lane MOE, W1) to the same byte
offsets the pack file itself contains, and does its per-expert row-range
lookup compute the correct intra-tensor offset for a routed and a shared
expert at every quant dtype the pack uses (Q4_K, Q8_0, Q6_K, F32), without
ever assuming a tensor's position in `index.txt`?

## Treatment

New module `serve/moe_pack.mojo`: `TensorInfo` (dtype, offset, n_elem) keyed
by tensor name in a `Dict`, `tensor_byte_size` (per-dtype byte-size formula),
`resolve_expert` (layer + projection + shared/routed + expert id + row range
-> byte offset and superblock geometry, looked up by name), and
`size_host_state` (KV pool and SSM state slot sizes from
`serve/model_qwen35moe.mojo`, W0's profile: H 2048, QF 8192, KV 512, 40
layers, 30 SSM, 10 attention, HD 256, NKVH 2). `serve/harness.mojo` is
untouched by this step (its existing q4/q8 `load_pack` path is unaffected;
`moe_pack.mojo` is a new, independent read path over the moe pack).

## Method

Independent oracle: a one-off Python read of `.work/moe-w1/pack/index.txt`
(same per-dtype byte-size formulas, transcribed by hand, not by copying the
Mojo) confirmed, before any Mojo code ran, that every one of the 733
recorded offsets equals the running sum of the preceding tensors' resolved
sizes, and that the total equals the real `pack.bin` size
(21,005,191,680 bytes) exactly.

The gate is `tools/moe-loader-probe.mojo` (built AOT into
`.work/moe-loader-probe`, `-I serve`, pure host code, no GPU):

1. Parse the real pack's `index.txt` into the name-keyed map; print the
   tensor count and the sum of resolved byte sizes against the pack's real
   file size (`f.seek(0, 2)` on `pack.bin`).
2. Resolve, for `blk.0`: the router (`ffn_gate_inp.weight`), the shared
   expert's gate/up/down projections, and the routed expert 0's gate/up/down
   projections. For each routed projection, also resolve the LAST expert
   (255) and check that its slice's end lands exactly on the tensor's
   recorded end (`offset + tensor_byte_size`) -- this is the strongest
   internal check available on a per-expert offset without a second,
   independently-computed ground truth file, and it is sensitive to an
   off-by-one or a swapped in/out dimension in `resolve_expert`.
3. Resolve `ffn_down_exps.weight` on layers 34, 38 and 39 -- the three
   layers this GGUF's UD quant keeps at Q6_K instead of Q4_K -- and check
   the loader reports `q6_k` for them and that expert 0's offset matches the
   tensor base.
4. Print the profile numbers read from `serve/model_qwen35moe.mojo` and the
   sizes `size_host_state` derives from them.

Regression: `./run-tests.sh` (unchanged) and `bench/force-ab.sh` between a
`main`-built engine and one built from this branch, 20/20 prompts, so the
existing qwen35 path (`serve/harness.mojo`'s q4/q8 `load_pack`, untouched)
is provably unaffected by this step. Both go through `gpu-wait`.

## Registered prediction

- Tensor count: 733.
- Resolved byte total: 21,005,191,680 (matches `pack.bin` exactly).
- `blk.0` offsets (from the independent Python read of the real index):
  router 329850880; shared gate 331956224, shared up 484065280, shared down
  177741824; expert0 gate 178855936, expert0 up 333070336, expert0 down
  26746880.
- Expert 255 of each `blk.0` routed projection ends exactly at that
  projection's recorded tensor end (no FAIL from the last-expert check).
- Layers 34, 38, 39 `ffn_down_exps.weight` report dtype `q6_k`; their
  expert-0 offset equals the tensor's own recorded offset.
- Profile printout: H 2048, QF 8192, KV 512, N_LAYERS 40, N_SSM 30, N_ATT 10,
  HD 256, NKVH 2 (read from `serve/model_qwen35moe.mojo`, not restated).
- `run-tests.sh` exit 0; `force-ab.sh` 20/20 prompts, min and mean 100.0%,
  void none (the existing qwen35 default path is untouched by this step).

## Scoring

Pass only if every number above matches exactly and the probe itself prints
`MOE LOADER PROBE: PASS` (it prints `FAIL` and exits 1 on any mismatch,
including a missing tensor, an unresolved name, or a wrong dtype). Any
regression failure or an inexact match fails the gate; no partial credit.

## Failure meanings

- Tensor count or total-byte mismatch means a dtype byte-size formula is
  wrong, or `index.txt` itself changed shape since the independent read.
- A wrong `blk.0` offset means `resolve_expert`'s name construction or its
  shared/routed base-offset branch is wrong.
- The expert-255 check failing means the per-expert stride (`out_dim *
  row_bytes`) is wrong even though expert 0 (offset 0 into the tensor)
  looked right -- expert 0 alone cannot catch a stride bug.
- A Q6_K layer reporting the wrong dtype, or a wrong expert-0 offset on
  those layers, means the loader's dtype-driven block geometry silently
  defaulted to Q4_K's 144-byte block instead of Q6_K's 210-byte block.
- A regression failure means this step's new, additive code somehow altered
  the existing `load_pack` path it was written not to touch.

## Outputs

Durable outputs: `serve/moe_pack.mojo`, `tools/moe-loader-probe.mojo`, this
protocol, the probe's printed receipt (recorded in the Result section and
the lane report), and `exchange/lane-MOE-LOADER-report.md`. No output is
written to `/tmp`.

## Result

PASSED on 2026-09-12. `tools/moe-loader-probe .work/moe-w1/pack`:

- `tensors resolved: 733`; `resolved bytes: 21005191680 pack bytes: 21005191680`
  (exact match, matches the independent Python read).
- `blk.0`: router 329850880; shared gate 331956224, shared up 484065280,
  shared down 177741824; expert0 gate 178855936, expert0 up 333070336,
  expert0 down 26746880 -- every one matches the registered prediction
  exactly, and the last-expert (255) end-of-tensor check for gate/up/down
  raised no FAIL.
- Layers 34, 38, 39 `ffn_down_exps.weight`: dtype `q6_k`, expert0 offsets
  13828560640 / 15869454336 / 16435647232, matching the tensor's own
  recorded offset -- no FAIL.
- Profile printout: `H 2048 QF 8192 KV 512 N_LAYERS 40 N_SSM 30 N_ATT 10
  HD 256 NKVH 2`; host state `kvpool 5898240 kvpool1 589824 conv_slot 737280
  ssm_slot 15728640 slots 9`.
- `MOE LOADER PROBE: PASS`.

Deviation from the original plan, disclosed: mid-lap, the maintainer asked for a pack/
profile mismatch guard in `serve/harness.mojo`'s `load_pack` (not part of the
original prediction above, added as an extra, in-scope finding -- see the
lane report for the before/after receipt and the exact detection rule). That
guard's own regression is the same `run-tests.sh` / `force-ab.sh` pair as
this protocol's; both are reported together.

Regression: `./run-tests.sh` exit 0. `bench/force-ab.sh` (main-built
`38478b34aa4aaf4d` vs this-branch-built `bd850620005e0e8c56`, distinct
hashes): 20/20 prompts, 64/64 each, min 100.0%, mean 100.0%, void none.
