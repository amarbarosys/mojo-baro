# Report: gemma4 (Forge-1 E2B) packer and dense-engine wiring (lane-gemma4-port, 2026-11-16)

Skills loaded before the first edit: `model-bringup` (route `arch-port`, rules), `gguf-forensics`,
`gguf-tokenizer-spec`, `gpu-kernels` (rules, `inference-kernels`), `bench-gates`, `mojo-dev` (`mojo-syntax`,
`mojo-lean`, rules), `mojo-nightly-lane-builder`, `memory-optimization`, `lanes`, `mojo-baro`, `clean-room-build`.
Provenance of the architecture facts: `docs/provenance/gemma4.md`.

## M1 packer: PASS (`d24f685`)

`tools/gemma4-pack.mojo` writes `index.txt` (541 lines: name, dtype, file offset, bytes), `profile.txt` (one line per
layer: kind, head dim, own KV, KV source, pool slot, MLP width), `profile.mojo` (comptime shapes) and links
`pack.bin` to the GGUF. The weights are read in place: 0 bytes on disk instead of a 9.26 GB copy.

Commands and receipt (PACKDIR is the pack directory, the binary is `.work/gemma4-pack`):

```
.work/gemma4-pack $HOME/Models/forge-1-gemma-4-e2b-f16/Forge-1-E2B-F16.gguf .work/gemma4/pack
tools/gemma4-pack-check.sh ~/Models/forge-1-gemma-4-e2b-f16/Forge-1-E2B-F16.gguf .work/gemma4/pack
```

PACKDIR = `$HOME/Projects/mojo/mojo-baro-lanes/gemma4-port/.work/gemma4/pack`. Receipt (local, `exchange/` is
not tracked): `exchange/receipts/gemma4-port/pack-check.log`, also `.work/gemma4/pack-check.log`:

```
== arm: pack ==
entries 541 tensors 541 bytes 9257711756 problems 0
== reverse arm: one offset moved by 2 bytes must fail ==
FAIL blk.0.attn_q.weight: index (5584049730,6291456,f16) vs gguf-py (5584049728,6291456,f16)
entries 541 tensors 541 bytes 9257711756 problems 1
PASS gemma4-pack-check: every tensor accounted for, bytes equal, mutant caught
58.05 s 19302220 KB
```

The check uses gguf-py (llama.cpp's reader) as the oracle: count, total bytes, every entry's offset, size and
type, sha256 of the bytes, f16 -> f32 round trip, and `profile.txt` against the kv_source rule recomputed in
Python from the GGUF keys. Profile: 12 S slots (layers 0-3, 5-8, 10-13), 3 F slots (4, 9, 14); shared layers
15-34 read layer 13 (S, slot 11) or 14 (F, slot 2); MLP 6144 on 0-14, 12288 on 15-34.

## M2 forward, layer by layer: PASS (`6e884fb`)

`serve/gemma4.mojo` (engine), `kernels/gemma4_ops.mojo` (f16 GEMV with gelu epilogues, row RMSNorm with residual
and scale, NEOX rope with `rope_freqs`, embed, PLE input). The prompt runs as single decode steps (no batched
prefill yet). Reference: llama.cpp 956591f `llama-debug` with the `LLAMA_DUMP_DIR` patch
(`mojo-baro-archive/exchange/receipts/latent-wa/llama-debug-cpp-dump.patch`), built CPU-only in
`.work/gemma4/llama-dump` (the shared tree is untouched), f32 KV, `--tensor-filter '^l_out-[0-9]+$'`.

```
bench/gemma4-layers.sh ~/Models/forge-1-gemma-4-e2b-f16/Forge-1-E2B-F16.gguf .work/gemma4/pack bench/spark-prefill-prompts/p0768.txt .work/gemma4/m2-b
```

Receipt `exchange/receipts/gemma4-port/m2-layers.log` (746 ids, so the sliding layers see the window edge):

```
PASS gemma4-layers: gate layers 0 1 4 15 34 min cos 1.000000 1.000000 1.000000 1.000000 1.000000, bar 0.9999, below: []
```

All 35 layers are above 0.9999, gated or not; the lowest is layer 11 at position 567, 0.9999339. That position is
sensitive in llama.cpp too: its f16-KV run against its own f32-KV run reads 0.9998 there, and against the f16-KV
run we read 0.9995 (first M2 run, `.work/gemma4/m2-a`). The gate therefore uses the f32-KV reference, as
`bench/dense-run.sh` does. llama.cpp keeps only the last row on layers 15 to 34 (it prunes rows before the first
shared-KV layer), so those layers compare one position.

Numbers from the gate job: weights 9.26 GB read from the GGUF in 1.23 s; 746 single steps 5.86 s (7.85 ms per step).

## M3 decode, teacher-forced agreement: PASS, reverse arm FAILs as it must

The gate is the repo's dense gate, `bench/dense-run.sh`, with three new knobs: `ENGINE_SRC=serve/gemma4.mojo`,
`TOKENIZER` (here a llama-tokenize shim, `.work/gemma4/llama-tok`, written before our tokenizer read gemma4)
and `LLAMA_SERVER`. The reference is llama.cpp 956591f `llama-server` on the CPU with f32 KV, `-c 16384`. Our
profile TMAX is 16384, so `min16k` passes. 20 held-out prompts (`bench/mtp-prompts`), 64 forced ids each. The bar
was frozen before the run: MIN_PCT 90 per prompt, at least 15 prompts exercised.

```
gpu-wait run --timeout 2400 --vram 12 -- env ENGINE_SRC=serve/gemma4.mojo TOKENIZER=$PWD/.work/gemma4/llama-tok \
  LLAMA_CACHE=$PWD/.work/gemma4/refcache bench/dense-run.sh GGUF .work/gemma4/pack .work/gemma4/pack .work/gemma4/m3-a
```

| arm | verdict | agreement |
|---|---|---|
| ours | PASS | min 100.0%, mean 100.0% over 17 prompts; 3 not exercised (reference < 16 ids: p06 13, p13 1, p14 7, all 100%) |
| reverse: `BARO_KV_ROUTE=own` (shared-KV layers read slot 0 of their pool, i.e. layer 0 / layer 4) | FAIL | min 0.0%, mean 3.7%, 17 of 17 below the bar |

Receipts: `exchange/receipts/gemma4-port/m3-SUMMARY.txt`, `m3-results.txt`, `m3-reverse-*.txt`. The
reference tokens were cached in the first job (`.work/gemma4/refcache`) and reused by the reverse job.
Decode in the forward arm: 119.1 to 120.2 tok/s over 64 tokens on every prompt.

What M3 does not cover: the held-out prompts are 8 to 50 ids long, so no forced id sits past the 512-token window
or in the split full-attention range (`g4_nsplit` > 1 from T 192). The step code there is covered by M2: the
engine has no separate prefill, each of the 746 M2 positions ran the same one-token decode step (window edge and
split attention included) and matched llama.cpp at cos >= 0.99993. Forced agreement at a long context is open.

## M3b long-context arm: PASS

The boss's add-on to M3, run before M4 is called done: forced ids at long positions, same bar (MIN_PCT 90), all
prompts exercised (`MIN_N=3`). The prompts are `bench/gemma4-long-prompts/l00600.txt`, `l05000.txt` and
`l15000.txt`: repo prose (README, CAPABILITIES, ENGINE-ROADMAP, KERNELS), cut mid-paragraph at 554, 4965 and 14967
ids. Each is followed by 64 forced ids, so the forced positions are 554 to 617, 4965 to 5028 and 14967 to 15030.
That puts them past the 512-token window and in the split full-attention range (`g4_nsplit` 8 from T 512). The
reference is the same: llama.cpp CPU, f32 KV, `-c 16384`.

```
gpu-wait run --timeout 3600 --vram 12 -- env PROMPT_DIR=bench/gemma4-long-prompts PROMPTS='l*' MIN_N=3 \
  ENGINE_SRC=serve/gemma4.mojo TOKENIZER=$PWD/.work/gemma4/llama-tok LLAMA_CACHE=$PWD/.work/gemma4/refcache \
  bench/dense-run.sh GGUF .work/gemma4/pack .work/gemma4/pack .work/gemma4/m3b
```

| prompt | ids | forced agreement | decode tok/s (64 tokens) | our prefill (single steps) |
|---|---|---|---|---|
| l00600 | 554 | 64/64 | 117.1 | |
| l05000 | 4965 | 64/64 | 115.7 | 38.7 s |
| l15000 | 14967 | 63/64 | 110.8 | 119.2 s |

`PASS agreement: min 98.4% mean 99.5% over 3 prompts, bar 90%` (receipts `exchange/receipts/gemma4-port/m3b-*`).
The one miss is forced index 36 at 15k: llama.cpp `Lat` (24469), ours `Int` (3898). Both are plausible next words.
It is not diagnosed, because no logits were captured, so a near-tie is a guess and not a finding. (The `tok: 20/20
tokenized` line in the summary is a fixed string in `dense-run.sh`; 3 prompts were tokenized.)

## M4 tokenizer, chat, speed: PASS

Tokenizer (`329f037`): `serve/tokenizer.mojo` reads `tokenizer.ggml.model` gemma4. The spec, from llama.cpp's
vocab code: BPE on raw UTF-8 codepoints, spaces as U+2581, the text split into newline runs and other runs, a
newline run that is one token stays whole, add_bos forced on. `bench/gemma4-tok-parity.sh` against llama-tokenize:
23/23 (20 held-out prompts, the 746-id prompt, a unicode/whitespace text, a turn text with specials). The change
moves nothing else: Llama-3.2-1B, Ornith-1.5-9B and Qwythos-9B-v2 read 23/23 each.

Chat (`bench/gemma4-chat.sh`, receipt `exchange/receipts/gemma4-port/m4-chat.log`): `baro-serve` spawns
`serve/gemma4.mojo` with `BARO_SERVE=1`. The pack carries `tokenizer.json` (a link to the base model's HF file) and
`tokenizer-meta.json` (the GGUF's embedded template, bos 2, eos `<turn|>` 106). `baro chat -p` at temperature 0,
max_tokens 16000:

```
server tokenize parity 20/20
reply: My name is Forge. 17 * 23 = 391.
stats: [19 tok · 111.6 tok/s · prefill 0.7 s · cached 0] [finish stop]
PASS gemma4-chat: reply names Forge and 391, tokenize 20/20
```

That is the answer in the llama.cpp receipt (`RUN.md`). The llama.cpp run there spent 273 tokens because it
thought first (`--jinja`, reasoning on); ours answered without a thinking block under baro-serve's default
render. Sampling is not wired: the engine parses the sampling fields and decodes greedy.

Speed (`bench/gemma4-speed.sh`, one gpu-wait job, interleaved A B x 3, 256 tokens after a 14-id prompt, F16,
full offload; receipt `exchange/receipts/gemma4-port/m4-speed.log`, engine sha `db2333f3`, load average 5.9 at
the start):

```
ours tok/s 118.5 118.4 117.8 median 118.4
llama.cpp tg256 tok/s 100.6 99.9 100.4 median 100.4
RESULT gemma4-speed: ours / llama.cpp = 1.179 (decode 256 tokens, F16, full offload)
```

Bytes per token are about 4.56 GB of f16 weights (MLP 1.56 G params, LM head 0.40 G, attention 0.28 G, PLE 0.04 G),
so 118 tok/s reads about 540 GB/s. Prefill is one decode step per position (7.85 ms each at 746 positions);
no batched prefill.

## Result and open items

| milestone | gate | verdict | commit |
|---|---|---|---|
| M1 packer | gguf-py: 541/541 tensors, 9257711756 B, sha + round trip, profile rule; mutant caught | PASS | `d24f685` |
| M2 forward | per-layer cos >= 0.9999 at 0, 1, 4, 15, 34 (all 35 >= 0.99993) | PASS | `6e884fb` |
| M3 decode | forced agreement, 20 held-out, bar 90: min 100% over 17; reverse arm 3.7% FAIL | PASS | `3e12269` |
| M3b long context | 64/64, 64/64, 63/64 at 554, 4965, 14967 ids | PASS | `f0446cc`, `eb3bc71` |
| M4 | tokenizer 23/23; `baro chat` "My name is Forge. 17 * 23 = 391."; 118.4 vs 100.4 tok/s | PASS | `329f037`, `f0446cc` |

`tools/ci-checks.sh` passes on the lane (`3687ca0`, after `BARO_KV_ROUTE` was registered). Not run:
`bench/preflight.sh`, `run-tests.sh`, `lane-merge`; the lane is not merged. `kernels/gemma4_attn.mojo` is not
changed.

Open:
- Prefill is one decode step per position: a 14967-id prompt takes 119 s. A batched prefill (row GEMM over the f16
  weights) is the next speed lever for chat with long history.
- Sampling: `temperature > 0` is parsed and ignored (greedy). No softcap is applied: the argmax does not need it,
  sampling will.
- VRAM: the 4.7 GB per-layer token table sits on the device, while each token reads 17.9 KB of it. Keeping it on
  the host would free that VRAM.
- The single miss at 15k (forced index 36, `Lat` against `Int`) is not diagnosed.
- No self-describing bake yet: `tools/bake.sh` needs a gemma4 arch entry and reference ids from this engine.

## Merge gate (boss, 2026-11-16)
- `bench/preflight.sh`: pf=0. `./run-tests.sh` under gpu-wait: PASS run-tests: all tests, rc=0.

## wavewait captures of the new kernels (`kernels/gemma4_ops.mojo`)

One warm SQTT capture per kernel with `bench/ww.sh` (rocprofv3 ATT on WGP 3, engine sha `db2333f3`, real pack,
p01 prompt). There is no earlier version of these kernels, so this is a baseline, not a diff. No pass bar.

```
WW_TIMEOUT=600 WW_ITER=12 bench/ww.sh g4_gemv .work/gemma4/engine .work/gemma4/pack .work/gemma4/ww/g4_gemv BARO_GEN=2
WW_ITER=12 bench/ww.sh g4_ple .work/gemma4/engine .work/gemma4/pack .work/gemma4/ww/g4_ple
```

| kernel | capture | what it waits on |
|---|---|---|
| `g4_gemv` | PASS, `exchange/receipts/gemma4-port/ww-g4_gemv/ww.json` (16 waves on WGP 3) | global loads: `s_waitcnt vmcnt(10)` avg 8593 cycles (max 14441), then the LDS of the wave-sum (1038). 0 spills, 0 barrier. Memory-bound, as the 540 GB/s decode number says |
| `g4_ple` | PASS, `exchange/receipts/gemma4-port/ww-g4_ple/ww.json` (2 waves) | the LDS read after the `block_sum` barrier, `s_waitcnt lgkmcnt(0)` 1729 cycles, then global loads (609, 443, 393). 0 spills |
| `g4_rmsnorm` | no wave on WGP 3 | launches 1 to 8 workgroups (one per row), so none lands on the traced WGP |
| `g4_rope` | no wave on WGP 3 | 1 or 8 workgroups (one per head) |
| `g4_embed` | no wave on WGP 3 | 6 workgroups |

The first `g4_gemv` and `g4_rmsnorm` captures died at the 300 s ATT timeout, because the full 78-step run issues
about 47k dispatches under rocprofv3. With `BARO_GEN=2` and `WW_TIMEOUT=600` the `g4_gemv` capture finished. The
three small kernels cannot be traced at their real launch shape on this card (wavewait skill: one workgroup lands
on any CU and WGP 0/1 give no instruction tokens). A bench-sized launch of each would trace, but it would not be
the engine's shape.
- Boss re-run at 3a7ad6a: `./run-tests.sh` 3291/3297, the 6 FAIL are tokenizer-meta fixture sidecars (kq, bos, nochat) naming another gguf, same on main e2ec66e per lane, receipt `.work/run-tests-3a7ad6a.log`.
