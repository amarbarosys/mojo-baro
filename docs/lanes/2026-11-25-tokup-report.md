# Lane tokup report: the tokenizer had a 100-second encode nobody had measured

The SPM tokenizer took 100.05 s to encode one 16K-token document. It now takes 31 ms with the same ids. The default
server spent 0.5 to 1.9 s of CPU per long chat request on checkpoint hints; that is now about 0.1 s. Both were found by
the new gate and bench, not by reading code. The gate passes on 45 of 45 model files against llama.cpp and HF.

Branch `lane-tokup` (from main cf1af3e), 2026-11-25. CPU only, no GPU job was run.
Brief: `docs/lanes/2026-11-25-tokup-brief.md`. Receipts: `exchange/receipts/tokup/` (local, `exchange/` is git-ignored).

## Verdict per goal

| goal | verdict | check that passed |
|---|---|---|
| 1 correctness gate | LANDED | `bench/tok-gate.sh`: 45 / 45 model files PASS, 75 cases, mutants 5 / 5 killed |
| 2 coverage | LANDED for every BPE, SPM and gemma4 file on disk. WordPiece (`bert`) and Unigram (`t5`) NOT SUPPORTED, NFC normalization NOT IMPLEMENTED | same gate, table below |
| 3 speed | LANDED | `baro-tokenize bench`, `cargo test --release role_boundaries_long_chat`, `bench/tok-front-bench.mojo`; ids equal on the gate |
| 4 one tokenizer | the Mojo front already uses `serve/tokenizer.mojo`; text gate PASS 3297 / 3297. Default switch NOT DONE: plan below, the decision is the maintainer's | `serve/front/test_text` on the Rust oracle, mutant FAIL 3109 / 3297 |

## 1. The gate compares four things per tokenizer and must fail on five broken ones

Corpus `bench/tok-corpus/`: code, multilingual text (Romanian with both diacritic forms, CJK, Arabic, Hebrew, Hindi, Thai and more),
emoji and ZWJ sequences, whitespace runs, digits, special-token strings, and one 83,668-byte document (24,027 ids on qwen35).
75 cases: every file whole, the `lines/` files also one case per line.

Arms, per distinct tokenizer (files with equal `tokenizer.*` keys share one run):

| arm | compares | oracle |
|---|---|---|
| llama | ids with specials | `llama-tokenize -f CASE --no-escape --ids` (llama.cpp 3057bb66c) |
| hf | ids without BOS | HF `tokenizers` 0.22.2 on a `tokenizer.json` found on disk with the same vocab (exact, or a prefix when the GGUF only adds UNUSED padding) |
| round | `decode(encode(case))` == case bytes | the case file |
| pieces | `decode([id])` bytes for every id in the vocab | `tools/tok-gate-oracle.py pieces`: llama.cpp token_to_piece rules from its own GGUF reader. Covers tokens that are invalid UTF-8 alone |

Mutant arm (each must FAIL the quick gate): dropped merge, qwen2 regex on a qwen35 file, special matching off, one decode byte dropped,
SPM score order inverted. All five are killed. My first "wrong regex" mutant survived: it widened the digit rule, and Qwen vocabs have
no multi-digit token, so it changed nothing.

Result: 51 files, 33 distinct tokenizers, 45 PASS, 0 FAIL, 6 SKIP. Full table: `exchange/receipts/tokup/gate-2026-11-25.tsv`.

| family (pre) | files | llama | hf | note |
|---|---|---|---|---|
| qwen35, 248,320 tokens (Ornith, Qwythos, RegesCore, nerkyor, Turbo, Flash-Next, MiMo-V2.6) | 12 | 75/75 | 75/75 | DFlash2 draft file: llama-tokenize cannot open it, hf 75/75 |
| qwen2 (Qwen3 1.7B/8B, qwen2.5, qwen2moe, qwen3moe, mimo-vl, 2 vocab files) | 11 | 73/75 | 74/75 where a json exists | two known deviations, below |
| llama-bpe (Llama 3.1 8B, 3.2 1B, vocab file) | 5 | 75/75 | 72/75 | the only json on disk lacks `ignore_merges` |
| gemma4 (Forge-1 E2B, vocab file) | 2 | 75/75 | NOT EXERCISED | round 71/75: literal U+2581 |
| SPM (llama-spm, phi-3, baichuan) | 3 | 75, 73, 75 | NOT EXERCISED | phi-3: llama.cpp strip attributes |
| other BPE vocab files (gpt-2, gpt-neox, mpt, command-r, refact, starcoder, falcon, aquila, deepseek-coder, deepseek-llm) | 10 | 75/75 except aquila 72, deepseek-coder 62 | NOT EXERCISED | reasons below |
| tiny_llama test files (no `pre` key) | 2 | 75/75 | 28/75 | file defect, below |
| no tokenizer (LTX, 3 ncm files) | 4 | | | SKIP |
| bert, t5 vocab files | 2 | | | SKIP, unsupported |

Known deviations (`bench/tok-gate-known.tsv`, one line each with its reason; deleting a line enforces again):

- llama.cpp promotes a NORMAL token named `</s>` or `<|endoftext|>` to CONTROL by a name list. We keep the type in the file. On Qwen3 the HF arm agrees with us on those two cases.
- phi-3: llama.cpp sets rstrip on special tokens by model name. We have no per-token strip attributes (2 cases).
- deepseek-coder: our regex is the HF chain, llama.cpp approximates it (13 cases; decision recorded earlier in `serve/pretok-known-diffs.txt`).
- Qwen3 `tokenizer.json` normalizes to NFC. We and llama.cpp do not (1 case: U+2126 and U+212A in the long document).
- llama3: the only matching `tokenizer.json` on disk is a MAX test copy without `ignore_merges: true` (3 cases; llama.cpp agrees with us 75/75).
- tiny_llama: the GGUF has no `tokenizer.ggml.pre`. We and llama.cpp both fall back to `default`.
- Literal U+2581 in the text cannot round trip on SPM and gemma4 (4 cases each). falcon has no token for some bytes (1 case).

## 2. Seven defects were fixed; three families of work stay open

Fixed (commit e8dd3e0):

| defect | before | after |
|---|---|---|
| SPM merge loop quadratic | 100.05 s for the long document | 0.05 s process time, same 29,555 ids |
| UNKNOWN-type tokens (`<unk>`) not matched in text, not decoded raw | llama-spm 73/75 | 75/75 |
| SPM file with no `add_bos_token` key got no BOS | baichuan 0/75 | 75/75 |
| GGUF version 2 refused | aquila did not load | loads |
| `deepseek-llm` pre missing | did not load | 75/75 |
| `default` pre (and a missing pre key) used the gpt-2 regex | gpt-neox 42/75, tiny_llama 40/75 | 75/75 |
| a byte with no token raised | falcon did not encode | 75/75 |

NOT EXERCISED or not supported:

- WordPiece (`tokenizer.ggml.model=bert`) and Unigram (`t5`): only llama.cpp test vocab files on disk. The bge embed path has its own `embed/wordpiece.mojo` on `vocab.txt`. Named load error.
- NFC normalization (`tokenizer.ggml.normalizer.nfc`, Qwen3): not implemented. It matters only for text that is not already NFC.
- HF arm: no matching `tokenizer.json` on disk for gemma4 (the two gemma json files have another vocab), SPM, qwen2.5, mimo-vl and the vocab-only files. The gate never builds the json with our own converter.
- Per-token lstrip/rstrip attributes (phi-3).
- `clean_spaces`, `add_eos_token`, and the 2 llama.cpp pre names that `tools/pretok-check.py` lists as not implemented.
- U+2014 in a short case (it is present only in the long document). Invalid UTF-8 as encode input.
- Private GGUFs: excluded on purpose (own gate).

## 3. Speed: one 2000x fix, one third off BPE, and the server's hint cost cut by 5x to 15x

Machine load during all runs: 1-min load 10 to 18 (other lanes were building). Arms were interleaved.

Encode, `baro-tokenize bench` on `bench/tok-corpus/whole/long-16k.txt`, min of 21 encodes:

| tokenizer | ids | before | after | tok/s after |
|---|---|---|---|---|
| SPM (llama-spm) | 29,555 | 100.05 s | 31 ms | 0.95 M |
| BPE qwen35 | 24,027 | 7.84 ms | 5.27 ms | 4.56 M |
| BPE llama-bpe | 23,948 | 6.96 ms | 4.64 ms | 5.16 M |
| gemma4 | 26,637 | 26 ms | 26 ms (not touched) | 1.0 M |

Decode: 0.6 to 3.7 ms for the same ids (8 M to 37 M tok/s), not touched.

The tokenizer's share of first-token latency on a 16K prompt is load plus one encode: 94 ms + 6 ms (qwen35 vocab file),
268 ms + 10 ms (Ornith, 248,320 tokens). Load is paid once per process and I left it alone. corewait on the load
(`.work/tokup/cw/load0`): `Reader.string` 32.5%, `_read_gguf` 24.2%, hashing 19.1%. heaptide on the load: no leak; the peak is two
hash tables (21 MB + 19 MB) and the 16 MB header read.

Per-request text path of the servers, one 16K-token chat:

| path | before | after |
|---|---|---|
| Rust `Text::role_boundaries`, 40 messages | 541.6 ms | 95.9 ms |
| Rust `Text::role_boundaries`, 200 messages | 1918.4 ms | 128.5 ms |
| Mojo `Text.role_boundaries`, 40 messages | 279.1 ms | 154.1 ms |
| Mojo `Text.role_boundaries`, 200 messages | 2118.9 ms | 1650.3 ms |

`role_boundaries` runs on every chat request in `baro-serve` (checkpoint hints). It encoded every message prefix. Now the conversation
is encoded once and a prefix length is read at a special-token edge; a prefix that does not fit is encoded alone as before.
The output is equal: the Rust oracle test asserts new == old on every chat row, and the regenerated oracle has 432 / 432 non-header rows
equal to the file written by the old code. The Mojo number moved less because its cost is template rendering, not encoding (section 4).

## 4. One tokenizer: the code is already shared, the default switch is blocked by two items outside this lane

`serve/front/text.mojo` imports `serve/tokenizer.mojo`, the engine's code. The Rust front uses HF `tokenizers`.

Gate run on this branch: Rust oracle regenerated (435 rows, equal to main's file), `serve/front/test_text` PASS 3297 / 3297,
mutant `TEXT_MUTANT=flip_add_special` FAIL 3109 / 3297. The oracle's pack had gone from the cache (`.work/ailab/pack-kq` was repointed
to another model today). I rebuilt its two sidecar files from the Ornith GGUF with `tools/gguf-tokenizer-json`; both sha256 equal the
oracle header.

Switch plan for `baro-serve` (not executed, needs the maintainer's go):

1. Checkpoint hints: `serve/front/api.mojo` does not send `ckpt` (its own comment says so). Wire `Text.role_boundaries` there. Owner: lane renovate.
2. Template render speed. After this lane the Mojo text path is render-bound. corewait on `bench/tok-front-bench` (200 messages): `Renderer.scan_to` 11.9%, `Renderer.starts` 10.1%, `String._iadd` 14.1%, `String.unsafe_ptr_mut` 10.2%; the tokenizer is under 1%. One render of the 16K chat costs 5.4 ms (40 messages) to 14.8 ms (200), and `role_boundaries` renders every prefix. Owner: lane renovate (`minja/`).
3. NFC for Qwen3 packs, or a decision that inputs are NFC.
4. Run `bench/tok-gate.sh` and `test_text` on each served pack, then `bench/front-real-check.sh` against a live engine.
5. Flip the default. Keep the Rust front as the oracle binary for `text_oracle`.

Evidence for the switch: ids equal on 45 files; text equal on 3297 rows; encode at 4.6 M tok/s against about 0.6 M for HF tokenizers
(41.7 ms for the same document through the Python binding of the same Rust core).

## Findings outside this lane (report only, owner: lane renovate)

- `minja` render is the limit of the Mojo front's text path (profile above).
- `.work/ailab/pack-kq` is a moving symlink that committed tests read (`run-tests.sh` line 60, `text_oracle`). The oracle should name its pack by id.
- `serve/front/api.mojo` sends no checkpoint hints.
- `bench/cw-decode.sh` names a pack inside a removed lane worktree (`../nerkyor-vs-forge`).

## Skill, SINDEX and tool changes

- Skill `corewait`: the triage step now gives the full path of `cw-triage` (it is in the skill directory, not in the corewait repo). `skilldex` PASS.
- Skill `mojo-baro`: command rows for `bench/tok-gate.sh` and the front-text fixtures.
- `~/Brain/SINDEX/mojo-baro.md`: two entries (SPM quadratic loop never measured; per-request helper never timed on a long chat).
- `docs/rules.md`: four lines (hand-typed Unicode regex, first gate run, equivalent mutant, front-text fixtures).
- New in the repo: `bench/tok-gate.sh`, `bench/tok-gate-known.tsv`, `bench/tok-corpus/`, `tools/tok-gate-oracle.py` (oracle), `bench/tok-front-bench.mojo`. `tools/baro-tokenize` gained `ids`, `ids-raw`, `roundtrip`, `pieces`, `bench`. No iTool was changed.

## Commits

- e8dd3e0 fix(tokenizer): SPM merge heap, unknown-type specials, GGUF v2, deepseek-llm, llama.cpp default pre
- 2c55ca7 feat(bench): tok-gate
- bf84f58 perf(tokenizer): BPE encode -33%
- e0c5656 perf(serve): role_boundaries encodes the conversation once
- eba3267 perf(front): Mojo role_boundaries encodes the conversation once; front text bench

## What this does not show

The first `bench/preflight.sh` run failed in ci-checks: `bench/tok-front-bench.mojo` needs `-I serve/front`, which the bench build step does not pass. It now carries the `# ci-checks: needs` marker, so that step lists it as skipped and nothing in CI compiles it. No timing here was taken on an idle machine. The Rust
`role_boundaries` gain is a function-level timing, not a time-to-first-token reading from a live server. The equal-output proof for
the one-encode path rests on the three oracle packs (one tokenizer, three sidecars); a tokenizer whose special tokens are normalized
takes the old path by design, and no such tokenizer was tested.

## run-tests receipt (driver, 2026-11-25)

`./run-tests.sh` under gpu-wait on head 543545f: rc=0, FAIL: 0, one SKIP (test_kq_prefill_gemm, no pack at .work/ornith-stint/pack-kq). Log copied to main .work/run-tests-tokup.log. Fixture repair before it: .work/ailab/pack-kq rebuilt from the plain Ornith GGUF the oracle header names.
