# Tokenizer — text in, text out, bit-equal to llama.cpp

> **Default path since 2026-09-08: the Mojo tokenizer** (`serve/tokenizer.mojo`, CLI `tools/baro-tokenize.mojo`, gate `tools/test_tokenizer_mojo.py`) — see the last section. The Python builder and gate that preceded it were deleted 2026-09-16 (git history has them); the Python CLI below stays for `--chat`.

The engine reads token ids and prints token ids. This layer turns text into
those ids and back, built from nothing but the GGUF's own metadata, and is
gated on producing exactly the ids llama.cpp produces on the same file.

| | |
|---|---|
| builder | `tools/gguf-tokenizer-json.mojo MODEL.gguf OUTDIR` (called by `tools/baro serve`) |
| gate | `tools/test_tokenizer_mojo.py`, `tools/test_gguf_tokenizer_json.py` |
| CLI | `tools/baro-tokenize.mojo encode\|decode\|count\|info`; `tools/retired/baro-tokenize.py` for `--chat` |
| runtime | HF `tokenizers` (Python dev dep; the Rust crate for the server) |

## Files next to a pack

```
.work/engine-pack-q4/
  pack.bin, index.txt        weights (tools/engine-pack.py)
  prompt-tokens.txt          the prompt the engine reads (ids, one per line)
  ref-tokens-64.txt          greedy reference for tools/check-tokens.sh
  tokenizer.json             HF tokenizers serialization, ~9.8 MB
  tokenizer-meta.json        ids, flags, special tokens, chat template
```

`tokenizer.json` is the standard HF format, loadable by the Python package
and the Rust crate without transformers. `tokenizer-meta.json`:

| field | Qwythos-9B value | from GGUF key |
|---|---|---|
| `model`, `pre` | `gpt2`, `qwen35` | `tokenizer.ggml.model`, `.pre` |
| `vocab_size`, `n_merges` | 248320, 247587 | `tokenizer.ggml.tokens`, `.merges` |
| `bos_token_id` / `bos_token` | null (none) | `tokenizer.ggml.bos_token_id` |
| `eos_token_id` / `eos_token` | 248046 `<\|im_end\|>` | `tokenizer.ggml.eos_token_id` |
| `pad_token_id` / `pad_token` | 248044 `<\|endoftext\|>` | `tokenizer.ggml.padding_token_id` |
| `add_bos`, `add_eos` | false, false | `tokenizer.ggml.add_bos_token`, `.add_eos_token` (absent = false, as llama.cpp) |
| `special_tokens` | 27 control tokens (`<\|im_start\|>` 248045, ...) | `token_type == 3` |
| `added_tokens` | 6 (`<think>`, `</think>`, `<tool_call>`, ...) | `token_type == 4` |
| `chat_template` | the Jinja string | `tokenizer.chat_template` |
| `source_gguf`, `built` | provenance | |

Every pack of this model family carries the same tokenizer (the metadata hash
is identical across the BF16, Q8_0 and Q4_0 GGUFs), so one build serves all
packs; the builder takes several OUTDIRs for that reason.

## How the tokenizer is assembled

Only `tokenizer.ggml.model == "gpt2"` (byte-level BPE) is supported:

- **vocab**: the GGUF token strings verbatim, id = array index. They are
  already byte-level (`Ġ` for space, `Ċ` for newline); the 243 `[PAD...]`
  entries of type UNUSED stay in the vocab so ids are dense and any id decodes.
- **merges**: `tokenizer.ggml.merges` verbatim, `"a b"` strings.
- **pre-tokenizer**: `Split(regex, isolated)` then `ByteLevel(use_regex=false)`,
  with the regex llama.cpp assigns to the recorded `pre` type
  (`src/llama-vocab.cpp`). `qwen2` is the Qwen2 pattern; `qwen35` is the same
  with `\p{M}` (combining marks) added to the letter runs. Any other `pre`
  aborts the build rather than guess.
- **no normalizer**: llama.cpp does not NFC-normalize BPE input, so neither do
  we (HF's Qwen `tokenizer.json` does; that is a deliberate divergence from HF
  to stay equal to llama.cpp — the NFD case in the hard set checks it).
- **added tokens**: CONTROL types are special (skipped by decode by default),
  USER_DEFINED types are plain added tokens. Both are matched in the raw
  text before BPE, which is llama.cpp's default `parse_special = true`
  behaviour (`llama-tokenize` without `--no-parse-special`).
- **no BOS**: nothing is prepended; `add_bos` is honoured by the CLI if a
  future GGUF sets it.
- **decoder**: `ByteLevel`; `decode(encode(x)) == x` for any valid UTF-8 text.

## Gate

```
.venv/bin/python tools/test_tokenizer_mojo.py              # same cases, reference llama-tokenize
```

1. every `bench/mtp-prompts/*.txt` (checked equal to `prompts.json`) encodes
   to its `.tokens` file, the llama.cpp reference used by every MTP number;
2. a hard set (NFD accents, Cyrillic/Greek/Arabic/Hebrew/Devanagari/Thai,
   Chinese/Japanese/Korean, emoji with ZWJ/flags/skin tones, tabs and
   4-space code, CRLF, NBSP/ZWJ, long whitespace runs, numbers, URLs, JSON,
   markdown, every special-token family, look-alike non-tokens, the pack's
   rendered chat template, control characters, the empty string) encodes to
   what `~/llama.cpp/build/bin/llama-tokenize -m <source_gguf> --ids -p TEXT`
   prints (`""` is refused by llama-tokenize; with `add_bos=false` the
   reference is `[]`);
3. `decode(encode(x), skip_special_tokens=False) == x` on all of the above.

Exit code is non-zero on any mismatch. It needs the GGUF (path from
`tokenizer-meta.json`, override `--gguf`) and the llama.cpp build, so it is
**not** in `tools/ci-checks.sh`; run it by hand after rebuilding a pack or
touching the builder. `--dump FILE` writes every case's ids for diffing.

## CLI

```
tools/retired/baro-tokenize.py encode "Water boils at 100 degrees"      # ids, one per line
tools/retired/baro-tokenize.py encode --chat --system "Be brief." "2+2?" # chat template applied
tools/retired/baro-tokenize.py encode -o /path/prompt-tokens.txt - < file.txt
tools/retired/baro-tokenize.py prompt --chat "What is 2+2?"             # writes <pack>/prompt-tokens.txt
tools/retired/baro-tokenize.py decode - < .work/engine-pack-q4/ref-tokens-64.txt
tools/retired/baro-tokenize.py decode --keep-special 248045 846 198
tools/retired/baro-tokenize.py info
```

`--pack DIR` (or `BARO_PACK`) selects the pack; default `.work/engine-pack-q4`.
The script re-executes itself under `.venv/bin/python` when the invoking
interpreter lacks `tokenizers`. `--chat` wraps TEXT as one user turn and
appends the generation prompt (`<|im_start|>assistant\n<think>\n` for this
template; pass `enable_thinking` yourself via a custom messages list in
Python if you need the no-think form).

## Engine contract

The engine has no text path. It reads `<pack>/prompt-tokens.txt` (override
with `BARO_PROMPT=/path`), decimal ids separated by any non-digit byte, one
per line by convention, and prints the generated ids on the `GENERATED:`
line. `baro-tokenize prompt` writes that file; `baro-tokenize decode` reads
the `GENERATED:` ids back (paste them, or `grep GENERATED: run.log | cut -d: -f2 |
tools/retired/baro-tokenize.py decode -`).

## Server: loading in Rust

The `tokenizers` crate reads the same file:

```rust
use tokenizers::Tokenizer;
let tok = Tokenizer::from_file(format!("{pack}/tokenizer.json"))?;
let ids: Vec<u32> = tok.encode(text, false)?.get_ids().to_vec(); // add_special_tokens=false; no post-processor anyway
let text = tok.decode(&ids, /*skip_special_tokens=*/ true)?;
```

- The pattern uses `\p{M}` and a look-ahead, so keep the crate's default
  `onig` regex feature (the `fancy-regex` alternative also handles both).
- Merges are `"a b"` strings, accepted by every crate version; no token
  contains a literal space (byte-level), so the split is unambiguous.
- EOS to stop on, the pad id and the chat template come from
  `tokenizer-meta.json`; render the template with a Jinja engine
  (`minijinja`) providing `raise_exception`, `strftime_now` and `tojson`,
  and pass `messages`, `add_generation_prompt`, `bos_token`/`eos_token`
  as the Python side does in `render_chat`.
- `decode` with `skip_special_tokens=true` drops the 27 control tokens but
  keeps `<think>`/`<tool_call>` text, matching HF semantics for Qwen.

## Residual risk

Identity is proven on 61 cases against this llama.cpp build (`ca3d5a3e1`,
2026-08-28). The one known class that can still diverge is Unicode table
version skew between llama.cpp's `unicode-data.cpp` and Oniguruma for code
points assigned after either was generated; add such a string to `HARD_SET`
and the gate decides.

## Mojo tokenizer (2026-09-08, default path)

`serve/tokenizer.mojo` reads the same `tokenizer.ggml.*` keys straight from the
GGUF header (96 MB cap, no tensors) and runs byte-level BPE in Mojo; pre-tokenizer
regexes run on `uregex/`, vendored at the repo root (`-I .`; upstream is
[`mojo-uregex`](~/Projects/mojo/mojo-uregex), `tools/ci-checks.sh` checks the
vendored copy against it when that upstream is on the machine).
Pre types: qwen2 / deepseek-r1-qwen, qwen35, llama3 / llama-bpe, spark2_5 (4-pass
Sequence), gpt-2 / default. Load 0.06 s for a 131k vocab.

```
./.venv/bin/mojo build tools/baro-tokenize.mojo -I . -I serve -o .work/baro-tokenize
.work/baro-tokenize (encode TEXT | decode IDS | decode-keep IDS | batch NUL_TEXTS | info) MODEL.gguf
./.venv/bin/python3 tools/test_tokenizer_mojo.py --gguf <Qwythos.gguf> \
    --extra <Spark.gguf>:.work/spark/prompt.txt:.work/spark/ref/prompt-tokens.txt
```

Spark directly: `~/Models/llama.cpp/build/bin/llama-tokenize` knows the
`spark2_5` architecture (the `~/llama.cpp` and `imports/llama.cpp` builds do
not), so the full gate runs against it:
`tools/test_tokenizer_mojo.py --gguf ~/Models/spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M.gguf --llama-tokenize ~/Models/llama.cpp/build/bin/llama-tokenize`
(61/61). The `.work/spark/` `--extra` files no longer exist on this machine.

Gate (ref = `llama-tokenize` on the source GGUF + Spark ids from llama-server
`/tokenize`): 63 cases, encode identical, `decode-keep(encode(x)) == x`.
`serve/spark.mojo` takes `BARO_PROMPT_TEXT=<file> BARO_GGUF=<gguf>` and tokenizes
in-process; `BARO_PROMPT=<ids>` still works. The Python tool above stays as the
oracle-side builder for HF `tokenizer.json` consumers.

### Dense family targets (preregistered 2026-09-11, lane TOK)

Adds SPM (`tokenizer.ggml.model == "llama"`, scores-based merge, byte-fallback via
`<0xXX>` tokens) and the `granite-docling` BPE pre-tokenizer (same regex as
`gpt-2`/`default`) to `serve/tokenizer.mojo`. Two real bugs found while wiring the
`llama-bpe` target and fixed in the same commit: `llama3`/`llama-bpe` GGUFs carry no
`tokenizer.ggml.add_bos_token` key, so `add_bos` must default true for that pre-type
(llama.cpp's `llm_tokenizer_bpe` constructor hardcodes it); the same pre-type sets
`ignore_merges` (a whole regex-split piece that is already a vocab token is emitted
directly, skipping BPE merges), needed because Llama-3.2's vocab has number tokens
with no reachable merge chain.

Targets and GGUFs (2026-09-11):

| target | pre / model | GGUF |
|---|---|---|
| Llama-3.2-1B-Instruct | `llama-bpe` (BPE) | `~/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| Qwen2.5-7B-Instruct | `qwen2` (BPE, already supported) | `~/Models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| granite-4.2-3b | `granite-docling` (BPE) | `~/Models/granite-4.2-3b-bf16/granite-4.2-3b-BF16.gguf` |
| lily-cybersecurity-7b | `llama` (SPM) | `~/Models/lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-q6_k.gguf` |

Gate: `tools/test_tokenizer_mojo.py --gguf <target>` (`HARD_SET` + the 20
`bench/mtp-prompts` prompts, both re-encoded live against `llama-tokenize` on that
GGUF rather than read from a stale `.tokens` file, so the same script is correct
for any target) must PASS 0 failures per target, plus `decode(encode(x)) == x`.
`HARD_SET` was 40 cases, not the plan's "41" (counted precisely, not by the grep
that inflated it); its own `"empty"` entry is now folded into the gate's single
dedicated empty-string case below (39 + empty + 20 prompts + chat = 61 cases per
target). Regression floor: the existing default target (Qwythos, `qwen35`) plus
the Spark `--extra` case, previously 63/63, now 62/62 after that fold (same
coverage, the duplicate empty-string expectation removed).

The empty-string case needed its own fix: `add_bos_token` defaults to `true` for
`llama3`/`llama-bpe` GGUFs that omit the key (llama.cpp hardcodes it in
`llm_tokenizer_bpe`'s constructor), so `encode("", add_special=True)` correctly
returns `[bos_id]`, not `[]`, on such models -- the test now reads
`add_bos_token` (or that pre-type default) straight from GGUF metadata instead of
inferring it from whether some case's reference ids happen to start with
`bos_id`, which produced a false positive on Qwen2.5 (`bos_token_id` there is the
same id as `<|endoftext|>`, and the `special-eot` case's text legitimately starts
with that token).

### Chat templates (2026-09-08)

The server can override the pack template without changing the pack:

```sh
tools/baro serve MODEL.gguf --chat-template-file custom-chat.jinja
```

The file is raw UTF-8 Jinja. It takes precedence over `chat_template` in
`tokenizer-meta.json`; the existing metadata template and ChatML fallback are
used when no override is supplied. The render context remains
`messages`, `tools`, `chat_template_kwargs`, `bos_token`, `eos_token`, and
`add_generation_prompt`.

`serve/spark.mojo` renders `tokenizer.chat_template` in-process with
`minja/` at the repo root (`-I .`; canonical here since 2026-10-06, byte-exact
vs jinja2 on `tests/minja-corpus/`, gate `tools/minja-gate.sh`):
`BARO_CHAT=<case.json> BARO_GGUF=<gguf>` where the JSON is
`{"messages": [...], "add_generation_prompt": true, "enable_thinking": false, ...}`.
Oracle `tools/spark-chat-ref.sh` (llama-server `/apply-template` → `/tokenize` →
greedy 64); gate = engine ids identical (Spark: 25 prompt ids, 43/43 generated).

## Pre-tokenizer coverage

The `gpt-4o` and `deepseek-coder` families use the HF regexes verbatim.
`gpt-4o` (gpt-oss, Phi-4-mini, MiniMax-M2, llama4, kanana2, talkie) is the o200k
regex with `\p{M}`; the earlier rewritten form split combining marks off their
base letter (332/400 corpus lines vs HF). `deepseek-coder` is HF's chain of five
Splits plus digits (explicit letter ranges, not `\p{L}`); it was 182/400 lines
vs HF, now 22215/22215.

`tools/pretok_harvest.py` measures coverage against the top N HF
text-generation models (tokenizer.json only, no weights): top 250 on
2026-10-08 = 164 covered, 68 without tokenizer.json, 17 without a regex,
1 unmatched (falcon-7b).

```
./.venv/bin/python3 tools/pretok_harvest.py --n 250 --corpus ../mojo-uregex/.work/bench/corpus.txt
```
