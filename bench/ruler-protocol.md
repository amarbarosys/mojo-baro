# RULER protocol — long-context gate for M5 (K8/V4, K4/V4 KV)

Binds to `bench/PROTOCOL-RULES.md` P1-P6. This is the gate design §9/§10 M5
needs before it exists: `docs/design/agent-engine-2026-09.md` says a
quantised KV format ships only if its RULER effective length equals bf16's.
Built ahead of M5 (`docs/design/agent-engine-2026-09.md` §10) so the chat
lane can budget against real numbers, not a guess.

Reference: `.work/design/nvidia/ruler.md` (NVIDIA/RULER README digest); task
generator source pulled live via `gh api repos/NVIDIA/RULER/contents/...`
(scripts/data/synthetic/{niah,variable_tracking,common_words_extraction,
constants}.py, scripts/synthetic.yaml, scripts/eval/synthetic/constants.py) —
task *definitions* copied, not the harness (see Deviations).

## Tasks

Four families, matched to RULER's own `scripts/synthetic.yaml` complexity
space:

| task | RULER config it maps to | haystack | complexity |
|---|---|---|---|
| `niah_single` | `niah_single_2`/`niah_single_3` merged | essay | 1 key, value alternates numbers/uuids by example parity |
| `niah_multikey` | `niah_multikey_1` (k=4 there) | essay | 3 keys, 1 value, 1 query |
| `vt` | `vt` | noise | 1 chain, hops rotate 3/4/5 across examples |
| `cwe` | `cwe` | word list | RULER's own defaults verbatim: freq_cw=30, freq_ucw=3, num_cw=10 |

Scoring: RULER's `string_match_all` (`scripts/eval/synthetic/constants.py`)
for all four — per example, the fraction of `answers` found as a
case-insensitive substring of the completion, averaged and x100.

Sizes: 4096 / 8192 / 16384 / 32768 / 65536 / 131072, measured in **our**
tokenizer's tokens (`serve/tokenizer.mojo` via `.work/baro-tokenize count`, GGUF from `$BARO_GGUF`),
per the brief — not nemo/hf/openai as RULER's own harness uses.

## Deviations from RULER's harness (not its task definitions)

- **Tokenizer**: ours, not nemo/hf/openai (brief requirement, and the whole
  point of a length gate is *our* tokens).
- **Word pool**: an adjective x noun list (40x40 = 1600 combos, extended
  with a numeric suffix past that) instead of vendoring `wonderwords` +
  RULER's 465k-word `english_words.json` fallback (8.5 MB; GitHub's
  contents API rejects files > 1 MB, would need the git blobs API for no
  real benefit) — the task mechanic (numbered list, frequency skew,
  substring-match scoring) doesn't need real dictionary words, only
  distinct strings.
- **Essay haystack**: local file `bench/ruler/data/essays.txt`, NOT in the repo since 2026-11-10
  (Paul Graham's essays, third-party text with no open licence; sha256
  `b3a70ebc054f2eab5057baf3c4b7e857711472be8086240a516fd29b648ad857`). Fetch (644 KB,
  ~112k words) — RULER's own `download_paulgraham_essay.py` HTML-scrapes
  paulgraham.com; instead pulled the 49 already-cleaned `.txt` files from
  `gkamradt/LLMTest_NeedleInAHaystack` (the same source RULER's script
  merges in) via `gh api`, no HTML parsing needed. Same text RULER would
  produce for the "essay" haystack minus paulgraham.com's own subset.
- **Sentence splitting**: a regex splitter (`(?<=[.!?])\s+`) instead of
  `nltk.sent_tokenize` — avoids an nltk + punkt-corpus dependency; only
  used to place needles at haystack sentence boundaries, never for scoring.
- **Sizing**: exponential-then-binary search for the haystack unit count
  that best fits the token target (tolerance checked at 2% in
  `test_gen.py`), not RULER's estimate-then-binary-search-with-retry loop —
  same idea, less code.
- **vt keeps RULER's one-shot ICL example** (same num_hops, a short 30-line
  noise haystack) prepended to the real task — it's part of the task
  definition (teaches the answer format), not harness plumbing.

## Generation (item 1) — `bench/ruler/gen.py`

`bench/ruler/gen.py -n 25 --out bench/ruler/prompts` (default pack
`.work/engine-pack-q4`, default seed 0). Deterministic for a fixed seed
(`test_gen.py::test_deterministic_for_seed`). Output: one JSONL per
(task, size), `{"id","prompt","answers":[...],"task","size","seed"}`.

Byte hashes of the generated files (sha256, N=25, seed=0):

```
niah_single_4096.jsonl     n=25 tok(min/med/max)=3950/3960/3963  target=4096   sha256=7a8d4832f5b0b150
niah_single_8192.jsonl     n=25 tok(min/med/max)=8044/8059/8064  target=8192   sha256=abe94d49ecad36b1
niah_single_16384.jsonl    n=25 tok(min/med/max)=16230/16246/16255 target=16384 sha256=bd8f1fe19e38087c
niah_single_32768.jsonl    n=25 tok(min/med/max)=32618/32633/32639 target=32768 sha256=72fd041c0be4332d
niah_single_65536.jsonl    n=25 tok(min/med/max)=65370/65373/65403 target=65536 sha256=fe3f5bc3be6c6a3f
niah_single_131072.jsonl   n=25 tok(min/med/max)=130905/130937/130943 target=131072 sha256=088cc109810cdd72
niah_multikey_4096.jsonl   n=25 tok(min/med/max)=3964/3966/3968  target=4096   sha256=174480f4f9f406fb
niah_multikey_8192.jsonl   n=25 tok(min/med/max)=8026/8027/8063  target=8192   sha256=db624527ef0e1746
niah_multikey_16384.jsonl  n=25 tok(min/med/max)=16246/16250/16254 target=16384 sha256=3a44ebb82aacaa12
niah_multikey_32768.jsonl  n=25 tok(min/med/max)=32633/32637/32640 target=32768 sha256=caeb1fc19fb207d0
niah_multikey_65536.jsonl  n=25 tok(min/med/max)=65375/65379/65382 target=65536 sha256=fec6d1dc0dd69518
niah_multikey_131072.jsonl n=25 tok(min/med/max)=130921/130925/130928 target=131072 sha256=1a11b2fed4f21908
vt_4096.jsonl               n=25 tok(min/med/max)=4043/4050/4059  target=4096   sha256=c299b03905012567
vt_8192.jsonl               n=25 tok(min/med/max)=8139/8148/8158  target=8192   sha256=591cef39bbc631ab
vt_16384.jsonl              n=25 tok(min/med/max)=16330/16348/16354 target=16384 sha256=e26fd2734ba662fa
vt_32768.jsonl              n=25 tok(min/med/max)=32718/32725/32732 target=32768 sha256=954b64fdc489910f
vt_65536.jsonl              n=25 tok(min/med/max)=65482/65500/65505 target=65536 sha256=98cbfd952e218a94
vt_131072.jsonl             n=25 tok(min/med/max)=131019/131027/131041 target=131072 sha256=bba09699ba31f455
cwe_4096.jsonl              n=25 tok(min/med/max)=3948/3963/3975  target=4096   sha256=b453695b27a2c75e
cwe_8192.jsonl              n=25 tok(min/med/max)=8000/8057/8072  target=8192   sha256=9720c01d2c3db63f
cwe_16384.jsonl             n=25 tok(min/med/max)=16181/16239/16263 target=16384 sha256=47352256e1bfda32
cwe_32768.jsonl             n=25 tok(min/med/max)=32556/32619/32648 target=32768 sha256=d215caa4abc861df
cwe_65536.jsonl             n=25 tok(min/med/max)=65308/65384/65415 target=65536 sha256=03779982c31ed768
cwe_131072.jsonl            n=25 tok(min/med/max)=130851/130916/130951 target=131072 sha256=12fbc7b2de0033ec
```

All 24 files land within the 2% tolerance `test_gen.py` checks; the full
sha256 digests are in `bench/ruler/prompts/*.jsonl` (regenerate with
`bench/ruler/gen.py -n 25 --out bench/ruler/prompts` to reproduce
byte-for-byte, seed 0 default).

## Running (item 2) — `bench/ruler/run.py`

Drives an OpenAI-compatible `/v1/chat/completions` endpoint
(`serve/PROTOCOL.md` "HTTP surface" — both llama-server and our future
baro-serve speak it), concurrency 1, temperature 0, resumable. Saves every
raw response (`<id>.raw.json`), the extracted completion text
(`<id>.response.txt`), and per-request metrics (`metrics.jsonl`: id,
prompt_tokens as reported by the server, wall_ms, ttft_ms if streamed).

**Reasoning-model finding (item 5, not anticipated by the brief):** the
9B baseline is a thinking model — by default it emits `reasoning_content`
separate from `content`, and at RULER's own `tokens_to_generate` budgets
(niah 128, vt 30, cwe 120) it exhausts the whole budget mid-reasoning,
never reaching the answer (`content` empty, `finish_reason: length`).
Measured: a niah-style question needs ~560 completion tokens (2080 chars
of reasoning) before it emits the actual answer in `content`. Fix, applied
in `run.py`: score `reasoning_content + "\n" + content` (the answer may be
stated inside an unclosed reasoning block if the budget still runs out),
and the baseline run below uses `--max-tokens 1024`, not RULER's per-task
default — a real cost line for the chat lane's M5 budget, see "cost" below.

## Baseline arms (item 4)

Model: `~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf`
(regenerated 2026-09-08 via `llama-quantize --pure BF16.gguf Q4_0-pure.gguf
Q4_0` after the original derived quants were cleared for disk space —
4937 MiB, 4.50 BPW; same path `bench/prefill-protocol.md` and
`tools/llama-mtp-prompts.sh` already reference). GGUF claims a 1M rope
scaling target (`Mythos-5-1M` in the name).

Common flags: `-ngl 99 -fa on -b 2048 -ub 512 -t 8 -np 1 --host 127.0.0.1`
(from `tools/llama-mtp-prompts.sh`, minus MTP; `-np 1` pins one KV slot so
`-c` isn't silently divided — read back from `/props`
`default_generation_settings.n_ctx`). `-c <size>+max_tokens+256` per size
(not RULER's own small per-task budget -- see "Reasoning-model finding"
above; a first version of this script used `+512` and got HTTP 400s from
llama.cpp once prompt+max_tokens exceeded the context, fixed same session).
Every
launch through `gpu-wait run --priority 50 --vram 8 --preemptible` (one job
per context size; a preempted size restarts cheaply via `run.py`'s own
resume logic); server killed
between sizes; `/props` saved next to each run's outputs as the P1 receipt.

- **Arm A — bf16 KV (baseline)**: llama.cpp's own default KV type is f16,
  not bf16 (there is no `bf16` vs `f16` distinction at the KV-cache level
  worth forcing here); run with no `-ctk`/`-ctv` flags, receipt = server
  stderr's KV cache alloc line + `/props`.
- **Arm B — KV-quantised (K8/V4)**: `-ctk q8_0 -ctv q4_0`. First data point
  for the design's K8/V4 claim, on a different engine than ours.

Run order: 4k/8k/16k/32k first at N=25 for both arms; 64k/128k at N=10 if
the 32k run finished under 30 min (brief's own gate on scope).

### Predictions, frozen before the run (P2)

No prior RULER measurement exists for this model in `~/Brain/mojo-baro/` —
these are order-of-magnitude, not modeled from data at this model's scale;
the run either confirms or falsifies them, that's the point.

**Effective length** (largest size with avg >= 0.85 x that arm's own 4k
avg — RULER's own convention uses a fixed anchor, Llama2-7B's 4k score
85.6; we don't have that reference model, so we anchor on each arm's own
4k score and report both the anchor and the threshold):

- Arm A (bf16 KV): **32k**. Reasoning: RULER's own table has no 9B-class
  merged/fine-tuned model near its claimed length; comparable-size models
  land at 16k-64k despite 128K-1M claims (Llama3.1-8B: claims 128K,
  effective 32K; Qwen2.5-7B-1M: claims 1M, effective 64K). This model's
  long-context behavior has never been measured by this repo; 32k is the
  middle of that comparison band, not a derived number.
- Arm B (K8/V4): **predicted equal to Arm A** (32k) — falsifies the M5 gate
  premise if it comes in lower.

**Wall per size** (prefill-dominated; extrapolated from
`bench/prefill-protocol.md`'s llama.cpp Q4_0-pure `prompt_ms` at 1024
tokens = 315 ms, i.e. ~0.31 ms/token linear component, plus an
unquantified attention-quadratic term expected to dominate above ~16k —
this model's actual attention shape (GQA head count, SSM-hybrid layers per
the `ssm_*` tensors seen in the quantize log) isn't accounted for in this
extrapolation, so treat these as rough, not tight):

| size | predicted prefill wall | predicted decode (1024 tok budget) | predicted total |
|---|---|---|---|
| 4k | ~2 s | ~10 s | ~12 s |
| 8k | ~5 s | ~10 s | ~15 s |
| 16k | ~12 s | ~10 s | ~22 s |
| 32k | ~35 s | ~10 s | ~45 s |
| 64k | ~120 s | ~10 s | ~130 s |
| 128k | ~450 s | ~10 s | ~460 s |

At N=25 x 4 tasks x 2 arms, predicted cost through 32k: (12+15+22+45) x 25
x 4 x 2 s ~= 18,800 s ~= 5.2 h. **This is the real finding for the chat
lane's M5 budget**: RULER at N=25 is not a quick gate on this hardware for
this model size; the brief's own 30-minute-per-run check (below) decides
whether 64k/128k narrow to N=10, but even the 4k-32k floor is hours, not
minutes. M5's actual gate should probably run a small N (5-10) except for
a final pre-ship confirmation.

## Result (item 5, filled in after the run)

<!-- RESULTS -->

### Arm A (bf16 KV), llama.cpp Q4_0-pure, N=25, 4k-32k — 2026-09-08

`-np 1 -ub 512` as frozen; 32k finished in-session after the 12:15 desktop
crash (run.py resume, 100/100). Score: `bench/ruler/score.py
.work/ruler-baseline/bf16/all` (per-size response dirs symlinked together).

| task          | 4k   | 8k   | 16k  | 32k   | effective_len |
|---------------|------|------|------|-------|---------------|
| niah_single   | 88.0 | 68.0 | 96.0 | 100.0 | 32768 |
| niah_multikey | 96.0 | 96.0 | 100.0| 88.0  | 32768 |
| vt            | 100.0| 100.0| 100.0| 100.0 | 32768 |
| cwe           | 93.6 | 100.0| 88.4 | 46.0  | **16384** |

Prediction "Arm A effective 32k" holds for 3 of 4 tasks; **cwe (common-word
extraction, aggregation) falls to 46 at 32k**, so the per-task effective
length is 16k. niah_single 68 at 8k is a dip, not a cliff (96/100 after).
Per-prompt wall at `-np 1`: 4k 5.3 s, 8k 8.3 s, 16k 10.9 s, 32k ~16 s.

Arm B (K8/V4) and 64k/128k: NOT run — stopped after arm A 32k (the maintainer,
2026-09-08 12:4x) on cost (~2 h more, 128k prefill dominated). The parallel
slot amendment below is committed and untested on a real run.

Deviation: `run-baseline.sh` was edited while the 32k run was executing;
bash read the modified tail and printed `unexpected EOF` (exit 2) AFTER the
last prompt completed — all 100 responses and metrics present, the server
was killed by the trap. Committed script passes `bash -n`.

### What the M5 gate should run

Once a format is built, the exact commands:

```
bench/ruler/gen.py -n 25 --out bench/ruler/prompts        # once, reused across formats
bench/ruler/run.py --base-url http://127.0.0.1:<port>/v1 --model qwythos \
    --out .work/ruler-<format>/ --max-tokens 1024
bench/ruler/score.py .work/ruler-<format>/ --json .work/ruler-<format>/table.json
```

A format ships only if its `effective_length` in `table.json` matches
bf16's for every task, not just the aggregate.

### Amendment 2026-09-08 12:5x (before arm B / 64k / 128k ran)

Arm A bf16 4k-32k ran as frozen (`-np 1 -ub 512`, N=25). Everything after
runs with parallel slots to cut wall time (the maintainer, "do 1-3"): `-np 4 -b 4096
-ub 2048` with `-c` scaled per slot (`-np 2` at 64k, 1 at 128k) and
`run.py --workers NP`; arm B at **N=10** (`LIMIT=10`) at every size. The gate
is accuracy at temperature 0, which none of these flags change; per-prompt
`wall_ms` from these runs is NOT comparable with the arm-A 4k-32k rows and is
not used. Receipt per run: `/props` `n_ctx` echoed into `run.log` next to
the `np=` line.
