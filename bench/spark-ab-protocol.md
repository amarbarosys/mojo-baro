# Spark X2.5 4B decode A/B: serve/spark.mojo vs llama.cpp master (fast config)

Preregistered 2026-09-08 ~17:25, engine at `d2ec40c`, before any run.

## Arms
- **A (ours)**: `.work/spark/spark-engine` built from `serve/spark.mojo` @ d2ec40c,
  `BARO_PROMPT_TEXT=<prompt.txt>` (Mojo tokenizer, add_bos false), greedy, `BARO_GEN=64`,
  pack `.work/spark/pack-q8` from `Spark-X2.5-4B-Q8_0-requant.gguf`.
- **B (bar)**: `~/llama.cpp-master/build/bin/llama-server` same GGUF, fast config
  `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -t 8`, `/tokenize` (add_special
  false) then `/completion` on the ids, n_predict 64, temperature 0, top_k 1,
  cache_prompt false; `predicted_per_second` from timings.
- Prompt set: `bench/mtp-prompts/p*.txt` (20 real prompts, reused from the Qwythos
  rounds), each prompt tokenized by its own arm's tokenizer (identity of the id
  lists is checked and reported).
- Same stint, A then B per PROTOCOL-RULES P4; power cap + vddgfx read back first (P1);
  binary sha256 recorded.

## Frozen prediction
- A median tok/s_gen: **138 – 146** (one-shot 142 on the 23-id gate prompt; prompts
  here are 4–30 ids longer, attention cost is ~4% of busy time).
- B median: **125 – 145** (f32-KV/fa-off reference ran 111; the fast config on this
  4B Q8 gains 15–30% in prior rounds).
- Ratio A/B: **0.97 – 1.15**. Claim if the whole A range clears 1.0; otherwise report
  "parity" and stop tuning the launch path.
- Token identity A vs B: NOT expected (B uses q8 KV + flash-attn); reported as
  informational. Identity vs the f32 ref is the separate gate (`.work/spark/gate.sh`).
- Tokenizer identity A vs B: expected 20/20.

## Result (2026-09-08 ~17:50, engine c29f4a5 = d2ec40c + prompt-id print, 290 W, vddgfx -100 mV)

Run 1, B as preregistered (`-fa on -ctk q8_0 -ctv q8_0`): A median **142.68** (spread 1.8%),
B **98.68** (0.5%), ratio 1.446. B missed its predicted range (125-145) from below:
the f32/fa-off reference had run 111. llama-bench read-back (`.work/spark/ab/llama-bench.md`):
tg64 f16 KV + fa on = 121.2, q8_0 KV + fa on = 100.7 -- q8_0 KV is the slow config on
this model/backend, so the preregistered B was not the fast config. Deviation: B re-run
with f16 KV (the fastest llama.cpp config found), q8_0 result kept above.

Run 2 (`.work/spark/ab-f16/`), B = `-fa on -ctk f16 -ctv f16`: A median **142.26**
(spread 2.9%), B **118.87** (0.3%), **ratio 1.197**. Tokenizer identity 20/20.
Generated tokens equal B's on 12/20 (informational; B uses f16 KV + flash-attn).

Verdict: A inside its predicted range; the whole A range clears B, so **claim: 1.20x
llama.cpp master's fastest config on the 20-prompt median.** Ratio range was predicted
0.97-1.15; the bar came in lower than predicted, the engine did not come in higher.
