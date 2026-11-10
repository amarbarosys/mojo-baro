# Arithmetic with a checker, predictions (frozen by commit before any run beyond a 30-step smoke test)

Task: digit-level expressions, 5 operands, values kept in 0..99. Tokens: digits, `+ - * ( ) = ;` and END, vocab 24 (LAT_V=24).
Families by shape: 0 flat `+ -`; 1 flat with `* + -` (at least one `*`, precedence); 2 `* + -` with one parenthesised operand pair.
Checker: `arithgen.value` (shunting-yard) against the number the model wrote before END, after greedy decode of the whole chain from the
prompt (`EXPR =`) with its own tokens fed back. `test_arith.mojo` proves the reduction chain equals the checker on 120000 samples and
the hand cases. Chain format (cot): `EXPR = S1 ; S2 ; S3 ; N END`, each S the whole expression after one reduction. nocot: `EXPR = N END`.
Example (n=4): `8*3+6-7 = 24+6-7 ; 30-7 ; 23 END`.
Model: 1.59M parameters class (d 256, 2 layers, 8 heads, vocab 24: 1,590,272 measured). 256 sequences x 40 rows = 10240 token rows per update,
3000 updates = 31M token rows per run, 20 tokens per parameter. lr 1e-3, warmup 50, linear decay to 0, AdamW, bf16 GEMMs.
Streams: train seed = run seed (fresh each update), val 10000+seed (512, every 1000 updates, no selection), test 20000+seed (2048 per family, read once).

Scripts: `miniagi/latent/arith.sh pilot` (seed 110: nocot and cot, trained on all 3 families), `arith.sh folds` (cot, hold out one family, seeds 110-112, test on all 3).
Runs: 2 pilot + 9 fold = 11 runs x 3000 updates = 33000 updates at 6-8 ms (smoke test: 8.2 ms/step incl. startup, params 1590272) = about 4 GPU minutes. Timeouts: pilot 150 s (2 x 24 s + evals), folds 420 s (9 x 24 s + evals).

Predictions:
- Q1 (validity) nocot on the pilot, mean test accuracy over the three families, stays below 50%, and below 90% on every family. If any family
  reaches 90% with no chain, that family is invalid as a CoT task and the report says so.
- Q2 (ceiling) cot-all (trained on all 3) reaches >= 90% on each family. A miss there means the 3000 update budget is too small: optimize
  (updates, lr), do not read it as a task property.
- Q3 (family fold) held-out accuracy of cot, mean over 3 seeds:
  out0 (flat +/- never trained) >= 70%, out1 (flat with `*`, no parens) >= 50%, out2 (parens never trained, `(` `)` embeddings untouched) <= 20%.
  Trained families in each fold stay >= 90%.
- Q4 the gap: cot accuracy on the held-out family is below its accuracy on trained families on at least 2 of the 3 folds (in-distribution advantage is real).
No kill line: a miss means optimize (updates, lr, data mix), not drop. Two failed attempts and I stop and report.
