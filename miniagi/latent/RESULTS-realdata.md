# Family hold-out and arithmetic with a checker: results (2026-11-02)

Predictions: `PREDICTIONS-family.md` (995757e), `PREDICTIONS-arith.md` (4fee47e). Model 1,590,272 parameters (d 256, 2 layers, 8 heads), bf16 GEMMs, one RX 7900 XTX.

## Step 1: family hold-out (pointer chasing with hop rules, 3 seeds 110-112)

Base trains on one rule family (4000 updates), adapt starts from it, scratch starts fresh; both 1000 updates max, stop at 90% on a 1024 val stream, final test 4096.
hit_step is in updates, first eval is at 25.

| direction | base on its rules | zero-update floor on held-out rules | adapt hit_step | scratch hit_step |
|---|---|---|---|---|
| train add 1-4, hold out xor 5 and 8 | 1.000 x3 | 0.068-0.113 | 25 on 6 of 6 | 175-225 |
| train xor 5-10, hold out add 1 and 3 | 0.9998, 0.9998, 1.000 | 0.066-0.096 | 25 on 6 of 6 | 175-225 |

adapt/scratch ratio 0.11-0.14 on all 12 pairs. The new rule token is the only thing the held-out rule needs to teach; the hop procedure carries over across families.

## Step 2: arithmetic, 5 operands, checker = shunting-yard value, greedy decode with own tokens

3000 updates, 256 x 40 rows = 10240 token rows per update (30.7M per run, 19 tokens per parameter), lr 1e-3. Test stream 20000+seed, 2048 per family.

Pilot (seed 110, trained on all three families):

| family | no-CoT | CoT |
|---|---|---|
| 0 flat + - | 0.999 | 1.000 |
| 1 flat with * | 0.792 | 0.9985 |
| 2 one paren pair | 0.647 | 0.9985 |

Folds (CoT, 3 seeds, mean test accuracy; trained families in brackets show the minimum over seeds and trained families):

| held out | held-out accuracy (s110, s111, s112) | mean | trained families |
|---|---|---|---|
| family 0 | 0.109, 0.107, 0.112 | 0.109 | >= 0.9985 |
| family 1 | 0.208, 0.172, 0.251 | 0.211 | >= 0.9946 |
| family 2 | 0.199, 0.108, 0.168 | 0.158 | >= 0.9956 |

GPU time (sum of run wall seconds): step 1 229 s, pilot 52 s, folds 235 s, total 516 s = 8.6 min plus floors and a 1 s smoke test.
