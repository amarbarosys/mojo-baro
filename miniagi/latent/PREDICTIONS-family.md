# Family hold-out, predictions (frozen by commit before any run)

Script: `miniagi/latent/family.sh add|xor`. Model 1.59M, lr 1e-3, text arm, soft-chain hit, seeds 110-112.
Rules: 1-4 add family, 5-10 xor family. add: base on rules 1-4, held out xor 5 and 8. xor: base on rules 5-10, held out add 1 and 3.
Base 4000 updates, adapt and scratch 1000 updates, eval every 25, bar 90% (1024 stream), final test stream 4096 untouched.
Cost: about 24000 updates per direction at about 7 ms/step (R4: ~60000 updates in ~7 min) = about 2.8 GPU min per direction.

Predictions:
- P1 base reaches >= 99% on its trained rules on 3 of 3 seeds in both directions.
- P2 zero-update read on the held-out family stays <= 15% (floor 6.25% chance; new rule token has untrained embedding).
- P3 R4 carry-over shrinks across families: adapt hit_step / scratch hit_step is above 0.5 on at least half of (rule, seed) pairs
  in both directions (R4 within family was <= 0.125). Adapt still beats scratch on the mean in both directions.
- P4 a direction where adapt does not reach 90% in 1000 updates counts as no transfer; scratch not reaching 90% in 1000 is recorded, not dropped.
No kill line: a miss means optimize (lr, updates), not drop.

Run: `gpu-wait run --timeout 420 -- bash miniagi/latent/family.sh add` then `... xor`. Binary built from run.mojo at this commit
(`mojo build miniagi/latent/run.mojo -D LAT_H=8 ... -o .work/spec/run`), 1.59M params (V 32, d 256, 2 layers).
Worst case 24000 updates per direction (adapt/scratch stop at the 90% hit), at most 5.6 GPU minutes for both directions.
Tokens per update: 256 sequences x 41 rows = 10.5k; 24000 updates = 252M tokens per direction.
Stricter than R4: the held-out family shares no rule with the base (R4 held out one rule per family while the family stayed in training).
