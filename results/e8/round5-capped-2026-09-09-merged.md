# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 17/20 / 19/20 | 10/100 | 27/120 / 29/120 |
| T | 16/20 / 18/20 | 63/100 | 79/120 / 81/120 |
| L8-raw | 17/20 / 19/20 | 10/100 | 27/120 / 29/120 |
| L8-soft | 17/20 / 19/20 | 7/100 | 24/120 / 26/120 |
| L32-soft | 17/20 / 19/20 | 9/100 | 26/120 / 28/120 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.523 | 120 |
| L8-raw | 0.174 | 120 |
| L8-soft | 0.222 | 120 |
| L32-soft | 0.557 | 120 |
