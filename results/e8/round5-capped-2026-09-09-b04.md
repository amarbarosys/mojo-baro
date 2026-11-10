# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)
VRAM before both loads: 1.03 GB, after both loads: 1.03 GB (delta -0.00 GB)
Topology 2: see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 4/5 / 5/5 | 4/25 | 8/30 / 9/30 |
| T | 4/5 / 5/5 | 16/25 | 20/30 / 21/30 |
| L8-raw | 4/5 / 5/5 | 4/25 | 8/30 / 9/30 |
| L8-soft | 4/5 / 5/5 | 3/25 | 7/30 / 8/30 |
| L32-soft | 4/5 / 5/5 | 3/25 | 7/30 / 8/30 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.527 | 30 |
| L8-raw | 0.175 | 30 |
| L8-soft | 0.222 | 30 |
| L32-soft | 0.558 | 30 |
