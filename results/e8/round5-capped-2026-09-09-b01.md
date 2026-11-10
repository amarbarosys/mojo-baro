# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)
VRAM before both loads: 1.04 GB, after both loads: 24.42 GB (delta 23.38 GB)
Topology 2: see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 5/5 / 5/5 | 4/25 | 9/30 / 9/30 |
| T | 5/5 / 5/5 | 11/25 | 16/30 / 16/30 |
| L8-raw | 5/5 / 5/5 | 4/25 | 9/30 / 9/30 |
| L8-soft | 5/5 / 5/5 | 3/25 | 8/30 / 8/30 |
| L32-soft | 5/5 / 5/5 | 4/25 | 9/30 / 9/30 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.507 | 30 |
| L8-raw | 0.173 | 30 |
| L8-soft | 0.220 | 30 |
| L32-soft | 0.553 | 30 |
