# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)
VRAM before both loads: 1.35 GB, after both loads: 24.72 GB (delta 23.37 GB)
Topology 2: see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 1/4 / 2/4 | 4/4 | 5/8 / 6/8 |
| T | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |
| L8-raw | 2/4 / 3/4 | 4/4 | 6/8 / 7/8 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.469 | 8 |
| L8-raw | 0.155 | 8 |
