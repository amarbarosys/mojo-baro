# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)
VRAM before both loads: 1.20 GB, after both loads: 24.55 GB (delta 23.35 GB)
Topology 2: see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| 0 | 1/20 / 4/20 | 20/20 | 21/40 / 24/40 |
| T | 2/20 / 5/20 | 20/20 | 22/40 / 25/40 |
| L8-raw | 2/20 / 6/20 | 20/20 | 22/40 / 26/40 |
| L8-soft | 1/20 / 4/20 | 20/20 | 21/40 / 24/40 |
| L32-soft | 1/20 / 7/20 | 20/20 | 21/40 / 27/40 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| 0 | - | 0 |
| T | 2.490 | 40 |
| L8-raw | 0.155 | 40 |
| L8-soft | 0.202 | 40 |
| L32-soft | 0.534 | 40 |
