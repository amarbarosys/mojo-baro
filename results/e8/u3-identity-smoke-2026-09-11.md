# E8 HARNESS results -- topology1-q4

pack: `.work/engine-pack-q4`  transport: in-process host copy (memfd transport is E7's, not measured here)
VRAM before both loads: 18.20 GB, after both loads: 24.04 GB (delta 5.84 GB)
Topology 2: see .work/e8-harness-topology2-blocker.md (structure does not transfer, Spark harness not attempted)

## Accuracy per arm per task type

json cells show `exact/subset` (`correct_exact`/`correct_subset`, round 2 defect 2); other task types have no subset concept so `correct_exact == correct_subset` and the cell shows one count.

| arm | json | math | overall (exact/subset) |
|---|---|---|---|
| T | 1/1 / 1/1 | - | 1/1 / 1/1 |
| KV | 1/1 / 1/1 | - | 1/1 / 1/1 |

## Producer time (median, s)

| arm | median producer_s | n |
|---|---|---|
| T | 0.203 | 1 |
| KV | 0.206 | 1 |
