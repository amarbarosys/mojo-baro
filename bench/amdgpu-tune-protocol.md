# amdgpu tuning for Flash decode (prereg)

FROZEN by the commit that adds this file, before any run. the maintainer 2026-10-14: levers 1, 2, 3; no power-cap raise.
OS note: `~/Brain/OS/2026-10-14-amdgpu-tune.md`.

## Levers

1. INT: `HSA_ENABLE_INTERRUPT=0`, ROCr busy-polls completion signals instead of sleeping on an interrupt.
2. Perf level: HIGH = `power_dpm_force_performance_level=high`; COMP = `manual` + `pp_power_profile_mode=5` (COMPUTE).
   AUTO (default) is the reference.
3. IOMMU passthrough (`iommu=pt`), needs a reboot: job 2 only.

## Instrument (`bench/amdgpu-tune-ab.sh`, one gpu-wait job per boot)

q4x binary `.work/q4x-serve/q4x2` (v0.5.10 source), resident `Q4X_STDIN=1`, default warm list, 10 prompts from
`bench/q4x-prompts` plus p01 again, 64 generated tokens. Per arm: steady gen ms/token = median over requests 2..10,
misses/token, CPU user+sys seconds of the process (read-back for INT: polling burns a core), sclk/mclk/power sampled
every 100 ms by a separate process, perf level and profile read back from sysfs at arm start. `pcie-bw`
(`bench/pcie-bandwidth.mojo`) once at the start and once at the end of the job, with the link state.

Job 1 order (power blocks ABCCBA, INT alternating inside each block):
AUTO(base, int) HIGH(int, base) COMP(base, int) COMP(int, base) HIGH(base, int) AUTO(int, base). 12 arms. Perf level
restored to auto on every exit path.

Job 2 (after `iommu=pt` and a reboot, link state read back): the same job 1, unchanged.

## Bars

1. Identity: every arm's ids equal arm 1's for every request (no lever may change the math), else the job is void.
2. Read-back: sysfs perf level per arm matches the arm; INT arms show CPU seconds at least 1.3x their paired base arm.
3. No minimum gain (CLAUDE.md 2026-10-09). A lever lands if its pooled ratio is below 1 and both of its ABBA pairs
   agree in sign; otherwise it is reported and not adopted.

## Prediction (frozen)

- INT vs base (pooled over power states): gen ms/token 0.96 to 1.00; CPU seconds 1.5x to 3x.
- HIGH vs AUTO: gen ms/token 0.93 to 0.99; mean sclk during the arm 20% or more above AUTO; the first-arm slowdown
  (seen in three jobs on 2026-10-14) gone under HIGH.
- COMP vs AUTO: 0.96 to 1.00.
- Job 2 vs job 1 (cross-boot, link state recorded): pinned H2D/D2H within 5%; AUTO base steady gen ms/token 0.95 to
  1.00; misses/token unchanged within 3%.

## Budget

12 arms of about 30 s (load 8 s, warm 2 s, 11 requests), two pcie-bw runs of about 10 s: under 8 minutes per job.
`--timeout 1200`.
