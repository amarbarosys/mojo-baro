# mojo-baro designs on a phone GPU (Xiaomi 25128PNA1C, Adreno 840), 2026-11-20

Mojo has no Adreno backend. The phone engine is the Mojo engine of `~/Android/baro` (`mojo-engine/`), whose
`--gpu 1` path drives OpenCL C kernels (`mojo-engine/gpu/baro_gpu.cl`) from Mojo through dlopen. This repo holds
the phone tooling: `phone/ph.sh` (adb under timeout, both phone locks), `phone/race.sh` (GEMV race through
wavewait's wwk), `phone/speed.sh` (decode tok/s, arms rotated), `phone/cl/` (race kernels, read-ceiling probes).

| measure (Qwen3-8B pure Q4_0, sha f4eb7f82) | value |
|---|---|
| bar: llama.cpp 956591f45 OpenCL tg128 | 11.56 +- 0.10 (llama-bench -r 3); 11.89 median of 5 in the confirm |
| engine `--gpu 1` decode | **13.88** median of 5, spread 0.1 % (1.167x) |
| prefill, GPU token by token | 13.8 tok/s vs llama pp512 134.6: no batched GPU prefill yet |
| token gate, 20 prompts x 32 ids | exact vs CPU ref 17/20 = llama.cpp OpenCL control 17/20; vs llama.cpp OpenCL 20/20 |
| read ceiling (any pattern) | 65.5 GB/s; token floor 70.4 ms, measured 72.0 ms (~98 %) |

GEMV race and the ceiling table: `~/Projects/mojo/wavewait/docs/DEVICES.md`. Full report with receipts:
`.work/briefs/2026-11-20-adreno-runner.report.md` (local) and `~/Brain/mojo/mojo-baro/2026-11-20-adreno-runner.md`.
Rules learned (vendor LD_LIBRARY_PATH, pkill -f self-match, ww-phone on the Xiaomi): `docs/rules.md`.
