# GTT warning, predictions (frozen before the first run)

Engine: serve/engine.mojo from the commit before this file, dense q4 pack `.work/engine-pack-q4`, BARO_TMAX=16384, BARO_PROMPT=bench/mtp-prompts/p01-water.tokens, one run per arm under `gpu-wait run`.

- A default env: one `GTT: card /sys/class/drm/card1/device/ gtt_used N` line (card1 = 24 GB card, not card0 iGPU 512 MB), no WARN, exit 0, GENERATED line present.
- B `BARO_GTT_WARN_BYTES=1`: pack load and buffer allocation grow GTT by more than 1 byte, so a `WARN gtt: stage after_pack_load bytes spilled <n>` line appears with n > 1; exit 0.
- C `BARO_GTT_WARN_BYTES=1 BARO_GTT_REFUSE=1`: WARN then `FAIL gtt:` line, exit 3, no GENERATED line.
- tok/s_gen of A within 3% of the main-tree champion row for this pack (not a claim, a sanity read); the check adds 0 reads inside the decode loop.
