# lane-gtt report

The engine warns when GTT grows past a threshold; BARO_GTT_REFUSE=1 exits 3. Commits on lane-gtt: 98b336c (feature), 092162c (bench/gtt-check.sh, settings docs).

run-tests: ./run-tests.sh under gpu-wait (timeout 5580 s), rc=0, log .work/gtt/run-tests.log (last line rc=0, previous line "PASS run-tests: all tests, pack .work/engine-pack-q4"). The only line starting with FAIL is the summary "FAIL: 0 (none)". Toolchain: worktree-local `uv sync` venv, Mojo 1.2.0 (the linked 1.3 nightly does not build the engine). State layer ran with the lane engine built from the lane tree.

Feature checks (bench/gtt-check.sh on the dense pack, TMAX 16384): default env no WARN exit 0; BARO_GTT_WARN_BYTES=1 WARN exit 0; plus BARO_GTT_REFUSE=1 FAIL exit 3 with no GENERATED. MoE pack: 21 GB real spill reported and exit 3. Receipts: .work/gtt/run1, .work/gtt/spill, .work/gtt/fit.

Unverified: no decode-speed A/B against an unpatched binary; tier/zero-copy paths not run; no spill-then-slowdown run.
