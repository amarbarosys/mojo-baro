#!/usr/bin/env bash
# CPU preflight before any GPU job (PROTOCOL-RULES P15).
#   bench/preflight.sh          run everything below, write .work/preflight.ok keyed to the content of every tracked and untracked (non-ignored) file
#   bench/preflight.sh --check  exit 1 unless .work/preflight.ok matches the current tree (gate scripts call this)
# Checks: tools/ci-checks.sh (docs, py/sh syntax, census, every bench/*.mojo builds), every run-tests.sh test
# binary builds, dense and MoE engines build (tools/build-many.sh: parallel, cached on an all-Mojo
# per-build import-closure key; the engines stay at .work/preflight/engine-{dense,moe} for lanes), baro-serve builds and its unit tests pass.
set -euo pipefail
cd "$(dirname "$0")/.."
accel=${BARO_TARGET_ACCELERATOR:-gfx1100}
# exchange/ holds reports and conference notes that other lanes write while a job is queued; nothing there feeds a build or a gate.
stamp() { git ls-files -co --exclude-standard | grep -v '^exchange/' | sort | while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f"; done | tr '\n' '\0' | xargs -0 sha256sum -- | sha256sum | cut -c1-32; }
if [ "${1:-}" = --check ]; then
  [ "$(cat .work/preflight.ok 2>/dev/null)" = "$(stamp)" ] && exit 0
  echo "FAIL preflight: no passing bench/preflight.sh for the current file contents; run it first" >&2
  exit 1
fi
rm -f .work/preflight.ok; mkdir -p .work/preflight
# stamp the tree BEFORE checking it: a stamp taken at the end certified edits made while the builds ran
# (latent-flash P1 2026-10-12: --check passed for a tree whose full run then failed twice).
st0=$(stamp) || { echo "FAIL preflight stamp: file hashing failed" >&2; exit 1; }
step() { echo "== $1"; }
fail() { echo "FAIL preflight $1: see $2" >&2; exit 1; }
step ci-checks;  tools/ci-checks.sh > .work/preflight/ci.log 2>&1 || fail ci-checks .work/preflight/ci.log
step rdna4-probe
tools/rdna4-probe/rdna4-probe.sh > .work/preflight/rdna4-probe.log 2>&1 || fail rdna4-probe .work/preflight/rdna4-probe.log
diff <(sed -n '/^probe /,$p' .work/preflight/rdna4-probe.log | head -13) <(head -13 docs/rdna4/probe-2026-11-12.txt) > /dev/null || fail rdna4-probe-drift .work/preflight/rdna4-probe.log
step builds
export S="$PWD/.work/shim-build"
{ sed -nE 's#^ *t ([^ ]+) "([^"]+)".*#./.venv/bin/mojo build \2 -o \1#p' run-tests.sh | envsubst '$S' | sed "s#\$#  --target-accelerator $accel#"
  grep -E '^\./\.venv/bin/mojo build ' run-tests.sh | sed "s#\$#  --target-accelerator $accel#"
  echo "./.venv/bin/mojo build --target-accelerator $accel serve/engine.mojo -I . -I kernels -o .work/preflight/engine-dense"
  echo "./.venv/bin/mojo build --target-accelerator gfx1201 serve/engine.mojo -I . -I kernels -o .work/preflight/engine-dense-gfx1201"
  echo "./.venv/bin/mojo build --target-accelerator $accel serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -o .work/preflight/engine-moe"
  echo "./.venv/bin/mojo build --target-accelerator $accel serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -D BARO_KVQ=f32 -o .work/preflight/engine-moe-f32"
} | tools/build-many.sh preflight > .work/preflight/build.log 2>&1 || fail builds .work/preflight/build.log
tail -1 .work/preflight/build.log
step baro-serve
cargo test --release --manifest-path serve/Cargo.toml > .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
cargo build --release --manifest-path serve/Cargo.toml >> .work/preflight/cargo.log 2>&1 || fail cargo .work/preflight/cargo.log
st=$(stamp) || fail stamp "file hashing failed"
[ "$st" = "$st0" ] || { echo "FAIL preflight tree-changed: files changed while preflight ran (${st0:0:12} -> ${st:0:12}); rerun on a quiet tree" >&2; exit 1; }
echo "$st" > .work/preflight.ok
echo "PASS preflight ${st:0:12}"
