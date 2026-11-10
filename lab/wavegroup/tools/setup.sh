#!/usr/bin/env bash
# .venv with Mojo and MAX on the same nightly date (mixed dates segfault builds that import max)
set -euo pipefail
cd "$(dirname "$0")/.."
NIGHTLY=2026102505
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python --prerelease allow --extra-index-url https://whl.modular.com/nightly/simple/ \
  --index-strategy unsafe-best-match "mojo==1.3.0.dev$NIGHTLY" "max==26.8.0.dev$NIGHTLY" || { echo "FAIL setup: uv pip install"; exit 1; }
.venv/bin/mojo --version | grep -q "dev$NIGHTLY" || { echo "FAIL setup: mojo is not dev$NIGHTLY"; exit 1; }
echo "OK mojo+max dev$NIGHTLY"
