#!/usr/bin/env bash
# minja gate: Mojo renderer must reproduce jinja2's output byte for byte on tests/minja-corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .work
./.venv/bin/mojo build tools/minja-cli.mojo -I . -o .work/minja-cli
./.venv/bin/python3 tools/minja-difftest.py
if ./.venv/bin/python3 tools/minja-difftest.py --negative >/dev/null 2>&1; then echo "FAIL minja-gate: negative control did not fail"; exit 1; fi
echo "negative control: FAILS as required"
