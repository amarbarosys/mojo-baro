#!/usr/bin/env bash
# Stand-in for the single-session engine in the dry run of parity-llama.sh: one "forced agreement: N / N" line per request line.
set -euo pipefail
while IFS= read -r line; do n=$(jq -r '.n' <<<"$line"); echo "forced agreement: $n / $n"; done
