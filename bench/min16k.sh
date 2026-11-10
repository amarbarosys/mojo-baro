#!/usr/bin/env bash
# the maintainer, 2026-10-18: no test, probe or gate runs under a 16K window or under a 16K generation cap.
# Exempt by definition: max_tokens:1 on POST /v1/checkpoints (a state snapshot, not a generation) the fixed 64 forced ids of bench/force-ab-serve.sh, and the fixed-length greedy id comparisons of lab/wavegroup (ref_story, stint, llama_self, ref_tokens: N ids with ignore_eos, compared id by id; window still >= 16384).
# usage (in a script):  . "$(dirname "$0")/min16k.sh"; min16k WINDOW [MAX_TOKENS]
# Refuses (exit 1) when WINDOW < 16384 or MAX_TOKENS < 16000. A cap above the free window is the caller's job to size.
min16k() {
  local w=${1:?min16k WINDOW [MAX_TOKENS]} m=${2:-16000}
  [ "$w" -ge 16384 ] || { echo "FAIL min16k: window $w < 16384 (the maintainer 2026-10-18: no run under 16K)"; exit 1; }
  [ "$m" -ge 16000 ] || { echo "FAIL min16k: max_tokens $m < 16000 (the maintainer 2026-10-18: no run under 16K)"; exit 1; }
}
