#!/usr/bin/env bash
# usage: bench/moe-launch-count.sh ENGINE OUTDIR [N_TOKENS]
#
# Per-token kernel launch count for the MoE decode path, from rocprofv3's own
# kernel trace rather than from counting enqueue_function call sites by eye.
# R4's prediction (bench/moe-persist-protocol.md) is stated in LAUNCHES
# (1216 -> about 1116), so the tok/s A/B alone cannot confirm or refute it:
# this is the receipt that can.
#
# The count is dispatches during the decode phase only. Load, prefill and the
# one-off allocation memsets are excluded by counting the trace for two runs
# that differ only in generated length and taking the difference, which cancels
# everything that does not scale with tokens. That is also why this needs no
# knowledge of which kernel names belong to which phase.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; out=$2; n2=${3:-32}; lcenv=${LC_ENV:-BARO_MEGA=0}
n1=$(( n2 / 2 ))
mkdir -p "$out"
command -v rocprofv3 >/dev/null || { echo "no rocprofv3 on PATH" >&2; exit 2; }
echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) tokens=$n1,$n2 env=$lcenv" | tee "$out/arm.txt"

count_for() {
  local n=$1 d="$out/gen$1"
  rm -rf "$d"; mkdir -p "$d"
  # Serve mode, one request: the one-shot path has no generated-length knob
  # (GEN_N is a comptime 64), the wire does. Same prompt both times, so the
  # prefill cancels in the difference along with the load.
  local p
  p=$(tr -s ' \n' ',' < bench/mtp-prompts/p01-water.tokens | sed 's/,$//')
  echo "{\"id\":1,\"prompt\":[$p],\"n\":$n,\"spec\":false}" > "$d/request.jsonl"
  env BARO_SERVE=1 BARO_SPEC=0 $lcenv BARO_PACK=.work/moe-w1/pack \
      rocprofv3 --kernel-trace -f csv -d "$d" -o trace -- "$eng" \
      < "$d/request.jsonl" > "$d/stdout.txt" 2> "$d/stderr.txt"
  grep -q '"done"' "$d/stdout.txt" || { echo "run at n=$n did not finish, see $d/stderr.txt" >&2; return 1; }
  local csv
  csv=$(find "$d" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] || { echo "no kernel trace csv under $d" >&2; return 1; }
  # one row per dispatch, minus the header
  echo $(( $(wc -l < "$csv") - 1 ))
}

c1=$(count_for "$n1") || exit 1
c2=$(count_for "$n2") || exit 1
per=$(python3 -c "print(f'{($c2 - $c1) / ($n2 - $n1):.1f}')")
{
  echo "dispatches at $n1 tokens: $c1"
  echo "dispatches at $n2 tokens: $c2"
  echo "launches per token: $per   (difference method, so load and prefill cancel)"
} | tee -a "$out/arm.txt"
