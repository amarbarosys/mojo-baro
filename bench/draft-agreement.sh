#!/usr/bin/env bash
# bench/draft-agreement.sh: top-token agreement of drafter candidates with the
# main model (llama.cpp oracle, teacher-forced on wikitext). Exploration for
# shared-weight speculation: the agreement rate is the draft acceptance ceiling.
# Arms: base (saves logits), then k4/k2 = same GGUF routed to 4/2 experts, plus
# any extra GGUF passed as NAME=PATH. Experts stay on the CPU (--cpu-moe) so the
# desktop keeps its VRAM. Run through gpu-wait.
# Usage: gpu-wait run ... -- bench/draft-agreement.sh MAIN.gguf OUTDIR [NAME=GGUF ...]
set -euo pipefail
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || { echo "FAIL draft-agreement: run through gpu-wait"; exit 1; }
main=$1; out=$2; shift 2
bin=$HOME/llama.cpp/build/bin/llama-perplexity
txt=$HOME/Models/quant-lab/wikitext-2-raw/wiki.test.raw
mkdir -p "$out"
common=(-f "$txt" --chunks 8 -c 512 -ngl 99 --cpu-moe -t 12)
if [ ! -s "$out/base.logits" ]; then
  "$bin" -m "$main" "${common[@]}" --save-all-logits "$out/base.logits" > "$out/base.log" 2>&1 \
    || { echo "FAIL base: $out/base.log"; exit 1; }
fi
failed=()
arm() { # name model [extra args]
  local name=$1 model=$2; shift 2
  "$bin" -m "$model" "${common[@]}" --kl-divergence --kl-divergence-base "$out/base.logits" "$@" > "$out/$name.log" 2>&1 \
    || { echo "FAIL $name: $out/$name.log"; failed+=("$name"); return 0; }
  printf '%-10s %s | %s\n' "$name" "$(grep -m1 'Same top p:' "$out/$name.log")" "$(grep -m1 'Mean    KLD' "$out/$name.log")"
}
arm k4 "$main" --override-kv qwen35moe.expert_used_count=int:4
arm k2 "$main" --override-kv qwen35moe.expert_used_count=int:2
for spec in "$@"; do arm "${spec%%=*}" "${spec#*=}"; done
[ ${#failed[@]} -eq 0 ] || { echo "FAIL ${#failed[@]} arms: ${failed[*]}"; exit 1; }
