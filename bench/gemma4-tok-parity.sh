#!/usr/bin/env bash
# gemma4 tokenizer parity: serve/tokenizer.mojo (via .work/baro-tokenize) vs llama.cpp llama-tokenize, exact ids,
# on the 20 held-out prompts, the 746-id prefill prompt and the stress texts under bench/gemma4-tok/.
# Usage: bench/gemma4-tok-parity.sh GGUF [LLAMA_BIN_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
G=$1; LB=${2:-.work/gemma4/llama-dump/build/bin}
ok=0; n=0
for f in bench/mtp-prompts/*.txt bench/spark-prefill-prompts/p0768.txt bench/gemma4-tok/*.txt; do
  n=$((n + 1))
  a=$(./.work/baro-tokenize encode "$f" "$G" 2>&1 | tr -s '\n' ' ' | xargs)
  b=$("$LB/llama-tokenize" -m "$G" -f "$f" --ids --log-disable 2>/dev/null | tr -d '[]' | tr ',' ' ' | xargs)
  if [ "$a" = "$b" ]; then ok=$((ok + 1)); else echo "DIFF $f"; echo " ours:  ${a:0:200}"; echo " llama: ${b:0:200}"; fi
done
[ "$ok" = "$n" ] && echo "PASS gemma4-tok-parity: $ok/$n" || { echo "FAIL gemma4-tok-parity: $ok/$n"; exit 1; }
