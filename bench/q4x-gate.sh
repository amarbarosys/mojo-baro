#!/usr/bin/env bash
# qwen4exp teacher-forced agreement gate (plan p8 step 8).
# REF : llama-server (-ngl 99 --cpu-moe) greedy N ids per prompt
# CTRL: llama-server CPU (-ngl 0) teacher-forced on REF ids (same-arm control, P5b)
# OURS: .work/p8/q4x with Q4X_FORCE=REF ids
# usage: bench/q4x-gate.sh OUT_DIR N_PROMPTS N_TOK   (runs itself under gpu-wait)
# 16K floor (lane q4x-serve16k): Q4X_PROMPT_DIR=bench/q4x-prompts-16k Q4X_PROMPT_TOK=16384 Q4X_CTX=16640 Q4X_TMAX=16640
#   Q4X_PREFILL=1 [Q4X_CHUNK=8192] Q4X_BIN=engine; Q4X_SELF=1 adds the instrument control: OURS forced with its own greedy ids
#   must read N/N. Peak VRAM and GTT of the OURS runs land in OUT/mem.txt. Bar (exchange/q4x-serve16k-prereg-item2.md) is in the scorer.
set -euo pipefail
OUT=${1:?out dir}; NP=${2:-3}; NT=${3:-32}
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/bench/min16k.sh"; min16k "${Q4X_TMAX:-8192}" 16000
GGUF=${Q4X_GGUF:-$HOME/Models/qwen3.8-flash-next/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf}
SRV=$HOME/Projects/imports/llama.cpp-master/build/bin/llama-server
if [ -z "${Q4X_GATE_INNER:-}" ]; then
  mkdir -p "$OUT"
  exec gpu-wait run --vram 16 --timeout 5400 -- env Q4X_GATE_INNER=1 Q4X_GATE_REUSE="${Q4X_GATE_REUSE:-}" Q4X_PROMPT_DIR="${Q4X_PROMPT_DIR:-}" Q4X_DENSE="${Q4X_DENSE:-}" Q4X_WARM="${Q4X_WARM:-}" Q4X_PROMPT_TOK="${Q4X_PROMPT_TOK:-}" Q4X_CTX="${Q4X_CTX:-}" Q4X_TMAX="${Q4X_TMAX:-}" Q4X_PREFILL="${Q4X_PREFILL:-}" Q4X_CHUNK="${Q4X_CHUNK:-}" Q4X_BIN="${Q4X_BIN:-}" Q4X_SELF="${Q4X_SELF:-}" Q4X_DEVARG="${Q4X_DEVARG:-}" Q4X_GGUF="$GGUF" HOME="$HOME" bash "$REPO/bench/q4x-gate.sh" "$OUT" "$NP" "$NT"
fi
cd "$REPO"
for v in Q4X_CHUNK Q4X_TMAX Q4X_PREFILL Q4X_CTX Q4X_PROMPT_TOK Q4X_SELF Q4X_BIN Q4X_REUSE Q4X_DEVARG; do [ -n "${!v:-}" ] || unset "$v"; done
PROMPTS=$(ls ${Q4X_PROMPT_DIR:-bench/mtp-prompts}/*.txt | head -n "$NP")
serve() {  # $1 = extra flags, $2 = log
  "$SRV" -m "$GGUF" $1 -c ${Q4X_CTX:-8192} -np 1 --port 8091 > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 600); do curl -sf localhost:8091/health >/dev/null 2>&1 && return 0; sleep 1; done
  echo "FAIL serve: llama-server not ready, log $2"; kill $SPID; exit 1
}
# Q4X_GATE_REUSE=1: keep the REF/CTRL ids already in OUT (llama.cpp arms are
# fixed across our builds); only the OURS arm reruns.
if [ -z "${Q4X_GATE_REUSE:-}" ]; then
serve "-ngl 99 --cpu-moe" "$OUT/ref-server.log"
for p in $PROMPTS; do
  b=$(basename "$p" .txt)
  jq -Rs '{content: .}' < "$p" | curl -sf localhost:8091/tokenize -d @- | jq -c ".tokens${Q4X_PROMPT_TOK:+[:$Q4X_PROMPT_TOK]}" > "$OUT/$b.prompt"
  [ -z "${Q4X_PROMPT_TOK:-}" ] || [ "$(jq length "$OUT/$b.prompt")" = "$Q4X_PROMPT_TOK" ] || { echo "FAIL prompt: $b has $(jq length "$OUT/$b.prompt") ids, Q4X_PROMPT_TOK=$Q4X_PROMPT_TOK"; exit 1; }
  jq -n --argjson pr "$(cat "$OUT/$b.prompt")" --argjson n "$NT" '{prompt: $pr, n_predict: $n, temperature: 0, top_k: 1, return_tokens: true, cache_prompt: false}' \
    | curl -sf localhost:8091/completion -d @- | jq -c .tokens > "$OUT/$b.ref"
  echo "ref $b $(jq length "$OUT/$b.ref") ids"
done
kill $SPID; wait $SPID 2>/dev/null || true
serve "-ngl 0" "$OUT/ctrl-server.log"
for p in $PROMPTS; do
  b=$(basename "$p" .txt)
  : > "$OUT/$b.ctrl"
  for i in $(seq 0 $((NT - 1))); do
    jq -n --argjson pr "$(cat "$OUT/$b.prompt")" --argjson r "$(cat "$OUT/$b.ref")" --argjson i "$i" '{prompt: ($pr + $r[:$i]), n_predict: 1, temperature: 0, top_k: 1, return_tokens: true, cache_prompt: true}' \
      | curl -sf localhost:8091/completion -d @- | jq -r '.tokens[0]' >> "$OUT/$b.ctrl"
  done
  echo "ctrl $b done"
done
kill $SPID; wait $SPID 2>/dev/null || true
fi
BIN=${Q4X_BIN:-.work/p8/q4x}
{ echo "gate arm: bin=$BIN sha256 $(sha256sum "$BIN" | cut -c1-16) gguf=$GGUF commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain serve kernels | wc -l)"
  echo "knobs: Q4X_TMAX=${Q4X_TMAX:-} Q4X_PREFILL=${Q4X_PREFILL:-} Q4X_CHUNK=${Q4X_CHUNK:-} Q4X_CTX=${Q4X_CTX:-8192} Q4X_PROMPT_TOK=${Q4X_PROMPT_TOK:-} NP=$NP NT=$NT self=${Q4X_SELF:-}"
  echo "vram_free_start $(~/iTools/bin/vram-free 2>/dev/null || echo ?)"; } | tee "$OUT/arm.txt"
ours() {  # $1 prompt base, $2 comma ids to force, $3 out ids file, $4 log
  MC="$OUT/mem.$1.$(basename "$4" .log).cur"; : > "$MC"
  ( while :; do for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_total ] && [ "$(cat $d/mem_info_vram_total)" -gt 20000000000 ] && echo "$(cat $d/mem_info_vram_used) $(cat $d/mem_info_gtt_used)" >> "$MC"; done; sleep 1; done ) & MPID=$!
  env ${2:+Q4X_FORCE="$2"} MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 ${Q4X_TMAX:+Q4X_TMAX=$Q4X_TMAX} ${Q4X_PREFILL:+Q4X_PREFILL=$Q4X_PREFILL} ${Q4X_CHUNK:+Q4X_CHUNK=$Q4X_CHUNK} ${Q4X_DEVARG:+Q4X_DEVARG=$Q4X_DEVARG} \
    /usr/bin/time -f '%e s %M KB' "$BIN" "$GGUF" "$NT" $(jq -r 'join(" ")' "$OUT/$1.prompt") > "$4" 2>&1 \
    || { kill $MPID; echo "FAIL ours: $1, log $4"; exit 1; }
  kill $MPID; wait $MPID 2>/dev/null || true
  echo "$1 $(basename "$4" .log) peak_vram_gb $(sort -n "$MC" | tail -1 | awk '{printf "%.2f", $1/2^30}') peak_gtt_gb $(awk '{print $2}' "$MC" | sort -n | tail -1 | awk '{printf "%.2f", $1/2^30}')" >> "$OUT/mem.txt"
  grep '^ids' "$4" | cut -d' ' -f2- | tr ' ' '\n' > "$3"
}
for p in $PROMPTS; do
  b=$(basename "$p" .txt)
  ours "$b" "$(jq -r 'join(",")' "$OUT/$b.ref")" "$OUT/$b.ours" "$OUT/$b.ours.log"
  echo "ours $b $(tail -1 "$OUT/$b.ours.log") $(grep -m1 '^prompt ' "$OUT/$b.ours.log")"
  if [ -n "${Q4X_SELF:-}" ]; then
    ours "$b" "" "$OUT/$b.self0" "$OUT/$b.self0.log"
    ours "$b" "$(tr '\n' ',' < "$OUT/$b.self0" | sed 's/,$//')" "$OUT/$b.self" "$OUT/$b.self.log"
    cmp -s "$OUT/$b.self0" "$OUT/$b.self" || { echo "FAIL self-control: $b forced with its own ids reads differently, see $OUT/$b.self.log"; exit 1; }
    echo "self-control $b PASS: $(wc -l < "$OUT/$b.self") of $(wc -l < "$OUT/$b.self0") ids"
  fi
done
python3 tools/q4x-gate-score.py "$OUT" $PROMPTS
