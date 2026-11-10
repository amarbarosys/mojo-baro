#!/usr/bin/env bash
# Re-runs the named check behind every README WORKS row, one log per check, one TSV of verdicts.
#   bench/claims-verify.sh OUT [cpu|gpu|all] [ONLY_NAME...]
# bundle = the gpu phase as ONE gpu-wait job (priority CV_PRIO, default 90). CV_TRAIN=1 adds the train gate.
# Engines come from $CV_BIN (default OUT/bin, built by the cpu phase). GPU steps queue through
# gpu-wait one at a time; steps that queue themselves are called bare. A failed step does not
# stop the sweep; the run ends PASS n/n or FAIL k/n <names> with a non-zero exit.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(realpath -m "${1:?usage: claims-verify.sh OUT [cpu|gpu|all] [NAME...]}"); phase=${2:-all}; shift 2 || shift $#
only=" $* "
bin=${CV_BIN:-$out/bin}
mkdir -p "$out" "$bin"
tsv=$out/verdicts.tsv
fails=(); n=0
export BARO_SERVE_BIN=${BARO_SERVE_BIN:-$PWD/serve/target/release/baro-serve} ROUTER_BIN=${ROUTER_BIN:-$PWD/serve/target/release/router}
gw() { [ -n "${GPU_WAITING_ROOM_JOB:-}" ] && { env BARO_ENGINE="$bin/engine" "${@:3}"; return; }
  gpu-wait run --priority "${CV_PRIO:-40}" --vram "$1" --timeout "$2" -- env HOME="$HOME" PATH="$PATH" BARO_ENGINE="$bin/engine" "${@:3}"; }
step() {
  local name=$1 rows=$2; shift 2
  [ "$only" = "  " ] || [[ $only == *" $name "* ]] || return 0
  n=$((n + 1)); local t0=$SECONDS rc=0
  "$@" > "$out/$name.log" 2>&1 || rc=$?
  local v=PASS; [ $rc -eq 0 ] || { v="FAIL($rc)"; fails+=("$name"); }
  printf '%s\t%s\t%ss\t%s\t%s\n' "$name" "$v" $((SECONDS - t0)) "$rows" "$out/$name.log" | tee -a "$tsv"
}

if [ "$phase" != gpu ]; then
  step preflight "ci-checks, kernel census, settings template, every test builds" bench/preflight.sh
  step build "dense + MoE engine, baro-serve, router" bash -c "
    ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o '$bin/engine' &&
    ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -D BARO_MODEL=qwen35moe -o '$bin/engine-moe' &&
    (cd serve && cargo build --release -q)"
  step router-affinity "router prefix affinity, rendezvous, cold-miss lock" bench/p8-router-affinity-gate.sh "$out/router-affinity"
  step minja "chat templates (minja)" tools/minja-gate.sh
  step pwa-wiring "PWA shell, pull, topk, hidden, HMAC, speech, tool-call deltas (fake engine)" bench/p6-pwa-gate.sh "$out/pwa"
fi

if [ "$phase" = bundle ]; then
  [ -n "${GPU_WAITING_ROOM_JOB:-}" ] || exec gpu-wait run --priority "${CV_PRIO:-90}" --vram 23 --timeout "${CV_TIMEOUT:-7200}" -- \
    env HOME="$HOME" PATH="$PATH" CV_BIN="$bin" CV_MOE_PACK="${CV_MOE_PACK:-}" BARO_SERVE_BIN="$BARO_SERVE_BIN" ROUTER_BIN="$ROUTER_BIN" bash "$(realpath "$0")" "$out" bundle "$@"
  phase=gpu
fi
if [ "$phase" != cpu ]; then
  step fixture "q4 pack reference: llama.cpp greedy 64 ids on the pack's source gguf, bake prompt" bash -c "
    p=.work/engine-pack-q4; src=\$(jq -r .source \$p/identity.json)
    [ -s \$p/prompt-tokens.txt ] || ./.venv/bin/python tools/gguf-extract.py '${CV_BAKE:-$HOME/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-04867e2.gguf}' --meta | jq -r '.[\"baro.run.prompt.tokens\"]' > \$p/prompt-tokens.txt
    [ -s \$p/ref-tokens-64.txt ] || { PROMPT_FILE=\$p/prompt-tokens.txt tools/llama-ref-run.sh \$src '$out/fixture' 8097 1 && cp '$out/fixture/ref-tokens-64-llama.txt' \$p/ref-tokens-64.txt; }
    [ \$(wc -l < \$p/ref-tokens-64.txt) -eq 64 ]"
  step run-tests "SSM scan, rmsnorm fold, sampler/penalties/logprobs/grammar, MoE rows, K-quant, prefix restore" \
    gw 22 1800 ./run-tests.sh
  step server "dense engine + MTP + megakernel ids == ref, SSE, stop, cancel, templates" \
    env BARO_ENGINE="$bin/engine" tools/test_server.sh "$out/server"
  step contract "embeddings, OpenAI contract" gw 23 3600 bench/p0a-contract-gate.sh "$out/contract" 1 11434
  step pool "engine pool BARO_POOL=2" env BARO_TMAX=4096 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 tools/test_pool.sh "$out/pool"
  step state "checkpoints, ckpt share, BAROST formats" bench/state-gates.sh "$bin/engine" .work/engine-pack-q4 "$out/state" 8120
  step state-roundtrip "LAT1 export/import one node" gw 23 900 bench/p1-state-roundtrip-smoke.sh "$out/state-roundtrip"
  step fork "single-node /v1/fork" gw 23 900 env QUICK=3 bench/fork-live-smoke.sh "$out/fork"
  step latent "LatentOS memfd handoff, identity 20/20" gw 23 1800 tools/latent-gate.sh "$out/latent"
  step router-live "router placement + failover" gw 23 1800 bench/p0b-gates12-live.sh "$out/router-live"
  step moe-prefill "MoE engine, batched prefill identity, tier mode" \
    gw 22 1800 env QUICK=3 bench/moe-prefill-identity.sh "$bin/engine-moe" "${CV_MOE_PACK:-missing-CV_MOE_PACK}" "$out/moe-prefill" tier
  step whisper "speech in (whisper sidecar)" bench/p3a-gate.sh "$out/whisper"
  [ -n "${CV_TRAIN:-}" ] && step train "miniagi/train/ parity, 200 steps, bit-identical rerun" tools/train-gate.sh
fi

if [ ${#fails[@]} -eq 0 ]; then echo "PASS $n/$n ($tsv)"; else echo "FAIL ${#fails[@]}/$n ${fails[*]} ($tsv)"; exit 1; fi
