#!/usr/bin/env bash
# Build the C++ shim, then build and run the Mojo verification tests.
# Builds go through tools/build-many.sh (import-closure cache). A test whose binary sha256, run
# command and named inputs match a previous PASS prints CACHED PASS and is not rerun (iTool
# pass-cache, markers under .work/test-pass/). RUN_ALL=1 reruns every test: before a merge, and for environmentally flaky probes.
set -euo pipefail
cd "$(dirname "$0")"
S="$PWD/.work/shim-build"; export S
cmake -S shim -B "$S" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$S" -j"$(nproc)" >/dev/null
mkdir -p .work/test-pass
pc() { PASS_CACHE_DIR=.work/test-pass "$HOME/iTools/bin/pass-cache" "$@"; }  # iTool: skip on a passed key
# t BIN "BUILD ARGS" [INPUT ...] -- RUN...: build BIN, run RUN unless its key already passed
t() {
  local bin=$1 bargs=$2; shift 2; local ins=(--in "$bin" --in "$S/libamarbaro_shim.so"); while [ "$1" != -- ]; do ins+=(--in "$1"); shift; done; shift
  echo "./.venv/bin/mojo build $bargs -o $bin  --target-accelerator ${BARO_TARGET_ACCELERATOR:-gfx1100}" | tools/build-many.sh "rt-$(basename "$bin")" > ".work/test-pass/$(basename "$bin").build" 2>&1 \
    || { echo "FAIL build $bin: see .work/test-pass/$(basename "$bin").build"; exit 1; }
  pc "${ins[@]}" -- bash -c "$*"
}
t .work/test_gemm "kernels/test_gemm.mojo -I kernels -Xlinker -L$S -Xlinker -lamarbaro_shim -Xlinker -rpath -Xlinker $S" -- ./.work/test_gemm

# MOEPF: row-batched MoE kernels bit-exact against their m=1 siblings
t .work/test_moe_rows "kernels/test_moe_rows.mojo -I kernels" -- ./.work/test_moe_rows
t .work/test_moe_wmma "kernels/test_moe_wmma.mojo -I kernels" -- ./.work/test_moe_wmma
t .work/test_ssm_rows "kernels/test_ssm_rows.mojo -I kernels" -- ./.work/test_ssm_rows
# GQA-batched prefill attention byte-equal to the per-head decode kernel (MoE int8 KV, dense f32 KV)
t .work/test_attn_gqa_moe "kernels/test_attn_gqa.mojo -I kernels -I serve -D BARO_MODEL=qwen35moe" -- ./.work/test_attn_gqa_moe
t .work/test_attn_gqa "kernels/test_attn_gqa.mojo -I kernels -I serve" -- ./.work/test_attn_gqa
# q4x device kernels vs the f32 host math they replaced (serve/q4x.mojo)
t .work/test_q4x_hostops "kernels/test_q4x_hostops.mojo -I kernels" -- ./.work/test_q4x_hostops
# native K-quant GEMV of the spark engine (Q4_K / Q6_K / split QKV): device bytes vs the same-order host run
t .work/test_gemv_kq "kernels/test_gemv_kq.mojo -I kernels -I tools" -- ./.work/test_gemv_kq
t .work/test_gemv_kq_rows "kernels/test_gemv_kq_rows.mojo -I kernels -I tools" -- ./.work/test_gemv_kq_rows
t .work/test_turbo_rows "kernels/test_turbo_rows.mojo -I kernels -I serve -I tools" -- ./.work/test_turbo_rows
t .work/test_doorbell "kernels/test_doorbell.mojo -I kernels" -- ./.work/test_doorbell
# kq batched prefill GEMM (dequant + WMMA) on real Ornith Q4_K / Q6_K rows vs the per-row GEMV, fp64 numpy oracle
# (bench/llama-stint-protocol.md run 9). Needs a --kq pack; skipped loudly like test_prefix.
kqpack=${BARO_KQ_PACK:-$HOME/Projects/mojo/mojo-baro/.work/ornith-stint/pack-kq}
if [ -f "$kqpack/pack.bin" ]; then
  t .work/test_kq_prefill_gemm "kernels/test_kq_prefill_gemm.mojo -I kernels -I tools" tools/kq-prefill-ref.py -- ": pack $(stat -c %s-%Y "$kqpack/pack.bin"); mkdir -p .work/kqpf && ./.work/test_kq_prefill_gemm $kqpack .work/kqpf && ./.venv/bin/python tools/kq-prefill-ref.py .work/kqpf"
else
  echo "SKIP test_kq_prefill_gemm (no kq pack at $kqpack)"
fi
# HIP stream capture of MAX launches (raw hipStream_t via libAsyncRTMojoBindings), graph replay value + node count
t .work/test_graph_capture "kernels/test_graph_capture.mojo -I kernels" -- ./.work/test_graph_capture
# q4x CPU expert path (Q4X_CPUMISS): VNNI Q2_0 FFN vs scalar reference, then us/expert
t .work/test_q4x_cpu "serve/test_q4x_cpu.mojo -I serve" -- ./.work/test_q4x_cpu
# Mojo HTTP front (serve/front/, CPU only): HTTP layer on a fake engine, engine client on a stub child,
# text parity with the Rust front on its oracle (local receipt + pack; skipped loudly without them).
t .work/test_http "serve/front/test_http.mojo -I . -I serve/front" -- ./.work/test_http
t .work/stub_engine "serve/front/stub_engine.mojo -I ." -- :
t .work/test_engine_client "serve/front/test_engine_client.mojo -I ." .work/stub_engine -- ./.work/test_engine_client .work/stub_engine
t .work/baro-front "serve/front/main.mojo -I . -I serve -I serve/front" -- :
t .work/engine_cpu "cpu/engine_cpu.mojo -I . -I kernels -I cpu -I serve" -- :  # CPU tier engine builds without max (bench/check-cpu-engine.sh is its gate)
if [ -f exchange/receipts/front/text-oracle.jsonl ] && [ -f .work/ailab/pack-kq/tokenizer.json ]; then
  t .work/test_text "serve/front/test_text.mojo -I . -I serve" exchange/receipts/front/text-oracle.jsonl -- MINJA_NOW=1700000000 ./.work/test_text
else
  echo "SKIP test_text (needs exchange/receipts/front/text-oracle.jsonl and .work/ailab/pack-kq)"
fi
# cold-cache helper: bandwidth ruler units, read-sweep flush effect, copy flush is worse (kernels/icflush.mojo)
t .work/test_icflush "kernels/test_icflush.mojo -I kernels" -- ./.work/test_icflush
# Freivalds exact one-pass check of integer GEMMs (exchange/freivalds-prereg-2026-10-12.md)
t .work/test_freivalds "kernels/test_freivalds.mojo -I kernels" -- ./.work/test_freivalds --no-speed

# M1a prefix checkpoints: byte-exact restore against the real engine path
# (needs the q4 pack at BARO_PACK, default .work/engine-pack-q4).
# A fresh clone has no pack: every other test below is self-contained, this one is
# skipped LOUDLY and the script ends with exit 3, not 0. Build a pack from any
# qwen35 GGUF with `tools/baro run MODEL.gguf --no-serve` (it prints the pack dir).
pack=${BARO_PACK:-.work/engine-pack-q4}; skipped=""
if [ -f "$pack/pack.bin" ]; then
  t .work/test_prefix "kernels/test_prefix.mojo -I . -I kernels -I serve" -- ": pack $(stat -c %s-%Y "$pack/pack.bin"); BARO_PACK=$pack ./.work/test_prefix"
  t .work/test_cpu_layer "kernels/test_cpu_layer.mojo -D BARO_SHARED_EXP=1 -I . -I kernels -I cpu -I serve" -- ": pack $(stat -c %s-%Y "$pack/pack.bin"); ./.work/test_cpu_layer $pack"  # CPU tier ops bitwise vs GPU, 27 stages
else
  skipped="test_prefix (no q4 pack at $pack)"
  echo "SKIP $skipped"
fi

# C3 host reference sampler: pure CPU, no accelerator needed, matched
# against kernels/sample.mojo's semantics (lane-KSAMP).
t .work/test_sample_ref "kernels/test_sample_ref.mojo -I kernels -I serve" -- ./.work/test_sample_ref

# Penalties and top-N logprobs: device kernels vs the host reference at real
# VOCAB width (briefs/2026-09-16-fable-sample-kernels.md).
t .work/test_sample_pen "kernels/test_sample_pen.mojo -I . -I kernels -I serve" -- ./.work/test_sample_pen

# Grammar-masked sampling: mask before truncation, masked argmax at T = 0,
# masked probs rows (briefs/2026-09-16-fable-masked-sampler.md).
t .work/test_sample_mask "kernels/test_sample_mask.mojo -I kernels -I serve" -- ./.work/test_sample_mask

# KATT head-dimension parity: the Spark attention path at HD 64/128/256,
# judged by the numpy float64 oracle (bench/dense-protocol.md, KATT).
mkdir -p .work/katt
# fused decode rope (q + k/v append, one launch) byte-equal to the two-launch path
t .work/test_rope_qkv "kernels/test_rope_qkv.mojo -I kernels -I serve" -- ./.work/test_rope_qkv
t .work/test_spark_attn "kernels/test_spark_attn.mojo -I kernels -I serve" tools/spark-attn-ref.py -- "./.work/test_spark_attn && ./.venv/bin/python tools/spark-attn-ref.py .work/katt"

# serve/latent.mojo sat broken from 506e91d to 2e7b5d3 because nothing here
# built it. This test names both mint and ingest, so the whole sidecar is
# type-checked: Mojo checks lazily per reached symbol, and building a tool that
# imports only one side leaves the other side's breakage invisible.
t .work/test_latent "kernels/test_latent.mojo -I . -I kernels -I serve" -- ./.work/test_latent

# JSON-enforcement lane item 1 prep: the request-line schema slice
# (briefs/2026-09-16-json-enforcement-lane.md), pure string handling.
t .work/test_serve_proto "serve/test_serve_proto.mojo -I . -I serve" -- ./.work/test_serve_proto
# engine wire words on the CPU (serve/wire.mojo): err_line escaping, done-line fields, finish words, is_cancel_line, decide_path
t .work/test_fsm_wire "serve/test_fsm_wire.mojo -I . -I serve -I kernels" -- ./.work/test_fsm_wire

t .work/test_semantic_latent "serve/test_semantic_latent.mojo -I . -I serve" latentos/tests/fixtures/binding.manifest latentos/tests/fixtures/expected.f32 -- ./.work/test_semantic_latent latentos/tests/fixtures/binding.manifest latentos/tests/fixtures/expected.f32 3 2

./.venv/bin/mojo build tools/kernel-census.mojo -o .work/kernel-census
./.work/kernel-census --check

# State layer (13-fifty-steps step 25, M3): the gates a user meets, on the GPU
# through gpu-wait (each script queues itself when not already inside a job):
#   bench/state-gates.sh   checkpoint-api 28/28 in both formats, S1 share, S4 budget + identity
#   bench/turn-api.sh      /v1/turn TTFT + tier, chat checkpoints (TOKENIZER set), /metrics,
#                          file pull + import across two servers
#   AMDHQ latentos-check   the LatentOS package tests, when that repo is on this machine
# Needs the pack and a built engine + baro-serve (STATE_ENGINE, default .work/engine);
# without them the block is SKIPPED loudly and the script ends with exit 3.
# STATE_TEST_BREAK=1 runs a failing gate on purpose: proves red propagates.
state_engine=${STATE_ENGINE:-.work/engine}
if [ "${STATE_TEST_BREAK:-0}" = 1 ]; then
  echo "state layer: STATE_TEST_BREAK=1, running a failing gate on purpose"; false
elif [ -x "$state_engine" ] && ldd "$state_engine" | grep -q 'not found'; then
  # a stale engine (built against a venv that is gone) fails each gate only as a VOID after its 181 s health timeout
  echo "FAIL state layer: $state_engine does not load: $(ldd "$state_engine" | grep 'not found' | head -1 | xargs)"
  echo "  rebuild: ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o $state_engine"; exit 1
elif [ -f "$pack/pack.bin" ] && [ -x "$state_engine" ] && [ -x serve/target/release/baro-serve ]; then
  pc --in "$state_engine" --in serve/target/release/baro-serve --in bench/state-gates.sh --in bench/turn-api.sh --in bench/checkpoint-api.sh -- bash -c ": state pack $(stat -c %s-%Y "$pack/pack.bin"); STATE_GATES_TIMEOUT=600 bench/state-gates.sh $state_engine $pack .work/run-tests-state-gates 8300 && bench/turn-api.sh $state_engine $pack .work/run-tests-turn 8310"
  if [ -x "$HOME/AMDHQ/tools/latent-os/latentos-check" ]; then "$HOME/AMDHQ/tools/latent-os/latentos-check"; else echo "SKIP latentos-check (AMDHQ not on this machine)"; fi
else
  skipped="$skipped state-layer (needs $pack/pack.bin, $state_engine and serve/target/release/baro-serve)"
  echo "SKIP state layer: needs $pack/pack.bin, an engine at $state_engine and serve/target/release/baro-serve"
fi
if [ -n "$skipped" ]; then
  echo "INCOMPLETE: every pack-free test passed; SKIPPED $skipped."
  echo "  Build one: tools/baro run MODEL.gguf --no-serve   then   BARO_PACK=<printed pack dir> ./run-tests.sh"
  exit 3
fi
echo "PASS run-tests: all tests, pack $pack"
