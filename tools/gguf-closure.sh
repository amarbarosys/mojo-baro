#!/usr/bin/env bash
# Strict identity closure: build the engine FROM the sources embedded in a BARO
# gguf (baro.kernel.src.* KVs), run it on .work/engine-pack/, gate on ref tokens.
# usage: tools/gguf-closure.sh MODEL-BARO.gguf [ref-tokens-file] [outdir]
# Spark-harness ggufs (spark_kernels.mojo in the file list -- spark2_5 itself and
# the profile-driven dense families llama/qwen2/granite all use serve/spark.mojo):
# harness = serve/spark.mojo at the gguf commit, profile/prompt/pack/reference all
# from the file itself (baro.kernel.src.profile.mojo, baro.run.prompt.tokens,
# baro.run.pack.tool + baro.run.src.<tool> + baro.run.pack.flags, baro.run.ref.tokens).
# Run under gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; ref=${2:-}; out=${3:-.work/gguf-src}
rm -rf "$out"; mkdir -p "$out"
./.venv/bin/python3 tools/gguf-extract.py "$model" --meta > "$out/meta.json"
jq -r '.["baro.kernel.files"]' "$out/meta.json" | tr ',' '\n' > "$out/FILES"
while read -r f; do mkdir -p "$out/$(dirname "$f")"; jq -r --arg k "baro.kernel.src.$f" '.[$k]' "$out/meta.json" > "$out/$f"; done < "$out/FILES"
jq -r '"commit: " + .["baro.kernel.commit"] + "  arch: " + .["baro.kernel.arch"] + "  files: " + .["baro.kernel.files"]' "$out/meta.json"
# The file's own prompt and reference ids (baro.run.*, B1). A bake that carries
# them verifies with nothing outside itself; before they existed the qwen35moe
# branch defaulted to .work/moe-w3/one.tokens, a local path the gguf never
# carried and which no longer exists, so tools/gguf-verify.sh exited 1 on every
# MoE bake.
jq -er '.["baro.run.prompt.tokens"]' "$out/meta.json" > "$out/prompt.tokens" 2>/dev/null \
  && echo "prompt: from the file ($(wc -w < "$out/prompt.tokens") ids)" || rm -f "$out/prompt.tokens"
# tools/check-tokens.sh reads its ref file with `mapfile` (one id per line);
# baro.run.ref.tokens is a single space-separated KV string (bake.sh joins it
# that way so it round-trips through GGUF's string type), so it is reflowed
# to one-per-line here rather than in every branch that reads $ref (the MoE
# branch below used to do this itself; a spark bake hit the same mismatch --
# "expected <all 64 ids>, got <first id>" -- the day this comment was added).
# `// empty` (not `-e`) so a missing/null key yields a truly empty file, not
# the literal text "null" surviving through tr/grep as a false "from the file".
jq -r '.["baro.run.ref.tokens"] // empty' "$out/meta.json" 2>/dev/null | tr -s ' ' '\n' | grep -v '^$' > "$out/ref-embedded.txt"
if [ -s "$out/ref-embedded.txt" ]; then
  echo "reference: from the file ($(wc -l < "$out/ref-embedded.txt") ids)"
else
  rm -f "$out/ref-embedded.txt"
fi
# A challenge beside the file wins over the file's own ref: a file that carries
# its own reference can never fail its own gate (tools/gguf-check.py diffs them).
if [ -z "$ref" ] && [ -f "$model.challenge.json" ]; then
  jq -r '.["baro.run.ref.tokens"] // empty' "$model.challenge.json" | tr -s ' ' '\n' | grep -v '^$' > "$out/ref-challenge.txt" || true
  [ -s "$out/ref-challenge.txt" ] && ref="$out/ref-challenge.txt" && echo "reference: from the challenge ($(wc -l < "$ref") ids)"
fi
[ -n "$ref" ] || { [ -s "$out/ref-embedded.txt" ] && ref="$out/ref-embedded.txt"; } || true
[ -n "${BARO_PROMPT:-}" ] || { [ -s "$out/prompt.tokens" ] && export BARO_PROMPT="$out/prompt.tokens"; } || true
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10

# Build a pack for $model into $out/pack from the pack tool the file itself
# carries (baro.run.pack.tool names it, baro.run.src.<that name> is its
# source, baro.run.pack.flags its flags). Used by both the spark branch and
# the dense qwen35 split-layout branch below -- both used to assume a
# pack already sat at a fixed local path (.work/spark/pack-q8,
# .work/engine-pack-q4) and crashed or silently used someone else's model's
# pack when it did not (2026-09-16, ornith/qwythos-v2 first bakes: the
# split-layout branch opened .work/engine-pack-q4/index.txt unconditionally,
# a path that happens to hold Qwythos's own weights and nothing else's).
# Not applied to the qwen35moe branch above: a 35B pack takes real time to
# build and that branch's local-pack reuse is deliberate, not a gap.
build_pack_from_file() {
  local packtool packflags
  packtool=$(jq -r '.["baro.run.pack.tool"] // empty' "$out/meta.json")
  [ -n "$packtool" ] || { echo "no baro.run.pack.tool: file carries no pack builder, cannot build a pack from itself"; exit 1; }
  # the pack tool gets a directory of its own: a Mojo tool builds with -I there, and the closure's kernel sources
  # in $out must not be on its path
  local pt=$out/packtool; rm -rf "$pt"; mkdir -p "$pt"
  jq -er --arg k "baro.run.src.$packtool" '.[$k]' "$out/meta.json" > "$pt/$packtool" 2>/dev/null \
    || { echo "no baro.run.src.$packtool embedded"; exit 1; }
  jq -er '.["baro.run.src.gguf-extract.py"]' "$out/meta.json" > "$pt/gguf-extract.py" 2>/dev/null \
    || cp tools/gguf-extract.py "$pt/gguf-extract.py"
  for dep in $(jq -r '.["baro.run.pack.deps"] // empty' "$out/meta.json" | tr ',' ' '); do
    jq -er --arg k "baro.run.src.$dep" '.[$k]' "$out/meta.json" > "$pt/$dep" 2>/dev/null \
      || { echo "no baro.run.src.$dep embedded (baro.run.pack.deps)"; exit 1; }
  done
  packflags=$(jq -r '.["baro.run.pack.flags"] // empty' "$out/meta.json")
  # shellcheck disable=SC2086
  tools/pack-run.sh "$pt/$packtool" "$model" "$out/pack" $packflags > "$out/pack.log" 2>&1 \
    || { tail -20 "$out/pack.log"; echo "pack build FAILED, see $out/pack.log"; exit 1; }
}
# vendor arm: if the gguf carries the hipBLASLt shim sources, build them too
if [ -f "$out/shim/CMakeLists.txt" ]; then
  cmake -S "$out/shim" -B "$out/shim-build" -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build "$out/shim-build" -j"$(nproc)" >/dev/null \
    && echo "shim built from gguf: $(ls "$out"/shim-build/*.so)" || { echo "shim build FAILED"; exit 1; }
fi
# Split layout (2026-09-08): no main() in the embedded sources; the harness is
# serve/engine.mojo at the gguf's commit, from git, never from the gguf.
entry="$out/engine.mojo"
# serve/engine.mojo imports latentos; vendored at the repo root since
# briefs/2026-09-15-vendor-latentos.md, so tools/embed-files.py's EXT walk
# pulls the package into the file's own baro.kernel.src.latentos/* KVs and
# this closure needs no include outside $out for it (no more than it does
# for kernels/ or serve/).
kmodel=$(jq -r '.["baro.kernel.model"] // empty' "$out/meta.json")
# qwen4exp: the harness serve/q4x.mojo comes from git at the gguf commit; the
# baked shard 1 is its own pack (its -0000N-of- siblings sit beside it), and the
# rebuilt engine must print the reference ids exactly.
# q4x_kernels.mojo also enters every qwen35 and spark closure (gemv_kq imports it): window.mojo and spark_kernels.mojo rule those out.
if grep -qx q4x_kernels.mojo "$out/FILES" && ! grep -qxE 'window.mojo|spark_kernels.mojo' "$out/FILES"; then
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/q4x.mojo" > "$out/q4x_main.mojo" || { echo "no serve/q4x.mojo at gguf commit $kcommit"; exit 1; }
  ./.venv/bin/mojo build "$out/q4x_main.mojo" -I "$out" -I "$out/tools" -o .work/q4x-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  [ -x .work/q4x-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
  [ -s "$out/prompt.tokens" ] || { echo "no prompt (baro.run.prompt.tokens missing)"; exit 1; }
  [ -n "$ref" ] || { echo "no reference tokens (baro.run.ref.tokens missing and none given)"; exit 1; }
  n=$(wc -w < "$ref")
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2 ./.work/q4x-closure "$model" "$n" $(cat "$out/prompt.tokens") > "$out/run.log" 2>&1 \
    || { tail -20 "$out/run.log"; echo "closure engine FAILED"; exit 1; }
  got=$(sed -n 's/^ids //p' "$out/run.log"); want=$(tr -s ' \n' ' ' < "$ref" | sed 's/ $//; s/^ //')
  [ "$got" = "$want" ] && { echo "PASS $n ids identical"; exit 0; }
  echo "FAIL ids differ"; echo " got  $got"; echo " want $want"; exit 1
fi
if [ "$kmodel" = "qwen35moe" ]; then
  # qwen35moe FULL ENGINE closure (2026-09-12): the file carries the whole
  # engine closure, not just the expert kernels, so rebuild the engine from it
  # exactly as the qwythos path does and gate on reference tokens. The harness
  # (serve/engine.mojo) comes from git at the gguf commit, never from the gguf,
  # so a baked file cannot carry its own clock.
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at gguf commit $kcommit"; exit 1; }
  # the harness's own modules: every embedded baro.run.src.*.mojo first (the bake's transitive harness closure:
  # latent_mlp -> semantic_latent is not a direct import), then git serve/, then git kernels/ for anything missing
  # (engine.mojo imports kernels/state.mojo flat; a serve/-only lookup stopped at "unable to locate module 'state'")
  for k in $(jq -r 'keys[] | select(startswith("baro.run.src.") and endswith(".mojo"))' "$out/meta.json"); do
    n=${k#baro.run.src.}; [ "$n" = engine.mojo ] || [ -f "$out/$n" ] || jq -r --arg k "$k" '.[$k]' "$out/meta.json" > "$out/$n"
  done
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] && continue
    jq -er --arg k "baro.run.src.$m.mojo" '.[$k]' "$out/meta.json" > "$out/$m.mojo" 2>/dev/null && continue
    rm -f "$out/$m.mojo"
    for d in serve kernels; do
      git cat-file -e "$kcommit:$d/$m.mojo" 2>/dev/null && { git show "$kcommit:$d/$m.mojo" > "$out/$m.mojo"; break; }
    done
  done
  ./.venv/bin/mojo build "$out/closure_main.mojo" -I "$out" -D BARO_MODEL=qwen35moe -o .work/moe-engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  [ -x .work/moe-engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
  moeref=${ref:-.work/moe-closure-ref.txt}
  [ -f "$moeref" ] || { echo "no reference token file $moeref: the gguf carries no baro.run.ref.tokens and no file was given"; exit 1; }
  # The reference holds bare ids. A file still carrying the engine's own
  # "GENERATED:" prefix compares that word against a token id and reports a
  # mismatch at position 1 on a run that is actually identical.
  sed 's/^GENERATED: *//' "$moeref" | tr -s ' ' '\n' | grep -v '^$' > "$out/ref-ids.txt"
  moeref="$out/ref-ids.txt"
  moeprompt=${BARO_PROMPT:-.work/moe-w3/one.tokens}
  [ -f "$moeprompt" ] || { echo "no prompt token file $moeprompt: the gguf carries no baro.run.prompt.tokens and BARO_PROMPT is unset"; exit 1; }
  # Tier pack only. The resident MoE pack (one 20 GB pack.bin) ran out of VRAM next to the desktop on 2026-10-12 and
  # the compositor's ring resets took the machine down; pack-fit had called it a fit. A split pack (experts.bin from
  # tools/pack-split-experts.py) runs the expert tier, bit-exact with resident decode.
  [ -n "${BARO_PACK:-}" ] || { echo "FAIL pack: set BARO_PACK to a tier pack (pack.bin + experts.bin, tools/pack-split-experts.py)"; exit 1; }
  [ -f "$BARO_PACK/experts.bin" ] || { echo "FAIL pack: $BARO_PACK has no experts.bin; a resident MoE pack is refused (OOM next to the desktop, 2026-10-12)"; exit 1; }
  env BARO_MEGA=0 BARO_PREFILL=1 BARO_TIER=${BARO_TIER:-64} BARO_TIER_PINNED=1 BARO_TIER_ZC=1 BARO_PACK="$BARO_PACK" \
      BARO_PROMPT="$moeprompt" ./.work/moe-engine-closure > "$out/run.log" 2>&1 \
      || { echo "closure engine FAILED"; tail -20 "$out/run.log"; exit 1; }
  grep -E "tok/s_gen|mega fail word" "$out/run.log"
  tools/check-tokens.sh "$moeref" "$out/run.log"
  exit
fi
if grep -qx moe.mojo "$out/FILES" && ! grep -qxE 'window.mojo|spark_kernels.mojo' "$out/FILES"; then
  # qwen35moe kernels-only bake (no engine closure in the file); the closure rebuilds the parity test from the embedded
  # kernels and checks blk.0 of THIS gguf against the numpy oracle (tools/moe-ref.py).
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:kernels/test_moe_block.mojo" > "$out/harness.mojo" || { echo "no kernels/test_moe_block.mojo at gguf commit $kcommit"; exit 1; }
  ./.venv/bin/mojo build "$out/harness.mojo" -I "$out" -o .work/moe-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  ./.venv/bin/python3 tools/moe-ref.py --gguf "$model" --layer 0 > "$out/oracle.log"
  ./.work/moe-closure > "$out/run.log" 2>&1 || { echo "closure test FAILED"; tail -20 "$out/run.log"; exit 1; }
  grep -E "PASS|FAIL|rel|exact" "$out/run.log" | tail -12
  exit
fi
if grep -qx spark_kernels.mojo "$out/FILES"; then
  # serve/spark.mojo is its own harness (baked from git at the gguf's commit,
  # same rule as every other arch: the file cannot carry its own clock), but
  # unlike qwythos/qwen35moe it needs three more things the file must carry,
  # because before this there was no local .work/spark/ fixture on this box
  # and the closure could not run at all (2026-09-16, coordinator diagnosis):
  # a profile.mojo (dims/rope/activation read from THIS gguf by
  # tools/gen-profile.mojo, extracted automatically above since it rides in
  # baro.kernel.files like any other source), a prompt (baro.run.prompt.tokens,
  # extracted above into $out/prompt.tokens), and a pack, built fresh from the
  # embedded pack tool (baro.run.pack.tool names which of engine-pack.py /
  # spark-pack.py; baro.run.src.<that name> is its source, baro.run.pack.flags
  # its flags) rather than reused from a local directory.
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/spark.mojo" > "$out/harness.mojo" || { echo "no serve/spark.mojo at gguf commit $kcommit"; exit 1; }
  [ -f "$out/profile.mojo" ] || { echo "no profile.mojo embedded (baro.kernel.src.profile.mojo missing; re-bake with BARO_PROFILE set)"; exit 1; }
  ./.venv/bin/mojo build "$out/harness.mojo" -I "$out" -I "$out/tools" -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
  [ -x .work/engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
  build_pack_from_file
  [ -s "$out/prompt.tokens" ] || { echo "no prompt (baro.run.prompt.tokens missing)"; exit 1; }
  [ -n "$ref" ] || { echo "no reference tokens (baro.run.ref.tokens missing and none given)"; exit 1; }
  BARO_PACK="$out/pack" BARO_PROMPT="$out/prompt.tokens" BARO_GEN=64 ./.work/engine-closure > "$out/run.log" 2>&1 \
    || { tail -20 "$out/run.log"; echo "closure engine FAILED"; exit 1; }
  grep -E "tok/s" "$out/run.log"
  sed 's/^generated:/GENERATED:/' "$out/run.log" | tools/check-tokens.sh "$ref" /dev/stdin
  exit
fi
if [ -f "$out/window.mojo" ] && ! grep -q '^def main' "$out/engine.mojo" 2>/dev/null; then
  kcommit=$(jq -r '.["baro.kernel.commit"]' "$out/meta.json")
  git show "$kcommit:serve/engine.mojo" > "$out/closure_main.mojo" || { echo "no serve/engine.mojo at gguf commit $kcommit"; exit 1; }
  # the harness's own modules: every embedded baro.run.src.*.mojo first (the bake's transitive harness closure:
  # latent_mlp -> semantic_latent is not a direct import), then git serve/, then git kernels/ for anything missing
  # (engine.mojo imports kernels/state.mojo flat; a serve/-only lookup stopped at "unable to locate module 'state'")
  for k in $(jq -r 'keys[] | select(startswith("baro.run.src.") and endswith(".mojo"))' "$out/meta.json"); do
    n=${k#baro.run.src.}; [ "$n" = engine.mojo ] || [ -f "$out/$n" ] || jq -r --arg k "$k" '.[$k]' "$out/meta.json" > "$out/$n"
  done
  for m in $(sed -n 's/^from \([a-z_]*\) import.*/\1/p' "$out/closure_main.mojo"); do
    [ -f "$out/$m.mojo" ] && continue
    jq -er --arg k "baro.run.src.$m.mojo" '.[$k]' "$out/meta.json" > "$out/$m.mojo" 2>/dev/null && continue
    rm -f "$out/$m.mojo"
    for d in serve kernels; do
      git cat-file -e "$kcommit:$d/$m.mojo" 2>/dev/null && { git show "$kcommit:$d/$m.mojo" > "$out/$m.mojo"; break; }
    done
  done
  entry="$out/closure_main.mojo"; echo "split layout: harness serve/engine.mojo@$kcommit"
fi
# Shape -D flags from this file's own metadata (OrcaSAQ-27B on the 9B qwen35 profile): without them the
# closure builds the 9B-shaped engine. Defaults come back empty, so a 9B bake builds exactly as before.
dflags=$(./.venv/bin/python tools/model-id.py "$model" --json | jq -r '.fields | to_entries[] | select(.key != "BARO_MODEL" and (.key | startswith("BARO_"))) | "-D \(.key)=\(.value)"' | tr '\n' ' ')
# Build flags the bake chose for this model (-D BARO_SEQ_CAP=1 on 27B: 4.72 GB less VRAM).
dflags="$dflags$(jq -r '.["baro.run.build.flags"] // empty' "$out/meta.json")"
echo "shape flags: ${dflags:-none}"
./.venv/bin/mojo build "$entry" -I "$out" $dflags -o .work/engine-closure 2>&1 | grep -E "error" -A3 && exit 1 || true
[ -x .work/engine-closure ] || { echo "closure build FAILED (no binary)"; exit 1; }
# This branch used to run with whatever BARO_PACK happened to be set, or the
# engine's own hardcoded default (.work/engine-pack-q4) otherwise -- a path
# that holds one specific model's weights, silently wrong for any other
# gguf verified on this box (2026-09-16: ornith and qwythos-v2's first
# bakes both opened Qwythos's own pack and either crashed on a missing
# tensor or, worse, would have "verified" against the wrong model's output
# if the shapes had happened to line up). Build this file's own pack unless
# the caller pinned BARO_PACK explicitly.
[ -n "${BARO_PACK:-}" ] || { build_pack_from_file; export BARO_PACK="$out/pack"; }
[ -n "$ref" ] || { echo "no reference tokens (baro.run.ref.tokens missing and none given)"; exit 1; }
./.work/engine-closure > "$out/run.log"
grep -E "tok/s|host_enqueue" "$out/run.log"
tools/check-tokens.sh "$ref" "$out/run.log"
