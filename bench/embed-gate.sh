#!/usr/bin/env bash
# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
# Gate for the standard embedder (bge-small-en-v1.5 behind baro-serve --embed-only). CPU only, no GPU job.
#   30 fixed texts (bench/embed-gate-texts.jsonl: short, unicode, multi-line, empty, whitespace, 301 tokens, 700 tokens
#   cut at 512), vectors from baro-serve against (a) the ONNX oracle on all 30 and (b) brain-ask's embed.mojo on the
#   texts its `embed lines` mode can take (one line, not empty, <= 256 tokens: it cuts at 256). Bar: cosine >= 0.9999
#   each, dims 384, unit norm. Falsifier: a build whose pooling is the mean over tokens instead of CLS must fail the
#   same bar. Arm identity is read back from /proc (model dir, norm table, token cap, binary sha) before any number.
#   Exit 0 PASS; 1 FAIL; 3 UNVERIFIED (EXPLORE=1 with an input missing). Receipt: $W/receipt.txt, copied to
#   docs/receipts/bge-embed/embed-gate.txt.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
ROOT=$PWD
W=${EMBED_GATE_W:-$ROOT/.work/embed-gate}
MODEL=${BARO_EMBED_MODEL:-$HOME/Models/bge-small-en-v1.5}
BRAIN=${BRAIN_EMBED_BIN:-$HOME/iTools/brain/brain-ask/.bin/embed}
PORT=${EMBED_GATE_PORT:-18381}
BAR=0.9999
MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo
fail() { echo "FAIL $1: $2"; exit 1; }
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
rm -rf "$W"; mkdir -p "$W"
exec > >(tee "$W/receipt.txt") 2>&1

[ -d "$MODEL" ] || fail model "$MODEL missing"
UNVER=""
if [ ! -x "$BRAIN" ]; then
  [ "${EXPLORE:-0}" = 1 ] || fail brain-ask "$BRAIN missing (EXPLORE=1 runs without that arm and exits 3)"
  UNVER="brain-ask arm NOT EXERCISED"
fi
G="python3 -I embed/oracle/embed_gate.py"
T=bench/embed-gate-texts.jsonl
[ "$(wc -l < $T)" = 30 ] || fail fixture "$T must hold 30 texts"

python3 -I embed/oracle/bge_norm_oracle.py | cmp -s - embed/norm.tsv || fail norm "embed/norm.tsv differs from bge_norm_oracle.py output"

build_embed() {  # SRC_DIR OUT_DIR
  mkdir -p "$2"
  "$MOJO" build "$1/serve_embed.mojo" -I "$1" -o "$2/serve_embed" > "$2/build.log" 2>&1 || fail build "$2/build.log"
  cp embed/norm.tsv "$2/norm.tsv"
}
build_embed embed "$W/real"
mkdir -p "$W/mutsrc"; cp embed/*.mojo embed/norm.tsv "$W/mutsrc/"
sed -i '/^    var n = V(0)$/i\    for i in range(D):\n        var a = Float32(0)\n        for t in range(T):\n            a += xp.unsafe_load[width=1](t * D + i)\n        xp.unsafe_store(i, a / Float32(T))' "$W/mutsrc/bge.mojo"
cmp -s embed/bge.mojo "$W/mutsrc/bge.mojo" && fail mutant "sed changed nothing"
build_embed "$W/mutsrc" "$W/mut"

BIN=${BARO_SERVE_BIN:-}
if [ -z "$BIN" ]; then
  (cd serve && CARGO_TARGET_DIR=$ROOT/.work/cargo-target cargo build --release --bin baro-serve > "$W/cargo.log" 2>&1) || fail cargo "$W/cargo.log"
  BIN=$ROOT/.work/cargo-target/release/baro-serve
fi

# serve NAME PORT DIR: baro-serve --embed-only with the embedder from DIR; read the arm back from /proc
serve() {
  local name=$1 port=$2 dir=$3
  BARO_EMBED_BIN=$dir/serve_embed BARO_EMBED_NORM=$dir/norm.tsv BARO_EMBED_MODEL=$MODEL BARO_EMBED_MAX_TOKENS=512 \
    "$BIN" --embed-only --port "$port" > "$W/$name.serve.log" 2>&1 &
  local pid=$!; PIDS+=("$pid")
  for _ in $(seq 100); do curl -sf "localhost:$port/health" > /dev/null 2>&1 && break; sleep 0.2; done
  curl -sf "localhost:$port/health" > /dev/null || fail "serve-$name" "no /health on :$port, see $W/$name.serve.log"
  local child; child=$(pgrep -P "$pid" | head -1)
  [ -n "$child" ] || fail "serve-$name" "baro-serve has no embedder child"
  local cmd; cmd=$(tr '\0' ' ' < "/proc/$child/cmdline")
  [ "$cmd" = "$dir/serve_embed $MODEL $dir/norm.tsv 512 " ] || fail "arm-$name" "child cmdline '$cmd'"
  [ "$(readlink -f /proc/$child/exe)" = "$(readlink -f "$dir/serve_embed")" ] || fail "arm-$name" "child exe is not $dir/serve_embed"
  echo "arm $name: child $(sha256sum "$dir/serve_embed" | cut -c1-12) cmdline: $cmd"
}
post() {  # PORT OUT.json  (all 30 texts, one request)
  python3 -I -c 'import json,sys;print(json.dumps({"input":[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]}))' "$T" \
    | curl -sf "localhost:$1/v1/embeddings" -H 'content-type: application/json' --data-binary @- > "$2" || fail post "request to :$1 failed, see $W"
}

serve real "$PORT" "$W/real"
post "$PORT" "$W/real.json"
$G json "$W/real.json" "$W/real.f32" || fail shape "served vectors are not 30 x 384 unit vectors"

$G onnx "$MODEL" "$T" 512 "$W/onnx.f32" "$W/ntok.txt" || fail onnx "oracle failed (python3 needs onnxruntime and numpy; tokenizers from .venv)"
read -r n min mean < <($G cmp "$W/real.f32" "$W/onnx.f32" all)
echo "baro-serve vs ONNX: $n texts, min cosine $min, mean $mean (bar $BAR)"
python3 -c "import sys; sys.exit(0 if $min >= $BAR and $n == 30 else 1)" || fail onnx-cosine "min cosine $min < $BAR"

if [ -z "$UNVER" ]; then
  rows=$($G eligible "$T" "$W/ntok.txt" 256 "$W/lines.txt")
  BRAIN_EMBED_MODEL=$MODEL BRAIN_EMBED_NORM=$ROOT/embed/norm.tsv "$BRAIN" lines "$W/lines.txt" "$W/brain.f32" 2> "$W/brain.log" || fail brain-ask "embed lines failed, see $W/brain.log"
  echo "arm brain-ask: $(sha256sum "$BRAIN" | cut -c1-12) $BRAIN lines, model $MODEL"
  $G pick "$W/real.f32" "$rows" "$W/real.sub.f32"
  read -r n min mean < <($G cmp "$W/real.sub.f32" "$W/brain.f32" all)
  echo "baro-serve vs brain-ask embed.mojo: $n texts (rows $rows), min cosine $min, mean $mean (bar $BAR)"
  echo "NOT EXERCISED against brain-ask: rows 16 (multi-line), 17 (empty), 20 (301 tokens), 21 (700 tokens): its lines mode cannot take or cuts them at 256"
  python3 -c "import sys; sys.exit(0 if $min >= $BAR and $n >= 20 else 1)" || fail brain-cosine "min cosine $min < $BAR"
fi

curl -sf "localhost:$PORT/api/embeddings" -H 'content-type: application/json' -d '{"model":"bge-small-en-v1.5","prompt":"What is the capital of France?"}' > "$W/api.json" || fail api "/api/embeddings failed"
python3 -I - "$W/api.json" "$W/real.f32" <<'E' || fail api "/api/embeddings vector differs from the batch row 0"
import json, sys
import numpy as np
a = np.array(json.load(open(sys.argv[1]))["data"][0]["embedding"], np.float64)
b = np.fromfile(sys.argv[2], "<f4").reshape(-1, 384)[0].astype(np.float64)
sys.exit(0 if a @ b / np.linalg.norm(a) / np.linalg.norm(b) >= 0.999999 else 1)
E
echo "/api/embeddings single text equals batch row 0 (cosine >= 0.999999)"

serve mut "$((PORT + 1))" "$W/mut"
post "$((PORT + 1))" "$W/mut.json"
$G json "$W/mut.json" "$W/mut.f32" > /dev/null || fail shape "mutant vectors are not 30 x 384 unit vectors"
read -r n min mean < <($G cmp "$W/mut.f32" "$W/onnx.f32" all)
echo "falsifier, mean pooling instead of CLS vs ONNX: $n texts, min cosine $min, mean $mean (must be < $BAR)"
python3 -c "import sys; sys.exit(0 if $min < $BAR else 1)" || fail falsifier "the mean-pooling mutant passed the bar: the gate cannot tell pooling apart"

mkdir -p docs/receipts/bge-embed
if [ -n "$UNVER" ]; then echo "UNVERIFIED: $UNVER"; sleep 0.2; cp "$W/receipt.txt" docs/receipts/bge-embed/embed-gate.txt; exit 3; fi
echo "PASS embed-gate: 30 texts, dims 384, unit norm, ONNX and brain-ask cosine >= $BAR, mean-pooling mutant rejected"
sleep 0.2; cp "$W/receipt.txt" docs/receipts/bge-embed/embed-gate.txt
