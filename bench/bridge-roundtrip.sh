#!/usr/bin/env bash
# usage: bench/bridge-roundtrip.sh [OUT] [P_TOKENS] [Q_TOKENS]
#
# CPU gate for tools/state-to-llama-slot.mojo (P1 item 4, the forward bridge). No GPU:
# llama-server runs with --device none and the GPU hidden.
#   1. llama-server prefills P on the CPU and saves its slot           -> orig.slot
#   2. tools/llama-slot-to-state (the reverse tool, landed)            -> BAROST01
#   3. tools/state-to-llama-slot (the tool under test)                 -> round.slot
#   4. round.slot must equal orig.slot byte for byte
#   5. a fresh llama-server restores round.slot and continues P+Q: cache_n must be the
#      restored length and 32 tokens must come out. Whether they equal the cold continuation
#      is printed (restored_equals_cold) but is llama.cpp's property, not the tool's: on qwen2
#      llama.cpp diverged from its own cold run on its own bytes (protocol amendment 1)
#      (a prompt equal to the restored prefix is re-evaluated whole by llama-server, a
#      recurrent state cannot step back one token, so the suffix Q is what makes reuse visible)
# BRIDGE_HD=N selects an attention-only model (qwen2, llama): the reverse leg is then
# tools/llama-slot-kv-oracle.py and the forward tool gets the geometry it prints.
# Step 4 proves the writer emits llama.cpp's own layout (cell ext, framing, row order,
# recurrent section); step 5 proves llama.cpp accepts and uses the file.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-.work/fork/bridge-roundtrip}
p=${2:-bench/mtp-prompts/p05-math.tokens}
q=${3:-bench/mtp-prompts/p02-python-fib.tokens}
model=${LH_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
server=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
kv=${BRIDGE_KV:-f16}
fwd=.work/fork/state-to-llama-slot
rev=.work/fork/llama-slot-to-state
mkdir -p "$out/slots"
exec > >(tee "$out/roundtrip.log") 2>&1

fail() { echo "FAIL $1: $2 (log $out/roundtrip.log)"; exit 1; }
for f in "$model" "$server" "$p" "$q"; do [ -e "$f" ] || fail setup "missing $f"; done
./.venv/bin/mojo build tools/state-to-llama-slot.mojo -I . -I serve -I kernels -o "$fwd" > "$out/build-fwd.log" 2>&1 || fail build "forward tool, $out/build-fwd.log"
./.venv/bin/mojo build tools/llama-slot-to-state.mojo -I . -I serve -I kernels -o "$rev" > "$out/build-rev.log" 2>&1 || fail build "reverse tool, $out/build-rev.log"
echo "arm: model=$(basename "$model") kv=$kv prompt=$p ($(wc -w < "$p") tokens) server=$("$server" --version 2>&1 | grep -m1 -o 'version: .*' || true)"

srv=""
stop() { [ -n "$srv" ] && { kill "$srv" 2>/dev/null || true; wait "$srv" 2>/dev/null || true; }; srv=""; }
trap stop EXIT
start() {
  port=$((20000 + RANDOM % 20000))
  env HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$server" -m "$model" --device none -ngl 0 -fa on -np 1 -c 2048 \
    -ctk "$kv" -ctv "$kv" --port "$port" --no-webui --slot-save-path "$out/slots" > "$out/server-$1.log" 2>&1 &
  srv=$!
  for _ in $(seq 1 600); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && return 0
    kill -0 "$srv" 2>/dev/null || fail "server-$1" "exited: $(tail -2 "$out/server-$1.log")"
    sleep 0.5
  done
  fail "server-$1" "no /health in 300 s"
}
# A .tokens file is ids in the Qwythos vocab; a .txt file is tokenized by the model's own
# server, so any vocabulary works (lily's is 32000 and rejects the .tokens ids outright).
ids_of() {  # FILE ADD_SPECIAL
  case "$1" in
    *.txt) python3 - "$1" "$2" "$port" <<'PYTOK' || fail tokenize "$1"
import json, sys, urllib.request
body = json.dumps({"content": open(sys.argv[1]).read(), "add_special": sys.argv[2] == "1"}).encode()
r = urllib.request.urlopen(urllib.request.Request(f"http://localhost:{sys.argv[3]}/tokenize", body, {"Content-Type": "application/json"}), timeout=60)
print(",".join(str(t) for t in json.load(r)["tokens"]))
PYTOK
      ;;
    *) tr -s ' \n' ',' < "$1" | sed 's/^,//; s/,$//' ;;
  esac
}
req() { printf '{"prompt":[%s],"n_predict":%s,"temperature":0,"cache_prompt":true,"return_tokens":true}' "$1" "$2"; }

# 1. cold continuation (32 tokens, the reference), then a clean prefill of P and the slot save
start a
pids=$(ids_of "$p" 1); qids=$(ids_of "$q" 0)
[ -n "$pids" ] && [ -n "$qids" ] || fail tokenize "empty id list"
echo "P: $(echo "$pids" | tr ',' ' ' | wc -w) ids, Q: $(echo "$qids" | tr ',' ' ' | wc -w) ids"
req "$pids,$qids" 32 > "$out/req32.json"; req "$pids" 0 > "$out/req0.json"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req32.json" > "$out/cold.json" || fail cold "completion"
curl -sf -X POST "localhost:$port/slots/0?action=erase" > /dev/null || fail erase "slot erase"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req0.json" > "$out/prefill.json" || fail prefill "completion"
curl -sf -X POST "localhost:$port/slots/0?action=save" -H 'content-type: application/json' -d '{"filename":"orig.slot"}' > "$out/save.json" || fail save "slot save"
stop

# 2 to 4. reverse, forward, compare
geom=()
if [ -n "${BRIDGE_HD:-}" ]; then
  # attention-only model (qwen2, llama): the Mojo reverse tool is qwen35-only, so the reverse
  # leg is the Python oracle, which also reads the geometry out of the slot file
  ./.venv/bin/python tools/llama-slot-kv-oracle.py "$out/slots/orig.slot" "$out/from-llama.state" --hd "$BRIDGE_HD" \
    > "$out/oracle.log" 2>&1 || fail reverse "oracle: $(tail -1 "$out/oracle.log")"
  cat "$out/oracle.log"
  read -r -a geom <<< "$(sed -n 's/^geometry: //p' "$out/oracle.log")"
  [ "${#geom[@]}" = 8 ] || fail reverse "oracle printed no geometry line"
else
  "$rev" "$out/slots/orig.slot" roundtrip-pack "$out/from-llama.state" || fail reverse "llama-slot-to-state"
fi
"$fwd" "$out/from-llama.state" "$out/slots/round.slot" --kv "$kv" "${geom[@]}" || fail forward "state-to-llama-slot"
if cmp "$out/slots/orig.slot" "$out/slots/round.slot" > "$out/cmp.txt" 2>&1; then
  echo "byte identity: PASS ($(stat -c%s "$out/slots/orig.slot") bytes)"
else
  fail identity "$(cat "$out/cmp.txt"); sizes $(stat -c%s "$out/slots/orig.slot") vs $(stat -c%s "$out/slots/round.slot")"
fi

# 5. a fresh server restores the written file and continues
start b
curl -sf -X POST "localhost:$port/slots/0?action=restore" -H 'content-type: application/json' -d '{"filename":"round.slot"}' > "$out/restore.json" || fail restore "slot restore refused: $(tail -3 "$out/server-b.log")"
curl -sf "localhost:$port/completion" -H 'content-type: application/json' --data @"$out/req32.json" > "$out/restored.json" || fail restored "completion"
stop
python3 - "$out" <<'PY'
import json, sys
o = sys.argv[1]
cold, rest, rs = (json.load(open(f"{o}/{n}.json")) for n in ("cold", "restored", "restore"))
n_restored = rs.get("n_restored")
cache_n = rest["timings"].get("cache_n")
same = cold["tokens"] == rest["tokens"]
first = next((k for k, (x, y) in enumerate(zip(cold["tokens"], rest["tokens"])) if x != y), None)
print(f"restore n_restored={n_restored} cache_n={cache_n} prompt_n={rest['timings'].get('prompt_n')} generated={len(rest['tokens'])}")
# Not a verdict on the tool: round.slot is byte-identical to llama.cpp's own file by step 4, so
# this line measures llama.cpp's restore against its own cold run (control L of
# bench/p1-bridge-protocol.md, amendment 1). It is printed on every run and never hidden.
print(f"restored_equals_cold={same} first_divergence={first}")
if not n_restored or cache_n != n_restored:
    print("FAIL reuse: the restored prefix was not reused"); sys.exit(1)
if len(rest["tokens"]) != 32 or len(cold["tokens"]) != 32:
    print("FAIL generate: the restored server did not produce 32 tokens"); sys.exit(1)
PY
echo "PASS bridge-roundtrip kv=$kv"
