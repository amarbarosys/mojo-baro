#!/usr/bin/env bash
# Gate for tools/gguf-check.py: an honest rebake passes; one flipped tensor byte,
# a tampered in-file ref, and a missing data hash each fail or say NOT CHECKED.
# CPU only. usage: tools/test-gguf-check.sh [SMALL-BARO.gguf]
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M-BARO-3c6c25c.gguf}
d=.work/test-gguf-check; rm -rf "$d"; mkdir -p "$d"
py=./.venv/bin/python3; [ -x "$py" ] || py=python3
die() { echo "FAIL $1: $2"; exit 1; }
# rebake: every baro.* key kept, data hash and challenge added
$py - "$src" "$d/ok.gguf" <<'PY'
import sys
from importlib.util import module_from_spec, spec_from_file_location
def load(n):
    s = spec_from_file_location(n.replace("-", "_"), f"tools/{n}.py"); m = module_from_spec(s); s.loader.exec_module(m); return m
e, ge = load("gguf-embed"), load("gguf-extract")
_, _, _, kv = ge.parse(sys.argv[1])
keep = [(k, v) for k, v in kv.items() if k.startswith(("baro.kernel.", "baro.run.", "baro.hw."))]
e.rewrite(sys.argv[1], sys.argv[2], keep); e.write_challenge(sys.argv[2])
PY
tools/gguf-check.py "$d/ok.gguf" > "$d/ok.log" || { cat "$d/ok.log"; die honest "rebake did not pass"; }
grep -q '^PASS tensor-bytes' "$d/ok.log" && grep -q '^PASS challenge' "$d/ok.log" || die honest "claims missing"
# 1. one tensor byte flipped (last byte of the file, inside the data region)
cp --reflink=auto "$d/ok.gguf" "$d/flip.gguf"; cp "$d/ok.gguf.challenge.json" "$d/flip.gguf.challenge.json"
$py -c "import sys; f=open(sys.argv[1],'r+b'); f.seek(-1,2); b=f.read(1); f.seek(-1,2); f.write(bytes([b[0]^1]))" "$d/flip.gguf"
tools/gguf-check.py "$d/flip.gguf" > "$d/flip.log" && die flip "flipped byte passed"
grep -q '^FAIL tensor-bytes' "$d/flip.log" || die flip "tensor-bytes not named"
# 2. in-file ref rewritten to match a cheat: challenge outside the file catches it
$py - "$d/ok.gguf" "$d/ref.gguf" <<'PY'
import sys
from importlib.util import module_from_spec, spec_from_file_location
def load(n):
    s = spec_from_file_location(n.replace("-", "_"), f"tools/{n}.py"); m = module_from_spec(s); s.loader.exec_module(m); return m
e, ge = load("gguf-embed"), load("gguf-extract")
_, _, _, kv = ge.parse(sys.argv[1])
keep = [(k, ("1 " + v) if k == "baro.run.ref.tokens" else v) for k, v in kv.items() if k.startswith(("baro.kernel.", "baro.run.", "baro.hw."))]
assert any(k == "baro.run.ref.tokens" for k, _ in keep), "source has no baro.run.ref.tokens"
e.rewrite(sys.argv[1], sys.argv[2], keep)
PY
cp "$d/ok.gguf.challenge.json" "$d/ref.gguf.challenge.json"
tools/gguf-check.py "$d/ref.gguf" > "$d/ref.log" && die ref "tampered ref passed"
grep -q '^FAIL challenge.*baro.run.ref.tokens' "$d/ref.log" || die ref "ref key not named"
# 3. an old bake (no data hash, no challenge) is NOT CHECKED, never PASS
tools/gguf-check.py "$src" > "$d/old.log" || true
grep -q '^NOT CHECKED tensor-bytes' "$d/old.log" || die old "missing hash not reported"
grep -q '^PASS' "$d/old.log" && die old "old bake printed a PASS"
echo "PASS test-gguf-check: honest pass, flip FAIL, ref-tamper FAIL, old NOT CHECKED"
rm -rf "$d"
