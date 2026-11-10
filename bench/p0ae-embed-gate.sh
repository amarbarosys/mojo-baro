#!/usr/bin/env bash
# P0a-e gate: the engine's "embed":true wire line against llama.cpp's last-token embedding.
# usage: bench/p0ae-embed-gate.sh            (re-execs itself under gpu-wait)
#        GATE_DRYRUN=1 bench/p0ae-embed-gate.sh   (CPU only: builds, tokenizes, stops before the GPU)
# Bars, frozen 2026-09-17 before the first run (coordinator):
#   B1 every embed line has H floats and |norm - 1| < 1e-4
#   B2 the same request twice gives the identical line
#   B3 cosine(ours_i, llama_i) >= 0.98 on all 8 prompts (q4 pack, bf16 activations vs llama.cpp Q4_0-pure)
#   B4 retrieval: for every i, argmax_j cosine(ours_i, llama_j) == i
#   B5 the pack's reference prompt with and without "embed":true both reproduce ref-tokens-64.txt
#   B6 a malformed flag ("embed":7) is an error line, not a crash
# Reference arm is an arm: llama-embedding runs on the GPU queue too, its flags are echoed to the log.
set -euo pipefail
cd "$(dirname "$0")/.."
out=.work/p0ae; mkdir -p "$out"
pack=${BARO_PACK:-.work/engine-pack-q4}
model=${LLAMA_MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
lbin=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
engine=${ENGINE:-$out/engine}
die() { echo "FAIL $1: $2 (log $out)"; exit 1; }
[ -x "$engine" ] || die build "no engine at $engine (mojo build serve/engine.mojo -I . -I kernels -o $engine)"
[ -f "$model" ] || die fixture "reference GGUF missing: $model"
[ -f "$pack/ref-tokens-64.txt" ] || die fixture "$pack/ref-tokens-64.txt missing"

cat > "$out/prompts.txt" <<'EOF'
The capital of France is Paris, a city on the Seine.
Paris is the capital city of France.
def quicksort(xs): return xs if len(xs) < 2 else quicksort([x for x in xs[1:] if x < xs[0]]) + [xs[0]] + quicksort([x for x in xs[1:] if x >= xs[0]])
The mitochondrion is the organelle that produces most of the cell's ATP.
Interest rates rose by 25 basis points after the central bank meeting on Thursday.
Wie spät ist es? Ich habe um drei Uhr einen Termin beim Zahnarzt.
SELECT name, COUNT(*) FROM orders WHERE total > 100 GROUP BY name ORDER BY 2 DESC;
The 7900 XTX has 24 GB of memory and 96 compute units.
EOF
: > "$out/ids.txt"
while IFS= read -r line; do
  "$lbin/llama-tokenize" -m "$model" -p "$line" --ids --no-bos --log-disable 2>/dev/null | tr -d '[],' >> "$out/ids.txt" || die tokenize "llama-tokenize failed"
done < "$out/prompts.txt"
[ "$(wc -l < "$out/ids.txt")" -eq 8 ] || die tokenize "expected 8 id lines, got $(wc -l < "$out/ids.txt")"
echo "preflight: 8 prompts tokenized, engine $(sha256sum "$engine" | cut -c1-12), pack $pack, model $(basename "$model")"
if [ "${GATE_DRYRUN:-0}" = 1 ]; then echo "GATE_DRYRUN stop: next step is the GPU"; exit 0; fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --vram 24 --timeout 1500 -- env BARO_PACK="$pack" LLAMA_MODEL="$model" LLAMA_BIN="$lbin" ENGINE="$engine" "$0"
fi

# reference arm: one llama-embedding call per prompt, last-token pooling, L2 normalization
: > "$out/llama.jsonl"
echo "reference flags: --pooling last --embd-normalize 2 -ngl 99 --embd-output-format array" | tee "$out/llama-flags.txt"
while IFS= read -r line; do
  "$lbin/llama-embedding" -m "$model" -p "$line" --pooling last --embd-normalize 2 -ngl 99 \
    --embd-output-format array 2>> "$out/llama.err" | tr -d '\n' >> "$out/llama.jsonl" || die reference "llama-embedding failed, see $out/llama.err"
  echo >> "$out/llama.jsonl"
done < "$out/prompts.txt"

python3 - "$engine" "$pack" "$out" <<'PY'
import json, math, os, subprocess, sys
engine, pack, out = sys.argv[1:4]
env = dict(os.environ, BARO_SERVE="1", BARO_PACK=pack)
p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(f"{out}/engine.err", "w"), env=env, text=True)
def fail(step, why):
    print(f"FAIL {step}: {why} (log {out}/engine.err)"); p.kill(); sys.exit(1)
for line in p.stdout:
    if line.startswith('{"ready"'): print("engine ready:", line.strip()[:120]); break
else: fail("start", "engine exited before ready")
nid = [0]
def req(prompt, n, extra):
    nid[0] += 1
    p.stdin.write(json.dumps(dict(id=nid[0], prompt=prompt, n=n, spec=False, **extra)) + "\n"); p.stdin.flush()
    toks, emb, raw = [], None, None
    for line in p.stdout:
        if not line.startswith("{"): continue
        m = json.loads(line)
        if m.get("id") != nid[0]: continue
        if "embed" in m: emb, raw = m["embed"], line
        elif "error" in m: return None, None, line
        elif m.get("done"): return toks, emb, raw
        elif "tok" in m: toks.append(m["tok"])
    fail("request", "engine closed stdout")
ids = [[int(x) for x in l.split()] for l in open(f"{out}/ids.txt")]
ref = [json.loads(l) for l in open(f"{out}/llama.jsonl")]
ref = [r[0] if isinstance(r[0], list) else r for r in ref]
ours, bad = [], []
for i, pr in enumerate(ids):
    _, e, raw = req(pr, 1, dict(embed=True))
    if e is None: fail("B1", f"prompt {i}: no embed line")
    _, e2, raw2 = req(pr, 1, dict(embed=True))
    H = len(e); nrm = math.sqrt(sum(x * x for x in e))
    if H != len(ref[i]): bad.append(f"B1 prompt {i}: H {H} vs llama {len(ref[i])}")
    if abs(nrm - 1) >= 1e-4: bad.append(f"B1 prompt {i}: norm {nrm}")
    if raw != raw2.replace(f'"id":{nid[0]}', f'"id":{nid[0]-1}'): bad.append(f"B2 prompt {i}: two runs differ")
    ours.append(e)
json.dump(ours, open(f"{out}/ours.json", "w"))
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
table = [[cos(o, r) for r in ref] for o in ours]
for i, row in enumerate(table):
    print(f"prompt {i}: cos(ours, llama) {row[i]:.4f}  best other {max(v for j, v in enumerate(row) if j != i):.4f}")
    if row[i] < 0.98: bad.append(f"B3 prompt {i}: cosine {row[i]:.4f} < 0.98")
    if max(range(len(row)), key=row.__getitem__) != i: bad.append(f"B4 prompt {i}: nearest llama vector is {max(range(len(row)), key=row.__getitem__)}")
want = [int(x) for x in open(f"{pack}/ref-tokens-64.txt").read().split()]
pt = [int(x) for x in open(f"{pack}/prompt-tokens.txt").read().split()]
for extra, name in ((dict(), "plain"), (dict(embed=True), "embed")):
    t, e, _ = req(pt, 64, extra)
    if t != want[:64]: bad.append(f"B5 {name}: tokens differ from ref-tokens-64 at {next((k for k,(a,b) in enumerate(zip(t or [], want)) if a != b), 'length')}")
    if name == "plain" and e is not None: bad.append("B5 plain: an embed line was printed without the flag")
t, e, err = req(pt[:4], 1, dict(embed=7))
if err is None: bad.append("B6: embed:7 was accepted")
p.stdin.close(); p.wait(timeout=60)
json.dump(dict(cos=[table[i][i] for i in range(len(table))], fails=bad), open(f"{out}/result.json", "w"))
if bad:
    print(f"FAIL {len(bad)} check(s):"); [print("  " + b) for b in bad]; sys.exit(1)
print(f"PASS P0a-e: 8/8 prompts, min cosine {min(table[i][i] for i in range(8)):.4f}, identity with and without the flag, malformed flag refused")
PY
