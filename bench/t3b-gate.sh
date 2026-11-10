#!/usr/bin/env bash
# T3b gate (AMDHQ docs/design/latent-os/06-experiments.md `## T3b`): two baro-serve engines on one
# GPU, A holds an 8k checkpoint, B forks it 5 times over HIP IPC (front relays 320 B + ack).
# Gates: ids == B's cold P+Q reference 5/5; sha of the moved region equal (A before export and
# after the last, B after every fork); a tampered pack and a tampered prefix_hash header -> 409
# with B's open count unchanged; A's live stages 0 at the end. Bar: median IPC TTFT < 0.100 s
# AND < median file-tier TTFT (B /v1/turn after a flush, same checkpoint, same job).
# Step 2 (T3B_WEIGHTS=shared, amendment "T3b step 2"): A exports its weight buffer, B maps it
# (no pack.bin read); adds: a tampered weight record refused at engine start, B's cold refs == the
# separate-weights run's (T3B_REFS=results.json), pair peak above the pre-launch base < 12.2e9 B
# (run 2 of 2a; run 1's absolute 13.0e9 bar moved with the desktop's base), device VRAM back to base
# within 0.3e9 B after both exit. Step 2b (T3B_EMBD=host): both engines keep token_embd in host
# memory (BARO_EMBD_HOST=1); read-back of that line, bar T3B_VRAM_BAR (default 12.2e9) above base.
# Run under gpu-wait as a script with literal paths: the queue passes no env and no stdin.
# usage: bench/t3b-gate.sh ENGINE PACK OUTDIR [PORT]
set -euo pipefail
cd "$(dirname "$0")/.."

eng=$(realpath "${1:?usage: bench/t3b-gate.sh ENGINE PACK OUTDIR [PORT]}")
pack=$(realpath "${2:?usage: bench/t3b-gate.sh ENGINE PACK OUTDIR [PORT]}")
out=$(realpath -m "${3:?usage: bench/t3b-gate.sh ENGINE PACK OUTDIR [PORT]}")
port=${4:-8260}
tmax=${T3B_TMAX:-9216}
weights=${T3B_WEIGHTS:-own}
refs_from=${T3B_REFS:-}
embd=${T3B_EMBD:-device}
vram_bar=${T3B_VRAM_BAR:-12.2e9}
shim=$(realpath -m .work/shim-build/libamarbaro_shim.so)
serve=serve/target/release/baro-serve
mkdir -p "$out/ckpts"
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$eng" ] || fail setup "no engine at $eng"
[ -x "$serve" ] || fail setup "no $serve"
[ -f "$shim" ] || fail setup "no shim at $shim (cmake -S shim -B .work/shim-build && cmake --build .work/shim-build)"

echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain -- serve bench | head -c 80)' engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve_sha=$(sha256sum "$serve" | cut -c1-16) shim=$shim shim_sha=$(sha256sum "$shim" | cut -c1-16) pack=$pack BARO_TMAX=$tmax weights=$weights embd=$embd vram_bar=$vram_bar refs_from=$refs_from job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"

B4_TARGET_TOKENS=8000 B4_PREP_ONLY=1 bench/b4-cross-host.sh "$eng" "$pack" "$out/prep" > "$out/prep.log" 2>&1 || fail prep "$out/prep.log"

[ "${GATE_DRYRUN:-0}" = 1 ] && { echo "VOID dry run: stop before the engine launch ($eng)"; exit 3; }

vram_used() {
  for d in /sys/class/drm/card[0-9]/device; do
    echo "$(< "$d/mem_info_vram_total") $(< "$d/mem_info_vram_used")"
  done | sort -n | tail -1 | cut -d' ' -f2
}
pids=()
stop_all() {
  for p in "${pids[@]}"; do kill -- -"$p" 2> /dev/null || kill "$p" 2> /dev/null || true; done
  wait 2> /dev/null || true
}
trap stop_all EXIT
echo "phase=base used=$(vram_used)" > "$out/vram.txt"
( while :; do echo "t=$(date +%s.%N) used=$(vram_used)" >> "$out/vram.txt"; sleep 0.5; done ) &
pids+=($!)

rec=$out/weights.rec
start() {  # $1 name, $2 port; both fronts share one checkpoint dir (B finds A's file for the file tier)
  local w=()
  [ "$embd" = host ] && w+=(BARO_EMBD_HOST=1)
  [ "$weights" = shared ] && { [ "$1" = a ] && w+=(BARO_WEIGHTS_EXPORT="$rec") || w+=(BARO_WEIGHTS_IMPORT="$rec"); }
  setsid env "${w[@]}" BARO_PACK="$pack" BARO_CKPT_DIR="$out/ckpts" BARO_SPEC=0 BARO_TMAX="$tmax" BARO_SHIM_LIB="$shim" \
    "$serve" --engine "$eng" --pack "$pack" --port "$2" > "$out/serve-$1.out" 2> "$out/serve-$1.err" &
  pids+=($!)
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$2/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$2/health" > "$out/health-$1.json" || fail "health $1" "$out/serve-$1.err"
}
start a "$port"
start b $((port + 1))
for n in a b; do
  echo "$n: $(grep -m1 -o 'kv dtype:.*seq_cap: [0-9]*' "$out/serve-$n.err") health=$(cat "$out/health-$n.json" | head -c 300)" >> "$out/arm.txt"
done
grep -q "seq_cap: 1$" <(grep -m1 -o 'seq_cap: [0-9]*' "$out/serve-a.err") || fail readback "engine A seq_cap is not 1 ($out/serve-a.err)"
grep -q "seq_cap: 1$" <(grep -m1 -o 'seq_cap: [0-9]*' "$out/serve-b.err") || fail readback "engine B seq_cap is not 1 ($out/serve-b.err)"
if [ "$weights" = shared ]; then
  grep -q "weights exported over IPC: [0-9]* bytes" "$out/serve-a.err" || fail readback "engine A exported no weights ($out/serve-a.err)"
  grep -q "weights imported over IPC: [0-9]* bytes" "$out/serve-b.err" || fail readback "engine B imported no weights ($out/serve-b.err)"
  grep -q "loading pack:" "$out/serve-b.err" && fail readback "engine B read pack.bin ($out/serve-b.err)"
  if [ "$embd" = host ]; then
    for n in a b; do grep -q "token_embd in host memory: [0-9]* bytes" "$out/serve-$n.err" || fail readback "engine $n kept token_embd in VRAM ($out/serve-$n.err)"; done
    grep -h "token_embd in host memory" "$out/serve-a.err" "$out/serve-b.err" >> "$out/arm.txt"
  fi
  grep -h "weights .*ported over IPC\|pack loaded in" "$out/serve-a.err" "$out/serve-b.err" >> "$out/arm.txt"
  # a record for another pack: the engine must refuse at start, before any HIP IPC open
  python3 -c "import sys; b = bytearray(open(sys.argv[1], 'rb').read()); b[16] ^= 1; open(sys.argv[2], 'wb').write(b)" "$rec" "$out/weights-tampered.rec"
  trc=0; timeout 120 env $([ "$embd" = host ] && echo BARO_EMBD_HOST=1) BARO_WEIGHTS_IMPORT="$out/weights-tampered.rec" BARO_PACK="$pack" BARO_TMAX="$tmax" BARO_SHIM_LIB="$shim" "$eng" < /dev/null > "$out/engine-c.out" 2>&1 || trc=$?
  grep -q "weights import refused: exported from a different pack" "$out/engine-c.out" && [ "$trc" != 0 ] \
    || fail tamper "engine with a tampered weight record did not refuse (exit $trc, $out/engine-c.out)"
  echo "tampered weight record: refused, exit $trc" >> "$out/arm.txt"
fi

rc=0
python3 - "$out" "$port" "$out/prep/P.tokens" "$weights" "$refs_from" "$vram_bar" <<'PY' || rc=$?
import glob, json, pathlib, statistics, sys, time, urllib.request, urllib.error
out, port, P = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
weights, refs_from, vram_bar = sys.argv[4], sys.argv[5], float(sys.argv[6])
A, B = f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port + 1}"
ids = lambda f: [int(x) for x in open(f).read().split()]
p = ids(P)
prompts = sorted(glob.glob("bench/mtp-prompts/p*.tokens"))
QS = [ids(f)[:8] for f in prompts[:5]]
F = ids(prompts[5])
assert all(len(q) == 8 for q in QS) and len({tuple(q) for q in QS}) == 5, "five distinct 8-token suffixes"
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}
def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw), time.perf_counter() - t
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")}, time.perf_counter() - t
log, fails = [], 0
def check(cond, label):
    global fails
    log.append(("PASS " if cond else "FAIL ") + label); fails += 0 if cond else 1
    print(log[-1], flush=True)
def vram_used():
    best = (-1, -1)
    for d in glob.glob("/sys/class/drm/card[0-9]/device"):
        try:
            best = max(best, (int(open(d + "/mem_info_vram_total").read()), int(open(d + "/mem_info_vram_used").read())))
        except OSError:
            pass
    return best[1]
flush = lambda base: call(base, "POST", "/v1/completions", {"prompt": F, "max_tokens": 4, "spec": False})

# cold references on B, each after a flush so no checkpoint of an earlier ref is reused
refs = []
for i, q in enumerate(QS):
    flush(B)
    s, r, _ = call(B, "POST", "/v1/completions", {"prompt": p + q, **greedy})
    check(s == 200, f"cold ref {i} 200 (got {s})")
    refs.append(r["choices"][0]["tokens"] if s == 200 else None)
if refs_from:
    want = json.load(open(refs_from))["refs"]
    check(refs == want, f"cold refs == {refs_from} ({sum(a == b for a, b in zip(refs, want))}/5 equal)")

s, cr, _ = call(A, "POST", "/v1/checkpoints", {"prompt": p, "ttl_s": 900, "max_tokens": 1, "spec": False})
ck = cr.get("checkpoint", {}); cid, pos = ck.get("id", ""), ck.get("pos")
check(s == 200 and pos == len(p) - 1, f"A created checkpoint {cid} at pos {pos}")
s, sa0, _ = call(A, "POST", "/v1/ipc/sha", {"pos": pos})
check(s == 200 and len(sa0.get("sha", "")) == 64, f"A sha before the first export ({sa0.get('sha_s')} s)")

rows = []
for i, q in enumerate(QS):
    s, fk, wall = call(A, "POST", "/v1/ipc/fork", {"checkpoint": cid, "target": f"127.0.0.1:{port + 1}", "suffix": q, **greedy})
    ok = s == 200
    check(ok, f"ipc fork {i} 200 (got {s}: {json.dumps(fk)[:240]})")
    if not ok:
        rows.append({"kind": "ipc", "i": i, "status": s, "body": fk}); continue
    ph = fk["ipc"]["prefix_hash"]
    s2, sb, _ = call(B, "POST", "/v1/ipc/sha", {"pos": pos, "prefix_hash": ph})
    check(fk.get("tokens") == refs[i], f"ipc fork {i}: ids == cold ref ({len(fk.get('tokens') or [])} ids, cached {fk.get('cached_tokens')})")
    check(fk.get("cached_tokens") == pos, f"ipc fork {i}: B restored the ingested checkpoint (cached {fk.get('cached_tokens')} == {pos})")
    check(s2 == 200 and sb.get("sha") == sa0.get("sha"), f"ipc fork {i}: B sha after ingest == A sha")
    check(fk["ipc"].get("live_after_release") == 0, f"ipc fork {i}: A freed its stage (live {fk['ipc'].get('live_after_release')})")
    if i == 0:
        used_after_first = vram_used()
    rows.append({"kind": "ipc", "i": i, "ttft_s": fk.get("ttft_s"), "wall_s": fk.get("wall_s"), "client_wall_s": wall,
                 "ingest_s": fk.get("ingest_s"), "b_ttft_s": fk.get("b_ttft_s"), "ipc": fk.get("ipc"), "sha_b": sb.get("sha"), "tokens": fk.get("tokens")})
s, sa1, _ = call(A, "POST", "/v1/ipc/sha", {"pos": pos})
check(s == 200 and sa1.get("sha") == sa0.get("sha"), "A sha after the last export == before the first")

s, st0, _ = call(B, "GET", "/v1/ipc/stats")
opens0 = st0.get("opens")
for what in ("pack", "prefix_hash"):
    s, rj, _ = call(A, "POST", "/v1/ipc/fork", {"checkpoint": cid, "target": f"127.0.0.1:{port + 1}", "suffix": QS[0], "tamper": what, **greedy})
    check(s == 409, f"tampered {what}: 409 (got {s}: {json.dumps(rj)[:200]})")
    rows.append({"kind": "409", "what": what, "status": s, "body": rj})
s, st1, _ = call(B, "GET", "/v1/ipc/stats")
check(st1.get("opens") == opens0, f"B opened no handle for the refused headers (opens {opens0} -> {st1.get('opens')})")
s, sta, _ = call(A, "GET", "/v1/ipc/stats")
check(s == 200 and sta.get("live") == 0, f"A live stages 0 after every ack (live {sta.get('live')})")
check(sta.get("stages") == 1, f"A reused one export stage for all 7 exports (stages {sta.get('stages')})")
grow = vram_used() - used_after_first
check(grow < 0.2e9, f"device VRAM grew {grow / 1e9:.3f} GB over 6 exports after the first (bar < 0.2; one stage is 0.58)")

for i, q in enumerate(QS):
    flush(B)
    s, t, wall = call(B, "POST", "/v1/turn", {"checkpoint": cid, "suffix": q, **greedy})
    check(s == 200 and t.get("tier") == "file", f"file fork {i}: 200, tier {t.get('tier')}")
    check(t.get("tokens") == refs[i], f"file fork {i}: ids == cold ref")
    rows.append({"kind": "file", "i": i, "ttft_s": t.get("ttft_s"), "wall_s": t.get("wall_s"), "client_wall_s": wall,
                 "load_s": (t.get("timings") or {}).get("load_s"), "tokens": t.get("tokens")})

ipc_t = [r["ttft_s"] for r in rows if r["kind"] == "ipc" and r.get("ttft_s") is not None]
file_t = [r["ttft_s"] for r in rows if r["kind"] == "file" and r.get("ttft_s") is not None]
mi = statistics.median(ipc_t) if len(ipc_t) == 5 else None
mf = statistics.median(file_t) if len(file_t) == 5 else None
print(f"INFO ipc ttft_s {[round(x, 4) for x in ipc_t]} median {mi}")
print(f"INFO file ttft_s {[round(x, 4) for x in file_t]} median {mf}")
print(f"INFO ingest_s {[r.get('ingest_s') for r in rows if r['kind'] == 'ipc']} bytes {[(r.get('ipc') or {}).get('bytes') for r in rows if r['kind'] == 'ipc']}")
check(mi is not None and mi < 0.100, f"bar: median IPC TTFT {mi} < 0.100 s")
check(mi is not None and mf is not None and mi < mf, f"bar: median IPC TTFT {mi} < file tier {mf}")
if weights == "shared":
    used = [int(l.split("used=")[1]) for l in open(out / "vram.txt") if "used=" in l and l.split("used=")[1].strip().isdigit()]
    base, peak = used[0], max(used)
    check(peak - base < vram_bar, f"bar: pair peak {peak / 1e9:.2f} GB - base {base / 1e9:.2f} = {(peak - base) / 1e9:.2f} GB < {vram_bar / 1e9:.2f}")
(out / "results.json").write_text(json.dumps({"rows": rows, "refs": refs, "sha_a": [sa0, sa1], "stats_b": [st0, st1], "stats_a": sta, "ckpt": ck}, indent=1))
(out / "gate.txt").write_text("\n".join(log) + f"\n{'PASS' if fails == 0 else 'FAIL'} t3b-gate: {len(log) - fails}/{len(log)}\n")
print(f"{'PASS' if fails == 0 else 'FAIL'} t3b-gate: {len(log) - fails}/{len(log)}")
sys.exit(0 if fails == 0 else 1)
PY
echo "phase=end used=$(vram_used) peak=$(grep -o 'used=[0-9]*' "$out/vram.txt" | cut -d= -f2 | sort -n | tail -1)" | tee -a "$out/arm.txt"
for n in a b; do grep -h "ipc ingest:\|state loaded:\|mega fail word: [1-9]" "$out/serve-$n.err" > "$out/engine-$n.grep" || true; done
grep -q "NOT-RESIDENT" "$out/serve-a.err" "$out/serve-b.err" && fail failword "a grid barrier timed out (serve-*.err)"
if [ "$weights" = shared ]; then
  stop_all; trap - EXIT
  base=$(grep -m1 -o 'phase=base used=[0-9]*' "$out/vram.txt" | cut -d= -f3)
  for _ in $(seq 1 20); do now=$(vram_used); [ $((now - base)) -lt 300000000 ] && break; sleep 0.5; done
  echo "phase=exit used=$now base=$base" | tee -a "$out/arm.txt"
  [ $((now - base)) -lt 300000000 ] || { echo "FAIL exit-vram: device VRAM $now after both engines exited, base $base (exported weights not reclaimed?)"; rc=1; }
fi
[ "$rc" = 0 ] || fail gate "$out/gate.txt"
echo "PASS t3b-gate: $out/gate.txt"
