#!/usr/bin/env bash
# usage: bench/turn-api.sh ENGINE PACK OUTDIR [PORT]
#
# QS batch gate (AMDHQ docs/design/latent-os/06-experiments.md "QS batch",
# 2026-09-24), the way a user meets it: one server, an 8k checkpoint, then
#   turn     POST /v1/turn x20 with a fresh 15-token suffix (3 rotating), x20
#            with an 8-token suffix, x20 repeating one suffix: ttft_s medians,
#            tier, ids == /v1/checkpoints/{id}/fork for the same suffix
#   chat     (TOKENIZER set) chat with checkpoint:true -> id; chat with
#            checkpoint_id -> ids == plain chat ids; a different prefix -> 409
#   metrics  GET /metrics parses, fork/turn counts equal the requests made
#   import   GET file + meta from server A; server B (other BARO_CKPT_DIR)
#            imports them (multipart) and forks: ids == A's fork ids
# Self-queues through gpu-wait. PASS list to OUTDIR/receipt.md, exit 1 on any FAIL.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); port=${4:-8240}
fail() { echo "FAIL $1: $2"; exit 1; }
serve=serve/target/release/baro-serve
[ -x "$eng" ] || fail setup "no engine at $eng"; [ -x "$serve" ] || fail setup "no $serve"
mkdir -p "$out"
if [ ! -e "$out/prep-8000/.done" ]; then
  B4_TARGET_TOKENS=8000 B4_PREP_ONLY=1 bench/b4-cross-host.sh "$eng" "$pack" "$out/prep-8000" > "$out/prep-8000.log" 2>&1 || fail prep "$out/prep-8000.log"
  touch "$out/prep-8000/.done"
fi
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority "${TURN_PRIORITY:-20}" --vram 22 --timeout 600 -- env ${TOKENIZER:+TOKENIZER="$TOKENIZER"} ${CHAT_TEMPLATE:+CHAT_TEMPLATE="$CHAT_TEMPLATE"} "$0" "$@"
fi
export LC_ALL=C
echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | head -c 80)' engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port tokenizer=${TOKENIZER:-none} job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"
spid=""
stop_server() {
  [ -n "$spid" ] || return 0
  kill -- -"$spid" 2>/dev/null || kill "$spid" 2>/dev/null || :
  pkill -TERM -P "$spid" 2>/dev/null || :
  wait "$spid" 2>/dev/null || :
  spid=""
  if pgrep -f "^$eng( |$)" > /dev/null; then pkill -KILL -f "^$eng( |$)" || :; sleep 1; fi
}
trap stop_server EXIT
start_server() {  # $1 ckpt dir, $2 port
  setsid env BARO_PACK="$pack" BARO_CKPT_DIR="$1" BARO_SPEC=0 BARO_TMAX=32768 \
    "$serve" --engine "$eng" --pack "$pack" --port "$2" ${TOKENIZER:+--tokenizer "$TOKENIZER"} ${CHAT_TEMPLATE:+--chat-template-file "$CHAT_TEMPLATE"} > "$out/serve-$2.out" 2> "$out/serve-$2.err" &
  spid=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$2/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$2/health" > /dev/null || fail "health $2" "$out/serve-$2.err"
}
rm -rf "$out/ckpts-a" "$out/ckpts-b"; mkdir -p "$out/ckpts-a" "$out/ckpts-b"
start_server "$out/ckpts-a" "$port"
python3 - "$out" "$port" "$pack" <<'PY' || fail turn "$out/receipt.md"
import json, pathlib, re, statistics as st, sys, time, urllib.request, urllib.error, os
out, port, pack = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ids = lambda f: [int(x) for x in open(f).read().split()]
P = ids(out / "prep-8000" / "P.tokens")
QS = [ids(f"bench/mtp-prompts/{n}.tokens") for n in ("p02-python-fib", "p05-math", "p11-email")]
Q8 = QS[0][:8]
F = ids("bench/mtp-prompts/p03-story.tokens")
log, fails = [], 0
def check(cond, label):
    global fails
    log.append(("PASS " if cond else "FAIL ") + label); fails += 0 if cond else 1
    (out / "receipt.md").write_text("# turn-api gate, server A\n\n" + "\n".join(f"- {l}" for l in log) + "\n")
def call(method, path, body=None, raw=None, headers=None, prt=port):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"} if raw is None else (headers or {})
    req = urllib.request.Request(f"http://127.0.0.1:{prt}{path}", data=data, method=method, headers=h)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            b = r.read(); return r.status, (json.loads(b) if r.headers.get("Content-Type", "").startswith("application/json") else b), time.monotonic() - t0, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.monotonic() - t0, {}
flush = lambda: call("POST", "/v1/completions", {"prompt": F, "max_tokens": 4, "spec": False})
s, cr, _, _ = call("POST", "/v1/checkpoints", {"prompt": P, "ttl_s": 900, "max_tokens": 1, "spec": False})
check(s == 200, f"create 8k checkpoint 200 (got {s} {str(cr)[:160]})")
if s != 200:
    print("\n".join(log)); sys.exit(1)
cid = cr["checkpoint"]["id"]
# reference fork ids per suffix
ref = {}
Q8B_ref = QS[1][:8]
for i, q in enumerate(QS + [Q8, Q8B_ref]):
    flush()
    s, fk, _, _ = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": q, "max_tokens": 16, "spec": False, "temperature": 0}]})
    ref[i] = fk["branches"][0]["tokens"] if s == 200 else None
    check(s == 200 and ref[i] is not None, f"fork reference for suffix {i} 200")
rows = []
def turns(kind, plan, mt=16, cached_min=7999, cached_max=7999):
    for j, q in plan:
        s, t, wall, _ = call("POST", "/v1/turn", {"checkpoint": cid, "suffix": q, "max_tokens": mt, "spec": False, "temperature": 0})
        ct = t.get("cached_tokens") if isinstance(t, dict) else None
        ok = s == 200 and ct is not None and cached_min <= ct <= cached_max and t.get("ttft_s") is not None
        rows.append({"kind": kind, "suffix": j, "ok": ok, "status": s, "ttft_s": t.get("ttft_s") if isinstance(t, dict) else None, "wall_s": wall, "tier": t.get("tier") if isinstance(t, dict) else None, "cached_tokens": ct, "tokens": t.get("tokens") if isinstance(t, dict) else None, "timings": t.get("timings") if isinstance(t, dict) else None})
# fresh suffixes, prefix hot: rotate three suffixes so each turn can only hit
# the checkpoint of P (no flush: a flush drops the chain and measures the file
# path instead, the first run's mistake). Q8 alternates with a 15-token
# neighbour for the same reason.
Q8B = QS[1][:8]
for k in range(20):
    turns("fresh15", [(k % 3, QS[k % 3])])
for k in range(20):
    turns("fresh8", [(3 if k % 2 == 0 else 4, Q8 if k % 2 == 0 else Q8B)])
for k in range(20):
    turns("repeat", [(0, QS[0])], cached_max=7999 + len(QS[0]))
for k in range(5):
    flush(); turns("file", [(k % 3, QS[k % 3])])
(out / "results.json").write_text(json.dumps(rows, indent=1))
for kind in ("fresh15", "fresh8", "repeat", "file"):
    r = [x for x in rows if x["kind"] == kind]
    good = [x for x in r if x["ok"]]
    ok = len(good) == len(r) and len(r) > 0
    check(ok, f"{kind}: {len(good)}/{len(r)} turns ok (cached_tokens 7999, ttft present); first bad: {json.dumps([x for x in r if not x['ok']][:1])[:200]}")
    if not good:
        continue
    med = st.median(x["ttft_s"] for x in good)
    check(all(x["tokens"] == ref[x["suffix"]] for x in good), f"{kind}: ids == fork reference for the same suffix")
    log.append(f"INFO {kind}: median ttft_s {med:.4f}, max {max(x['ttft_s'] for x in good):.4f}, tiers {sorted(set(str(x['tier']) for x in good))}")
    if kind == "fresh8": check(med < 0.060, f"fresh8 (hot prefix): median ttft_s {med:.4f} < 0.060 (QS bar)")
    if kind == "repeat": check(med < 0.035, f"repeat: median ttft_s {med:.4f} < 0.035")
    if kind in ("fresh15", "fresh8"): check(all(x["tier"] == "hot" for x in good), f"{kind}: every turn hit the hot tier")
# metrics
s, m, _, _ = call("GET", "/metrics")
mt = m.decode() if isinstance(m, (bytes, bytearray)) else str(m)
forks = sum(int(v) for v in re.findall(r'baro_checkpoint_forks_total\{tier="[a-z]+"\} (\d+)', mt))
turns_n = int(re.search(r"baro_turns_total (\d+)", mt).group(1))
check(s == 200 and re.search(r"baro_turn_ttft_seconds_count (\d+)", mt), "metrics: scrape parses")
check(turns_n == len(rows), f"metrics: baro_turns_total {turns_n} == turns made {len(rows)}")
check(forks == len(rows) + len(ref), f"metrics: forks total {forks} == turns + reference forks {len(rows) + len(ref)}")
(out / "metrics.txt").write_text(mt)
# chat section
tok = os.environ.get("TOKENIZER")
if tok:
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three planets."}]
    s, plain, _, _ = call("POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "spec": False, "temperature": 0})
    check(s == 200, f"chat plain 200 (got {s})")
    s, made, _, _ = call("POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "spec": False, "temperature": 0, "checkpoint": True})
    check(s == 200 and made.get("checkpoint"), "chat checkpoint:true returns a checkpoint")
    check(s == 200 and made["choices"][0]["tokens"] == plain["choices"][0]["tokens"], "chat with checkpoint:true ids == plain ids")
    ck = made["checkpoint"]["id"]
    s, cont, _, _ = call("POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "spec": False, "temperature": 0, "checkpoint_id": ck})
    check(s == 200 and cont["choices"][0]["tokens"] == plain["choices"][0]["tokens"], "chat with checkpoint_id ids == plain ids")
    check(s == 200 and cont.get("checkpoint_from") == ck, "chat with checkpoint_id reports checkpoint_from")
    s, bad, _, _ = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Different prefix entirely."}], "max_tokens": 4, "checkpoint_id": ck})
    check(s == 409, f"chat with checkpoint_id on another prefix -> 409 (got {s})")
else:
    log.append("INFO chat section skipped: TOKENIZER not set")
# export for the import section
s, body, _, hdr = call("GET", f"/v1/checkpoints/{cid}/file")
check(s == 200 and isinstance(body, (bytes, bytearray)) and len(body) > 1000, f"file pull 200 ({len(body) if isinstance(body,(bytes,bytearray)) else 0} bytes)")
(out / "pulled.baro").write_bytes(body); (out / "pulled.meta").write_text(json.dumps(call("GET", f"/v1/checkpoints/{cid}/meta")[1]))
check("tokens" not in json.loads(hdr.get("X-Baro-Meta", hdr.get("x-baro-meta", "{}"))) or not json.loads(hdr.get("X-Baro-Meta", hdr.get("x-baro-meta", "{}")))["tokens"], "X-Baro-Meta carries no token ids (curl 100 KB header cap)")
(out / "pulled.sha").write_text(hdr.get("X-Baro-Sha256", hdr.get("x-baro-sha256", "")))
(out / "ref.json").write_text(json.dumps({"cid": cid, "ref": ref}))
(out / "results.json").write_text(json.dumps(rows, indent=1))
(out / "receipt.md").write_text("# turn-api gate, server A\n\n" + "\n".join(f"- {l}" for l in log) + "\n")
print("\n".join(log)); sys.exit(1 if fails else 0)
PY
stop_server
start_server "$out/ckpts-b" "$((port + 1))"
python3 - "$out" "$((port + 1))" <<'PY' || fail import "$out/receipt.md"
import json, pathlib, sys, urllib.request, urllib.error, uuid, hashlib
out, port = pathlib.Path(sys.argv[1]), sys.argv[2]
log, fails = [], 0
def check(cond, label):
    global fails
    log.append(("PASS " if cond else "FAIL ") + label); fails += 0 if cond else 1
refj = json.loads((out / "ref.json").read_text()); cid = refj["cid"]; ref = {int(k): v for k, v in refj["ref"].items()}
body = (out / "pulled.baro").read_bytes(); meta = (out / "pulled.meta").read_text(); sha = (out / "pulled.sha").read_text().strip()
check(hashlib.sha256(body).hexdigest() == sha, "pulled file sha256 == X-Baro-Sha256")
b = uuid.uuid4().hex.encode()
parts = [(b"meta", b"application/json", meta.encode()), (b"sha256", b"text/plain", sha.encode()), (b"file", b"application/octet-stream", body)]
buf = bytearray()
for name, ctype, data in parts:
    buf += b"--" + b + b"\r\nContent-Disposition: form-data; name=\"" + name + b"\"; filename=\"x\"\r\nContent-Type: " + ctype + b"\r\n\r\n" + data + b"\r\n"
buf += b"--" + b + b"--\r\n"
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/checkpoints/import", data=bytes(buf), method="POST", headers={"Content-Type": f"multipart/form-data; boundary={b.decode()}"})
try:
    with urllib.request.urlopen(req, timeout=600) as r: s, imp = r.status, json.loads(r.read())
except urllib.error.HTTPError as e: s, imp = e.code, json.loads(e.read() or b"{}")
check(s == 200 and imp.get("created") is True and imp["checkpoint"]["id"] == cid, f"import on server B 200 created (got {s} {str(imp)[:120]})")
ids = lambda f: [int(x) for x in open(f).read().split()]
QS = [ids(f"bench/mtp-prompts/{n}.tokens") for n in ("p02-python-fib", "p05-math", "p11-email")]
for i, q in enumerate(QS):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/checkpoints/{cid}/fork", data=json.dumps({"branches": [{"prompt_suffix": q, "max_tokens": 16, "spec": False, "temperature": 0}]}).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: s, fk = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: s, fk = e.code, {}
    br = (fk.get("branches") or [{}])[0]
    check(s == 200 and br.get("cached_tokens") == 7999 and br.get("tokens") == ref[i], f"server B fork suffix {i}: ids == server A's, cached 7999")
with open(out / "receipt.md", "a") as f: f.write("\n# import section, server B\n\n" + "\n".join(f"- {l}" for l in log) + "\n")
print("\n".join(log)); sys.exit(1 if fails else 0)
PY
stop_server
echo "PASS turn-api: receipt=$out/receipt.md"
