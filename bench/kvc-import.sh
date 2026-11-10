#!/usr/bin/env bash
# usage: bench/kvc-import.sh INT8_ENGINE PACK RECON_BAROST03 PAYLOAD SIDECAR_JSON OUT [PORT]
# HTTP import of an own-codec payload (BAROKV4/5/6), the way a second host meets it:
#   server A (ckpts-a)  POST /v1/checkpoints/import RECON (BAROST03, the same bytes the
#                       payload decodes to) -> fork 3 suffixes x 16 greedy ids = reference
#   server B (ckpts-b)  POST /v1/checkpoints/import PAYLOAD with the same meta (bytes = payload
#                       size) -> created; fork the same suffixes: ids == A's, cached_tokens ==
#                       pos; GET /v1/checkpoints/ID/file returns the payload bytes (sha256)
# WIN=1: the client is the Windows 11 VM (windows-mcp at 127.0.0.1:18000, PowerShell tool):
#   it pulls the file from A over the guest NIC (host = 10.0.2.2), A stops, B starts, the VM
#   posts the multipart import with curl.exe, then the fork check runs from here.
# The servers run one at a time (two 32k int8 engines do not fit next to each other).
# Identity: the sidecar's runtime/pack may differ from this build's (409 identity_mismatch names
# only the field); the client reads the server's identity from its startup line and adopts it.
# Self-queues through gpu-wait. Exit 1 on any FAIL.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); pack=$(realpath "$2"); recon=$(realpath "$3"); pay=$(realpath "$4"); side=$(realpath "$5"); out=$(realpath -m "$6"); port=${7:-8260}
fail() { echo "FAIL $1: $2"; exit 1; }
serve=serve/target/release/baro-serve
[ -x "$eng" ] && [ -x "$serve" ] || fail setup "engine or $serve missing"
mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  {
    echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$(sha256sum "$serve" | cut -c1-16) commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ' | head -c 80)'"
    echo "pack=$pack recon=$(sha256sum "$recon" | cut -c1-16) payload=$pay $(sha256sum "$pay" | cut -c1-16) $(stat -c%s "$pay") B sidecar=$side win=${WIN:-0}"
  } > "$out/arm.txt"; cat "$out/arm.txt"
  exec gpu-wait run --priority 20 --vram 22 --timeout 900 -- env WIN="${WIN:-0}" "$0" "$@"
fi
echo "$GPU_WAITING_ROOM_JOB" > "$out/gpu-wait-job.txt"
export LC_ALL=C
spid=""
stop_server() { [ -n "$spid" ] || return 0; kill "$spid" 2>/dev/null || :; wait "$spid" 2>/dev/null || :; spid=""; sleep 2; }
trap 'stop_server' EXIT
start_server() {  # $1 ckpt dir, $2 port
  setsid env BARO_PACK="$pack" BARO_CKPT_DIR="$1" BARO_SPEC=0 BARO_TMAX=32768 \
    "$serve" --engine "$eng" --pack "$pack" --port "$2" > "$out/serve-$2.out" 2> "$out/serve-$2.err" &
  spid=$!
  for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$2/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$2/health" > /dev/null || fail "health $2" "$out/serve-$2.err"
}
cat > "$out/client.py" <<'PY'
import hashlib, json, pathlib, sys, urllib.request, urllib.error, uuid
import http.client
http.client._MAXLINE = 4 << 20   # X-Baro-Meta carries the 32k token ids in one header line (turn-api.sh pulls 8k)
def call(port, method, path, body=None, headers=None, timeout=900):
    data = None if body is None else (body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode())
    h = {"Content-Type": "application/json"} if body is not None and not isinstance(body, (bytes, bytearray)) else {}
    h.update(headers or {})
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(); ct = r.headers.get("Content-Type", "")
            return r.status, (json.loads(raw) if "json" in ct else raw), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try: return e.code, json.loads(raw), dict(e.headers)
        except Exception: return e.code, raw, dict(e.headers)
def multipart(meta, blob, sha):
    b = uuid.uuid4().hex.encode(); buf = bytearray()
    for name, ctype, data in ((b"meta", b"application/json", json.dumps(meta).encode()), (b"sha256", b"text/plain", sha.encode()), (b"file", b"application/octet-stream", blob)):
        buf += b"--" + b + b"\r\nContent-Disposition: form-data; name=\"" + name + b"\"; filename=\"x\"\r\nContent-Type: " + ctype + b"\r\n\r\n" + data + b"\r\n"
    return bytes(buf + b"--" + b + b"--\r\n"), {"Content-Type": f"multipart/form-data; boundary={b.decode()}"}
def server_identity(err_path):
    """The front prints its identity at startup: identity Identity { pack: "..", runtime: "..", tokenizer_sha: None, weights_uuid: Some(".."), portable: true }"""
    import re
    m = re.search(r"identity Identity \{(.*?)\}", pathlib.Path(err_path).read_text())
    if not m: return {}
    out = {}
    for k, v in re.findall(r'(\w+): (Some\("[^"]*"\)|"[^"]*"|None|true|false)', m.group(1)):
        out[k] = None if v == "None" else True if v == "true" else False if v == "false" else v.strip('Some()').strip('"')
    return out
def import_file(port, meta, path, log, err_path=None):
    blob = pathlib.Path(path).read_bytes(); meta = dict(meta); meta["bytes"] = len(blob)
    sha = hashlib.sha256(blob).hexdigest()
    if err_path:
        ours = server_identity(err_path)
        for k in ("pack", "runtime", "tokenizer_sha", "weights_uuid"):
            if k in ours and meta["identity"].get(k) != ours[k]:
                log.append(f"INFO identity {k}: sidecar {meta['identity'].get(k)} -> server {ours[k]}"); meta["identity"][k] = ours[k]
    body, h = multipart(meta, blob, sha)
    s, r, _ = call(port, "POST", "/v1/checkpoints/import", body, h)
    return s, r, sha, meta
def forks(port, cid, n=16):
    ids = lambda f: [int(x) for x in open(f).read().split()]
    out = []
    for name in ("p02-python-fib", "p05-math", "p11-email"):
        s, r, _ = call(port, "POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": ids(f"bench/mtp-prompts/{name}.tokens"), "max_tokens": n, "spec": False, "temperature": 0}]})
        br = (r.get("branches") or [{}])[0] if isinstance(r, dict) else {}
        out.append((s, br.get("cached_tokens"), br.get("tokens")))
    return out
PY
PYRC=0
start_server "$out/ckpts-a" "$port"
python3 - "$out" "$recon" "$side" "$port" <<'PY' || fail A "$out/A.log"
import json, pathlib, sys
sys.path.insert(0, sys.argv[1]); from client import *
out, recon, side, port = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
log = []; meta = json.load(open(side))
s, r, sha, meta = import_file(port, meta, recon, log, out / f"serve-{port}.err")
ok = s == 200 and isinstance(r, dict) and r.get("created") is True
log.append(f"{'PASS' if ok else 'FAIL'} A import RECON (BAROST03) 200 created (got {s} {str(r)[:100]})")
fk = forks(port, meta["id"]) if ok else []
(out / "A.json").write_text(json.dumps({"meta": meta, "forks": fk, "sha": sha}))
(out / "A.log").write_text("\n".join(log) + "\n"); print("\n".join(log))
sys.exit(0 if ok and all(f[0] == 200 and f[1] == meta["pos"] for f in fk) else 1)
PY
win_ps() {  # $1 script; prints the tool's text
  python3 - "$1" <<'PY'
import json, sys, urllib.request
H = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
def rpc(sid, payload):
    h = dict(H); h.update({"mcp-session-id": sid} if sid else {})
    req = urllib.request.Request("http://127.0.0.1:18000/mcp", data=json.dumps(payload).encode(), method="POST", headers=h)
    with urllib.request.urlopen(req, timeout=900) as r:
        sid2 = r.headers.get("mcp-session-id", sid); body = r.read().decode()
    data = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    return sid2, (data[-1] if data else None)
sid, _ = rpc(None, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "kvc-import", "version": "0"}}})
rpc(sid, {"jsonrpc": "2.0", "method": "notifications/initialized"})
_, r = rpc(sid, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "PowerShell", "arguments": {"command": sys.argv[1]}}})
print("".join(c.get("text", "") for c in (r or {}).get("result", {}).get("content", [])))
PY
}
if [ "${WIN:-0}" = 1 ]; then
  cid=$(python3 -c "import json;print(json.load(open('$out/A.json'))['meta']['id'])")
  win_ps "New-Item -ItemType Directory -Force C:\\kvc | Out-Null; Remove-Item C:\\kvc\\pulled.bin -ErrorAction SilentlyContinue; \$t=[Diagnostics.Stopwatch]::StartNew(); \$rc = (Start-Process -FilePath curl.exe -ArgumentList '-s','-S','-o','C:\\kvc\\pulled.bin','-D','C:\\kvc\\pulled.hdr','http://10.0.2.2:$port/v1/checkpoints/$cid/file' -NoNewWindow -Wait -PassThru -RedirectStandardError C:\\kvc\\pulled.err).ExitCode; \$t.Stop(); 'curl_exit=' + \$rc; 'bytes=' + \$(if (Test-Path C:\\kvc\\pulled.bin) { (Get-Item C:\\kvc\\pulled.bin).Length } else { -1 }); 'seconds=' + \$t.Elapsed.TotalSeconds; 'sha=' + \$(if (Test-Path C:\\kvc\\pulled.bin) { (Get-FileHash C:\\kvc\\pulled.bin -Algorithm SHA256).Hash.ToLower() } else { 'none' }); 'stderr=' + (Get-Content C:\\kvc\\pulled.err -Raw -ErrorAction SilentlyContinue); Get-Content C:\\kvc\\pulled.hdr -ErrorAction SilentlyContinue | Select-String 'X-Baro-Sha256|HTTP/'" > "$out/win-pull.txt" 2>&1
  want=$(stat -c%s "$recon"); wsha=$(sha256sum "$recon" | cut -c1-64)
  grep -q "bytes=$want" "$out/win-pull.txt" && grep -q "sha=$wsha" "$out/win-pull.txt" && echo "PASS VM pull from A: $want bytes, sha matches RECON" >> "$out/A.log" || { echo "FAIL VM pull from A: $(grep -E '^(curl_exit|bytes|stderr)=' "$out/win-pull.txt" | tr '\n' ' ')" >> "$out/A.log"; PYRC=1; }
  cat "$out/win-pull.txt"
fi
stop_server
start_server "$out/ckpts-b" "$((port + 1))"
if [ "${WIN:-0}" = 1 ]; then
  python3 - "$out" "$pay" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); A = json.loads((out / "A.json").read_text()); meta = A["meta"]; meta["bytes"] = pathlib.Path(sys.argv[2]).stat().st_size
(out / "meta-b.json").write_text(json.dumps(meta))
PY
  # the VM does not hold the payload: it pulled RECON from A. Ship the payload to the VM the honest way, over HTTP from a static server on the host.
  ( cd "$(dirname "$pay")" && python3 -m http.server "$((port + 7))" --bind 0.0.0.0 > "$out/http-static.log" 2>&1 & echo $! > "$out/http-static.pid" )
  sleep 1
  win_ps "curl.exe -s -o C:\\kvc\\payload.bin 'http://10.0.2.2:$((port + 7))/$(basename "$pay")'; 'bytes=' + (Get-Item C:\\kvc\\payload.bin).Length" > "$out/win-fetch.txt" 2>&1 || :
  grep -q "bytes=$(stat -c%s "$pay")" "$out/win-fetch.txt" && echo "PASS VM fetched the payload ($(stat -c%s "$pay") bytes)" >> "$out/A.log" || { echo "FAIL VM payload fetch: $(cat "$out/win-fetch.txt" | tr '\n' ' ')" >> "$out/A.log"; PYRC=1; }
  kill "$(cat "$out/http-static.pid")" 2>/dev/null || :
  cat "$out/win-fetch.txt"
fi
python3 - "$out" "$pay" "$((port + 1))" "${WIN:-0}" <<'PY' || PYRC=1
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1]); from client import *
out, pay, port, win = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
A = json.loads((out / "A.json").read_text()); meta = A["meta"]; log = []
paysha = hashlib.sha256(pathlib.Path(pay).read_bytes()).hexdigest()
if win:
    pass  # the VM posts below; here only the checks
else:
    s, r, sha, meta = import_file(port, meta, pay, log, out / f"serve-{port}.err")
    log.append(f"{'PASS' if s == 200 and isinstance(r, dict) and r.get('created') is True else 'FAIL'} B import PAYLOAD 200 created (got {s} {str(r)[:100]})")
(out / "meta-b.json").write_text(json.dumps(dict(meta, bytes=pathlib.Path(pay).stat().st_size)))
(out / "B.log").write_text("\n".join(log) + "\n"); print("\n".join(log))
sys.exit(0 if all(l.startswith(("PASS", "INFO")) for l in log) else 1)
PY
if [ "${WIN:-0}" = 1 ]; then
  cp "$out/meta-b.json" "$(dirname "$pay")/meta-b.json"
  ( cd "$(dirname "$pay")" && python3 -m http.server "$((port + 7))" --bind 0.0.0.0 > "$out/http-static2.log" 2>&1 & echo $! > "$out/http-static.pid" ); sleep 1
  win_ps "curl.exe -s -o C:\\kvc\\meta.json 'http://10.0.2.2:$((port + 7))/meta-b.json'; \$t=[Diagnostics.Stopwatch]::StartNew(); curl.exe -s -F 'meta=@C:\\kvc\\meta.json;type=application/json' -F 'file=@C:\\kvc\\payload.bin;type=application/octet-stream' 'http://10.0.2.2:$((port + 1))/v1/checkpoints/import'; \$t.Stop(); ''; \$t.Elapsed.TotalSeconds" > "$out/win-import.txt" 2>&1 || :
  kill "$(cat "$out/http-static.pid")" 2>/dev/null || :; rm -f "$(dirname "$pay")/meta-b.json"
  cat "$out/win-import.txt"
  grep -q '"created":true' "$out/win-import.txt" || { echo "FAIL win import: see $out/win-import.txt"; PYRC=1; }
fi
python3 - "$out" "$pay" "$((port + 1))" <<'PY' || PYRC=1
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1]); from client import *
out, pay, port = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
A = json.loads((out / "A.json").read_text()); meta = A["meta"]; log = []
fk = forks(port, meta["id"])
for i, (a, b) in enumerate(zip(A["forks"], fk)):
    log.append(f"{'PASS' if b[0] == 200 and b[1] == meta['pos'] and b[2] == a[2] else 'FAIL'} B fork suffix {i}: ids == A's (RECON), cached {b[1]} (A {a[2][:4] if a[2] else None} B {b[2][:4] if b[2] else None})")
s, blob, h = call(port, "GET", f"/v1/checkpoints/{meta['id']}/file")
got = hashlib.sha256(blob).hexdigest() if isinstance(blob, (bytes, bytearray)) else ""
log.append(f"{'PASS' if s == 200 and got == hashlib.sha256(pathlib.Path(pay).read_bytes()).hexdigest() else 'FAIL'} B file pull returns the payload bytes (sha {got[:16]}, {len(blob) if isinstance(blob,(bytes,bytearray)) else 0} B)")
with open(out / "B.log", "a") as f: f.write("\n".join(log) + "\n")
print("\n".join(log))
sys.exit(0 if all(l.startswith("PASS") for l in log) else 1)
PY
stop_server
{ echo "# kvc-import ($( [ "${WIN:-0}" = 1 ] && echo 'Windows VM client' || echo 'local client'))"; echo; sed 's/^/- /' "$out/A.log" "$out/B.log"; [ -f "$out/win-pull.txt" ] && { echo; echo '## VM pull (bytes, seconds, sha, headers)'; sed 's/^/    /' "$out/win-pull.txt"; }; [ -f "$out/win-import.txt" ] && { echo '## VM import (response, seconds)'; sed 's/^/    /' "$out/win-import.txt"; }; } > "$out/receipt.md"
[ "$PYRC" = 0 ] && echo "PASS kvc-import: $out/receipt.md" || fail kvc-import "$out/receipt.md"
