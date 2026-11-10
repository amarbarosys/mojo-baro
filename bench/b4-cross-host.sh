#!/usr/bin/env bash
# usage: bench/b4-cross-host.sh ENGINE PACK OUTDIR [PORT]
#
# B4-mini: payload vs recipe on a shaped link (docs/design/latent-os/06-experiments.md
# section "B4-mini", AMDHQ repo). Veth pair + netns stands in for A3's two-VM rig.
# Engine stays on the host, one GPU; every HTTP call in both arms crosses the veth
# link from a peer network namespace back to baro-serve in the root namespace.
#
#   GPU phase (server up): flush -> create checkpoint (P) once; for each of 3
#     items (distinct Q suffix): flush -> fork (Q) -> ids, fork_s; flush -> cold
#     /v1/completions (P+Q) -> ids, recipe_s. Then the server stops.
#   link phase (no GPU): for each profile in unshaped 100mbit 1gbit 10gbit:
#     tc qdisc replace dev veth-a root tbf <profile>; move the .baro file across
#     the shaped link via nc, sha256 both ends; payload_s = transfer + fork_s
#   compare payload ids == recipe ids per item; report breakeven
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; port=${4:-8099}
mkdir -p "$out" "$out/ckpts" "$out/xfer"
target_tokens=${B4_TARGET_TOKENS:-1000}
case "$target_tokens" in
  ''|*[!0-9]*) echo "FAIL setup: B4_TARGET_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
[ "$target_tokens" -gt 0 ] || { echo "FAIL setup: B4_TARGET_TOKENS must be positive" >&2; exit 2; }
serve=serve/target/release/baro-serve
[ -x "$eng" ] || { echo "no engine at $eng" >&2; exit 2; }
[ -x "$serve" ] || { echo "no $serve; cargo build --release --manifest-path serve/Cargo.toml" >&2; exit 2; }

NS=b4peer
VA=veth-b4a
VB=veth-b4b
HOST_IP=10.99.7.1
PEER_IP=10.99.7.2
XFER_PORT=9911

P="$out/P.tokens"
python3 - "$P" "$target_tokens" <<'PY'
import glob, pathlib, sys
out, target = pathlib.Path(sys.argv[1]), int(sys.argv[2])
toks = []
for f in sorted(glob.glob("bench/mtp-prompts/p*.tokens")):
    toks += [int(x) for x in open(f).read().split()]
if not toks:
    raise SystemExit("FAIL setup: tokenizer corpus is empty")
rep = (toks * ((target + len(toks) - 1) // len(toks)))[:target]
out.write_text(" ".join(str(x) for x in rep) + "\n")
actual = len([int(x) for x in out.read_text().split()])
if actual != target:
    raise SystemExit(f"FAIL setup: tokenizer id count {actual} != target {target}")
print(f"P tokenizer ids: {actual} target: {target}")
PY
if [ "${B4_PREP_ONLY:-0}" = 1 ]; then
  exit 0
fi

cleanup() {
  [ -n "${spid:-}" ] && pkill -P "$spid" 2>/dev/null
  [ -n "${spid:-}" ] && kill "$spid" 2>/dev/null
  pkill -f "baro-serve --engine .* --port $port" 2>/dev/null
  sudo firewall-cmd --zone=trusted --remove-interface="$VA" 2>/dev/null
  sudo ip netns exec "$NS" true 2>/dev/null && sudo ip netns del "$NS" 2>/dev/null
  sudo ip link del "$VA" 2>/dev/null
  true
}
trap cleanup EXIT

sudo ip netns del "$NS" 2>/dev/null; sudo ip link del "$VA" 2>/dev/null
sudo ip netns add "$NS"
sudo ip link add "$VA" type veth peer name "$VB"
sudo ip link set "$VB" netns "$NS"
sudo ip addr add "$HOST_IP/24" dev "$VA"
sudo ip link set "$VA" up
# firewalld's default zone drops unsolicited inbound on veth-b4a (verified: TCP
# "no route to host" while ICMP passes); runtime-only, reverted in cleanup.
sudo firewall-cmd --zone=trusted --add-interface="$VA"
sudo ip netns exec "$NS" ip addr add "$PEER_IP/24" dev "$VB"
sudo ip netns exec "$NS" ip link set "$VB" up
sudo ip netns exec "$NS" ip link set lo up
peer() { sudo ip netns exec "$NS" "$@"; }

Q1=bench/mtp-prompts/p02-python-fib.tokens
Q2=bench/mtp-prompts/p05-math.tokens
Q3=bench/mtp-prompts/p11-email.tokens
FLUSH=bench/mtp-prompts/p03-story.tokens
for f in "$P" "$Q1" "$Q2" "$Q3" "$FLUSH"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
{
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
  p_tokens=$(python3 -c 'print(len([int(x) for x in open("'$P'").read().split()]))')
  echo "P=$P ($p_tokens tokenizer ids, target=$target_tokens) Q1=$Q1 Q2=$Q2 Q3=$Q3 flush=$FLUSH"
  echo "rig: netns=$NS host=$HOST_IP peer=$PEER_IP veth=$VA/$VB"
} | tee "$out/arm.txt"

# BARO_TMAX / BARO_STATE_INT8 are forwarded only when the caller set them (B4-long).
gpu-wait run --priority 20 --vram 22 --timeout "${B4_GPU_TIMEOUT:-1800}" -- \
  env BARO_PACK="$pack" BARO_CKPT_DIR="$out/ckpts" BARO_SPEC=0 \
  ${BARO_TMAX:+BARO_TMAX=$BARO_TMAX} ${BARO_STATE_INT8:+BARO_STATE_INT8=$BARO_STATE_INT8} "$serve" --engine "$eng" --pack "$pack" --port "$port" --host 0.0.0.0 \
  > "$out/serve.out" 2>"$out/serve.err" &
spid=$!
ready=0
for _ in $(seq 1 360); do
  peer curl -sf "http://$HOST_IP:$port/health" -o "$out/health.json" 2>/dev/null && { ready=1; break; }
  sleep 5
done
[ "$ready" = 1 ] || { echo "VOID: no /health on $HOST_IP:$port from peer ns; see $out/serve.err" >&2; exit 3; }
echo "health: $(cat "$out/health.json")" | tee -a "$out/arm.txt"

python3 - "$out" "$HOST_IP" "$port" "$P" "$Q1" "$Q2" "$Q3" "$FLUSH" <<'PY'
import json, pathlib, sys, urllib.request, urllib.error
out, host, port, P, Q1, Q2, Q3, F = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], *sys.argv[4:9]
ids = lambda f: [int(x) for x in open(f).read().split()]
p = ids(P); items = [("q1", Q1, ids(Q1)), ("q2", Q2, ids(Q2)), ("q3", Q3, ids(Q3))]
fl = ids(F)
base = f"http://{host}:{port}"
def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}
def flush():
    call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False})
import time
# One checkpoint of P (every item shares it), then per item: fork (payload
# side, restore + suffix) and cold P+Q (recipe side). Both are link-independent
# (a few KB of ids; B4-mini recipe medians 0.572-0.577 s across profiles), so
# they run once, and the shaped profiles below only time the file transfer,
# after the server has released the GPU.
flush()
s, cr = call("POST", "/v1/checkpoints", {"prompt": p, "ttl_s": 600, "max_tokens": 1, "spec": False})
ck = cr.get("checkpoint", {}); cid = ck.get("id", "")
print(f"CREATE: {s} id={cid} bytes={ck.get('bytes')}", flush=True)
measured = []
for name, qfile, q in items:
    flush()
    t0 = time.monotonic()
    s1, fk = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": q, **greedy}]})
    fork_s = time.monotonic() - t0
    br = (fk.get("branches") or [{}])[0]
    flush()
    t0 = time.monotonic()
    s2, rc = call("POST", "/v1/completions", {"prompt": p + q, **greedy})
    recipe_s = time.monotonic() - t0
    recipe_ids = rc["choices"][0]["tokens"] if s2 == 200 else None
    measured.append({"item": name, "fork_ok": s1 == 200, "recipe_ok": s2 == 200, "fork_s": fork_s,
                     "recipe_s": recipe_s, "ids_match": br.get("tokens") == recipe_ids,
                     "restored": br.get("restored"), "cached_tokens": br.get("cached_tokens")})
    print(f"{name}: fork_s={fork_s:.3f} recipe_s={recipe_s:.3f} ids_match={measured[-1]['ids_match']}", flush=True)
(out / "measured.json").write_text(json.dumps(measured, indent=2))
PY

commonfile=$(ls -t "$out"/ckpts/*.baro 2>/dev/null | head -1)
[ -n "$commonfile" ] || { echo "VOID: no checkpoint file created" >&2; exit 3; }
srcsha=$(sha256sum "$commonfile" | cut -d' ' -f1)
echo "checkpoint file: $commonfile ($(stat -c%s "$commonfile") bytes) sha256=$srcsha" | tee -a "$out/arm.txt"
# GPU work is done: stop the server so the transfers below do not hold the card.
pkill -P "$spid" 2>/dev/null; kill "$spid" 2>/dev/null
pkill -f "baro-serve --engine .* --port $port" 2>/dev/null; wait "$spid" 2>/dev/null; spid=
echo "[]" > "$out/results.json"

run_profile() {
  prof=$1; tcspec=$2
  echo "=== profile $prof ===" | tee -a "$out/arm.txt"
  if [ -n "$tcspec" ]; then
    sudo tc qdisc replace dev "$VA" root $tcspec
  else
    sudo tc qdisc del dev "$VA" root 2>/dev/null || true
  fi
  tc qdisc show dev "$VA" | tee -a "$out/$prof.tc.txt"

  # transfer: nc listener in peer ns, sender in root ns, both cross the shaped veth link
  xferfile="$out/xfer/$prof.baro"
  rm -f "$xferfile"
  # 60 s floor plus 5 MB/s worst case: a 2 GB file at 100mbit takes about 180 s.
  peer bash -c "timeout $(( 60 + $(stat -c%s "$commonfile") / 5000000 )) nc -l -p $XFER_PORT > '$xferfile'" &
  ncpid=$!
  sleep 0.3
  t0=$(date +%s.%N)
  nc -N "$PEER_IP" "$XFER_PORT" < "$commonfile"
  wait "$ncpid" 2>/dev/null
  t1=$(date +%s.%N)
  xfer_s=$(python3 -c "print($t1-$t0)")
  xfersha=$(sha256sum "$xferfile" | cut -d' ' -f1)
  match="FAIL"; [ "$xfersha" = "$srcsha" ] && match="PASS"
  echo "$prof transfer_s=$xfer_s bytes=$(stat -c%s "$xferfile" 2>/dev/null) sha_match=$match" | tee -a "$out/arm.txt"

  python3 - "$out" "$prof" "$xfer_s" "$match" <<'PY'
import json, pathlib, sys
out, prof, xfer_s, xmatch = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
measured = json.loads((pathlib.Path(out) / "measured.json").read_text())
rows = [{**m, "profile": prof, "xfer_s": xfer_s, "payload_s": xfer_s + m["fork_s"], "sha_match": xmatch == "PASS"}
        for m in measured]
for r in rows:
    print(f"{prof} {r['item']}: payload_s={r['payload_s']:.3f} recipe_s={r['recipe_s']:.3f} ids_match={r['ids_match']}", flush=True)
resfile = pathlib.Path(out) / "results.json"
existing = json.loads(resfile.read_text()) if resfile.exists() else []
resfile.write_text(json.dumps(existing + rows, indent=2))
PY
}

run_profile unshaped ""
run_profile 100mbit "tbf rate 100mbit burst 32kbit latency 400ms"
run_profile 1gbit "tbf rate 1gbit burst 320kbit latency 100ms"
run_profile 10gbit "tbf rate 10gbit burst 3200kbit latency 50ms"

sudo tc qdisc del dev "$VA" root 2>/dev/null || true

# B4_PREDICT (B4-long), e.g. "100mbit=recipe,1gbit=none,10gbit=payload": gate the
# per-profile winner by median over the items instead of B4-mini's breakeven band.
python3 - "$out" "$commonfile" "$P" "${B4_PREDICT:-}" <<'PY'
import json, pathlib, statistics, sys
out, ckfile, P, predict = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = json.loads((pathlib.Path(out) / "results.json").read_text())
plen = len(open(P).read().split())
fails = [r for r in rows if not r["ids_match"] or not r["sha_match"]]
void = [r for r in rows if r["cached_tokens"] != plen - 1]
size = pathlib.Path(ckfile).stat().st_size
recipe_fast = min(r["recipe_s"] for r in rows if r["profile"] == "10gbit")
breakeven = size / recipe_fast
lines = [f"# B4 receipt (P = {plen} ids)", "", f"checkpoint file bytes: {size}",
         f"fastest recipe_s (10gbit profile): {recipe_fast:.4f} s",
         f"measured breakeven: {breakeven/1e6:.2f} MB/s ({breakeven*8/1e9:.3f} Gbit/s)"]
if predict:
    calls = dict(kv.split("=") for kv in predict.split(","))
    lines += ["", "## direction calls (median over items)", ""]
    misses = []
    for prof, call in calls.items():
        pr = [r for r in rows if r["profile"] == prof]
        pay = statistics.median(r["payload_s"] for r in pr)
        rec = statistics.median(r["recipe_s"] for r in pr)
        won = "payload" if pay < rec else "recipe"
        ok = call == "none" or call == won
        if not ok:
            misses.append(prof)
        lines.append(f"- {prof}: payload {pay:.3f} s, recipe {rec:.3f} s, faster {won}, called {call}: "
                     + ("not gated" if call == "none" else "PASS" if ok else "FAIL"))
    gate_ok = not misses
else:
    pred = 152.75e6  # B4-mini: 03 sec 5, 1088 tok: 118 MiB / 0.81 s
    gate_ok = (pred / 2) <= breakeven <= (pred * 2)
    lines += ["03 sec 5 prediction (1088 tok): 152.75 MB/s (1.22 Gbit/s)", f"within 2x: {gate_ok}"]
verdict = "VOID" if void else "PASS" if not fails and gate_ok else "FAIL"
lines += ["", "## per-profile, per-item", ""]
for r in rows:
    lines.append(f"- {r['profile']}/{r['item']}: payload_s={r['payload_s']:.3f} recipe_s={r['recipe_s']:.3f} "
                  f"ids_match={r['ids_match']} sha_match={r['sha_match']} cached={r['cached_tokens']}")
lines += ["", f"## VERDICT: {verdict}", ""]
if void:
    lines.append(f"VOID rows (cached_tokens != {plen - 1}): {json.dumps(void, indent=2)}")
if fails:
    lines.append(f"FAILING rows: {json.dumps(fails, indent=2)}")
(pathlib.Path(out) / "receipt.md").write_text("\n".join(lines))
print("RESULT", verdict)
sys.exit(0 if verdict == "PASS" else 1)
PY
rc=$?
exit $rc
