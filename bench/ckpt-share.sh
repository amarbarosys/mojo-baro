#!/usr/bin/env bash
# S1: one checkpoint registry survives a baro-serve restart and is forkable
# by a second process sharing BARO_CKPT_DIR. The live command is run inside one
# gpu-wait allocation; A, B and the wrong-pack peer never overlap.
set -euo pipefail
cd "$(dirname "$0")/.."

engine=${1:?usage: bench/ckpt-share.sh ENGINE PACK OUTDIR [PORT]}
pack=${2:?usage: bench/ckpt-share.sh ENGINE PACK OUTDIR [PORT]}
out=${3:?usage: bench/ckpt-share.sh ENGINE PACK OUTDIR [PORT]}
port=${4:-8098}
other_pack=${BARO_PACK_OTHER:-$out/other-pack}
mkdir -p "$out"
exec > >(tee "$out/s1.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
ckpt_dir=${BARO_CKPT_DIR:-$out/ckpts}
mkdir -p "$ckpt_dir"
pidfile="$out/server-pids"
arm="$out/arm.txt"
{
  echo "engine=$engine"
  echo "pack=$pack"
  echo "other_pack=$other_pack"
  echo "BARO_CKPT_DIR=$ckpt_dir"
  echo "BARO_STATE_INT8=${BARO_STATE_INT8-<unset-default-int8>}"
  echo "ttl_expired_s=1"
  echo "shared_processes=A->B->wrong-pack"
} > "$arm"

cleanup() {
  set +e
  [ -f "$pidfile" ] || return 0
  for sig in INT TERM KILL; do
    while read -r pid pgid; do [ -n "${pid:-}" ] && kill -"$sig" -- "-$pgid" 2>/dev/null; done < "$pidfile"
    [ "$sig" = KILL ] && break
    for _ in $(seq 1 100); do
      alive=0
      while read -r pid _; do [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && alive=1; done < "$pidfile"
      [ "$alive" = 0 ] && break
      sleep 0.1
    done
  done
  while read -r pid _; do [ -n "${pid:-}" ] && wait "$pid" 2>/dev/null; done < "$pidfile"
}
trap cleanup EXIT

assert_no_engine() {
  command -v pgrep >/dev/null || fail cleanup "pgrep is required"
  for pid in $(pgrep -f -- "$engine" || true); do
    [ "$(readlink "/proc/$pid/exe" 2>/dev/null || true)" = "$engine" ] || continue
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -KILL -- "-$pgid" 2>/dev/null || true
    fail cleanup "engine pid $pid from $engine survived server cleanup"
  done
  echo "PASS cleanup: no engine from worktree survives"
}

if [ "${GATE_DRYRUN:-0}" = 1 ]; then
  python3 - "$out/dry-replay.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
recorded = {"result": "PASS", "stages": ["A", "B", "wrong-pack"], "cleanup": "no engine"}
path.write_text(json.dumps(recorded) + "\n")
assert recorded["result"] == "PASS" and len(recorded["stages"]) == 3
assert recorded["cleanup"] == "no engine"
print("PASS dry-run: replayed post-GPU result and cleanup assertion")
PY
  echo "FAIL GPU: GATE_DRYRUN stops before baro-serve launch"
  exit 97
fi
[ -x "$engine" ] || fail setup "missing engine $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run inside gpu-wait"

# Build a tiny alternate pack directory around the same payload. Its changed
# identity digest makes the server reject A's metadata before the engine call,
# while the engine remains a valid compatible pack.
mkdir -p "$other_pack"
for source in "$pack"/*; do
  name=${source##*/}
  [ "$name" = identity.json ] && continue
  ln -sfn "$(readlink -f "$source")" "$other_pack/$name"
done
python3 - "$pack/identity.json" "$other_pack/identity.json" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:])
data = json.loads(src.read_text())
data["pack_sha256"] = ("0" if data["pack_sha256"][0] != "0" else "1") + data["pack_sha256"][1:]
dst.write_text(json.dumps(data) + "\n")
PY

python3 - "$engine" "$pack" "$other_pack" "$ckpt_dir" "$out" "$port" "$pidfile" <<'PY'
import json, os, pathlib, signal, subprocess, sys, time, urllib.error, urllib.request

engine, pack, other_pack, ckpt_dir, out, port, pidfile = sys.argv[1:]
out = pathlib.Path(out)
base = f"http://127.0.0.1:{port}"
logs = {}

def start(name, selected_pack):
    stdout = open(out / f"{name}.stdout", "w")
    stderr = open(out / f"{name}.stderr", "w")
    env = os.environ.copy()
    env["BARO_CKPT_DIR"] = ckpt_dir
    env["BARO_SPEC"] = "0"
    p = subprocess.Popen(["serve/target/release/baro-serve", "--engine", engine, "--pack", selected_pack, "--port", port], stdout=stdout, stderr=stderr, env=env, start_new_session=True)
    pgid = os.getpgid(p.pid)
    with open(pidfile, "a") as f:
        f.write(f"{p.pid} {pgid}\n")
    logs[name] = (p, pgid, stdout, stderr)
    for _ in range(180):
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"{name} exited before /health, see {out}/{name}.stderr")
        time.sleep(1)
    raise RuntimeError(f"{name} did not expose /health")

def stop(name):
    p, pgid, stdout, stderr = logs[name]
    if p.poll() is None:
        os.killpg(pgid, signal.SIGINT)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL); p.wait(timeout=10)
    stdout.close(); stderr.close()
    assert_no_engine()

def assert_no_engine():
    result = subprocess.run(["pgrep", "-f", "--", engine], capture_output=True, text=True)
    for raw in result.stdout.split():
        pid = int(raw)
        if pathlib.Path(f"/proc/{pid}/exe").resolve() == pathlib.Path(engine).resolve():
            raise RuntimeError(f"engine pid {pid} from {engine} survived server cleanup")

def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def check(ok, label):
    if not ok:
        raise RuntimeError(label)
    print(f"PASS {label}", flush=True)

P = [int(x) for x in open("bench/mtp-prompts/p01-water.tokens").read().split()]
Q = [int(x) for x in open("bench/mtp-prompts/p02-python-fib.tokens").read().split()]
F = [int(x) for x in open(sorted(pathlib.Path("bench/mtp-prompts").glob("p03-*.tokens"))[0]).read().split()]
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}

try:
    start("A", pack)
    s, ref = call("POST", "/v1/completions", {"prompt": P + Q, **greedy})
    check(s == 200, "1 cold P+Q reference 200")
    ref_ids = ref["choices"][0]["tokens"]
    s, created = call("POST", "/v1/checkpoints", {"prompt": P, "ttl_s": 600, "max_tokens": 1, "spec": False})
    check(s == 200, "1 A creates live checkpoint")
    checkpoint = created["checkpoint"]
    cid = checkpoint["id"]
    check(cid.startswith("lat:state@pos"), "1 checkpoint id uses lat:state@pos format")
    identity_a = checkpoint["identity"]
    s, expired = call("POST", "/v1/checkpoints", {"prompt": F, "ttl_s": 1, "max_tokens": 1, "spec": False})
    check(s == 200, "3 A creates ttl=1 checkpoint")
    expired_id = expired["checkpoint"]["id"]
    (out / "create-a.json").write_text(json.dumps(created, indent=2) + "\n")
    stop("A")

    time.sleep(2)
    start("B", pack)
    s, listing = call("GET", "/v1/checkpoints")
    check(s == 200 and len(listing.get("data", [])) == 1, "1 B lists one live checkpoint after restart")
    s, got = call("GET", f"/v1/checkpoints/{cid}")
    check(s == 200 and got.get("id") == cid and got.get("identity") == identity_a, "1 B gets id, bytes and identity from sidecar")
    s, forked = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": Q, **greedy}]})
    branch = (forked.get("branches") or [{}])[0]
    check(s == 200 and branch.get("tokens") == ref_ids, "1 B fork ids equal cold P+Q reference")
    check(branch.get("restored") is True and branch.get("cached_tokens") == len(P) - 1, "1 B reports restored and cached_tokens=len(P)-1")
    check(not pathlib.Path(ckpt_dir, expired_id.replace(":", "_").replace("@", "_") + ".baro").exists(), "3 B sweep removes expired state file")
    stop("B")

    start("wrong-pack", other_pack)
    s, refused = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": Q, **greedy}]})
    check(s == 409 and refused.get("error", {}).get("field") == "pack", "2 different pack refuses fork with 409 field pack")
    stop("wrong-pack")
    print("RESULT PASS", flush=True)
except Exception as exc:
    for name in list(logs):
        try: stop(name)
        except Exception: pass
    print(f"FAIL s1: {exc}", flush=True)
    raise
PY
assert_no_engine

echo "PASS S1: numbered items 1, 2 and 3; receipt=$out/s1.log"
