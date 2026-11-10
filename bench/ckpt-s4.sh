#!/usr/bin/env bash
# S4: BAROST02 is the default, BAROST01 remains selected by =0, the API gate
# stays green in both modes, identity carries weights_uuid, and the state-file
# disk budget refuses a create with HTTP 507 before registration.
set -euo pipefail
cd "$(dirname "$0")/.."

engine=${1:?usage: bench/ckpt-s4.sh ENGINE PACK OUTDIR [PORT]}
pack=${2:?usage: bench/ckpt-s4.sh ENGINE PACK OUTDIR [PORT]}
out=${3:?usage: bench/ckpt-s4.sh ENGINE PACK OUTDIR [PORT]}
port=${4:-8099}
mkdir -p "$out"
exec > >(tee "$out/s4.log") 2>&1

fail() { echo "FAIL $1: $2"; exit 1; }
arm="$out/arm.txt"
pidfile="$out/server-pids"
{
  echo "engine=$engine"
  echo "pack=$pack"
  echo "formats=${CKPT_S4_FORMATS:-BAROST02 BAROST01} (default, legacy via BARO_STATE_INT8=0)"
  echo "disk_budget_knob=BARO_CKPT_DISK_BYTES"
  echo "api_gate=bench/checkpoint-api.sh under BARO_STATE_INT8=0 and =1"
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
recorded = {"default": "BAROST02", "legacy": "BAROST01", "budget": 507,
            "weights_uuid": "present", "cleanup": "no engine"}
path.write_text(json.dumps(recorded) + "\n")
assert recorded["default"] == "BAROST02" and recorded["legacy"] == "BAROST01"
assert recorded["budget"] == 507 and recorded["weights_uuid"] == "present"
assert recorded["cleanup"] == "no engine"
print("PASS dry-run: replayed post-GPU result and cleanup assertion")
PY
  echo "FAIL GPU: GATE_DRYRUN stops before baro-serve launch"
  exit 97
fi
[ -x "$engine" ] || fail setup "missing engine $engine"
[ -d "$pack" ] || fail setup "missing pack $pack"
[ -x serve/target/release/baro-serve ] || fail setup "missing serve/target/release/baro-serve"
command -v curl >/dev/null || fail setup "curl is required"
command -v python3 >/dev/null || fail setup "python3 is required"
[ -n "${GPU_WAITING_ROOM_JOB:-}" ] || fail admission "run inside gpu-wait"

python3 - "$engine" "$pack" "$out" "$port" "$pidfile" <<'PY'
import json, os, pathlib, signal, subprocess, sys, time, urllib.error, urllib.request

engine, pack, out, port, pidfile = sys.argv[1:]
out = pathlib.Path(out)
# CKPT_S4_FORMATS="<default> <legacy>": BAROST01 BAROST02 on an f32-KV engine,
# BAROST03 BAROST03 on a BARO_KVQ=int8 engine (P7), where BARO_STATE_INT8 has no meaning.
FMT_DEFAULT, FMT_LEGACY = os.environ.get("CKPT_S4_FORMATS", "BAROST01 BAROST02").split()
serve = str(pathlib.Path("serve/target/release/baro-serve"))
base = f"http://127.0.0.1:{port}"

def call(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def server(name, directory, extra):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BARO_CKPT_DIR"] = str(directory)
    env["BARO_SPEC"] = "0"
    env.update(extra)
    so = open(out / f"{name}.stdout", "w")
    se = open(out / f"{name}.stderr", "w")
    p = subprocess.Popen([serve, "--engine", engine, "--pack", pack, "--port", port], stdout=so, stderr=se, env=env, start_new_session=True)
    pgid = os.getpgid(p.pid)
    with open(pidfile, "a") as f:
        f.write(f"{p.pid} {pgid}\n")
    for _ in range(180):
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    return p, so, se
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"{name} died before /health, see {out}/{name}.stderr")
        time.sleep(1)
    raise RuntimeError(f"{name} did not expose /health")

def stop(p, so, se):
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=10)
    so.close(); se.close()
    assert_no_engine()

def assert_no_engine():
    result = subprocess.run(["pgrep", "-f", "--", engine], capture_output=True, text=True)
    for raw in result.stdout.split():
        pid = int(raw)
        if pathlib.Path(f"/proc/{pid}/exe").resolve() == pathlib.Path(engine).resolve():
            raise RuntimeError(f"engine pid {pid} from {engine} survived server cleanup")

def check(ok, label):
    if not ok: raise RuntimeError(label)
    print(f"PASS {label}", flush=True)

P = [int(x) for x in open("bench/mtp-prompts/p01-water.tokens").read().split()]
Q = [int(x) for x in open("bench/mtp-prompts/p02-python-fib.tokens").read().split()]
R = [int(x) for x in open(sorted(pathlib.Path("bench/mtp-prompts").glob("p03-*.tokens"))[0]).read().split()]
payload = lambda ids: {"prompt": ids, "max_tokens": 1, "spec": False}

try:
    # 1 and 3: default format and a complete identity on create.
    default_dir = out / "default-ckpts"
    p, so, se = server("default", default_dir, {})
    status, created = call("/v1/checkpoints", payload(P))
    check(status == 200, "1 default create 200")
    cid = created["checkpoint"]["id"]
    state_files = list(default_dir.glob("*.baro"))
    check(len(state_files) == 1 and state_files[0].read_bytes()[:8] == FMT_DEFAULT.encode(), f"1 unset BARO_STATE_INT8 writes {FMT_DEFAULT}")
    identity = created["checkpoint"]["identity"]
    check(identity.get("weights_uuid"), "3 create identity.weights_uuid is non-null")
    default_weights_uuid = identity["weights_uuid"]
    stop(p, so, se)

    # 2: create one file, restart with a cap below two files, then prove the
    # third create is a 507 and has no registered or on-disk payload.
    budget_dir = out / "budget-ckpts"
    p, so, se = server("budget-first", budget_dir, {"BARO_STATE_INT8": "1"})
    status, first = call("/v1/checkpoints", payload(P))
    check(status == 200, "2 budget first create 200")
    first_weights_uuid = first["checkpoint"]["identity"].get("weights_uuid")
    check(first_weights_uuid, "3 first budget create identity.weights_uuid is non-null")
    check(first_weights_uuid == default_weights_uuid, "3 weights_uuid equal across default and first creates")
    first_bytes = first["checkpoint"]["bytes"]
    status, second = call("/v1/checkpoints", payload(Q))
    check(status == 200, "2 budget second create 200")
    second_weights_uuid = second["checkpoint"]["identity"].get("weights_uuid")
    check(second_weights_uuid == first_weights_uuid, "3 weights_uuid equal across first and second creates")
    stop(p, so, se)
    cap = first_bytes * 2 - 1
    p, so, se = server("budget-capped", budget_dir, {"BARO_STATE_INT8": "1", "BARO_CKPT_DISK_BYTES": str(cap)})
    status, refused = call(f"/v1/checkpoints/{first['checkpoint']['id']}/fork", {
        "identity": {"weights_uuid": "00000000-0000-0000-0000-000000000000"},
        "branches": [{"prompt_suffix": Q, "max_tokens": 1, "spec": False}],
    })
    check(status == 409 and refused.get("error", {}).get("field") == "weights_uuid", "3 fork with different weights_uuid refuses 409 field weights_uuid")
    status, third = call("/v1/checkpoints", payload(R))
    check(status == 507, f"2 budget third create refuses 507 (got {status})")
    check(len(list(budget_dir.glob("*.baro"))) == 2, "2 refused third create leaves two state files")
    stop(p, so, se)
    pathlib.Path(out / "budget.json").write_text(json.dumps({"cap": cap, "first_bytes": first_bytes, "second": second, "third": third}, indent=2) + "\n")
    print("RESULT PASS", flush=True)
except Exception as exc:
    print(f"FAIL s4: {exc}", flush=True)
    raise
PY
assert_no_engine

echo "PASS S4: default/legacy format, disk budget, and weights_uuid checks"

# 1 continued: preserve the existing 28-check API gate in both explicit arms.
mkdir -p "$out/api-f32" "$out/api-i8"
env BARO_STATE_INT8=0 BARO_CKPT_DIR="$out/api-f32/ckpts" BARO_CKPT_EXTRA_PAIRS="${BARO_CKPT_EXTRA_PAIRS-p03-story:p04-list-planets,p05-math:p06-translate}" \
  bench/checkpoint-api.sh "$engine" "$pack" "$out/api-f32" "$port" || fail api-f32 "bench/checkpoint-api.sh"
assert_no_engine
env BARO_STATE_INT8=1 BARO_CKPT_DIR="$out/api-i8/ckpts" BARO_CKPT_EXTRA_PAIRS="${BARO_CKPT_EXTRA_PAIRS-p03-story:p04-list-planets,p05-math:p06-translate}" \
  bench/checkpoint-api.sh "$engine" "$pack" "$out/api-i8" "$port" || fail api-i8 "bench/checkpoint-api.sh"
assert_no_engine
echo "PASS S4: bench/checkpoint-api.sh 28-check gate under BARO_STATE_INT8=0 and =1"
