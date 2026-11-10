#!/usr/bin/env bash
# usage: bench/checkpoint-api.sh ENGINE PACK OUTDIR [PORT]
#
# DONE check for the checkpoint API (docs/design/latent-os/10-checkpoint-api.md
# sec 6 and 8), the way a user meets it: two HTTP clients, a state file, a
# refusal. Greedy, spec off. GPU through gpu-wait by the caller.
#
#   ref     cold server: P+Q via /v1/completions          -> reference ids
#   flush   an unrelated prompt (drops the chain, overwrites the KV ring)
#   create  client A: POST /v1/checkpoints with P          -> id, state file
#   flush   again, so nothing of P survives in the engine
#   fork    client B: POST /v1/checkpoints/{id}/fork, suffix Q
#           -> ids == ref, restored true, cached_tokens == len(P) - 1
#   refuse  fork with a wrong identity.pack                -> 409 IDENTITY_MISMATCH
#   list / get / delete / get-after-delete                 -> 1 live, 200, deleted true, 404
#
# BARO_CKPT_EXTRA_PAIRS (default "p03-story:p04-list-planets,p05-math:p06-translate")
# runs the create/flush/fork/id-match cycle again for each extra P:Q pair, on
# top of the P/Q pair above, so a caller can gate on N prompt pairs agreeing
# (C2-mini, AMDHQ docs/design/latent-os/06-experiments.md). BARO_CKPT_P and
# BARO_CKPT_Q swap the main pair for any token files (C2-long: 8k/32k P);
# BARO_CKPT_EXTRA_PAIRS="" (set, empty) runs no extra pairs.
set -uo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; out=$3; port=${4:-8098}
mkdir -p "$out"
export BARO_CKPT_DIR="${BARO_CKPT_DIR:-$out/ckpts}"
mkdir -p "$BARO_CKPT_DIR"
serve=serve/target/release/baro-serve
pidfile="$out/server-pids"
[ -x "$serve" ] || { echo "no $serve; cargo build --release --manifest-path serve/Cargo.toml" >&2; exit 2; }
P=${BARO_CKPT_P:-bench/mtp-prompts/p01-water.tokens}
Q=${BARO_CKPT_Q:-bench/mtp-prompts/p02-python-fib.tokens}
F=bench/mtp-prompts/p03-*.tokens
F=$(ls $F | head -1)
extra_pairs=${BARO_CKPT_EXTRA_PAIRS-"p03-story:p04-list-planets,p05-math:p06-translate"}
for f in "$P" "$Q" "$F"; do [ -f "$f" ] || { echo "missing prompt file $f" >&2; exit 2; }; done
{
  echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port"
  echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ';')'"
  echo "P=$P ($(wc -w < "$P") tokens) Q=$Q ($(wc -w < "$Q") tokens) flush=$F"
} | tee "$out/arm.txt"

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
assert_no_engine() {
  for pid in $(pgrep -f -- "$eng" || true); do
    [ "$(readlink "/proc/$pid/exe" 2>/dev/null || true)" = "$eng" ] || continue
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -KILL -- "-$pgid" 2>/dev/null || true
    echo "FAIL cleanup: engine pid $pid from $eng survived server cleanup" >&2
    exit 1
  done
}
trap cleanup EXIT

setsid env BARO_PACK="$pack" BARO_CKPT_DIR="$BARO_CKPT_DIR" BARO_SPEC=0 "$serve" --engine "$eng" --pack "$pack" --port "$port" \
  > "$out/serve.out" 2>"$out/serve.err" &
spid=$!
ready=0
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$port/health" -o "$out/health.json" 2>/dev/null && { ready=1; break; }
  sleep 1
done
# The group is read only once the server answers: read right after `setsid ... &` it races
# setsid() and can be this script's own group, and cleanup's kill -INT then takes the whole
# run-tests job down (mega2 merge, 2026-10-17: SIGINT after the gate's RESULT PASS).
spgid=$(ps -o pgid= -p "$spid" | tr -d ' ')
[ -n "$spgid" ] && [ "$spgid" != "$(ps -o pgid= -p $$ | tr -d ' ')" ] || spgid=$spid
printf '%s %s\n' "$spid" "$spgid" > "$pidfile"
[ "$ready" = 1 ] || { echo "VOID: no /health on $port; see $out/serve.err" >&2; exit 3; }
echo "health: $(cat "$out/health.json")" | tee -a "$out/arm.txt"

python3 - "$out" "$port" "$P" "$Q" "$F" "$extra_pairs" <<'PY'
import json, pathlib, sys, urllib.request, urllib.error
out, port, P, Q, F, extra_pairs = pathlib.Path(sys.argv[1]), sys.argv[2], *sys.argv[3:7]
ids = lambda f: [int(x) for x in open(f).read().split()]
p, q, fl = ids(P), ids(Q), ids(F)
base = f"http://127.0.0.1:{port}"
def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
greedy = {"max_tokens": 32, "spec": False, "temperature": 0}
log, fails = [], 0
def check(cond, label):
    global fails
    log.append(("PASS " if cond else "FAIL ") + label); fails += 0 if cond else 1
    print(log[-1], flush=True)

s, ref = call("POST", "/v1/completions", {"prompt": p + q, **greedy}); check(s == 200, f"ref completion 200 (got {s})")
ref_ids = ref["choices"][0]["tokens"]
s, _ = call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False}); check(s == 200, "flush 1")
s, cr = call("POST", "/v1/checkpoints", {"prompt": p, "ttl_s": 600, "max_tokens": 1, "spec": False}); check(s == 200, f"create 200 (got {s}: {json.dumps(cr)[:200]})")
ck = cr.get("checkpoint", {}); cid = ck.get("id", "")
check(cr.get("created") is True and cid.startswith("lat:state@pos"), f"create returned id {cid}")
check(ck.get("pos") == len(p) - 1, f"pos == len(P)-1 ({ck.get('pos')} vs {len(p)-1})")
sf = out / "ckpts"
files = list(sf.glob("*.baro")) if sf.exists() else []
check(len(files) == 1 and files[0].stat().st_size == ck.get("bytes"), f"one state file, bytes {ck.get('bytes')}")
s, _ = call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False}); check(s == 200, "flush 2")
s, fk = call("POST", f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": q, **greedy}]}); check(s == 200, f"fork 200 (got {s}: {json.dumps(fk)[:200]})")
br = (fk.get("branches") or [{}])[0]
check(br.get("tokens") == ref_ids, f"fork ids == ref ids ({len(br.get('tokens', []))} tokens)")
check(br.get("restored") is True and br.get("cached_tokens") == len(p) - 1, f"restored from file, cached_tokens {br.get('cached_tokens')} == {len(p)-1}")
s, rj = call("POST", f"/v1/checkpoints/{cid}/fork", {"identity": {"pack": "deadbeef"}, "branches": [{"prompt_suffix": q, **greedy}]})
check(s == 409 and rj.get("error", {}).get("field") == "pack", f"wrong identity refused 409 field pack (got {s} {json.dumps(rj)[:120]})")
s, ls = call("GET", "/v1/checkpoints"); check(s == 200 and len(ls.get("data", [])) == 1, "list shows 1 live")
s, g1 = call("GET", f"/v1/checkpoints/{cid}"); check(s == 200 and g1.get("id") == cid, "get 200")
s, d = call("DELETE", f"/v1/checkpoints/{cid}"); check(s == 200 and d.get("deleted") is True, "delete true")
s, _ = call("GET", f"/v1/checkpoints/{cid}"); check(s == 404, "get after delete 404")
check(not list(sf.glob("*.baro")), "state file removed")

# extra prompt pairs (C2-mini identity gate): each pair gets its own cold
# ref completion, checkpoint, flush, fork, compared for exact id equality.
pair_results = []
for spec in [x for x in extra_pairs.split(",") if x]:
    pn, qn = spec.split(":")
    pf = pathlib.Path("bench/mtp-prompts") / f"{pn}.tokens"
    qf = pathlib.Path("bench/mtp-prompts") / f"{qn}.tokens"
    ep, eq = ids(pf), ids(qf)
    s, eref = call("POST", "/v1/completions", {"prompt": ep + eq, **greedy}); check(s == 200, f"[{spec}] ref completion 200 (got {s})")
    eref_ids = eref["choices"][0]["tokens"]
    s, _ = call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False}); check(s == 200, f"[{spec}] flush before create")
    s, ecr = call("POST", "/v1/checkpoints", {"prompt": ep, "ttl_s": 600, "max_tokens": 1, "spec": False}); check(s == 200, f"[{spec}] create 200 (got {s})")
    ecid = ecr.get("checkpoint", {}).get("id", "")
    s, _ = call("POST", "/v1/completions", {"prompt": fl, "max_tokens": 4, "spec": False}); check(s == 200, f"[{spec}] flush before fork")
    s, efk = call("POST", f"/v1/checkpoints/{ecid}/fork", {"branches": [{"prompt_suffix": eq, **greedy}]}); check(s == 200, f"[{spec}] fork 200 (got {s})")
    ebr = (efk.get("branches") or [{}])[0]
    id_match = ebr.get("tokens") == eref_ids
    check(id_match, f"[{spec}] fork ids == ref ids ({len(ebr.get('tokens', []))} tokens)")
    call("DELETE", f"/v1/checkpoints/{ecid}")
    pair_results.append({"pair": spec, "ref_ids": eref_ids, "fork_ids": ebr.get("tokens"), "match": id_match})

(out / "receipt.md").write_text("# checkpoint-api DONE check\n\n" + "\n".join(f"- {l}" for l in log)
    + f"\n\nref ids: {ref_ids}\nfork ids: {br.get('tokens')}\ncreate timings: {json.dumps(cr.get('completion', {}).get('timings'))}\nfork timings: {json.dumps(br.get('timings'))}\n"
    + f"\nextra pairs: {json.dumps(pair_results)}\n")
print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
PY
rc=$?
cleanup
assert_no_engine
grep -E 'state (saved|loaded)' "$out/serve.err" | tee -a "$out/receipt.md"
exit $rc
