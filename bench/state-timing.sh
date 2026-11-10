#!/usr/bin/env bash
# usage: bench/state-timing.sh ENGINE PACK OUTDIR [PORT]
#
# Where fork time goes, and warm time to first token (AMDHQ
# docs/design/latent-os/11-state-layer.md S2 open measurement, S3 baseline).
# One gpu-wait job; per state format (f32, then int8) one server:
#   for P = 8,000 and 32,000 ids: create a checkpoint of P, then 3 x (flush,
#   fork with a 15-token suffix, 32 tokens) timed client-side next to the
#   engine's own load_s / restore_s / prefill_s / decode_s / wall_s;
#   at 8,000 also 20 x (flush, fork, 1 token) = warm time to first token.
#   P2a/P3 (plan 12): per P also fork32_q32 (flush, fork with a 32-token
#   suffix: linear in suffix tokens = m=1 replay, P3's falsifier), then
#   fork32_hot x3 and ttft_hot x20 with NO flush between forks (the chain
#   still holds the prefix: load_s must be ~0 and ids must equal the flushed
#   fork32's, P2a's check). fork32 and fork32_hot rotate three 15-token
#   suffixes (p02, p05, p11) in the same order, so a hot fork can only hit the
#   checkpoint of P (with one suffix the second hot fork hit the previous
#   fork's own suffix checkpoint, P2a amendment); ids compare per suffix.
# summary.md splits client wall into engine timers, engine gap (wall_s minus
# its timers) and front gap (client wall minus engine wall_s).
# bench/state-roundtrip.sh is the one-shot identity gate; this one times the
# checkpoint API. STATE_TIMING_SUMMARY_ONLY=1 re-runs only the summary.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); pack=$2; out=$(realpath -m "$3"); port=${4:-8097}
fail() { echo "FAIL $1: $2"; exit 1; }

summary() {
python3 - "$out" <<'PY'
import json, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1]); rows = json.loads((out / "results.json").read_text())
lines = ["# state-timing summary", "", "| fmt | P | kind | n | client s | engine wall_s | load_s | read | parse | pin | host | h2d | restore_s | prefill_s | decode_s | engine gap | front gap |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for fmt in ("f32", "int8"):
    for n in (8000, 32000):
        for kind in ("fork32", "fork32_q32", "ttft", "fork32_hot", "ttft_hot"):
            r = [x for x in rows if x["fmt"] == fmt and x["P"] == n and x["kind"] == kind and x["ok"]]
            if not r:
                continue
            m = lambda k: st.median(x[k] for x in r)
            eng_gap = st.median(x["wall_s"] - x["load_s"] - x["restore_s"] - x["prefill_s"] - x["decode_s"] for x in r)
            front_gap = st.median(x["client_s"] - x["wall_s"] for x in r)
            lines.append(f"| {fmt} | {n} | {kind} | {len(r)} | {m('client_s'):.3f} | {m('wall_s'):.3f} | {m('load_s'):.3f} | {m('load_read_s'):.3f} | {m('load_parse_s'):.3f} | {m('load_pin_s'):.3f} | {m('load_host_s'):.3f} | {m('load_h2d_s'):.3f} | {m('restore_s'):.4f} | {m('prefill_s'):.3f} | {m('decode_s'):.3f} | {eng_gap:.3f} | {front_gap:.3f} |")
bad = [x for x in rows if not x["ok"]]
lines += ["", f"requests: {len(rows)}, failed: {len(bad)}"] + [f"- FAILED {json.dumps(x)[:200]}" for x in bad]
idmis = []
for fmt in ("f32", "int8"):
    for n in (8000, 32000):
        cold = {x.get("suffix", 0): x["tokens"] for x in rows if x["fmt"] == fmt and x["P"] == n and x["kind"] == "fork32" and x["ok"]}
        hot = [(x.get("suffix", 0), x["tokens"]) for x in rows if x["fmt"] == fmt and x["P"] == n and x["kind"] == "fork32_hot" and x["ok"]]
        if cold and hot:
            same = all(sfx in cold and t == cold[sfx] for sfx, t in hot)
            lines.append(f"- ids {fmt} {n}: fork32_hot x{len(hot)} == fork32 of the same suffix: {'PASS' if same else 'FAIL'}")
            if not same: idmis.append((fmt, n))
hot_loads = [x for x in rows if x["kind"].endswith("_hot") and x["ok"]]
if hot_loads:
    lines.append(f"- hot forks: {len(hot_loads)}, max load_s {max(x['load_s'] for x in hot_loads):.4f}")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if bad or not rows or idmis else 0)
PY
}
if [ "${STATE_TIMING_SUMMARY_ONLY:-0}" = 1 ]; then summary; exit; fi

serve=serve/target/release/baro-serve
[ -x "$eng" ] || fail setup "no engine at $eng"
[ -x "$serve" ] || fail setup "no $serve"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$out"
  for n in 8000 32000; do
    B4_TARGET_TOKENS=$n B4_PREP_ONLY=1 bench/b4-cross-host.sh "$eng" "$pack" "$out/prep-$n" > "$out/prep-$n.log" 2>&1 \
      || fail prep "$out/prep-$n.log"
  done
  exec gpu-wait run --priority "${STATE_TIMING_PRIORITY:-10}" --vram 22 --timeout 600 -- "$0" "$@"
fi

echo "[]" > "$out/results.json"
echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | head -c 80)' engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port tmax=32768 spec=0 job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"
spid=""
stop_server() {
  [ -n "$spid" ] || return 0
  kill -- -"$spid" 2>/dev/null || kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  spid=""
}
trap stop_server EXIT
for fmt in f32 int8; do
  int8=0; [ "$fmt" = int8 ] && int8=1
  rm -rf "$out/ckpts-$fmt"; mkdir -p "$out/ckpts-$fmt"
  setsid env BARO_PACK="$pack" BARO_CKPT_DIR="$out/ckpts-$fmt" BARO_SPEC=0 BARO_TMAX=32768 BARO_STATE_INT8=$int8 \
    "$serve" --engine "$eng" --pack "$pack" --port "$port" > "$out/serve-$fmt.out" 2> "$out/serve-$fmt.err" &
  spid=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$port/health" > /dev/null || fail "health $fmt" "$out/serve-$fmt.err"
  python3 - "$out" "$port" "$fmt" <<'PY'
import json, pathlib, sys, time, urllib.request, urllib.error
out, port, fmt = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ids = lambda f: [int(x) for x in open(f).read().split()]
Q, F = ids("bench/mtp-prompts/p02-python-fib.tokens"), ids("bench/mtp-prompts/p03-story.tokens")
Q32 = ids("bench/mtp-prompts/p12-rust.tokens")
QS = [Q, ids("bench/mtp-prompts/p05-math.tokens"), ids("bench/mtp-prompts/p11-email.tokens")]
assert len(Q) == 15 and len(Q32) == 32 and all(len(q) == 15 for q in QS), (len(Q), len(Q32), [len(q) for q in QS])
def call(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.monotonic() - t0
flush = lambda: call("/v1/completions", {"prompt": F, "max_tokens": 4, "spec": False})
rows = json.loads((out / "results.json").read_text())
for n in (8000, 32000):
    P = ids(out / f"prep-{n}" / "P.tokens")
    flush()
    s, cr, _ = call("/v1/checkpoints", {"prompt": P, "ttl_s": 600, "max_tokens": 1, "spec": False})
    if s != 200:
        rows.append({"fmt": fmt, "P": n, "kind": "create", "ok": False, "status": s, "body": cr}); continue
    cid = cr["checkpoint"]["id"]
    plan = [("fork32", 32)] * 3 + [("fork32_q32", 32)] * 3 + ([("ttft", 1)] * 20 if n == 8000 else []) \
        + [("fork32_hot", 32)] * 3 + [("ttft_hot", 1)] * 20
    seq = {}
    import subprocess, glob
    def fincore():
        fs = glob.glob(str(out / f"ckpts-{fmt}" / "*.baro"))
        try: return subprocess.run(["fincore", "--raw", "--noheadings", "-o", "RES,SIZE"] + fs, capture_output=True, text=True).stdout.strip().replace("\n", ";")
        except Exception as e: return f"fincore-error {e}"
    for kind, mt in plan:
        if not kind.endswith("_hot"):
            flush()
        fc = fincore() if kind == "fork32" else ""
        i = seq.get(kind, 0); seq[kind] = i + 1
        sfx = Q32 if kind == "fork32_q32" else QS[i % 3]
        s, fk, wall = call(f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": sfx, "max_tokens": mt, "spec": False, "temperature": 0}]})
        b = (fk.get("branches") or [{}])[0]; t = b.get("timings") or {}
        ok = s == 200 and b.get("cached_tokens") == n - 1 and "wall_s" in t
        rows.append({"fmt": fmt, "P": n, "kind": kind, "ok": ok, "status": s, "client_s": wall,
                     **{k: float(t.get(k, 0.0)) for k in ("wall_s", "load_s", "load_read_s", "load_parse_s", "load_pin_s", "load_host_s", "load_h2d_s", "restore_s", "prefill_s", "decode_s")},
                     "cached_tokens": b.get("cached_tokens"), "bytes": cr["checkpoint"].get("bytes"), "tokens": b.get("tokens"), "suffix": i % 3, "fincore": fc})
(out / "results.json").write_text(json.dumps(rows, indent=1))
PY
  stop_server
done
if pgrep -f "^$eng( |$)" > /dev/null; then fail cleanup "an engine from $eng survived its server"; fi
summary
