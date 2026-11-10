#!/usr/bin/env bash
# usage: bench/replay-m.sh ENGINE PACK OUTDIR [PORT]
#
# P3 stage B (AMDHQ docs/design/latent-os/06-experiments.md, P3 amendment
# 2026-09-24): the replay window cap BARO_REPLAY_M swept over REPLAY_MS
# (default "8 4 2 3 6"; 8 = today's behaviour = the control), one server per
# value, int8 default, P = 8,000 ids, 15-token suffix p02. Per m: create the
# checkpoint, 3 x (flush, fork 32 tokens) for prefill_s and ids, 10 x (fork,
# 1 token, no flush) for the hot TTFT. summary.md: prefill_s medians per m,
# ids of every arm against the m=8 arm (FAIL on any difference), the knob
# read back from the engine log. Self-queues through gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$(realpath "$1"); pack=$(realpath "$2"); out=$(realpath -m "$3"); port=${4:-8150}
ms=${REPLAY_MS:-"8 4 2 3 6"}
fail() { echo "FAIL $1: $2"; exit 1; }
summary() {
python3 - "$out" <<'PY'
import json, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1]); rows = json.loads((out / "results.json").read_text())
lines = ["# replay-m summary (P3 stage B)", "", "| m | knob read back | fork32 n | prefill_s med | client s med | ttft_hot n | ttft client med | ttft wall med | ids == m=8 |", "|---|---|---|---|---|---|---|---|---|"]
ref = [x["tokens"] for x in rows if x["m"] == 8 and x["kind"] == "fork32" and x["ok"]]
bad = [x for x in rows if not x["ok"]]; idmis = []
for m in sorted({x["m"] for x in rows}):
    f = [x for x in rows if x["m"] == m and x["kind"] == "fork32" and x["ok"]]
    t = [x for x in rows if x["m"] == m and x["kind"] == "ttft_hot" and x["ok"]]
    knob = (out / f"knob-{m}.txt").read_text().strip() if (out / f"knob-{m}.txt").exists() else "MISSING"
    same = bool(ref) and bool(f) and all(x["tokens"] == ref[0] for x in f) and all(r == ref[0] for r in ref)
    if not same: idmis.append(m)
    med = lambda rs, k: f"{st.median(x[k] for x in rs):.4f}" if rs else "-"
    lines.append(f"| {m} | {knob} | {len(f)} | {med(f,'prefill_s')} | {med(f,'client_s')} | {len(t)} | {med(t,'client_s')} | {med(t,'wall_s')} | {'PASS' if same else 'FAIL'} |")
lines += ["", f"requests: {len(rows)}, failed: {len(bad)}, id mismatches vs m=8: {idmis}"] + [f"- FAILED {json.dumps(x)[:200]}" for x in bad]
knobbad = [m for m in sorted({x['m'] for x in rows}) if not (out / f"knob-{m}.txt").exists() or f"BARO_REPLAY_M: {m} " not in (out / f"knob-{m}.txt").read_text()]
lines.append(f"knob read-back mismatches: {knobbad}")
(out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
sys.exit(1 if bad or not rows or idmis or knobbad else 0)
PY
}
if [ "${REPLAY_M_SUMMARY_ONLY:-0}" = 1 ]; then summary; exit; fi
serve=serve/target/release/baro-serve
[ -x "$eng" ] || fail setup "no engine at $eng"; [ -x "$serve" ] || fail setup "no $serve"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  mkdir -p "$out"
  B4_TARGET_TOKENS=8000 B4_PREP_ONLY=1 bench/b4-cross-host.sh "$eng" "$pack" "$out/prep-8000" > "$out/prep-8000.log" 2>&1 || fail prep "$out/prep-8000.log"
  exec gpu-wait run --priority "${REPLAY_M_PRIORITY:-20}" --vram 22 --timeout 600 -- "$0" "$@"
fi
export LC_ALL=C
echo "[]" > "$out/results.json"
echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | head -c 80)' engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) serve=$serve sha=$(sha256sum "$serve" | cut -c1-16) pack=$pack port=$port tmax=32768 spec=0 int8=default ms='$ms' job=${GPU_WAITING_ROOM_JOB:-}" > "$out/arm.txt"
spid=""
stop_server() {
  [ -n "$spid" ] || return 0
  kill -- -"$spid" 2>/dev/null || kill "$spid" 2>/dev/null || :
  pkill -TERM -P "$spid" 2>/dev/null || :
  wait "$spid" 2>/dev/null || :
  spid=""
}
trap stop_server EXIT
for m in $ms; do
  rm -rf "$out/ckpts-$m"; mkdir -p "$out/ckpts-$m"
  setsid env BARO_PACK="$pack" BARO_CKPT_DIR="$out/ckpts-$m" BARO_SPEC=0 BARO_TMAX=32768 BARO_REPLAY_M=$m \
    "$serve" --engine "$eng" --pack "$pack" --port "$port" > "$out/serve-$m.out" 2> "$out/serve-$m.err" &
  spid=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$port/health" > /dev/null || fail "health m=$m" "$out/serve-$m.err"
  grep -m1 'BARO_REPLAY_M:' "$out/serve-$m.err" > "$out/knob-$m.txt" || fail "knob m=$m" "BARO_REPLAY_M line missing in $out/serve-$m.err"
  python3 - "$out" "$port" "$m" <<'PY'
import json, pathlib, sys, time, urllib.request, urllib.error
out, port, m = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ids = lambda f: [int(x) for x in open(f).read().split()]
Q, F = ids("bench/mtp-prompts/p02-python-fib.tokens"), ids("bench/mtp-prompts/p03-story.tokens")
P = ids(out / "prep-8000" / "P.tokens")
def call(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r: return r.status, json.loads(r.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}"), time.monotonic() - t0
flush = lambda: call("/v1/completions", {"prompt": F, "max_tokens": 4, "spec": False})
rows = json.loads((out / "results.json").read_text())
flush()
s, cr, _ = call("/v1/checkpoints", {"prompt": P, "ttl_s": 600, "max_tokens": 1, "spec": False})
if s != 200:
    rows.append({"m": m, "kind": "create", "ok": False, "status": s, "body": cr})
else:
    cid = cr["checkpoint"]["id"]
    for kind, mt in [("fork32", 32)] * 3 + [("ttft_hot", 1)] * 10:
        if kind == "fork32": flush()
        s, fk, wall = call(f"/v1/checkpoints/{cid}/fork", {"branches": [{"prompt_suffix": Q, "max_tokens": mt, "spec": False, "temperature": 0}]})
        b = (fk.get("branches") or [{}])[0]; t = b.get("timings") or {}
        ok = s == 200 and b.get("cached_tokens") == 7999 and "wall_s" in t
        rows.append({"m": m, "kind": kind, "ok": ok, "status": s, "client_s": wall, "cached_tokens": b.get("cached_tokens"), "tokens": b.get("tokens"),
                     **{k: float(t.get(k, 0.0)) for k in ("wall_s", "load_s", "restore_s", "prefill_s", "decode_s")}})
(out / "results.json").write_text(json.dumps(rows, indent=1))
PY
  stop_server
  if pgrep -f "^$eng( |$)" > /dev/null; then fail cleanup "an engine from $eng survived its server (m=$m)"; fi
done
summary
