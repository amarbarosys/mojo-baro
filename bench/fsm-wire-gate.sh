#!/usr/bin/env bash
# fsm-wire-gate: the engine-side state-machine fixes (docs/lanes/2026-11-09-fsm-fixes.md, fix-engine) on the raw wire.
#   usage: bench/fsm-wire-gate.sh CAND_ENGINE BASE_ENGINE OUT
#   env:   BARO_PACK (default .work/engine-pack-q4)   EXPLORE=1 (skip the preflight check; best verdict UNVERIFIED, exit 3)
#          GATE_DRYRUN=1 (CPU: stop at the first engine step)
# One resident engine per probe, requests over stdin, BARO_TMAX=16384. Every probe runs on BASE (the pre-fix champion)
# and on CAND. PASS needs: every CAND check true AND every BASE symptom seen. A BASE that does not show the symptom
# means the probe is inert (it would pass without the fix): VOID, exit 1.
#   probes: ready (B-M9) | plain: done fields B-M1 B-M3 B-M6, idle cancel B-M12 | resume (B-M5) | fork (B-M5)
#           | window (B-M1 post-extension) | slens (B-M15) | steer: request A with a gain, plain B equals B alone (B-M2 no-regress)
# NOT EXERCISED here (no knob forces them): B-M1 live NOT-RESIDENT, B-M7 KV exhaustion, B-M13 captured, C6 grejected on a
# served grammar, B-M4/B-M10 (CPU tests only), spark B-M14. The verdict line lists them.
# serve-drive (iTools/harness) was surveyed: it cannot send raw control lines nor expect an error line or EOF, so this
# gate keeps a small driver of its own (candidate: extend serve-drive with --line).
# CPU dry run of this script: bench/fsm-wire-stub.sh --make DIR (a stub engine; proves this script, not the engine).
# PASS means all of the above held; receipts: OUT/arm.txt, OUT/<probe>-{base,cand}.log, OUT/verdict.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
cand=${1:?CAND_ENGINE}; base=${2:?BASE_ENGINE}; out=${3:?OUT}   # 1. arguments before the re-exec
pack=${BARO_PACK:-.work/engine-pack-q4}; explore=${EXPLORE:-0}; dry=${GATE_DRYRUN:-0}
fail() { echo "FAIL $1: $2" >&2; exit 1; }
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "$dry" != 1 ] && command -v gpu-wait >/dev/null; then
  echo "cost: 2 engines x 7 probes x (pack load ~25 s + 3 short requests) ~ 9 GPU min, one job"
  rc=0
  gpu-wait run --priority 70 --timeout 1500 --vram 24 -- env EXPLORE="$explore" BARO_PACK="$pack" "$0" "$@" || rc=$?   # 2. env explicit
  gpu-wait stats --days 1 >> "$out/verdict.txt" 2>/dev/null || true
  exit "$rc"
fi
mkdir -p "$out"
if [ "$explore" = 1 ]; then echo "EXPLORE=1: preflight check skipped, the best verdict is UNVERIFIED"
else bench/preflight.sh --check || fail preflight "no passing bench/preflight.sh for this tree"; fi
[ -f "$pack/index.txt" ] || fail setup "no pack at $pack"
[ -f "$pack/prompt-tokens.txt" ] || fail setup "no $pack/prompt-tokens.txt"
sha() { if [ -e "$1" ]; then sha256sum "$1" | cut -c1-16; else echo MISSING; fi; }
{ echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ' ')'"
  echo "cand=$cand sha=$(sha "$cand")"; echo "base=$base sha=$(sha "$base")"
  echo "pack=$pack index_sha=$(sha "$pack/index.txt")"
  echo "env BARO_SERVE=1 BARO_TMAX=16384 BARO_SPEC=0 (steer probe adds BARO_STEER_SERVE=1)"
  echo "explore=$explore dry=$dry"; } | tee "$out/arm.txt"
if [ "$dry" = 1 ]; then echo "FAIL GPU step: the engine never printed its ready line (dry run stops before the first engine start)"; exit 1; fi
[ -x "$cand" ] || fail setup "CAND_ENGINE not executable: $cand"
[ -x "$base" ] || fail setup "BASE_ENGINE not executable: $base"
[ "$(sha "$cand")" != "$(sha "$base")" ] || fail arms "cand and base are the same binary: the symptoms cannot differ"

python3 -I - "$cand" "$base" "$out" "$pack" "$explore" <<'PY'
import json, os, select, subprocess, sys, time
cand, base, out, pack, explore = sys.argv[1:6]
toks = [int(x) for x in open(f"{pack}/prompt-tokens.txt").read().split()][:32]
TMAX = 16384
READY_S, REQ_S = 300, 120

class Eng:
    def __init__(self, path, tag, env_extra):
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "BARO_SERVE": "1", "BARO_TMAX": str(TMAX),
               "BARO_SPEC": "0", "BARO_PACK": pack, **env_extra}
        self.log = open(f"{out}/{tag}.log", "w")
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, bufsize=0)
        self.buf = b""; self.lines = []; self.dead = False; self.toks = {}; self.tok_by_id = {}; self.all_logs = []
        self.ready = self.read_until(lambda l: l.get("ready") is True, READY_S)
        if self.ready is None: raise RuntimeError("the engine never printed its ready line (see " + self.log.name + ")")
    def send(self, obj):
        data = (obj if isinstance(obj, str) else json.dumps(obj)).encode() + b"\n"
        try: self.p.stdin.write(data)
        except BrokenPipeError: self.dead = True
    def read_until(self, pred, timeout):
        # the first queued JSON line matching pred, or None on EOF or timeout; every line lands in self.lines and the log
        end = time.time() + timeout
        while time.time() < end:
            for i, l in enumerate(self.lines):
                if isinstance(l, dict) and pred(l):
                    del self.lines[:i + 1]; return l
            r, _, _ = select.select([self.p.stdout], [], [], 0.5)
            if not r:
                if self.p.poll() is not None: self.dead = True; return None
                continue
            chunk = os.read(self.p.stdout.fileno(), 65536)
            if not chunk:
                self.dead = True; return None
            self.buf += chunk
            while b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                s = raw.decode(errors="replace"); self.log.write(s + "\n"); self.log.flush()
                try: obj = json.loads(s)
                except ValueError: self.lines.append(s); self.all_logs.append(s); continue
                if isinstance(obj, dict) and "tok" in obj: self.tok_by_id.setdefault(obj.get("id"), []).append(obj["tok"])  # counted once, on arrival
                else: self.lines.append(obj)
        return None
    def ask(self, req_id, obj, t=REQ_S, extra=()):
        # a request ends at its done line or at an error line (or EOF); tok lines of req_id are kept in self.toks.
        # A control line answers in milliseconds: its probes pass t=30 and the extra reply keys they accept.
        self.send(obj)
        got = self.read_until(lambda l: l.get("done") is True or "error" in l or any(k in l for k in extra), t)
        self.toks[req_id] = self.tok_by_id.pop(req_id, [])
        return got
    def logs(self): return self.all_logs
    def close(self):
        try: self.p.stdin.close()
        except Exception: pass
        try: self.p.wait(timeout=30)
        except subprocess.TimeoutExpired: self.p.kill()
        self.log.close()

def req(i, n=8, **kw): return {"id": i, "prompt": toks, "n": n, "temperature": 0, **kw}
def plain_ok(e, i):
    r = e.ask(i, req(i)); return bool(r) and r.get("done") is True and r.get("n", 0) > 0

def probe_ready(e): return {"ready_has_tier_seqs": "tier" in e.ready and "seqs" in e.ready}
def probe_plain(e):
    d = e.ask(1, req(1)); res = {}
    res["done_fail0"] = bool(d) and d.get("fail") == 0
    res["done_path"] = bool(d) and d.get("path") in ("mega", "mega_win", "spec", "launch")
    res["done_grammar_none"] = bool(d) and d.get("grammar") == "none"
    e.send('{"cancel":99}')
    r = e.ask(2, req(2))   # a cancel at idle must not put an error line in front of the next answer
    res["idle_cancel_silent"] = bool(r) and r.get("done") is True
    res["idle_cancel_logged"] = any("not decoding" in l for l in e.logs())
    return res
def probe_resume(e):
    r = e.ask(1, {"id": 1, "latent_resume": "lat_nope"}, t=30)
    return {"resume_error_line": bool(r) and "error" in r, "resume_engine_alive": (not e.dead) and plain_ok(e, 2)}
def probe_fork(e):
    r = e.ask(1, {"id": 1, "latent_fork": 1}, t=30, extra=("latent_id",))
    return {"fork_error_line": bool(r) and "error" in r and "no checkpoint" in str(r.get("error")),
            "fork_engine_alive": (not e.dead) and plain_ok(e, 2)}
def probe_window(e):
    st = f"{out}/window-state.bin"
    first = e.ask(1, req(1, n=4, state_save=st))
    if not first or first.get("done") is not True: return {"window_setup": False}
    r = e.ask(2, {"id": 2, "prompt": [toks[0]], "n": TMAX - 1, "temperature": 0, "state_load": st, "state_prefix": True})
    named = bool(r) and r.get("id") == 2 and "error" in r and "exceeds" in str(r.get("error"))
    return {"window_setup": True, "window_error_with_id": named, "window_engine_alive": (not e.dead) and plain_ok(e, 3)}
def probe_slens(e):
    r = e.ask(1, req(1, n=2, slens_ctx=5, slens_k=1))
    saw_err = bool(r) and "error" in r
    d = e.read_until(lambda l: l.get("done") is True, 30) if saw_err else None
    return {"slens_error_line": saw_err, "slens_finish_error": bool(d) and d.get("finish") == "error",
            "slens_engine_alive": (not e.dead) and plain_ok(e, 2)}
def probe_steer(e):
    e.ask(1, req(1)); e.ask(2, req(2, gain="0:0:2.0")); e.ask(3, req(3))
    return {"steer_B2_equals_B1": len(e.toks[1]) > 0 and e.toks[1] == e.toks[3]}

# name, fn, extra env, CAND checks that must be True, BASE symptoms {check: value on BASE} (empty = identity arm, must hold on BASE too)
PROBES = [
    ("ready", probe_ready, {}, ["ready_has_tier_seqs"], {"ready_has_tier_seqs": False}),
    ("plain", probe_plain, {}, ["done_fail0", "done_path", "done_grammar_none", "idle_cancel_silent", "idle_cancel_logged"],
        {"done_fail0": False, "idle_cancel_logged": False}),
    ("resume", probe_resume, {}, ["resume_error_line", "resume_engine_alive"], {"resume_engine_alive": False}),
    ("fork", probe_fork, {}, ["fork_error_line", "fork_engine_alive"], {"fork_error_line": False}),
    ("window", probe_window, {}, ["window_setup", "window_error_with_id", "window_engine_alive"], {"window_engine_alive": False}),
    ("slens", probe_slens, {}, ["slens_error_line", "slens_finish_error", "slens_engine_alive"], {"slens_engine_alive": False}),
    ("steer", probe_steer, {"BARO_STEER_SERVE": "1"}, ["steer_B2_equals_B1"], {}),
]
verdict = []; bad = 0; void = 0
for name, fn, env, want, symptom in PROBES:
    res = {}
    for tag, path in (("base", base), ("cand", cand)):
        e = None
        try:
            e = Eng(path, f"{name}-{tag}", env); res[tag] = fn(e)
        except Exception as ex:
            res[tag] = {"_exception": str(ex)}
        finally:
            if e: e.close()
    for k in want:
        ok = res["cand"].get(k) is True
        verdict.append(f"{'ok  ' if ok else 'FAIL'} cand {name}.{k} = {res['cand'].get(k)}")
        bad += 0 if ok else 1
    for k, v in symptom.items():
        seen = res["base"].get(k) == v
        verdict.append(f"{'ok  ' if seen else 'VOID'} base {name}.{k} = {res['base'].get(k)} (symptom wants {v})")
        void += 0 if seen else 1
    if not symptom:
        k = want[0]; same = res["base"].get(k) is True
        verdict.append(f"{'ok  ' if same else 'FAIL'} base {name}.{k} = {res['base'].get(k)} (identity arm, holds on both)")
        bad += 0 if same else 1
not_ex = "NOT EXERCISED: B-M1 live NOT-RESIDENT (fail word injected on CPU only), B-M7 KV exhaustion, B-M13 captured, C6 grejected on a served grammar, B-M4, B-M10, spark B-M14"
open(f"{out}/verdict.txt", "w").write("\n".join(verdict) + "\n" + not_ex + "\n")
print("\n".join(verdict)); print(not_ex)
if bad: print(f"FAIL fsm-wire-gate: {bad} candidate check(s) failed, see {out}"); sys.exit(1)
if void: print(f"FAIL fsm-wire-gate VOID: {void} base symptom(s) not seen, the probe cannot fail on the old engine"); sys.exit(1)
if explore == "1": print("UNVERIFIED fsm-wire-gate: EXPLORE=1 skipped the preflight check, not a gate result"); sys.exit(3)
print(f"PASS fsm-wire-gate: {len(verdict)} checks, receipts in {out}")
PY
