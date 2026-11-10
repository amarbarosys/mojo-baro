#!/usr/bin/env python3
# usage: bench/a2-arm.py ref|cand REQS.jsonl RECDIR STATEDIR MIN_PCT FAILFAST [REFDIR] -- ENGINE_CMD...
# One arm of bench/a2-gate.sh, resumable per prompt. REQS lines carry id, set (L8192 ...), doclen and
# the request. Every prompt is independent: one doc-only request per set saves STATEDIR/<set>.st
# ("state_save", doc tokens, n=1) and each prompt loads it ("state_load"), so no prompt depends on an
# earlier prompt of the run. A finished prompt is written to RECDIR/<id>-<sha>.json the moment its
# done line arrives (sha = the request line); a rerun skips every prompt with a record, so a kill at
# any point loses at most the prompt in flight. cand mode adds "force": the REFDIR record's ids, and
# with FAILFAST=1 stops at the first prompt under MIN_PCT. Engine stdout goes to RECDIR/engine.log.
import hashlib, json, pathlib, re, subprocess, sys, threading

mode, reqs, recdir, stdir, minpct, ff = sys.argv[1:7]
rest = sys.argv[7:]; refdir = None
if rest[0] != "--": refdir = pathlib.Path(rest[0]); rest = rest[1:]
cmd = rest[1:]; minpct = float(minpct); ff = ff == "1"
recdir = pathlib.Path(recdir); stdir = pathlib.Path(stdir); recdir.mkdir(parents=True, exist_ok=True); stdir.mkdir(parents=True, exist_ok=True)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
def rec_path(d, r): return d / f"{r['id']}-{sha(json.dumps(r['req'], sort_keys=True))}.json"

todo, lines, docs = [], [], {}
for l in open(reqs):
    r = json.loads(l)
    if rec_path(recdir, r).exists(): continue
    req = dict(r["req"])
    if mode == "cand":
        rp = rec_path(refdir, r)
        if not rp.exists(): sys.exit(f"FAIL a2-arm: no reference record for request {r['id']} ({rp})")
        req["force"] = json.loads(rp.read_text())["ids"]
    st = stdir / f"{r['set']}.st"
    if r["set"] not in docs and not st.exists():
        docs[r["set"]] = st
        lines.append({"id": int(r["id"]) // 100 * 100, "prompt": r["req"]["prompt"][: r["doclen"]], "n": 1, "spec": False, "temperature": 0, "state_save": str(st.resolve())})
    req["state_load"] = str(st.resolve())
    todo.append(r); lines.append(req)
if not todo: print(f"{mode}: all prompts already recorded in {recdir}"); sys.exit(0)
print(f"{mode}: {len(todo)} prompts to run ({len(docs)} doc states to save), records in {recdir}", flush=True)

byid = {int(r["id"]): r for r in todo}
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(recdir / "engine.err", "a"), text=True)
def feed():  # own thread: the engine's stdout must drain while ~10 MB of prompts go in
    try:
        for x in lines: p.stdin.write(json.dumps(x) + "\n"); p.stdin.flush()
        p.stdin.close()
    except (BrokenPipeError, OSError): pass  # engine died: the records say how far it got
threading.Thread(target=feed, daemon=True).start()
toks, agree, ready, failed = {}, None, False, None
with open(recdir / "engine.log", "a") as log, open(recdir / "receipts.txt", "a") as rc:
    for line in p.stdout:
        log.write(line)
        if re.match(r"^(BARO_[A-Z_]+):", line) or '"ready"' in line: rc.write(line); ready = ready or '"ready":true' in line
        m = re.match(r"forced agreement: (\d+) / (\d+)", line)
        if m: agree = (int(m[1]), int(m[2])); continue
        if not line.startswith("{"): continue
        try: d = json.loads(line)
        except ValueError: continue
        i = d.get("id")
        if "error" in d: failed = f"engine refused request {i}: {d['error']}"; break
        if "tok" in d: toks.setdefault(i, []).append(d["tok"]); continue
        if not d.get("done") or i not in byid: continue
        r = byid[i]; out = {k: d.get(k, 0) for k in ("cached", "tok_s", "prefill_s", "restore_s")}
        if mode == "ref": out["ids"] = toks.pop(i, [])
        else:
            if agree is None: failed = f"request {i}: done line without a forced-agreement line"; break
            out["agree"], out["checked"] = agree; agree = None
        tmp = rec_path(recdir, r).with_suffix(".tmp"); tmp.write_text(json.dumps(out)); tmp.replace(rec_path(recdir, r))
        pct = 100.0 * out.get("agree", 1) / max(out.get("checked", 1), 1)
        print(f"  {r['set']} {r['name']}: cached {out['cached']} tok/s {out['tok_s']:.2f}" + (f" agreement {out['agree']}/{out['checked']} = {pct:.1f}%" if mode == "cand" else f" ids {len(out['ids'])}"), flush=True)
        if mode == "cand" and ff and pct < minpct: failed = f"failfast: {r['set']} {r['name']} {pct:.1f}% < {minpct}%"; break
if p.poll() is None: p.kill()
p.wait()
if not ready: sys.exit(f"FAIL a2-arm {mode}: engine never printed its ready line, see {recdir}/engine.err")
if failed: sys.exit(f"FAIL a2-arm {mode}: {failed}")
left = [r["id"] for r in todo if not rec_path(recdir, r).exists()]
if left: sys.exit(f"FAIL a2-arm {mode}: {len(left)} prompts without a record (engine exit), rerun to resume")
