#!/usr/bin/env python3
"""P4 helper (called by parity-llama.sh): answer parity of baro-serve against llama-server on the runner's own requests.

  gen     --llama URL --out DIR [--states DIR] [--n 20 | --only s14,s05]   drive street-sim's ask_openai (fake game) with llama-server (--parallel 1)
                                                          as the model; every post becomes a unit: the exact greedy body bytes,
                                                          prompt ids, ONE unconstrained /completion generation (ids, text, calls
                                                          parsed from the text; llama's chat path adds a lazy tool grammar and is
                                                          not reproducible against it). Round 2+ bodies carry that reply.
  burst   --url URL --units DIR --out DIR --kind llama|ours [--ids]   the same body bytes, one round at a time, all states of
                                                          the round at once (lockstep); records call, ids, http code.
  fctl    --url URL --units DIR --out DIR [--workers 20 --budget 1500]                forced control: llama --parallel 20, one-token argmax per position of the p1 ids.
  negctl  --url URL --units DIR --tmax T                  the non-200 counter's negative control (expects HTTP 400).
  forcejl --units DIR --src ref|DIR2 --out FILE           engine wire lines with "force" = those ids (one resident process).
  compare --units DIR --ctl DIR --ours DIR --forced LOG --fctl DIR [--own LOG] [--own1 DIR] [--full]   frozen bar, verdict, exit 0/1/3.

Bar (docs/plan-ailab-server.md, frozen before any run), ONE metric on both sides: a unit's greedy agreement is the matching prefix
of an arm's ids over llama --parallel 1's ids, divided by the length of llama's ids. Control = llama --parallel 20 vs --parallel 1,
candidate = baro-serve N=20 vs llama --parallel 1. Pass: tool name equal on >= control count - 1; candidate mean >= control mean - 1 pp
and candidate min >= control min - 3 pp. Teacher-forced agreement of llama's ids through our single-session engine is REPORTED (it
answers a different question: per-position argmax, re-anchored after a flip); since 2026-11-07 it IS a required bar term against the forced control (llama --parallel 20, same per-position metric), added after QUICK=2 and before the FULL run. Greedy id equality is reported too.
(First version of this bar compared ours forced against a prefix control: two metrics, an early flip lowers the prefix, so the control
sat lower and the bar was easier for us. Driver caught it before any run.)
"""
import json, os, re, sys, threading, time, urllib.error, urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture as C  # noqa: E402


def post(url, d, raw=False, timeout=900):
    """Real HTTP even while capture_state has faked urllib.request.urlopen (the responder runs inside the fake)."""
    r = urllib.request.Request(url, d if raw else json.dumps(d).encode(), {"Content-Type": "application/json"})
    return json.load(C.REAL_URLOPEN(r, timeout=timeout))

MEAN_MARGIN, MIN_MARGIN, NAME_MARGIN = 1.0, 3.0, 1


def norm_call(msg):
    """[(tool name, parsed args)] of an assistant message."""
    out = []
    for c in (msg or {}).get("tool_calls") or []:
        try:
            a = json.loads(c["function"].get("arguments") or "{}")
        except ValueError:
            a = {"_unparsed": c["function"].get("arguments")}
        out.append([c["function"]["name"], a])
    return out


CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    """[(name, args)] from Qwen3.5 XML tool-call text, like baro-serve's parser: a parameter value is JSON when it parses, else the string.
    Unknown tool names are kept (llama's chat grammar would hide them; neither /completion nor baro-serve constrains)."""
    out = []
    for name, body in CALL.findall(text):
        args = {}
        for k, v in PARAM.findall(body):
            try:
                args[k] = json.loads(v)
            except ValueError:
                args[k] = v
        out.append([name, args])
    return out


def message_of(text):
    """The assistant message of one unconstrained generation: prose before the first call, then its parsed calls."""
    calls = parse_calls(text)
    msg = {"role": "assistant", "content": text.split("<tool_call>")[0]}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{i + 1}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return msg


def prompt_ids(llama, raw):
    b = json.loads(raw)
    p = post(llama + "/apply-template", {k: b[k] for k in ("messages", "tools", "chat_template_kwargs")})["prompt"]
    return post(llama + "/tokenize", {"content": p, "add_special": True, "parse_special": True})["tokens"]


def completion_ids(llama, ids):
    j = post(llama + "/completion", {"prompt": ids, "n_predict": 16000, "temperature": 0, "top_k": 1, "return_tokens": True})
    return j["tokens"], j.get("content", "")


def chat(url, raw):
    j = post(url + "/v1/chat/completions", raw, raw=True)
    return j["choices"][0], j.get("usage", {}).get("completion_tokens"), (j.get("timings") or {}).get("cached")


def greedy_body(body):
    """The bytes both arms receive: max_tokens floored to 16000 and temperature 0. The runner sends NO temperature, so llama-server
    samples at its own default (0.8, top_k 40, top_p 0.95) and baro-serve decodes greedily: without this the two arms, and two calls
    to llama itself, answer differently (QUICK=2 first run: chat and /completion on one prompt disagreed on 4 of 6 units)."""
    b = json.loads(C.floor_body(json.dumps(body).encode()))
    b["temperature"] = 0
    return json.dumps(b).encode()


def cmd_gen(llama, out, states, n, only=None):
    items = C.load_states(states)
    items = [i for i in items if i[0] in only.split(",")] if only else items[:n]
    os.makedirs(f"{out}/units", exist_ok=True)
    for item in items:
        def responder(k, body, sid=item[0]):
            raw = greedy_body(body)
            ids = prompt_ids(llama, raw)
            ref, text = completion_ids(llama, ids)
            if not ref:
                sys.exit(f"FAIL gen: {sid} k={k}: /completion returned no ids")
            uid = f"{sid}-r{k + 1}"
            open(f"{out}/units/{uid}.body.json", "wb").write(raw)
            json.dump({"prompt_ids": ids, "ref_ids": ref, "text": text, "call": parse_calls(text)}, open(f"{out}/units/{uid}.ref.json", "w"))
            return message_of(text)
        C.capture_state(item, responder)
    units = sorted(f[:-9] for f in os.listdir(f"{out}/units") if f.endswith(".ref.json"))
    nocall = [u for u in units if not json.load(open(f"{out}/units/{u}.ref.json"))["call"]]
    print(f"gen: {len(items)} states, {len(units)} units (requests), one /completion generation per unit (ids, text and calls from it); units without a tool call: {len(nocall)}")


def cmd_burst(url, units, out, kind, ids):
    os.makedirs(out, exist_ok=True)
    uids = sorted(f[:-10] for f in os.listdir(units) if f.endswith(".body.json"))
    by_round = {}
    for u in uids:
        by_round.setdefault(int(u.split("-r")[1]), []).append(u)
    codes = []

    def one(u):
        raw = open(f"{units}/{u}.body.json", "rb").read()
        rec, t0 = {"http": 0, "call": [], "ids": []}, time.time()
        try:
            if kind == "llama":  # the control repeats the reference's own path: one /completion, calls parsed from its text
                cids, text = completion_ids(url, json.load(open(f"{units}/{u}.ref.json"))["prompt_ids"])
                rec.update(http=200, call=parse_calls(text), text=text, ids=cids)
            else:
                ch, usage, cached = chat(url, raw)
                rec.update(http=200, call=norm_call(ch["message"]), text=ch["message"].get("content") or "", ids=ch.get("tokens") or [], cached=cached)
        except Exception as e:  # recorded, the compare step names it
            rec["error"] = str(e)[:200]
            rec["http"] = getattr(e, "code", 0)
            try:  # the response body says WHICH 429 / 400 it was (kv pool full, queue full, --max-inflight, exceed_context)
                rec["body"] = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
        rec["secs"] = round(time.time() - t0, 2)
        json.dump(rec, open(f"{out}/{u}.json", "w"))
        codes.append(rec["http"])
    for k in sorted(by_round):
        ts = [threading.Thread(target=one, args=(u,)) for u in by_round[k]]
        [t.start() for t in ts]
        [t.join() for t in ts]
    bad = sum(1 for c in codes if c != 200)
    print(f"burst {kind}: {len(codes)} requests, rounds {sorted(by_round)}, non-200: {bad}")
    if bad:
        bodies = sorted({json.load(open(f"{out}/{u}.json")).get("body", "")[:160] for u in uids if os.path.exists(f"{out}/{u}.json")} - {""})
        print(f"non-200 response bodies: {bodies}")
        sys.exit(f"FAIL burst {kind}: {bad} requests not HTTP 200 (see {out}/*.json)")


def cmd_fctl(url, units, out, workers=20, budget=1500):
    """Forced control (plan, added 2026-11-07): llama --parallel 20 scored at each position of llama --parallel 1's ids, the same
    per-position metric as our teacher-forced agreement: /completion on prompt + ids[:i], one token, temperature 0, cache_prompt on;
    agree when that token is ids[i]. A pool of `workers` threads (= the slots) takes units from a queue; worker k pins id_slot k, so
    each stream keeps its cached prefix (73 threads on 20 slots thrashed it: 30 positions a minute, FULL run 2026-11-07 cancelled).
    FAIL fast when `budget` seconds pass. Measured on the card (FULL 2026-11-07): the control server's total rate is about 5 to 9 one-token
    requests a second however many workers (QUICK 8.3 with 6 workers, FULL 5.2 with 20), so 4,737 positions need about 15 min; budget 1500 s."""
    import queue
    os.makedirs(out, exist_ok=True)
    uids = sorted(f[:-9] for f in os.listdir(units) if f.endswith(".ref.json"))
    q, errs, done_pos, t0 = queue.Queue(), [], [0], time.time()
    for u in uids:
        q.put(u)
    total = sum(len(json.load(open(f"{units}/{u}.ref.json"))["ref_ids"]) for u in uids)

    def work(slot):
        while not errs:
            try:
                u = q.get_nowait()
            except queue.Empty:
                return
            r = json.load(open(f"{units}/{u}.ref.json"))
            ids, agree, flips = r["ref_ids"], 0, []
            try:
                for i in range(len(ids)):
                    if time.time() - t0 > budget:
                        raise RuntimeError(f"over the {budget} s budget at {done_pos[0]} of {total} positions")
                    j = post(url + "/completion", {"prompt": r["prompt_ids"] + ids[:i], "n_predict": 1, "temperature": 0, "top_k": 1,
                                                   "return_tokens": True, "cache_prompt": True, "id_slot": slot})
                    tok = (j.get("tokens") or [None])[0]
                    done_pos[0] += 1
                    if tok == ids[i]:
                        agree += 1
                    elif len(flips) < 5:
                        flips.append([i, ids[i], tok])
            except Exception as e:
                errs.append(f"{u}: {str(e)[:120]}")
                return
            json.dump({"agree": agree, "n": len(ids), "flips": flips}, open(f"{out}/{u}.json", "w"))
    ts = [threading.Thread(target=work, args=(k,)) for k in range(min(workers, len(uids)))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errs:
        sys.exit("FAIL fctl: " + errs[0])
    tot = [json.load(open(f"{out}/{u}.json")) for u in uids]
    print(f"fctl: {len(uids)} units, {sum(t['n'] for t in tot)} positions, agree {sum(t['agree'] for t in tot)}, {time.time() - t0:.0f} s, {workers} workers pinned to slots")


def cmd_negctl(url, units, tmax):
    """Instrument self-check, no engine work: a body padded past tmax - 16000 prompt tokens must be answered 400
    exceed_context_size_error by baro-serve; a 200 means the non-200 counter cannot be trusted."""
    uid = sorted(f[:-10] for f in os.listdir(units) if f.endswith(".body.json"))[0]
    b = json.load(open(f"{units}/{uid}.body.json"))
    b["messages"][-1]["content"] += " filler" * (tmax // 2)
    try:
        post(url + "/v1/chat/completions", json.dumps(b).encode(), raw=True)
        sys.exit("FAIL negctl: an over-length body was answered 200, the counter is blind")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        if e.code != 400 or "exceed_context_size_error" not in body:
            sys.exit(f"FAIL negctl: HTTP {e.code} {body}")
    print(f"negctl: over-length body (tmax {tmax}) -> HTTP 400 exceed_context_size_error, as required")


def cmd_selftest():
    """CPU self-test of the burst driver's non-200 path: a local server answers 429 with a JSON body; the record must carry code and body."""
    import http.server, tempfile
    body = '{"error":{"code":429,"message":"kv pool full: selftest"}}'

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/units")
        open(f"{d}/units/s01-r1.body.json", "wb").write(b'{"messages":[]}')
        try:
            cmd_burst(f"http://127.0.0.1:{srv.server_port}", f"{d}/units", f"{d}/out", "ours", False)
            sys.exit("FAIL selftest: a 429 did not fail the burst")
        except SystemExit as e:
            if "not HTTP 200" not in str(e):
                raise
        r = json.load(open(f"{d}/out/s01-r1.json"))
        if r["http"] != 429 or "kv pool full: selftest" not in r.get("body", ""):
            sys.exit(f"FAIL selftest: record is {r}")
    srv.shutdown()
    print("selftest: a 429 fails the burst and its record carries the code and the response body")
    selftest_ownid()


def selftest_ownid():
    """ownid-compare on synthetic records: an equal pair, a mismatch at position 2, and a pair at different restore points must read as such."""
    import contextlib, io, tempfile
    with tempfile.TemporaryDirectory() as d:
        for arm, recs in {"x": {"s01-r1": ([1, 2, 3, 4], 3569), "s01-r2": ([1, 2, 3, 4], 3569), "s02-r1": ([1, 2], 3569)},
                          "y": {"s01-r1": ([1, 2, 3, 4], 3569), "s01-r2": ([1, 2, 9, 4], 3569), "s02-r1": ([1, 2], 0)}}.items():
            os.makedirs(f"{d}/{arm}")
            for u, (ids, c) in recs.items():
                json.dump({"http": 200, "ids": ids, "cached": c, "call": []}, open(f"{d}/{arm}/{u}.json", "w"))
        open(f"{d}/scored.txt", "w").write("s01-r1\ns01-r2\ns02-r1\n")
        open(f"{d}/roles.txt", "w").write("s01-r1\tcontrol\ns01-r2\tmismatch\ns02-r1\tcontrol\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_ownid_compare(f"{d}/scored.txt", [f"a1={d}/x", f"b={d}/y"])
        t = buf.getvalue()
        row = lambda u, role, r: f"   {u:8s} {role:9s} {r}"  # noqa: E731  the output's own row format
        want = ["a1 vs b: 1 mismatches of 2 comparable at equal restore point, 1 not comparable", row("s01-r1", "control", "equal"),
                row("s01-r2", "mismatch", "MISMATCH at position 2"), row("s02-r1", "control", "restore points differ")]
        miss = [w for w in want if w not in t]
        if miss:
            sys.exit(f"FAIL selftest ownid: missing {miss} in:\n{t}")
    print("selftest: ownid-compare reads an equal pair, a mismatch at position 2 and a restore-point difference correctly")


def _rec(d, u):
    try:
        return json.load(open(f"{d}/{u}.json"))
    except OSError:
        return None


def _firstdiff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else -1)


def cmd_ownid_select(old20, old1, units, out):
    """P4c selection, from the P4 FULL records only: the units whose ids differ between ours N=20 and the serial arm at an EQUAL restore point
    (timings.cached), plus 5 controls that are equal there, chosen blind: the first equal unit of each round r1..r4 in unit-id order, then the
    next equal unit overall. The subset dir holds ALL rounds of the states involved (earlier rounds seed the checkpoints that give the same
    restore points); only the 19 are scored."""
    uids = sorted(f[:-9] for f in os.listdir(units) if f.endswith(".ref.json"))
    mism, eq = [], []
    for u in uids:
        a, b = _rec(old20, u), _rec(old1, u)
        if not a or not b or a.get("cached") != b.get("cached") or not a["ids"] or not b["ids"]:
            continue
        (mism if a["ids"] != b["ids"] else eq).append(u)
    ctl = []
    for rnd in "1234":
        ctl += [next((u for u in eq if u.endswith("-r" + rnd) and u not in ctl), None)]
    ctl = [u for u in ctl if u]
    ctl += [u for u in eq if u not in ctl][:5 - len(ctl)]
    scored = sorted(mism + ctl)
    states = sorted({u.split("-r")[0] for u in scored})
    os.makedirs(f"{out}/units", exist_ok=True)
    for u in uids:
        if u.split("-r")[0] in states:
            for suf in (".body.json", ".ref.json"):
                open(f"{out}/units/{u}{suf}", "wb").write(open(f"{units}/{u}{suf}", "rb").read())
    open(f"{out}/scored.txt", "w").write("\n".join(scored) + "\n")
    open(f"{out}/roles.txt", "w").write("\n".join(f"{u}\t{'mismatch' if u in mism else 'control'}" for u in scored) + "\n")
    print(f"ownid-select: {len(mism)} mismatching, {len(ctl)} control units scored, {len(states)} states, "
          f"{len(os.listdir(out + '/units')) // 2} units in the subset -> {out}")


def cmd_ownid_compare(scored, arms):
    """arms: name=DIR ... Pairs (the driver's verdict lines): a1 vs a2 (same binary twice: batch nondeterminism), a1 vs b (HEAD seqs vs HEAD
    single: path difference), old1 vs b (stale d18eba5 serial engine vs HEAD single). Comparable = equal timings.cached in both arms."""
    uids = [u for u in open(scored).read().split() if u]
    A = {n: d for n, d in (x.split("=", 1) for x in arms)}
    roles = dict(l.split("\t") for l in open(os.path.join(os.path.dirname(scored), "roles.txt")).read().splitlines())

    def pair(x, y):
        rows, mism, comp = [], 0, 0
        for u in uids:
            a, b = _rec(A[x], u), _rec(A[y], u)
            if not a or not b or not a["ids"] or not b["ids"]:
                rows.append((u, "no record"))
                continue
            if a.get("cached") != b.get("cached"):
                rows.append((u, f"restore points differ ({a.get('cached')} vs {b.get('cached')})"))
                continue
            comp += 1
            d = _firstdiff(a["ids"], b["ids"])
            mism += d >= 0
            rows.append((u, "equal" if d < 0 else f"MISMATCH at position {d} (cached {a.get('cached')}, len {len(a['ids'])} vs {len(b['ids'])})"))
        return mism, comp, rows
    out = []
    for x, y, what in (("a1", "a2", "HEAD seqs N=20 twice, same units (batch nondeterminism)"),
                       ("a1", "b", "HEAD seqs N=20 vs HEAD single-session (path difference)"),
                       ("old1", "b", "stale d18eba5 serial engine-main vs HEAD single-session"),
                       ("S", "Q", "P4d wire, one request at a time, pinned creator: HEAD single vs HEAD seqs (C0)"),
                       ("S", "Q0", "P4d wire: single vs seqs with BARO_SEQS_ROWS=0 (C1)"),
                       ("S", "M0", "P4d wire: single megakernel on vs BARO_MEGA=0 (C2 control)"),
                       ("Q", "Q0", "P4d wire: seqs with rows kernels vs without"),
                       ("b", "S", "single path: P4c burst vs P4d wire (does the driver itself change anything)"),
                       ("a1", "Q", "seqs path: P4c burst vs P4d wire")):
        if x in A and y in A:
            m, c, rows = pair(x, y)
            print(f"{x} vs {y}: {m} mismatches of {c} comparable at equal restore point, {len(uids) - c} not comparable  [{what}]")
            out.append((x, y, rows))
    for x, y, rows in out:
        print(f"-- {x} vs {y} per unit (role from the P4 FULL selection)")
        for u, r in rows:
            print(f"   {u:8s} {roles.get(u, '?'):9s} {r}")


def cmd_p4b_select(units, out):
    """P4b: the 6 units, fixed BEFORE the oracle runs and blind to every agreement number (design file, 'Decision rule'): among units with >= 30
    forced positions, the shortest prompt of EACH round r1..r4, then the 2 next shortest overall; ties by unit id."""
    rows = []
    for f in sorted(os.listdir(units)):
        if f.endswith(".ref.json"):
            r = json.load(open(f"{units}/{f}"))
            if len(r["ref_ids"]) >= 30:
                rows.append((len(r["prompt_ids"]), f[:-9], len(r["ref_ids"])))
    rows.sort()
    pick = []
    for rnd in "1234":
        pick += [next((x for x in rows if x[1].endswith("-r" + rnd) and x not in pick), None)]
    pick = [x for x in pick if x]
    pick += [x for x in rows if x not in pick][: 6 - len(pick)]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("".join(f"{u}\t{P}\t{n}\n" for P, u, n in pick))
    print("p4b-select:", ", ".join(f"{u} (prompt {P}, {n} forced)" for P, u, n in pick))


def cmd_forcejl(units, src, out):
    uids = sorted(f[:-9] for f in os.listdir(units) if f.endswith(".ref.json"))
    with open(out, "w") as f:
        for i, u in enumerate(uids):
            ref = json.load(open(f"{units}/{u}.ref.json"))
            ids = ref["ref_ids"] if src == "ref" else json.load(open(f"{src}/{u}.json"))["ids"]
            if not ids and os.environ.get("DRY") == "1":
                ids = [1]  # dry run: the stub engine answers nothing; never in a real run
            if not ids:
                sys.exit(f"FAIL forcejl: {u} has no ids in {src}")
            f.write(json.dumps({"id": i, "prompt": ref["prompt_ids"], "n": len(ids), "spec": False, "stop": [], "ckpt": [], "force": ids}) + "\n")
    print(f"forcejl: {len(uids)} lines -> {out}")


def agreement(log, n):
    rows = [(int(a), int(b)) for a, b in re.findall(r"forced agreement: (\d+) / (\d+)", open(log).read())]
    if len(rows) != n:
        sys.exit(f"FAIL compare VOID: {len(rows)} agreement lines in {log} for {n} requests")
    return [100.0 * a / b if b else 0.0 for a, b in rows]


def stats(x):
    return sum(x) / len(x), min(x)


def cmd_compare(units, ctl, ours, forced, own, own1, fctl, full):
    uids = sorted(f[:-9] for f in os.listdir(units) if f.endswith(".ref.json"))
    ref = {u: json.load(open(f"{units}/{u}.ref.json")) for u in uids}
    c = {u: json.load(open(f"{ctl}/{u}.json")) for u in uids}
    o = {u: json.load(open(f"{ours}/{u}.json")) for u in uids}
    names = lambda r: [x[0] for x in r["call"]]  # noqa: E731
    cn = sum(names(c[u]) == names(ref[u]) for u in uids)
    on = sum(names(o[u]) == names(ref[u]) for u in uids)
    oa = sum(o[u]["call"] == ref[u]["call"] for u in uids)
    pre = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))  # noqa: E731
    ctl_agree = [100.0 * pre(c[u]["ids"], ref[u]["ref_ids"]) / max(1, len(ref[u]["ref_ids"])) for u in uids]
    ours_agree = [100.0 * pre(o[u]["ids"], ref[u]["ref_ids"]) / max(1, len(ref[u]["ref_ids"])) for u in uids]
    fa = agreement(forced, len(uids))
    fc = []
    for u in uids:
        t = json.load(open(f"{fctl}/{u}.json"))
        fc.append(100.0 * t["agree"] / max(1, t["n"]))
    (cm, cmin), (om, omin), (fm, fmin), (fcm, fcmin) = stats(ctl_agree), stats(ours_agree), stats(fa), stats(fc)
    eq_text = sum(o[u]["ids"] == ref[u]["ref_ids"] for u in uids if o[u]["ids"])
    lines = [f"units {len(uids)} (states x rounds), rounds {sorted({int(u.split('-r')[1]) for u in uids})}",
             f"tool name equal: control {cn}/{len(uids)}, ours {on}/{len(uids)} (bar: ours >= control - {NAME_MARGIN}); name+args equal ours {oa}/{len(uids)}",
             f"greedy prefix match vs llama --parallel 1 ids: ours N=20 mean {om:.2f}% min {omin:.2f}%; control (llama --parallel 20) mean {cm:.2f}% min {cmin:.2f}%",
             f"bar: ours mean >= {cm - MEAN_MARGIN:.2f}%, min >= {cmin - MIN_MARGIN:.2f}%; whole-answer ids equal ours {eq_text}/{len(uids)} (reported)",
             f"teacher-forced agreement of llama's ids, per position: ours (single-session engine) mean {fm:.2f}% min {fmin:.2f}%; control (llama --parallel 20, one-token argmax) mean {fcm:.2f}% min {fcmin:.2f}%",
             f"bar (added 2026-11-07, required): forced mean >= {fcm - MEAN_MARGIN:.2f}%, min >= {fcmin - MIN_MARGIN:.2f}%, AND the prefix term"]
    if own:
        oc = agreement(own, len(uids))
        lines.append(f"instrument control: ours multi-session ids forced through the single-session engine: mean {sum(oc) / len(oc):.2f}% min {min(oc):.2f}% (reads 100% unless restore points differ)")
    bad = []
    n200 = sum(1 for d in (c, o) for r in d.values() if r["http"] != 200)
    lines.append(f"non-200 posts (llama control, ours N=20): {n200} of {2 * len(uids)} (any non-200 is a FAIL: the runner re-asks without tools on an HTTPError)")
    if n200:
        bad.append(f"{n200} non-200 posts")
    if own1:  # own identity: ours at N=20 vs ours at N=1, same prompt, equal restore point (timings.cached) only
        o1 = {u: json.load(open(f"{own1}/{u}.json")) for u in uids}
        same = [u for u in uids if o[u].get("cached") == o1[u].get("cached")]
        mism = [u for u in same if o[u]["ids"] != o1[u]["ids"]]
        lines.append(f"own identity ours N=20 vs N=1: {len(same) - len(mism)}/{len(same)} equal at equal restore point, {len(uids) - len(same)} at different restore points (no reference), mismatches {mism[:5]}")
        if mism or not same:
            bad.append(f"own identity: {len(mism)} mismatches of {len(same)} comparable")
    if on < cn - NAME_MARGIN:
        bad.append(f"tool name {on} < control {cn} - {NAME_MARGIN}")
    if om < cm - MEAN_MARGIN:
        bad.append(f"mean {om:.2f} < {cm - MEAN_MARGIN:.2f}")
    if omin < cmin - MIN_MARGIN:
        bad.append(f"min {omin:.2f} < {cmin - MIN_MARGIN:.2f}")
    if fm < fcm - MEAN_MARGIN:
        bad.append(f"forced mean {fm:.2f} < {fcm - MEAN_MARGIN:.2f}")
    if fmin < fcmin - MIN_MARGIN:
        bad.append(f"forced min {fmin:.2f} < {fcmin - MIN_MARGIN:.2f}")
    print("\n".join(lines))
    if bad:
        sys.exit("FAIL answer parity: " + "; ".join(bad))
    if not full:
        print("UNVERIFIED answer parity: quick or dry run, bar met on the subset only")
        sys.exit(3)
    print("PASS answer parity: bar met on the full set")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd, opt, i = a[0], {}, 1
    while i < len(a):
        if a[i] in ("--ids", "--full"):
            opt[a[i][2:]], i = True, i + 1
        else:
            opt[a[i][2:]], i = a[i + 1], i + 2
    if cmd == "gen":
        cmd_gen(opt["llama"], opt["out"], opt.get("states", f"{HERE}/states"), int(opt.get("n", 20)), opt.get("only"))
    elif cmd == "burst":
        cmd_burst(opt["url"], opt["units"], opt["out"], opt["kind"], bool(opt.get("ids")))
    elif cmd == "p4b-select":
        cmd_p4b_select(opt["units"], opt["out"])
    elif cmd == "ownid-select":
        cmd_ownid_select(opt["old20"], opt["old1"], opt["units"], opt["out"])
    elif cmd == "ownid-compare":
        cmd_ownid_compare(opt["scored"], opt["arms"].split(","))
    elif cmd == "selftest":
        cmd_selftest()
    elif cmd == "fctl":
        cmd_fctl(opt["url"], opt["units"], opt["out"], int(opt.get("workers", 20)), int(opt.get("budget", 1500)))
    elif cmd == "negctl":
        cmd_negctl(opt["url"], opt["units"], int(opt["tmax"]))
    elif cmd == "forcejl":
        cmd_forcejl(opt["units"], opt["src"], opt["out"])
    elif cmd == "compare":
        cmd_compare(opt["units"], opt["ctl"], opt["ours"], opt["forced"], opt.get("own"), opt.get("own1"), opt["fctl"], bool(opt.get("full")))
    else:
        sys.exit(f"FAIL args: unknown command {cmd}")


if __name__ == "__main__":
    main()
