#!/usr/bin/env python3
"""P0 oracle: record every request body street-sim's own ask_openai posts, for the ailab server gates.

  bench/ailab/capture.py OUT_DIR [--states DIR] [--script state,actions,hold] [--selfcheck]

ask_openai is called unmodified (street-sim is imported read-only: no bytecode written there). Only two seams are
replaced: urllib.request.urlopen (records the exact bytes, answers from a `responder`) and ailab_run.bridge (canned
game replies, no socket). One body per file: OUT_DIR/sNN-rK.json, NN = state, K = round (1 = first post).
MANIFEST.tsv: id, round, md5, bytes. SET.md5 = md5 of MANIFEST.tsv.

Importable by step.py and parity-llama.sh: capture_state(item, responder), install_bridge(states_by_port),
floor_body(bytes) (max_tokens -> 16000, the repo floor, every other byte unchanged), load_states(dir).
Stand-ins, not read from the game: `jobs` [], `task` {}, `gt`; the diary lines and one handbook line come from the
existing bench/ailab/states/ user text. Bodies keep the runner's own max_tokens 200: the floor is applied at send time.
"""
import glob, hashlib, io, json, os, re, shutil, sys, urllib.request

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STREET = os.environ.get("STREET_SIM", os.path.expanduser("~/Projects/street-sim"))
sys.path.insert(0, f"{STREET}/tools")
import ailab_run as R  # noqa: E402  read only

REAL_URLOPEN = urllib.request.urlopen  # capture_state swaps urllib.request.urlopen for the fake: a responder that talks HTTP uses this
HOMES = f"{REPO}/.work/ailab/capture-home"
ACTION_NAMES = ["move_forward", "move_back", "move_left", "move_right", "interact", "jump", "sprint"]
USER = re.compile(r"You are (.+?)\. Backstory: (\w+)\.\nState: (\{.*\})\nEvents since.*?Your diary, last lines:\n- (.*)\n- (.*)\nChoose", re.S)


def load_states(d):
    """-> [(id, name, backstory, state dict, [diary lines])] from bench/ailab states dir, sorted by number."""
    out = []
    for f in sorted(glob.glob(f"{d}/s*.json"), key=lambda p: int(re.search(r"s(\d+)\.json", p).group(1))):
        m = USER.search(json.load(open(f))["messages"][1]["content"])
        if not m:
            raise SystemExit(f"FAIL load_states: {f} user text does not match")
        n = int(re.search(r"s(\d+)\.json", f).group(1))
        out.append((f"s{n:02d}", m.group(1), m.group(2), json.loads(m.group(3)), [m.group(4), m.group(5)]))
    if not out:
        raise SystemExit(f"FAIL load_states: no s*.json in {d}")
    return out


def runner_state(i, st, diary):
    s = dict(st)
    s.update(jobs=[], task={}, diary=diary[-4:], gt=round(100.0 + i * 0.5, 3), step=i)
    return s


def fake_home(h, diary, handbook):
    shutil.rmtree(h, ignore_errors=True)
    os.makedirs(f"{h}/asks")
    with open(f"{h}/diary.jsonl", "w") as f:
        f.write("".join(json.dumps({"step": 0, "text": t}) + "\n" for t in diary))
    with open(f"{h}/handbook.md", "w") as f:
        f.write("".join("- " + t + "\n" for t in handbook))


def install_bridge(states_by_port):
    """Replace ailab_run.bridge: state -> that body's state, actions -> a name list, any move -> ok. Returns the call log."""
    log = []

    def bridge(port, cmd, **a):
        r = dict(states_by_port[port]) if cmd == "state" else {"actions": ACTION_NAMES} if cmd == "actions" else {"ok": True}
        log.append((port, cmd, a, r))
        return r
    R.bridge = bridge
    return log


def canned(script):
    """responder: round k (0-based) -> assistant message calling script[k]; the last entry ends the ask (an act)."""
    calls = {"state": {}, "actions": {}, "hold": {"action": "move_forward", "seconds": 2.0}, "press": {"action": "interact"},
             "diary": {"text": "kept moving"}}

    def responder(k, body):
        name = script[min(k, len(script) - 1)]
        return {"role": "assistant", "content": "" if k < len(script) - 1 else "moving on",
                "tool_calls": [{"id": f"call_{k + 1}", "type": "function", "function": {"name": name, "arguments": json.dumps(calls[name])}}]}
    return responder


class Fake:
    def __init__(self, responder):
        self.responder, self.posts = responder, []

    def urlopen(self, req, timeout=None):
        k = len(self.posts)
        body = json.loads(req.data)
        self.posts.append({"url": req.full_url, "data": req.data, "ctype": req.get_header("Content-type"), "timeout": timeout})
        msg = self.responder(k, body)
        self.posts[-1]["reply"] = msg
        return io.BytesIO(json.dumps({"choices": [{"message": msg}]}).encode())


def capture_state(item, responder, port=7910):
    """Run ask_openai for one state item with urlopen faked; -> (posts, bridge log, said). Posts carry the exact bytes."""
    sid, name, back, st, diary = item
    i = int(sid[1:])
    state = runner_state(i, st, diary)
    h = f"{HOMES}/{sid}"
    fake_home(h, diary, [diary[0]] if i % 2 else [])
    blog = install_bridge({port: state})
    fk, real = Fake(responder), REAL_URLOPEN
    urllib.request.urlopen = fk.urlopen
    try:
        b = {"name": name, "backend": "openai", "url": "http://fake.local/v1", "backstory": back}
        said = R.ask(b, h, port, state, False)
    finally:
        urllib.request.urlopen = real
    return fk.posts, blog, said


def check_posts(sid, posts, responder_msgs, blog):
    """The recorded bytes are what ask_openai passed to urlopen, and later rounds carry what the runner built."""
    for k, p in enumerate(posts):
        body = json.loads(p["data"])
        assert isinstance(p["data"], bytes), f"{sid} r{k + 1}: data not bytes"
        assert json.dumps(body).encode() == p["data"], f"{sid} r{k + 1}: bytes are not the runner's json.dumps"
        assert p["url"] == "http://fake.local/v1/chat/completions" and p["timeout"] == 120 and p["ctype"] == "application/json", f"{sid} r{k + 1}: request envelope"
        assert body["model"] == "local" and body["chat_template_kwargs"] == {"enable_thinking": False} and "tool_choice" not in body
        assert all("required" not in t["function"]["parameters"] for t in body["tools"]), f"{sid}: a tool has required"
        assert body["messages"][0] == {"role": "system", "content": R.MANUAL} and body["messages"][1]["role"] == "user"
        assert len(body["messages"]) == 2 + 2 * k, f"{sid} r{k + 1}: {len(body['messages'])} messages"
        if k:
            a, t = body["messages"][-2], body["messages"][-1]
            assert a == {"role": "assistant", "content": "", "tool_calls": responder_msgs[k - 1]["tool_calls"]}, f"{sid} r{k + 1}: assistant history"
            want = json.dumps(blog[k - 1][3])[:1500]
            assert t == {"role": "tool", "tool_call_id": f"call_{k}", "content": want}, f"{sid} r{k + 1}: tool message"


def md5(b):
    return hashlib.md5(b).hexdigest()


def capture_dir(states_dir, out, script):
    items = load_states(states_dir)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    rows, n = [], 0
    for item in items:
        msgs = []

        def rec(k, body, base=canned(script)):
            m = base(k, body)
            msgs.append(m)
            return m
        posts, blog, said = capture_state(item, rec)
        assert len(posts) == len(script), f"{item[0]}: {len(posts)} posts for script {script}"
        check_posts(item[0], posts, msgs, blog)
        for k, p in enumerate(posts):
            open(f"{out}/{item[0]}-r{k + 1}.json", "wb").write(p["data"])
            rows.append(f"{item[0]}-r{k + 1}\t{k + 1}\t{md5(p['data'])}\t{len(p['data'])}")
            n += 1
    man = "\n".join(rows) + "\n"
    open(f"{out}/MANIFEST.tsv", "w").write(man)
    open(f"{out}/SET.md5", "w").write(md5(man.encode()) + "\n")
    return n, md5(man.encode())


def floor_body(data, floor=16000):
    """max_tokens up to the repo floor; json round trip keeps key order, so every other byte is the runner's."""
    b = json.loads(data)
    b["max_tokens"] = max(b.get("max_tokens", 0), floor)
    return json.dumps(b).encode()


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("-"):
        sys.exit(__doc__)
    out, sd, script, selfcheck = a[0], f"{HERE}/states", ["state", "actions", "hold"], False
    i = 1
    while i < len(a):
        if a[i] == "--states":
            sd, i = a[i + 1], i + 2
        elif a[i] == "--script":
            script, i = a[i + 1].split(","), i + 2
        elif a[i] == "--selfcheck":
            selfcheck, i = True, i + 1
        else:
            sys.exit(f"FAIL args: unknown {a[i]}")
    n, h = capture_dir(sd, out, script)
    print(f"capture: {n} bodies, script {','.join(script)}, set md5 {h} -> {out}")
    if selfcheck:
        n2, h2 = capture_dir(sd, out + ".again", script)
        r = os.system(f"diff -r {out} {out}.again >/dev/null")
        shutil.rmtree(out + ".again")
        if r or h != h2:
            sys.exit("FAIL selfcheck: second capture differs from the first")
        print(f"selfcheck: second capture byte-identical ({n2} bodies, md5 {h2})")
    print("PASS capture")


if __name__ == "__main__":
    main()
