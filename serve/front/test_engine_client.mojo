"""Tests of serve/front/engine_client.mojo against serve/front/stub_engine.mojo: the request line bytes of
serve/src/protocol.rs, the line classifier, every state transition, death seen by poll (idle and mid-stream),
cancel, shutdown, 1000 spawn/shutdown cycles without a leaked fd or zombie. They prove the module against a
stub, not the front against the real engine.

Build and run (CPU only; the capped scope is typed by hand, never committed):
  mojo build serve/front/stub_engine.mojo -I . -o .work/stub_engine
  mojo build serve/front/test_engine_client.mojo -I . -o .work/test_engine_client
  .work/test_engine_client .work/stub_engine
Prints `PASS n/m` and exits 0, or `FAIL n/m` and exits 1.
"""
from json import parse
from posix import kill, reap, PollSet, SIGKILL
from engine_client import (
    ENG_STARTING, ENG_IDLE, ENG_BUSY, ENG_DEAD, ENG_STOPPING, EV_TOKEN, EV_DONE, EV_ERROR, EV_READY, EV_DEAD, EV_OTHER,
    Request, Event, EngineClient, classify, cancel_line,
)
from std.os import listdir
from std.sys import argv, exit
from std.time import perf_counter_ns


struct Check:
    var n: Int
    var bad: Int

    def __init__(out self):
        self.n = 0
        self.bad = 0

    def ok(mut self, cond: Bool, what: String):
        self.n += 1
        if not cond:
            self.bad += 1
            print("FAIL", what)

    def eq(mut self, got: String, want: String, what: String):
        self.ok(got == want, what + String(": got ", repr(got), " want ", repr(want)))


def terminal(kind: Int) -> Bool:
    return kind == EV_DONE or kind == EV_ERROR or kind == EV_DEAD


def pump(mut c: EngineClient, mut evs: List[Event], terminals: Int, ms: Int) raises:
    """Poll until `terminals` events of kind done, error or dead are in evs, or ms pass."""
    var t0 = perf_counter_ns()
    while (perf_counter_ns() - t0) // 1000000 < ms:
        _ = c.poll(evs, 50)
        var seen = 0
        for i in range(len(evs)):
            if terminal(evs[i].kind):
                seen += 1
        if seen >= terminals:
            return


def until_ready(mut c: EngineClient, mut evs: List[Event]) raises:
    var t0 = perf_counter_ns()
    while c.state() == ENG_STARTING and (perf_counter_ns() - t0) // 1000000 < 5000:
        _ = c.poll(evs, 50)


def start(stub: String, mode: String) raises -> EngineClient:
    return EngineClient([stub, mode], [])


def kinds(evs: List[Event], kind: Int) -> Int:
    var n = 0
    for i in range(len(evs)):
        if evs[i].kind == kind:
            n += 1
    return n


def refused(c: EngineClient) -> Bool:
    for t in c.trail():
        if t.startswith("REFUSED"):
            return True
    return False


def fds() raises -> Int:
    return len(listdir("/proc/self/fd"))


def req(id: Int, n: Int) -> Request:
    return Request(id, [760, 6511, 314], n)


# ---- the request line: the expected strings are copied from the tests of serve/src/protocol.rs ----


def t_request_line(mut k: Check) raises:
    k.eq(
        Request(7, [760, 6511, 314], 64).line(),
        '{"id":7,"prompt":[760,6511,314],"n":64,"spec":false,"stop":[],"ckpt":[]}\n',
        "request_line_is_the_engine_shape",
    )
    var stops = Request(1, [1], 8)
    stops.stop = [[151645], [9707, 11]]
    k.eq(stops.line(), '{"id":1,"prompt":[1],"n":8,"spec":false,"stop":[[151645],[9707,11]],"ckpt":[]}\n', "stop sequences")
    var ck = Request(2, [1, 2, 3], 8)
    ck.ckpt = [7914, 8020]
    k.eq(ck.line(), '{"id":2,"prompt":[1,2,3],"n":8,"spec":false,"stop":[],"ckpt":[7914,8020]}\n', "ckpt hints")
    # sample parameters: the decimals go through json.parse (a float literal folds at compile time)
    var s = Request(3, [1], 8)
    s.temperature = Float32(parse("0.8").num())
    s.top_p = Float32(parse("0.9").num())
    s.top_k = 40
    s.min_p = Float32(parse("0.05").num())
    s.seed = 42
    k.eq(
        s.line(),
        '{"id":3,"prompt":[1],"n":8,"spec":false,"stop":[],"ckpt":[],"temperature":0.8,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":42}\n',
        "sample params only when set",
    )
    var sc = Request(4, [1], 8)
    sc.temperature = Float32(parse("0.7").num())
    sc.schema = parse('{"type":"object","properties":{"a":{"type":"string"}}}')
    sc.reasoning = False
    k.eq(
        sc.line(),
        '{"id":4,"prompt":[1],"n":8,"spec":false,"stop":[],"ckpt":[],"temperature":0.7,"schema":{"type":"object","properties":{"a":{"type":"string"}}},"reasoning":false}\n',
        "schema and reasoning only when set",
    )
    var em = Request(6, [1], 1)
    em.embed = True
    k.eq(em.line(), '{"id":6,"prompt":[1],"n":1,"spec":false,"stop":[],"ckpt":[],"embed":true}\n', "embed only when set")
    k.eq(cancel_line(42), '{"cancel":42}\n', "cancel_line_is_a_bare_id")
    # the f32 text of values a client sends: the shortest decimal that reads back, as serde writes it
    for t in ["0.8", "0.95", "0.05", "0.01", "1.0", "1.5", "0.15", "0.6", "2.0", "0.3"]:
        var r = Request(1, [1], 1)
        r.top_p = Float32(parse(String(t)).num())
        k.ok(String(t) in r.line(), "f32 shortest text of " + String(t) + ": " + r.line())
    # every optional field, in the order of the Rust struct
    var all = Request(9, [1], 2, True)
    all.state_save = String("a.st")
    all.state_load = String("b.st")
    all.latent_fork = True
    all.latent_resume = String("L1")
    all.state_prefix = True
    all.top_logprobs = 5
    all.hidden = True
    all.gain = String("1:2:0.5")
    all.steer_layer = 3
    all.steer_scale = parse("0.25").num()
    all.capture = String("run1")
    all.embeds = ["x.bemb", "y.bemb"]
    all.ipc_stats = True
    k.eq(
        all.line(),
        '{"id":9,"prompt":[1],"n":2,"spec":true,"stop":[],"ckpt":[],"state_save":"a.st","state_load":"b.st","latent_fork":true,"latent_resume":"L1","state_prefix":true,"top_logprobs":5,"hidden":true,"gain":"1:2:0.5","steer_layer":3,"steer_scale":0.25,"capture":"run1","embeds":["x.bemb","y.bemb"],"ipc_stats":true}\n',
        "all optional fields in struct order",
    )


# ---- the one classifier: the malformed lines of protocol.rs are logs, never trusted ----


def t_classify(mut k: Check) raises:
    var bad = [
        '{"id":',
        '{"id":"x","tok":5}',
        '{"id":1,"tok":-5}',
        '{"id":1,"tok":4294967296}',
        '{"id":1,"tok":"5"}',
        '{"tok":5}',
        '{"id":1,"done":true}',
        '{"id":1,"done":true,"n":1,"prefill_s":"a","decode_s":0,"tok_s":0}',
        '{"id":1,"error":5}',
        "[1,2,3]",
        '{"id":1,"done":true,"n":1,"prefill_s":1e999,"decode_s":0,"tok_s":0}',
        "tok/s_gen: 126.5",
        "GENERATED: 1 2 3 ",
        "",
        '{"progress":0.5}',
    ]
    for b in bad:
        k.ok(classify(String(b)).kind == -1, "a log line: " + String(b))
    var t = classify('{"id":3,"tok":11751}\n')
    k.ok(t.kind == EV_TOKEN and t.id == 3 and t.tok == 11751 and t.text == "", "tok line")
    t = classify('{"id":1,"tok":2,"extra":{"x":[1]}}')
    k.ok(t.kind == EV_TOKEN and t.tok == 2, "extra fields are ignored")
    t = classify('{"id":9,"tok":42,"logprob":-0.5,"top_logprobs":[{"id":1,"logprob":-1.0}]}')
    k.ok(t.kind == EV_TOKEN and "top_logprobs" in t.text, "logprob line keeps its raw text")
    var d = classify('{"id":5,"done":true,"n":9,"prefill_s":0.01,"decode_s":0.1,"tok_s":1.0,"finish":"stop"}')
    k.ok(d.kind == EV_DONE and d.id == 5 and '"finish":"stop"' in d.text, "done line")
    var e = classify('{"id":9,"error":"prompt+n exceeds TMAX 128"}')
    k.ok(e.kind == EV_ERROR and e.id == 9 and e.text == "prompt+n exceeds TMAX 128", "error line")
    var r = classify('{"ready":true,"tmax":128,"mrows":8,"kmax":8,"spec_k":2,"pack":".work/engine-pack-q4"}\n')
    k.ok(r.kind == EV_READY and "engine-pack-q4" in r.text, "ready line")
    for f in ["tmax", "mrows", "kmax", "spec_k", "pack"]:
        var line = String('{"ready":true,"tmax":1,"mrows":1,"kmax":1,"spec_k":1,"pack":"p"}')
        line = line.replace('"' + String(f) + '":1', '"x":1').replace('"pack":"p"', '"x":"p"') if f != "pack" else line.replace('"pack":"p"', '"x":"p"')
        var m = classify(line)
        k.ok(m.kind == EV_ERROR and m.id == -1 and m.text == "ready line: missing " + String(f), "named ready error for " + String(f) + ": " + m.text)
    k.ok(classify('{"ready":true}').text == "ready line: missing tmax", "ready without limits is named")
    k.ok(classify('{"id":6,"embed":[0.1,0.2]}').kind == EV_OTHER, "embed is EV_OTHER")


# ---- transitions ----


def t_normal(mut k: Check, stub: String) raises:
    var c = start(stub, "normal")
    k.ok(c.state() == ENG_STARTING and c.state_name() == "STARTING", "starting right after spawn")
    k.ok(c.fd() >= 0, "fd is open")
    var evs = List[Event]()
    until_ready(c, evs)
    k.ok(c.state() == ENG_IDLE and kinds(evs, EV_READY) == 1, "STARTING -> IDLE on the ready line")
    var info = c.ready_info()
    k.ok(info.get("tmax").int() == 16384 and info.get("mrows").int() == 8 and info.get("spec_k").int() == 2, "ready_info carries the limits")
    k.eq(info.text("pack"), "stub", "ready_info pack")
    k.eq(info.text("kv"), "int8", "ready_info kv")
    var r = req(1, 5)
    c.submit(r)
    k.ok(c.state() == ENG_BUSY, "IDLE -> BUSY on submit")
    evs = List[Event]()
    pump(c, evs, 1, 5000)
    k.ok(kinds(evs, EV_TOKEN) == 5 and kinds(evs, EV_DONE) == 1, "5 tokens then one done")
    var ids_ok = True
    for i in range(len(evs)):
        if evs[i].kind == EV_TOKEN:
            ids_ok = ids_ok and evs[i].id == 1 and Int(evs[i].tok) == 1000 + i
    k.ok(ids_ok, "token ids and order")
    k.ok(evs[len(evs) - 1].kind == EV_DONE and '"finish":"length"' in evs[len(evs) - 1].text, "done is last and carries the raw line")
    k.ok(c.state() == ENG_IDLE, "BUSY -> IDLE when the last request ends")
    # two requests in flight: BUSY until both end
    c.submit(req(2, 3))
    c.submit(req(3, 3))
    k.ok(c.state() == ENG_BUSY, "two in flight")
    evs = List[Event]()
    pump(c, evs, 2, 5000)
    k.ok(kinds(evs, EV_DONE) == 2 and kinds(evs, EV_TOKEN) == 6 and c.state() == ENG_IDLE, "both end, back to IDLE")
    # an error line ends its request like a done
    var t0 = perf_counter_ns()
    c.shutdown(2000)
    var ms = (perf_counter_ns() - t0) // 1000000
    k.ok(ms < 1000, String("shutdown on a healthy stub in ", ms, " ms"))
    k.ok(c.state() == ENG_DEAD and c.dead_reason().startswith("stopped, exit code 0"), "STOPPING -> DEAD: " + c.dead_reason())
    k.ok(c.fd() == -1, "fd closed after shutdown")
    var tr = c.trail()
    var want = ["NEW -> STARTING", "STARTING -> IDLE", "IDLE -> BUSY", "BUSY -> IDLE", "IDLE -> BUSY", "BUSY -> IDLE", "IDLE -> STOPPING", "STOPPING -> DEAD"]
    k.ok(len(tr) == len(want), String("transition count ", len(tr)))
    for i in range(min(len(tr), len(want))):
        k.ok(tr[i].startswith(want[i]), String("transition ", i, ": ", tr[i]))
    c.shutdown(10)  # a second call is harmless
    k.ok(len(c.trail()) == len(tr), "shutdown twice logs once")
    try:
        c.submit(req(9, 1))
        k.ok(False, "submit after shutdown must raise")
    except e:
        k.ok("engine dead: stopped" in String(e), "submit after shutdown: " + String(e))
    k.ok(not refused(c), "no refused transition")


def t_log_before_ready(mut k: Check, stub: String) raises:
    var c = start(stub, "log-before-ready")
    var evs = List[Event]()
    until_ready(c, evs)
    k.ok(c.state() == ENG_IDLE and len(evs) == 1 and evs[0].kind == EV_READY, "log lines and JSON without a known key are skipped")
    c.shutdown(2000)


def t_ready_missing_field(mut k: Check, stub: String) raises:
    var c = start(stub, "ready-missing-field")
    var evs = List[Event]()
    pump(c, evs, 1, 5000)
    k.ok(c.state() == ENG_DEAD, "a ready line missing a field is DEAD, not a 600 s wait")
    k.eq(c.dead_reason(), "ready line: missing spec_k", "named error")
    k.ok(kinds(evs, EV_DEAD) == 1 and evs[0].id == -1, "one EV_DEAD for the engine")
    try:
        _ = reap(c.pid())
        k.ok(False, "the child must already be reaped")
    except e:
        k.ok("errno 10)" in String(e), "child killed and reaped")
    k.ok(not refused(c), "no refused transition")


def t_never_ready(mut k: Check, stub: String) raises:
    var c = start(stub, "never-ready")
    c.ready_timeout_ms = 300
    var evs = List[Event]()
    pump(c, evs, 1, 5000)
    k.ok(c.state() == ENG_DEAD and c.dead_reason().startswith("ready timeout"), "named ready timeout: " + c.dead_reason())
    try:
        c.submit(req(1, 1))
        k.ok(False, "submit when dead must raise")
    except e:
        k.ok(String(e).startswith("engine dead: ready timeout"), "submit when DEAD names the cause: " + String(e))


def t_starting_submit(mut k: Check, stub: String) raises:
    var c = start(stub, "normal")
    try:
        c.submit(req(1, 1))
        k.ok(False, "submit while STARTING must raise")
    except e:
        k.eq(String(e), "engine not accepting requests: STARTING", "submit while STARTING")
    c.shutdown(2000)
    k.ok(c.state() == ENG_DEAD, "shutdown while STARTING")
    k.ok("STARTING -> STOPPING" in c.trail()[1], "STARTING -> STOPPING is a transition")


def t_length_guard(mut k: Check, stub: String) raises:
    var c = start(stub, "normal")
    var evs = List[Event]()
    until_ready(c, evs)
    try:
        c.submit(req(1, 16384))
        k.ok(False, "prompt+n over TMAX must raise")
    except e:
        k.eq(String(e), "prompt+n exceeds TMAX 16384", "length guard text")
    k.ok(c.state() == ENG_IDLE, "a refused request changes nothing")
    c.submit(req(2, 16381))  # 3 + 16381 = TMAX: fits
    k.ok(c.state() == ENG_BUSY, "exactly TMAX fits")
    c.shutdown(200)


def t_idle_death(mut k: Check, stub: String) raises:
    var c = start(stub, "dies-idle")
    var evs = List[Event]()
    until_ready(c, evs)
    k.ok(c.state() == ENG_IDLE, "ready before it dies")
    var out = List[Event]()
    var t0 = perf_counter_ns()
    var n = c.poll(out, 3000)  # one call
    var ms = (perf_counter_ns() - t0) // 1000000
    k.ok(n == 1 and len(out) == 1 and out[0].kind == EV_DEAD and out[0].id == -1, "idle death is one EV_DEAD from one poll")
    k.ok(ms < 1500, String("seen in ", ms, " ms"))
    k.eq(c.dead_reason(), "exit code 7", "dead_reason of an idle death")
    k.ok(len(out) == 1 and out[0].text == "exit code 7", "EV_DEAD text is the reason")
    k.ok(c.state() == ENG_DEAD and c.fd() == -1, "DEAD, fd closed")
    var again = List[Event]()
    k.ok(c.poll(again, 10) == 0, "poll after DEAD adds nothing")
    var t_dead = perf_counter_ns()
    _ = c.poll(again, 60)
    _ = c.poll(again, 60)
    k.ok((perf_counter_ns() - t_dead) // 1000000 >= 100, "poll on a dead engine waits its timeout (the owner loop does not spin)")
    try:
        c.submit(req(1, 1))
        k.ok(False, "submit when DEAD must raise")
    except e:
        k.eq(String(e), "engine dead: exit code 7", "submit when DEAD")
    k.ok(not c.cancel(1), "cancel when DEAD is False")
    var tr = c.trail()
    k.ok(len(tr) == 3 and tr[2].startswith("IDLE -> DEAD"), "IDLE -> DEAD logged once")
    k.ok(not refused(c), "no refused transition")


def t_midstream_death(mut k: Check, stub: String, mode: String, want: String) raises:
    var c = start(stub, mode)
    var evs = List[Event]()
    until_ready(c, evs)
    c.submit(req(1, 5))
    c.submit(req(2, 5))
    evs = List[Event]()
    pump(c, evs, 2, 5000)
    k.ok(kinds(evs, EV_TOKEN) == 2 and kinds(evs, EV_DEAD) == 2, mode + ": 2 tokens, then one EV_DEAD per request in flight")
    var ids = 0
    for i in range(len(evs)):
        if evs[i].kind == EV_DEAD:
            ids += evs[i].id
            k.eq(evs[i].text, want, mode + " EV_DEAD text")
    k.ok(ids == 3, mode + ": the dead events name requests 1 and 2")
    k.eq(c.dead_reason(), want, mode + " dead_reason")
    k.ok(c.state() == ENG_DEAD and not refused(c), mode + ": BUSY -> DEAD")
    try:
        c.submit(req(3, 1))
        k.ok(False, "submit after a mid-stream death must raise")
    except e:
        k.ok(String(e).startswith("engine dead: "), "submit: " + String(e))


def t_write_to_dead_child(mut k: Check, stub: String) raises:
    """A write to a child that is already gone must not end this process (SIGPIPE) and names the cause."""
    var c = start(stub, "normal")
    var evs = List[Event]()
    until_ready(c, evs)
    _ = kill(c.pid(), SIGKILL)
    var ps = PollSet()
    _ = ps.wait(150)  # the kernel closes its end when it reaps the process image
    try:
        c.submit(req(1, 1))
        k.ok(False, "submit to a killed child must raise")
    except e:
        k.ok("killed by signal 9" in String(e), "write failure names the signal: " + String(e))
    k.ok(c.state() == ENG_DEAD, "write failure is DEAD")
    var out = List[Event]()
    _ = c.poll(out, 10)
    k.ok(kinds(out, EV_DEAD) == 1, "the death is still reported by the next poll")


def t_cancel(mut k: Check, stub: String) raises:
    var c = start(stub, "slow")
    var evs = List[Event]()
    until_ready(c, evs)
    c.submit(req(1, 50))
    c.submit(req(2, 4))
    k.ok(not c.cancel(2), "a queued request is not cancelled now (it is remembered)")
    k.ok(not c.cancel(99), "an unknown id is not cancelled")
    evs = List[Event]()
    var t0 = perf_counter_ns()
    while kinds(evs, EV_TOKEN) < 2 and (perf_counter_ns() - t0) // 1000000 < 5000:
        _ = c.poll(evs, 50)
    k.ok(c.cancel(1), "the running request is cancelled")
    pump(c, evs, 1, 5000)
    var done = Event(-1, 0, 0, "")
    for i in range(len(evs)):
        if evs[i].kind == EV_DONE:
            done = evs[i].copy()
    k.ok(done.id == 1 and '"finish":"cancelled"' in done.text, "done says cancelled: " + done.text)
    k.ok(kinds(evs, EV_TOKEN) < 50, String("stopped early after ", kinds(evs, EV_TOKEN), " tokens"))
    k.ok(c.state() == ENG_BUSY, "request 2 still in flight")
    pump(c, evs, 2, 5000)
    k.ok(kinds(evs, EV_DONE) == 2 and c.state() == ENG_IDLE, "request 2 finishes after the cancel")
    var done2 = Event(-1, 0, 0, "")
    var toks2 = 0
    for i in range(len(evs)):
        if evs[i].kind == EV_DONE and evs[i].id == 2:
            done2 = evs[i].copy()
        if evs[i].kind == EV_TOKEN and evs[i].id == 2:
            toks2 += 1
    k.ok('"finish":"cancelled"' in done2.text, "the queued request is cancelled when it reaches the head: " + done2.text)
    k.ok(toks2 < 4, String("the queued request stopped early after ", toks2, " of 4 tokens"))
    k.ok(not c.cancel(1), "a finished request is not cancelled")
    c.shutdown(2000)


def t_stream_timeout(mut k: Check, stub: String) raises:
    """Mutant of the stub: it drops the done line. The client must name a timeout, not wait for ever."""
    var c = start(stub, "drops-done")
    c.stream_timeout_ms = 400
    var evs = List[Event]()
    until_ready(c, evs)
    c.submit(req(1, 3))
    evs = List[Event]()
    var t0 = perf_counter_ns()
    pump(c, evs, 1, 8000)
    var ms = (perf_counter_ns() - t0) // 1000000
    k.ok(kinds(evs, EV_TOKEN) == 3 and kinds(evs, EV_DONE) == 0, "3 tokens and no done")
    k.ok(kinds(evs, EV_DEAD) == 1 and evs[len(evs) - 1].id == 1, "the request gets one EV_DEAD")
    k.ok(c.dead_reason().startswith("stream timeout: no event for 400 ms"), "named timeout: " + c.dead_reason())
    k.ok(ms < 3000, String("timeout after ", ms, " ms"))
    try:
        _ = reap(c.pid())
        k.ok(False, "the wedged child must be reaped")
    except e:
        k.ok("errno 10)" in String(e), "wedged child killed and reaped")


def cycle(stub: String) raises -> Int:
    """One spawn, one request, shutdown: the milliseconds shutdown took, or -1 when it did not end right."""
    var c = start(stub, "normal")
    var evs = List[Event]()
    until_ready(c, evs)
    c.submit(req(1, 2))
    evs = List[Event]()
    pump(c, evs, 1, 5000)
    var t0 = perf_counter_ns()
    c.shutdown(2000)
    var ms = (perf_counter_ns() - t0) // 1000000
    var good = kinds(evs, EV_DONE) == 1 and c.state() == ENG_DEAD and c.dead_reason().startswith("stopped, exit code 0") and not refused(c)
    return Int(ms) if good else -1


def t_cycles(mut k: Check, stub: String) raises:
    var before = fds()
    var bad = 0
    var worst = 0
    var t0 = perf_counter_ns()
    for _ in range(1000):
        var ms = cycle(stub)
        if ms < 0:
            bad += 1
        worst = max(worst, ms)
    var total = (perf_counter_ns() - t0) // 1000000
    print("1000 cycles in", total, "ms, slowest shutdown", worst, "ms")
    k.ok(bad == 0, String(bad, " of 1000 cycles did not end right"))
    k.ok(worst < 1000, String("slowest shutdown ", worst, " ms"))
    k.ok(fds() == before, String("fd count ", before, " before, ", fds(), " after 1000 cycles"))
    try:
        _ = reap(-1)
        k.ok(False, "a zombie or a running child is left")
    except e:
        k.ok("errno 10)" in String(e), "no child left to wait for")


def t_cost() raises:
    """Information, not a gate: what the hot path costs per engine line and per request line (single core)."""
    var tok = String('{"id":7,"tok":11751}\n')
    var t0 = perf_counter_ns()
    var sum = 0
    for _ in range(100000):
        sum += Int(classify(tok).tok)
    var per = (perf_counter_ns() - t0) // 100000
    var r = Request(7, [760, 6511, 314], 64)
    t0 = perf_counter_ns()
    for _ in range(100000):
        sum += r.line().byte_length()
    print("hot path: classify of a token line", per, "ns, Request.line", (perf_counter_ns() - t0) // 100000, "ns, check", sum > 0)


def t_front_http_assumptions(mut k: Check, stub: String) raises:
    """What the front-http lane builds on: (1) the state is BUSY right after submit, before any poll;
    (2) poll(out, 2) returns in about 2 ms; (3) dead_reason() holds the reason of a death, whatever an
    EV_DEAD carries in its text."""
    var c = start(stub, "normal")
    var evs = List[Event]()
    until_ready(c, evs)
    k.eq(c.dead_reason(), "", "dead_reason is empty while the engine lives")
    c.submit(req(1, 3))
    k.ok(c.state() == ENG_BUSY and c.state_name() == "BUSY", "(1) BUSY right after submit, before any poll")
    evs = List[Event]()
    pump(c, evs, 1, 5000)
    k.ok(c.state() == ENG_IDLE, "(1) back to IDLE after the done")
    var us = List[Int]()
    for _ in range(100):
        var out = List[Event]()
        var t0 = perf_counter_ns()
        _ = c.poll(out, 2)
        us.append(Int((perf_counter_ns() - t0) // 1000))
    for i in range(1, len(us)):  # insertion sort: 100 values
        var j = i
        while j > 0 and us[j - 1] > us[j]:
            var t = us[j]
            us[j] = us[j - 1]
            us[j - 1] = t
            j -= 1
    print("poll(out, 2) on an idle engine, 100 calls: median", us[50], "us, max", us[99], "us")
    k.ok(us[50] < 3500, String("(2) median poll(out, 2) ", us[50], " us"))
    k.ok(us[99] < 20000, String("(2) slowest poll(out, 2) ", us[99], " us"))
    _ = kill(c.pid(), SIGKILL)
    var dead = List[Event]()
    pump(c, dead, 1, 5000)
    k.ok(kinds(dead, EV_DEAD) == 1, "(3) one EV_DEAD")
    k.eq(c.dead_reason(), "killed by signal 9", "(3) dead_reason holds the reason")
    for i in range(len(dead)):
        k.ok(dead[i].text == "" or dead[i].text == c.dead_reason(), "(3) an EV_DEAD text is empty or the reason")


def main() raises:
    var stub = String(argv()[1]) if len(argv()) > 1 else String(".work/stub_engine")
    var k = Check()
    t_request_line(k)
    t_classify(k)
    t_normal(k, stub)
    t_log_before_ready(k, stub)
    t_ready_missing_field(k, stub)
    t_never_ready(k, stub)
    t_starting_submit(k, stub)
    t_length_guard(k, stub)
    t_idle_death(k, stub)
    t_midstream_death(k, stub, "dies-mid-stream", "exit code 3")
    t_midstream_death(k, stub, "dies-by-signal", "killed by signal 9")
    t_write_to_dead_child(k, stub)
    t_cancel(k, stub)
    t_stream_timeout(k, stub)
    t_front_http_assumptions(k, stub)
    t_cycles(k, stub)
    t_cost()
    if k.bad > 0:
        print("FAIL", k.n - k.bad, "/", k.n)
        exit(1)
    print("PASS", k.n, "/", k.n)
