"""The engine child behind the Mojo front: one `serve/engine.mojo` (BARO_SERVE=1) driven over its stdin and
stdout, which are ONE unix socketpair end (`posix.spawn_pair`). Contract: serve/front/README.md; the line
protocol: serve/PROTOCOL.md, serve/src/protocol.rs. Single-threaded: one owner thread calls every method.

State is one tag with one writer, `_set`. Death is seen by `poll` (end of stream or the child's exit status),
also while idle. Everything not JSON with a known key is a log line and never becomes an event.
"""
from json import Json, parse, INT, STR
from posix import spawn_pair, reap, kill, SIGKILL, PollSet, recv, send_all, shutdown_write, close
from std.sys import stderr
from std.time import perf_counter_ns

comptime ENG_STARTING = 0
comptime ENG_IDLE = 1
comptime ENG_BUSY = 2
comptime ENG_DEAD = 3
comptime ENG_STOPPING = 4

comptime EV_TOKEN = 0
comptime EV_DONE = 1
comptime EV_ERROR = 2
comptime EV_READY = 3
comptime EV_DEAD = 4
comptime EV_OTHER = 5
comptime _LOG = -1  # classify() only: a line that is not an event


struct Request(Copyable, Movable):
    """The fields of serve/src/protocol.rs `Request`, same names and order. An absent Optional is not
    written. Floats the engine reads as f32 are Float32, printed as their shortest text."""

    var id: Int
    var prompt: List[Int32]
    var n: Int
    var spec: Bool
    var stop: List[List[Int32]]
    var ckpt: List[Int32]
    var state_save: Optional[String]
    var state_load: Optional[String]
    var latent_fork: Optional[Bool]
    var latent_resume: Optional[String]
    var state_prefix: Bool
    # SampleParams
    var temperature: Optional[Float32]
    var top_p: Optional[Float32]
    var top_k: Optional[Int]
    var min_p: Optional[Float32]
    var seed: Optional[Int]
    var presence_penalty: Optional[Float32]
    var frequency_penalty: Optional[Float32]
    var top_logprobs: Optional[Int]
    var hidden: Optional[Bool]
    var logits_topk: Optional[Int]
    var gain: Optional[String]
    var steer_file: Optional[String]
    var steer_bank: Optional[String]
    var steer_layer: Optional[Int]
    var steer_row: Optional[Int]
    var steer_scale: Optional[Float64]
    var capture: Optional[String]
    var schema: Json
    var reasoning: Optional[Bool]
    var embed: Optional[Bool]
    var embeds: List[String]
    # IpcFields
    var ipc_export: Optional[String]
    var ipc_release: Optional[Int]
    var ipc_sha: Optional[Int]
    var ipc_ph: Optional[String]
    var ipc_stats: Optional[Bool]
    var ipc_ingest: Optional[String]
    var ipc_prefix: Optional[String]

    def __init__(out self, id: Int, var prompt: List[Int32], n: Int, spec: Bool = False):
        self.id = id
        self.prompt = prompt^
        self.n = n
        self.spec = spec
        self.stop = List[List[Int32]]()
        self.ckpt = List[Int32]()
        self.state_save = None
        self.state_load = None
        self.latent_fork = None
        self.latent_resume = None
        self.state_prefix = False
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.min_p = None
        self.seed = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.top_logprobs = None
        self.hidden = None
        self.logits_topk = None
        self.gain = None
        self.steer_file = None
        self.steer_bank = None
        self.steer_layer = None
        self.steer_row = None
        self.steer_scale = None
        self.capture = None
        self.schema = Json()  # null = absent
        self.reasoning = None
        self.embed = None
        self.embeds = List[String]()
        self.ipc_export = None
        self.ipc_release = None
        self.ipc_sha = None
        self.ipc_ph = None
        self.ipc_stats = None
        self.ipc_ingest = None
        self.ipc_prefix = None

    def line(self) -> String:
        """The bytes Rust writes: one JSON object and a newline. Text goes through json (ASCII
        escapes), where serde writes UTF-8 as is: the same value, other bytes, for non-ASCII text."""
        var s = String('{"id":', self.id, ',"prompt":', _ints(self.prompt), ',"n":', self.n, ',"spec":', _b(self.spec))
        s += ',"stop":['
        for i in range(len(self.stop)):
            s += ("," if i > 0 else "") + _ints(self.stop[i])
        s += '],"ckpt":' + _ints(self.ckpt)
        _str(s, "state_save", self.state_save)
        _str(s, "state_load", self.state_load)
        _flag(s, "latent_fork", self.latent_fork)
        _str(s, "latent_resume", self.latent_resume)
        if self.state_prefix:
            s += ',"state_prefix":true'
        _f32(s, "temperature", self.temperature)
        _f32(s, "top_p", self.top_p)
        _int(s, "top_k", self.top_k)
        _f32(s, "min_p", self.min_p)
        _int(s, "seed", self.seed)
        _f32(s, "presence_penalty", self.presence_penalty)
        _f32(s, "frequency_penalty", self.frequency_penalty)
        _int(s, "top_logprobs", self.top_logprobs)
        _flag(s, "hidden", self.hidden)
        _int(s, "logits_topk", self.logits_topk)
        _str(s, "gain", self.gain)
        _str(s, "steer_file", self.steer_file)
        _str(s, "steer_bank", self.steer_bank)
        _int(s, "steer_layer", self.steer_layer)
        _int(s, "steer_row", self.steer_row)
        if self.steer_scale:
            s += ',"steer_scale":' + Json.of(self.steer_scale.value()).dumps()
        _str(s, "capture", self.capture)
        if not self.schema.is_null():
            s += ',"schema":' + self.schema.dumps()
        _flag(s, "reasoning", self.reasoning)
        _flag(s, "embed", self.embed)
        if len(self.embeds) > 0:
            s += ',"embeds":['
            for i in range(len(self.embeds)):
                s += ("," if i > 0 else "") + Json.of(self.embeds[i]).dumps()
            s += "]"
        _str(s, "ipc_export", self.ipc_export)
        _int(s, "ipc_release", self.ipc_release)
        _int(s, "ipc_sha", self.ipc_sha)
        _str(s, "ipc_ph", self.ipc_ph)
        _flag(s, "ipc_stats", self.ipc_stats)
        _str(s, "ipc_ingest", self.ipc_ingest)
        _str(s, "ipc_prefix", self.ipc_prefix)
        return s + "}\n"


def _b(v: Bool) -> String:
    return "true" if v else "false"


def _ints(v: List[Int32]) -> String:
    var s = String("[")
    for i in range(len(v)):
        s += ("," if i > 0 else "") + String(Int(v[i]))
    return s + "]"


def _str(mut s: String, key: String, v: Optional[String]):
    if v:
        s += ',"' + key + '":' + Json.of(v.value()).dumps()


def _int(mut s: String, key: String, v: Optional[Int]):
    if v:
        s += ',"' + key + '":' + String(v.value())


def _flag(mut s: String, key: String, v: Optional[Bool]):
    if v:
        s += ',"' + key + '":' + _b(v.value())


def _f32(mut s: String, key: String, v: Optional[Float32]):
    if v:
        s += ',"' + key + '":' + String(v.value())


def cancel_line(id: Int) -> String:
    return String('{"cancel":', id, "}\n")


@fieldwise_init
struct Event(Copyable, Movable):
    var kind: Int
    var id: Int
    var tok: Int32
    var text: String  # the error, the dead reason, or the raw JSON line of a done, ready, other or logprob line


def _num(j: Json, key: String, hi: Int) -> Int:
    """A non-negative integer member up to hi, else -1 (a float, a string and a missing key are all -1)."""
    var v = j.get(key)
    return v.int() if v.kind() == INT and v.int() >= 0 and v.int() <= hi else -1


def _finite(j: Json, key: String) -> Bool:
    var v = j.get(key)
    return v.is_num() and v.num() == v.num() and abs(v.num()) < 1.7976931348623157e308


def classify(line: String) -> Event:
    """The one line classifier (serve/src/protocol.rs parse_line): a line that is not a JSON object with a known
    key is `_LOG`. A ready line missing a field is an EV_ERROR with id -1 and the named error, where Rust logs
    it and waits 600 s."""
    var raw = String(line.strip("\r\n"))
    var log = Event(_LOG, 0, 0, raw)
    if not raw.startswith("{"):
        return log^
    var j: Json
    try:
        j = parse(raw)
    except:
        return log^
    if j.has("ready") and j.get("ready").truthy() and j.get("ready").kind() != INT:
        for k in ["tmax", "mrows", "kmax", "spec_k"]:
            if _num(j, k, 0xFFFFFFFF) < 0:
                return Event(EV_ERROR, -1, 0, "ready line: missing " + String(k))
        if j.get("pack").kind() != STR:
            return Event(EV_ERROR, -1, 0, "ready line: missing pack")
        return Event(EV_READY, 0, 0, raw)
    var id = _num(j, "id", 0x7FFFFFFFFFFFFFFF)
    if id < 0:
        return log^
    if j.has("error"):
        return Event(EV_ERROR, id, 0, j.get("error").str()) if j.get("error").kind() == STR else log^
    if j.has("ipc_op") or j.has("latent_id"):
        return Event(EV_OTHER, id, 0, raw)
    if j.get("done").truthy() and j.get("done").kind() != INT:
        var ok = _num(j, "n", 0xFFFFFFFF) >= 0 and _finite(j, "prefill_s") and _finite(j, "decode_s") and _finite(j, "tok_s")
        return Event(EV_DONE, id, 0, raw) if ok else log^
    if j.has("embed") or j.has("hidden") or j.has("logits_topk"):
        return Event(EV_OTHER, id, 0, raw)
    var tok = _num(j, "tok", 0xFFFFFFFF)
    if tok >= 0:
        return Event(EV_TOKEN, id, Int32(tok), raw if j.has("logprob") else String(""))
    return log^


def _reason(code: Int) -> String:
    return String("exit code ", code) if code >= 0 else String("killed by signal ", -code)


struct EngineClient(Movable):
    var ready_timeout_ms: Int  # the pack load; Rust waits 600 s
    var stream_timeout_ms: Int  # a busy engine that prints nothing this long is dead
    var _pid: Int
    var _fd: Int32
    var _state: Int
    var _reason: String
    var _ready: Json
    var _tmax: Int
    var _buf: List[UInt8]
    var _active: List[Int]
    var _pending: List[Event]
    var _trail: List[String]
    var _since: Int
    var _closed: Bool

    def __init__(out self, argv: List[String], env: List[String]) raises:
        var r = spawn_pair(argv, env)
        self.ready_timeout_ms = 600000
        self.stream_timeout_ms = 600000
        self._pid = r[0]
        self._fd = r[1]
        self._state = -1
        self._reason = String("")
        self._ready = Json()
        self._tmax = 0
        self._buf = List[UInt8]()
        self._active = List[Int]()
        self._pending = List[Event]()
        self._trail = List[String]()
        self._since = perf_counter_ns()
        self._closed = False
        self._set(ENG_STARTING, String("spawned pid ", self._pid))

    def __deinit__(deinit self):
        if not self._closed:
            _ = kill(self._pid, SIGKILL)
            try:
                _ = reap(self._pid, True)
            except:
                pass
            close(self._fd)

    def fd(self) -> Int32:
        """The child's end of the socketpair, non-blocking, for an aio loop; -1 once the engine is dead."""
        return -1 if self._closed else self._fd

    def pid(self) -> Int:
        return self._pid

    def state(self) -> Int:
        return self._state

    def state_name(self) -> String:
        return _name(self._state)

    def dead_reason(self) -> String:
        return self._reason

    def ready_info(self) -> Json:
        return self._ready.copy()

    def trail(self) -> List[String]:
        """Every transition, as logged: `OLD -> NEW (why)`."""
        return self._trail.copy()

    def _set(mut self, new: Int, why: String):
        """The only writer of the state. A transition outside the table is refused and logged as such."""
        var old = self._state
        var legal = (
            (old == -1 and new == ENG_STARTING)
            or (old == ENG_STARTING and (new == ENG_IDLE or new == ENG_DEAD or new == ENG_STOPPING))
            or (old == ENG_IDLE and (new == ENG_BUSY or new == ENG_DEAD or new == ENG_STOPPING))
            or (old == ENG_BUSY and (new == ENG_IDLE or new == ENG_DEAD or new == ENG_STOPPING))
            or (old == ENG_STOPPING and new == ENG_DEAD)
        )
        var msg = String(_name(old), " -> ", _name(new), " (", why, ")")
        if not legal:
            msg = "REFUSED " + msg
        else:
            self._state = new
        self._trail.append(msg)
        print("engine:", msg, file=stderr)

    def submit(mut self, req: Request) raises:
        """Queue one request line. Raises, naming the cause, unless the engine is IDLE or BUSY, or when
        prompt + n cannot fit TMAX (the one length check; the engine exits on an oversized request)."""
        if self._state == ENG_DEAD:
            raise Error("engine dead: " + self._reason)
        if self._state != ENG_IDLE and self._state != ENG_BUSY:
            raise Error("engine not accepting requests: " + _name(self._state))
        if len(req.prompt) + req.n > self._tmax:
            raise Error(String("prompt+n exceeds TMAX ", self._tmax))
        if not send_all(self._fd, req.line().as_bytes()):
            self._die(self._exit_reason("write to engine failed"))
            raise Error("engine dead: " + self._reason)
        if self._state == ENG_IDLE:
            self._since = perf_counter_ns()
            self._set(ENG_BUSY, String("request ", req.id))
        self._active.append(req.id)

    def cancel(mut self, id: Int) -> Bool:
        """Write the cancel line when id is the request decoding now. False for a queued, finished or
        unknown id (the engine cancels only the running one)."""
        if self._state != ENG_BUSY or len(self._active) == 0 or self._active[0] != id:
            return False
        if not send_all(self._fd, cancel_line(id).as_bytes()):
            self._die(self._exit_reason("write to engine failed"))
            return False
        return True

    def poll(mut self, mut out: List[Event], timeout_ms: Int) raises -> Int:
        """Wait up to timeout_ms for engine output, append the events, return how many were added. Sees the
        child's death (end of stream or exit status) in the same call, also when idle."""
        var n0 = len(out)
        out.extend(self._pending^)
        self._pending = List[Event]()
        if self._closed:
            return len(out) - n0
        var ps = PollSet()
        _ = ps.add(self._fd)
        if len(out) == n0:
            _ = ps.wait(timeout_ms)
        var eof = self._drain(out)
        if not eof and not self._closed:
            var code = reap(self._pid)
            if code:
                eof = self._drain(out)
                self._die(_reason(code.value()))
        if eof and not self._closed:
            self._die(self._exit_reason("engine closed its output"))
        if not self._closed:
            var quiet = (perf_counter_ns() - self._since) // 1000000
            if self._state == ENG_STARTING and quiet > self.ready_timeout_ms:
                self._kill_dead(String("ready timeout: no ready line in ", self.ready_timeout_ms, " ms"))
            elif self._state == ENG_BUSY and quiet > self.stream_timeout_ms:
                self._kill_dead(String("stream timeout: no event for ", self.stream_timeout_ms, " ms"))
        out.extend(self._pending^)
        self._pending = List[Event]()
        return len(out) - n0

    def shutdown(mut self, grace_ms: Int):
        """Close the engine's stdin (it exits at end of file), wait up to grace_ms for it, then kill."""
        if self._closed:
            return
        self._set(ENG_STOPPING, "shutdown")
        shutdown_write(self._fd)
        var ps = PollSet()
        _ = ps.add(self._fd)
        var sink = List[Event]()
        var t0 = perf_counter_ns()
        var code = Optional[Int]()
        while not code and (perf_counter_ns() - t0) // 1000000 < grace_ms:
            _ = ps.wait(10)
            _ = self._drain(sink)  # the child must never block on a full output
            try:
                code = reap(self._pid)
            except:
                break
        var why = String("stopped")
        if code:
            why += ", " + _reason(code.value())
        else:
            why += ", killed after " + String(grace_ms) + " ms"
            _ = kill(self._pid, SIGKILL)
            try:
                why += ", " + _reason(reap(self._pid, True).value())
            except:
                pass
        self._finish(why)

    def _drain(mut self, mut out: List[Event]) -> Bool:
        """Read what the socket holds, turn each full line into an event or a log line. True at end of stream."""
        var eof = False
        while True:
            var n = recv(self._fd, self._buf)
            if n == 0:
                eof = True
                break
            if n < 0:
                break
        var start = 0
        for i in range(len(self._buf)):
            if self._buf[i] == 10:
                self._line(String(unsafe_from_utf8=Span(self._buf)[start : i]), out)
                start = i + 1
        if start > 0:
            var rest = List[UInt8](Span(self._buf)[start:])
            self._buf = rest^
        return eof

    def _line(mut self, text: String, mut out: List[Event]):
        var e = classify(text)
        if e.kind == _LOG:
            print("engine:", e.text, file=stderr)
            return
        self._since = perf_counter_ns()
        if e.kind == EV_READY:
            if self._state != ENG_STARTING:
                print("engine: unexpected ready line after startup", file=stderr)
                return
            try:
                self._ready = parse(e.text)
            except:
                pass
            self._tmax = self._ready.get("tmax").int()
            self._set(ENG_IDLE, "ready")
            out.append(e^)
            return
        if e.id == -1:  # the named ready error
            if self._state == ENG_STARTING:
                self._kill_dead(e.text)
            return
        var at = -1
        for i in range(len(self._active)):
            if self._active[i] == e.id:
                at = i
        if at < 0:
            print("engine: line for unknown request ignored:", text, file=stderr)
            return
        if e.kind == EV_DONE or e.kind == EV_ERROR:
            _ = self._active.pop(at)
            if len(self._active) == 0 and self._state == ENG_BUSY:
                self._set(ENG_IDLE, String("request ", e.id, " finished"))
        out.append(e^)

    def _exit_reason(mut self, lead: String) -> String:
        """The child's exit status as text: waits up to 1 s for it, then kills a child that closed stdout and lives."""
        var ps = PollSet()
        for _ in range(200):
            try:
                var code = reap(self._pid)
                if code:
                    self._pid = -1
                    return _reason(code.value())
            except:
                return lead
            _ = ps.wait(5)
        _ = kill(self._pid, SIGKILL)
        try:
            return lead + "; killed after 1 s: " + _reason(reap(self._pid, True).value())
        except:
            return lead

    def _kill_dead(mut self, why: String):
        _ = kill(self._pid, SIGKILL)
        try:
            _ = reap(self._pid, True)
        except:
            pass
        self._die(why)

    def _die(mut self, why: String):
        """The child is gone (reaped). Every request in flight, or the engine itself when idle, gets one EV_DEAD."""
        self._finish(why)

    def _finish(mut self, why: String):
        self._reason = why
        if len(self._active) == 0:
            self._pending.append(Event(EV_DEAD, -1, 0, why))
        for id in self._active:
            self._pending.append(Event(EV_DEAD, id, 0, why))
        self._active = List[Int]()
        close(self._fd)
        self._closed = True
        self._set(ENG_DEAD, why)


def _name(state: Int) -> String:
    if state == ENG_STARTING:
        return "STARTING"
    if state == ENG_IDLE:
        return "IDLE"
    if state == ENG_BUSY:
        return "BUSY"
    if state == ENG_DEAD:
        return "DEAD"
    if state == ENG_STOPPING:
        return "STOPPING"
    return "NEW"
