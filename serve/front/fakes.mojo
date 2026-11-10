"""Test doubles for the HTTP front: `FakeText` (byte-level tokenizer) and `FakeEngine` (scripted events).

They prove the module, not the front: a real tokenizer and a real engine child are the boss's verification.
`Probe` is how a test thread reads what the dispatcher-owned `FakeEngine` saw (cancels, submits).
"""
from api import (
    DetokLike, TextLike, EngineLike, GenRequest, Ev, ENG_IDLE, ENG_BUSY, ENG_DEAD, EV_TOKEN, EV_DONE, EV_ERROR, EV_DEAD
)
from json import Json
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import alloc, Layout

comptime FAKE_BOS: Int32 = 1
comptime FAKE_EOS: Int32 = 2
comptime FAKE_IM_START: Int32 = 259  # `<|im_start|>`
comptime FAKE_IM_END: Int32 = 260  # `<|im_end|>`; also a stop id, as in Rust
comptime _BASE: Int32 = 3  # byte b is token b + 3

comptime P_SUBMITS = 0
comptime P_CANCELS = 1
comptime P_LAST_CANCEL = 2
comptime P_LAST_PROMPT_LEN = 3
comptime P_LAST_N = 4


struct Probe(Movable):
    """A few counters on the heap, shared by address, read and written atomically."""

    var p: Pointer[Int32, MutUntrackedOrigin]

    def __init__(out self):
        self.p = alloc(Layout[Int32](count=8)).unsafe_leak()
        for i in range(8):
            self.p.unsafe_offset(i).unsafe_write(Int32(0))

    def addr(self) -> Int:
        return Int(self.p)

    def get(self, i: Int) -> Int:
        return Int(Atomic[Int32].load[ordering=Ordering.ACQUIRE](self.p.unsafe_offset(i)))

    def __deinit__(deinit self):
        self.p.unsafe_free()


def _probe_set(addr: Int, i: Int, v: Int):
    var p = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)
    Atomic[Int32].store[ordering=Ordering.RELEASE](p.unsafe_offset(i), Int32(v))


def _probe_add(addr: Int, i: Int):
    var p = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)
    _ = Atomic[Int32].fetch_add(p.unsafe_offset(i), Int32(1))


# ---- text ---------------------------------------------------------------------


def _complete_len(b: Span[UInt8, _]) -> Int:
    """Length of the longest prefix of b that does not end inside a UTF-8 sequence."""
    var n = len(b)
    var back = 1
    while back <= 3 and back <= n:
        var c = b[n - back]
        if (c & 0xC0) != 0x80:
            var need = 1
            if (c & 0xE0) == 0xC0:
                need = 2
            elif (c & 0xF0) == 0xE0:
                need = 3
            elif (c & 0xF8) == 0xF0:
                need = 4
            return n if need <= back else n - back
        back += 1
    return n


def _piece(id: Int32, mut out: List[UInt8]):
    """The bytes one id decodes to: a byte token is one byte, the ChatML markers their name, ids 0 to 2 nothing
    (what the Rust oracle printed for them, see oracle-http/capture.py `detokenize-special`)."""
    if id >= _BASE and id < _BASE + 256:
        out.append(UInt8(Int(id - _BASE)))
        return
    var name = String()
    if id == FAKE_IM_START:
        name = "<|im_start|>"
    elif id == FAKE_IM_END:
        name = "<|im_end|>"
    out.extend(name.as_bytes())


struct FakeDetok(DetokLike):
    var held: List[UInt8]

    def __init__(out self):
        self.held = List[UInt8]()

    def push(mut self, id: Int32) -> String:
        _piece(id, self.held)
        var k = _complete_len(Span(self.held))
        var out = String(unsafe_from_utf8=Span(self.held)[0:k])
        var rest = List[UInt8](Span(self.held)[k:])
        self.held = rest^
        return out^

    def flush(mut self) -> String:
        var out = String(unsafe_from_utf8=Span(self.held))
        self.held = List[UInt8]()
        return out^


struct FakeText(TextLike):
    comptime D = FakeDetok

    var xml: Bool
    var fail_template: Bool

    def __init__(out self, xml: Bool = True, fail_template: Bool = False):
        self.xml = xml
        self.fail_template = fail_template

    def encode(self, s: String, add_special: Bool) raises -> List[Int32]:
        var out = List[Int32]()
        if add_special:
            out.append(FAKE_BOS)
        var b = s.as_bytes()
        var i = 0
        while i < len(b):
            var hit = False
            for m in range(2):  # the two ChatML markers are single tokens, as in the oracle tokenizer
                var name = "<|im_start|>" if m == 0 else "<|im_end|>"
                var nb = name.as_bytes()
                if i + len(nb) <= len(b):
                    var same = True
                    for k in range(len(nb)):
                        if b[i + k] != nb[k]:
                            same = False
                            break
                    if same:
                        out.append(FAKE_IM_START if m == 0 else FAKE_IM_END)
                        i += len(nb)
                        hit = True
                        break
            if not hit:
                out.append(Int32(Int(b[i])) + _BASE)
                i += 1
        return out^

    def decode(self, ids: Span[Int32, _]) -> String:
        var b = List[UInt8]()
        for id in ids:
            _piece(id, b)
        return String(unsafe_from_utf8=Span(b))

    def apply_chat_template(self, messages: Json, tools: Json, kwargs: Json) raises -> String:
        if self.fail_template:
            raise Error("template: boom")
        # ChatML, the Rust front's template when the pack carries none: it ignores tools and kwargs
        var out = String()
        for m in messages.items():
            out += "<|im_start|>" + m.get("role").str() + "\n" + m.get("content").str()
            var tc = m.get("tool_calls")
            if not tc.is_null():
                out += tc.dumps()
            out += "<|im_end|>\n"
        return out + "<|im_start|>assistant\n"

    def tool_xml(self) -> Bool:
        return self.xml

    def stop_ids(self) -> List[Int32]:
        return [FAKE_EOS, FAKE_IM_END]

    def detok(self) -> FakeDetok:
        return FakeDetok()


# ---- engine -------------------------------------------------------------------

comptime M_NORMAL = 0
comptime M_ERR_BEFORE = 1  # an error line, no token
comptime M_ERR_AFTER = 2  # two tokens, then an error line
comptime M_NO_TERMINAL = 3  # two tokens, then the engine is idle with the request unfinished
comptime M_SLOW = 4  # one token per poll (10 ms) until n or a cancel
comptime M_DEAD_MID = 5  # two tokens, then the child dies
comptime M_DEAD_IDLE = 6  # dead before any request
comptime M_SUBMIT_FAIL = 7  # submit raises, the engine stays up


struct FakeEngine(EngineLike):
    var mode: Int
    var script: List[Int32]  # the tokens a normal request answers with
    var st: Int
    var reason: String
    var probe: Int  # address of a Probe, 0: none
    var queue: List[Int]  # request ids in arrival order; the head is being answered
    var ns: List[Int]  # max_tokens of each queued request
    var pending_id: Int  # the head, 0: none
    var sent: Int  # tokens already sent for it
    var n: Int
    var cancelled: Bool

    def __init__(out self, mode: Int = M_NORMAL, var script: List[Int32] = List[Int32](), probe: Int = 0):
        self.mode = mode
        self.script = script^
        self.st = ENG_DEAD if mode == M_DEAD_IDLE else ENG_IDLE
        self.reason = "exit code 3" if mode == M_DEAD_IDLE else ""
        self.probe = probe
        self.queue = List[Int]()
        self.ns = List[Int]()
        self.pending_id = 0
        self.sent = 0
        self.n = 0
        self.cancelled = False

    def submit(mut self, req: GenRequest) raises:
        if self.st != ENG_IDLE and self.st != ENG_BUSY:
            raise Error("engine is " + self.state_name())
        if self.mode == M_SUBMIT_FAIL:
            raise Error("request line too long")
        if self.probe != 0:
            _probe_add(self.probe, P_SUBMITS)
            _probe_set(self.probe, P_LAST_PROMPT_LEN, len(req.prompt))
            _probe_set(self.probe, P_LAST_N, req.n)
        self.queue.append(req.id)
        self.ns.append(req.n)
        if self.pending_id == 0:
            self._head()
        self.st = ENG_BUSY

    def _head(mut self):
        self.pending_id = self.queue[0]
        self.n = self.ns[0]
        self.sent = 0
        self.cancelled = False

    def cancel(mut self, id: Int) -> Bool:
        if self.probe != 0:
            _probe_add(self.probe, P_CANCELS)
            _probe_set(self.probe, P_LAST_CANCEL, id)
        if id == self.pending_id:
            self.cancelled = True
            return True
        for i in range(len(self.queue)):
            if self.queue[i] == id:
                _ = self.queue.pop(i)
                _ = self.ns.pop(i)
                return True
        return False

    def _done(self, finish: String) -> Ev:
        var j = Json.object()
        j.set("id", Json.of(self.pending_id))
        j.set("done", Json.of(True))
        j.set("n", Json.of(self.sent))
        j.set("prefill_s", Json.of(Float64(0.5)))
        j.set("decode_s", Json.of(Float64(0.25)))
        j.set("tok_s", Json.of(Float64(4.0)))
        j.set("cached", Json.of(0))
        j.set("prefill_rows", Json.of(7))
        j.set("finish", Json.of(finish))
        return Ev(EV_DONE, self.pending_id, 0, j.dumps())

    def poll(mut self, mut out: List[Ev], timeout_ms: Int) raises -> Int:
        var before = len(out)
        if self.pending_id == 0:
            _ = external_call["usleep", Int32](Int32(min(timeout_ms, 5) * 1000))
            return 0
        var id = self.pending_id
        if self.mode == M_ERR_BEFORE:
            out.append(Ev(EV_ERROR, id, 0, "prompt rejected"))
            self._finish()
        elif self.mode == M_SLOW:
            _ = external_call["usleep", Int32](Int32(10000))
            if self.cancelled:
                out.append(self._done("cancelled"))
                self._finish()
            elif self.sent >= self.n:
                out.append(self._done("length"))
                self._finish()
            else:
                out.append(Ev(EV_TOKEN, id, Int32(Int(_BASE) + 97 + self.sent % 26), ""))
                self.sent += 1
        else:
            var k = len(self.script)
            if self.mode == M_ERR_AFTER or self.mode == M_NO_TERMINAL or self.mode == M_DEAD_MID:
                k = min(k, 2)
            for i in range(k):
                out.append(Ev(EV_TOKEN, id, self.script[i], ""))
            self.sent = k
            if self.mode == M_ERR_AFTER:
                out.append(Ev(EV_ERROR, id, 0, "boom"))
                self._finish()
            elif self.mode == M_NO_TERMINAL:
                self._finish()  # idle again, nothing said
            elif self.mode == M_DEAD_MID:
                self.st = ENG_DEAD
                self.reason = "killed by signal 9"
                out.append(Ev(EV_DEAD, 0, 0, ""))  # like the real client: the reason is read from dead_reason()
                self.pending_id = 0
                self.queue.clear()
                self.ns.clear()
            else:
                out.append(self._done("length"))
                self._finish()
        return len(out) - before

    def _finish(mut self):
        _ = self.queue.pop(0)
        _ = self.ns.pop(0)
        self.pending_id = 0
        if len(self.queue) > 0:
            self._head()
        else:
            self.st = ENG_IDLE

    def state(self) -> Int:
        return self.st

    def state_name(self) -> String:
        if self.st == ENG_IDLE:
            return "idle"
        if self.st == ENG_BUSY:
            return "busy"
        if self.st == ENG_DEAD:
            return "dead"
        return "starting"

    def dead_reason(self) -> String:
        return self.reason

    def ready_info(self) -> Json:
        var j = Json.object()
        j.set("ready", Json.of(True))
        j.set("tmax", Json.of(16384))
        j.set("mrows", Json.of(4))
        j.set("kmax", Json.of(8))
        j.set("spec_k", Json.of(0))
        j.set("kv", Json.of("f32"))
        j.set("pack", Json.of("fake-pack"))
        return j^
