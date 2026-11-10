"""The Mojo HTTP front: `Front[T, E]` is a bento `Handler` for the OpenAI routes.

Threads: bento runs `handle` on many workers at once on one shared `Front`. The engine has ONE owner, the
dispatcher thread started here. A handler never touches the engine: it registers a `Slot` (its own event
queue) in the hub, hands the dispatcher a command, and waits on the slot's condition variable. One mutex
(`Hub.mu`) guards the inbox, the slot table and every slot; the engine itself is touched by the dispatcher alone.

A streaming handler holds its bento worker for the whole stream: size `Server.serve(workers)` to the number of
concurrent streams you want, and state it (bento has no idle-connection timeout, a finding for upstream).
"""
from api import (
    TextLike, EngineLike, GenRequest, Ev, ApiError, Call, Acc, ToolStream, ToolDelta, ENG_IDLE, ENG_BUSY, ENG_DEAD, ENG_STOPPING,
    EV_TOKEN, EV_DONE, EV_ERROR, EV_DEAD, plain, bad, parse_body, parse_completion, parse_chat, check_gen,
    tokenize_response, detokenize_response, cancel_id, cancel_response, stats_json, completion_response,
    completion_delta, completion_finish, chat_response, chat_delta, chat_finish, parse_tool_calls, models_response,
    props_response, health_response, engine_error_chunk, now_secs, fail_word, invalid_tokens_error, invalid_tokens_chunk
)
from bento.http import Request, Response
from bento.server import Handler
from bento.sync import Mutex, Cond
from json import Json
from std.ffi import external_call
from std.memory import alloc, Layout
from std.time import perf_counter_ns


struct Slot(Movable):
    """One request's event queue; every field is guarded by `Hub.mu`."""

    var events: List[Ev]
    var cv: Cond
    var submitted: Bool  # the dispatcher has handed it to the engine
    var seen: Bool  # at least one engine event arrived
    var term: Bool  # a terminal event was delivered
    var ended: Bool  # the dispatcher closed it without a terminal event

    def __init__(out self):
        self.events = List[Ev]()
        self.cv = Cond()
        self.submitted = False
        self.seen = False
        self.term = False
        self.ended = False


struct Hub[E: EngineLike](Movable):
    var mu: Mutex
    var engine: Self.E  # the dispatcher thread only
    var inbox: List[GenRequest]
    var cancels: List[Int]
    var slots: Dict[Int, Int]  # request id -> address of its Slot
    var next_id: Int
    var stop: Bool
    var snap_state: Int
    var snap_name: String
    var snap_reason: String
    var snap_info: Json

    def __init__(out self, var engine: Self.E):
        self.mu = Mutex()
        self.snap_state = engine.state()
        self.snap_name = engine.state_name()
        self.snap_reason = engine.dead_reason()
        self.snap_info = engine.ready_info()
        self.engine = engine^
        self.inbox = List[GenRequest]()
        self.cancels = List[Int]()
        self.slots = Dict[Int, Int]()
        self.next_id = 1
        self.stop = False


comptime _SlotPtr = Pointer[Slot, MutUntrackedOrigin]


def _slot(addr: Int) -> _SlotPtr:
    return _SlotPtr(unsafe_from_address=addr)


def _is_terminal(kind: Int) -> Bool:
    return kind == EV_DONE or kind == EV_ERROR or kind == EV_DEAD


def _deliver[E: EngineLike](mut h: Hub[E], var ev: Ev):
    """Caller holds h.mu. EV_DEAD has no request id: it ends every live request."""
    if ev.kind == EV_DEAD:
        for e in h.slots.items():
            var s = _slot(e.value)
            if not s[].term:
                s[].events.append(ev.copy())
                s[].term = True
                s[].seen = True
                s[].cv.broadcast()
        return
    var addr = h.slots.get(ev.id, 0)
    if addr == 0:
        return  # a request whose handler already left (disconnect): drop its events
    var s = _slot(addr)
    var kind = ev.kind
    s[].events.append(ev^)
    s[].seen = True
    if _is_terminal(kind):
        s[].term = True
    s[].cv.broadcast()


def _dispatch_main[E: EngineLike](arg: Int) abi("C") -> Int:
    var hp = Pointer[Hub[E], MutUntrackedOrigin](unsafe_from_address=arg)
    _dispatch(hp[])
    return 0


def _dispatch[E: EngineLike](mut h: Hub[E]):
    """The engine's owner thread: submit, cancel, poll, route events, publish the state snapshot."""
    var evs = List[Ev]()
    while True:
        h.mu.lock()
        if h.stop:
            h.mu.unlock()
            return
        var subs = h.inbox^
        h.inbox = List[GenRequest]()
        var cans = h.cancels^
        h.cancels = List[Int]()
        h.mu.unlock()
        for g in subs:
            var err = String()
            try:
                h.engine.submit(g)
            except e:
                err = String(e)
            h.mu.lock()
            var addr = h.slots.get(g.id, 0)
            if addr != 0:
                if err == "":
                    _slot(addr)[].submitted = True
                elif h.engine.state() == ENG_DEAD or h.engine.state() == ENG_STOPPING:
                    var why = h.engine.dead_reason()
                    _deliver(h, Ev(EV_DEAD, g.id, 0, why if why != "" else err))  # ends every live request
                else:
                    _deliver(h, Ev(EV_ERROR, g.id, 0, err))  # this request only
            h.mu.unlock()
        for id in cans:
            _ = h.engine.cancel(id)
        evs.clear()
        var perr = String()
        try:
            _ = h.engine.poll(evs, 2)
        except e:
            perr = String(e)
        h.mu.lock()
        for ev in evs:
            var e2 = ev.copy()
            if e2.kind == EV_DEAD and e2.text == "":
                e2.text = h.engine.dead_reason()  # the client puts the exit code or signal there
            _deliver(h, e2^)
        if perr != "":
            _deliver(h, Ev(EV_DEAD, 0, 0, "engine poll failed: " + perr))
        var st = h.engine.state()
        if st != h.snap_state:
            h.snap_state = st
            h.snap_name = h.engine.state_name()
            h.snap_reason = h.engine.dead_reason()
            h.snap_info = h.engine.ready_info()
        if st == ENG_IDLE:
            # idle with a submitted request that never got a terminal event: close it with a name
            for e in h.slots.items():
                var s = _slot(e.value)
                if s[].submitted and not s[].term and len(evs) == 0:
                    s[].ended = True
                    s[].cv.broadcast()
        h.mu.unlock()


struct Batch(Movable):
    var evs: List[Ev]
    var ended: Bool
    var timed_out: Bool

    def __init__(out self):
        self.evs = List[Ev]()
        self.ended = False
        self.timed_out = False


struct Front[T: TextLike, E: EngineLike](Handler):
    var text: Self.T
    var hub: Int  # address of the heap Hub[E]
    var tid: UInt64
    var model: String
    var max_inflight: Int
    var keepalive_s: Float64

    def __init__(out self, var text: Self.T, var engine: Self.E, model: String = "mojo-baro", max_inflight: Int = 64, keepalive_s: Float64 = 15.0) raises:
        self.text = text^
        self.model = model
        self.max_inflight = max_inflight
        self.keepalive_s = keepalive_s
        var hp = alloc(Layout[Hub[Self.E]](count=1)).unsafe_leak()
        hp.unsafe_write(Hub[Self.E](engine^))
        self.hub = Int(hp)
        self.tid = 0
        var tids = List[UInt64](length=1, fill=0)
        var rc = external_call["pthread_create", Int32](tids.unsafe_ptr(), Int(0), _dispatch_main[Self.E], self.hub)
        if rc != 0:
            _ = hp.unsafe_take_pointee()
            hp.unsafe_free()
            raise Error("front: pthread_create rc ", rc)
        self.tid = tids[0]

    def __deinit__(deinit self):
        var hp = Pointer[Hub[Self.E], MutUntrackedOrigin](unsafe_from_address=self.hub)
        hp[].mu.lock()
        hp[].stop = True
        hp[].mu.unlock()
        _ = external_call["pthread_join", Int32](self.tid, Int(0))
        hp[].engine.shutdown(5000)  # stdin closed first: the engine ends its own way, a kill only after 5 s
        _ = hp.unsafe_take_pointee()
        hp.unsafe_free()

    def draining(self):
        pass

    def _h(self) -> Pointer[Hub[Self.E], MutUntrackedOrigin]:
        return Pointer[Hub[Self.E], MutUntrackedOrigin](unsafe_from_address=self.hub)

    # ---- routing --------------------------------------------------------------

    def handle(self, req: Request, mut resp: Response) raises:
        try:
            self._route(req, resp)
        except e:
            if resp.state == 0:
                self._json(resp, e.status, e.body)

    def _json(self, mut resp: Response, status: Int, body: String):
        resp.status = status
        resp.add("content-type", "application/json")
        resp.body.extend(body.as_bytes())

    def _route(self, req: Request, mut resp: Response) raises ApiError:
        var p = req.path
        var m = req.method
        if p == "/health" or p == "/props" or p == "/models" or p == "/v1/models":
            if m != "GET":
                return self._not_allowed(resp, "GET")
            if p == "/health":
                # 503 unless the engine can take work: `curl -sf /health` and the router's probe read the code
                return self._json(resp, 200 if self._alive() else 503, self._health())
            if p == "/props":
                return self._json(resp, 200, props_response(self.model, self._snap_tmax()))
            return self._json(resp, 200, models_response(self.model))
        var post = p == "/tokenize" or p == "/detokenize" or p == "/v1/cancel" or p == "/completions" or p == "/v1/completions" or p == "/chat/completions" or p == "/v1/chat/completions"
        if not post:
            resp.status = 404
            return
        if m != "POST":
            return self._not_allowed(resp, "POST")
        var j = parse_body(Span(req.body))
        if p == "/tokenize":
            return self._json(resp, 200, tokenize_response(self.text, j))
        if p == "/detokenize":
            return self._json(resp, 200, detokenize_response(self.text, j))
        if p == "/v1/cancel":
            return self._json(resp, 200, cancel_response(self._cancel(cancel_id(j))))
        var spec_k = self._snap_info().get("spec_k").int()
        if p == "/completions" or p == "/v1/completions":
            self._generate(parse_completion(self.text, j, spec_k), resp)
        else:
            self._generate(parse_chat(self.text, j, spec_k), resp)

    def _not_allowed(self, mut resp: Response, allow: String):
        resp.status = 405
        resp.add("allow", allow)

    # ---- hub access (handlers) ------------------------------------------------

    def _snap_info(self) -> Json:
        var h = self._h()
        h[].mu.lock()
        var j = h[].snap_info.copy()
        h[].mu.unlock()
        return j^

    def _snap_tmax(self) -> Int:
        return self._snap_info().get("tmax").int()

    def _alive(self) -> Bool:
        var h = self._h()
        h[].mu.lock()
        var st = h[].snap_state
        h[].mu.unlock()
        return st == ENG_IDLE or st == ENG_BUSY

    def _health(self) -> String:
        var h = self._h()
        h[].mu.lock()
        var out = health_response(h[].snap_state, h[].snap_name, h[].snap_reason, len(h[].slots), h[].snap_info)
        h[].mu.unlock()
        return out^

    def _cancel(self, id: Int) -> String:
        """`queued`: submitted, no event yet; `running`: events are flowing; `unknown`: not a live request."""
        var h = self._h()
        h[].mu.lock()
        var addr = h[].slots.get(id, 0)
        var state = String("unknown")
        if addr != 0 and not _slot(addr)[].term:
            state = "running" if _slot(addr)[].seen else "queued"
            h[].cancels.append(id)
        h[].mu.unlock()
        return state^

    def _request_cancel(self, id: Int):
        var h = self._h()
        h[].mu.lock()
        h[].cancels.append(id)
        h[].mu.unlock()

    def _release(self, id: Int):
        var h = self._h()
        h[].mu.lock()
        _ = h[].slots.pop(id, 0)
        h[].mu.unlock()

    def _next(self, sp: Int, seconds: Float64) -> Batch:
        """Wait up to `seconds` for events on a slot; takes everything queued."""
        var h = self._h()
        var s = _slot(sp)
        var b = Batch()
        var deadline = Int(perf_counter_ns()) + Int(seconds * 1e9)
        h[].mu.lock()
        while len(s[].events) == 0 and not s[].ended:
            var left = deadline - Int(perf_counter_ns())
            if left <= 0:
                b.timed_out = True
                break
            _ = s[].cv.timedwait(h[].mu, Float64(left) / 1e9)
        swap(b.evs, s[].events)
        b.ended = s[].ended and len(b.evs) == 0
        h[].mu.unlock()
        return b^

    # ---- generation -----------------------------------------------------------

    def _generate(self, var call: Call, mut resp: Response) raises ApiError:
        var h = self._h()
        var slot = Slot()
        var sp = Int(Pointer(to=slot))
        var id = 0
        var err = Optional[ApiError]()
        # one critical section: engine state, the length and admission check, and the registration, so two
        # requests cannot both slip under max_inflight
        h[].mu.lock()
        if h[].snap_state != ENG_IDLE and h[].snap_state != ENG_BUSY:
            var why = h[].snap_reason
            err = plain(503, "engine is " + h[].snap_name + (": " + why if why != "" else ""))
        else:
            try:
                check_gen(call.gen, h[].snap_info.get("tmax").int(), len(h[].slots), self.max_inflight, call.requested_n)
            except e:
                err = e^
        if not err:
            id = h[].next_id
            h[].next_id += 1
            h[].slots[id] = sp
            call.gen.id = id
            h[].inbox.append(call.gen.copy())
        h[].mu.unlock()
        if err:
            _ = len(slot.events)
            raise err.take()
        var oid = ("chatcmpl-" if call.chat else "cmpl-") + String(id)
        try:
            if call.stream:
                self._stream(call, oid, id, sp, resp)
            else:
                self._collect(call, oid, sp, resp)
        except e:
            self._release(id)
            _ = len(slot.events)  # the slot outlives every use of its address
            raise e^
        self._release(id)
        _ = len(slot.events)

    def _collect(self, call: Call, oid: String, sp: Int, mut resp: Response) raises ApiError:
        var acc = Acc[Self.T](self.text)
        var text_out = String()
        var stats = Json()
        var done = False
        while not done:
            var b = self._next(sp, 3600.0)
            if b.ended:
                raise plain(502, "engine stream ended without a terminal event")
            for ev in b.evs:
                if ev.kind == EV_TOKEN:
                    var d = acc.take(ev.tok)
                    if d:
                        text_out += d.value()
                elif ev.kind == EV_DONE:
                    stats = stats_json(ev.text)
                    if fail_word(stats) != 0:
                        raise invalid_tokens_error(fail_word(stats))
                    done = True
                    break
                elif ev.kind == EV_ERROR:
                    if len(acc.tokens) == 0 and ev.text.startswith("kv pool full"):
                        raise plain(429, "engine: " + ev.text)
                    raise plain(400 if len(acc.tokens) == 0 else 502, "engine: " + ev.text)
                elif ev.kind == EV_DEAD:
                    raise plain(503 if len(acc.tokens) == 0 else 502, "engine died: " + ev.text)
        var fin = stats.get("finish")
        var reason = acc.finish_reason(fin.str() if fin.kind() == 4 else "")
        var body: String
        if call.chat:
            var parsed = parse_tool_calls(call.force + text_out)
            body = chat_response(oid, now_secs(), self._model(call), parsed[0], parsed[1], acc.tokens, reason, call.n_prompt, stats)
        else:
            body = completion_response(oid, now_secs(), self._model(call), text_out, acc.tokens, reason, call.n_prompt, stats)
        self._json(resp, 200, body)

    def _model(self, call: Call) -> String:
        return call.model if call.model != "" else self.model

    def _send(self, mut resp: Response, data: String) -> Bool:
        """One SSE event; False when the client is gone."""
        try:
            resp.sse(data)
            return True
        except:
            return False

    def _stream(self, call: Call, oid: String, id: Int, sp: Int, mut resp: Response) raises ApiError:
        resp.add("cache-control", "no-cache")
        try:
            resp.begin_stream("text/event-stream")
        except:
            self._request_cancel(id)
            return
        var created = now_secs()
        var model = self._model(call)
        var acc = Acc[Self.T](self.text)
        var ts = ToolStream()
        var first = True
        var done = False
        while not done:
            var b = self._next(sp, self.keepalive_s)
            if b.timed_out:
                try:
                    resp.write(": keep-alive\n\n")
                except:
                    self._request_cancel(id)
                    return
                continue
            if b.ended:
                if not self._send(resp, engine_error_chunk("engine stream ended without a terminal event")):
                    self._request_cancel(id)
                    return
                break
            for ev in b.evs:
                var chunk = String()
                if ev.kind == EV_TOKEN:
                    var d = acc.take(ev.tok)
                    if not d:
                        continue
                    if call.chat:
                        var text = (call.force + d.value()) if first else d.value()
                        var tool = Optional[ToolDelta]()
                        var content = ts.push(text, tool)
                        chunk = chat_delta(oid, created, model, content, tool, first, ev.tok)
                        first = False
                    else:
                        chunk = completion_delta(oid, created, model, d.value(), ev.tok)
                elif ev.kind == EV_DONE:
                    var stats = stats_json(ev.text)
                    var fin = stats.get("finish")
                    var reason = acc.finish_reason(fin.str() if fin.kind() == 4 else "")
                    if fail_word(stats) != 0:
                        chunk = invalid_tokens_chunk(fail_word(stats))
                    elif call.chat:
                        chunk = chat_finish(oid, created, model, ts.finish_content(), reason, ts.has_tool(), call.n_prompt, len(acc.tokens), stats)
                    else:
                        chunk = completion_finish(oid, created, model, reason, call.n_prompt, acc.tokens, stats)
                    done = True
                elif ev.kind == EV_ERROR:
                    chunk = engine_error_chunk(ev.text)
                    done = True
                elif ev.kind == EV_DEAD:
                    chunk = engine_error_chunk("engine died: " + ev.text)
                    done = True
                if not self._send(resp, chunk):
                    self._request_cancel(id)
                    return
                if done:
                    break
        if not self._send(resp, "[DONE]"):
            return
        resp.end_stream()
