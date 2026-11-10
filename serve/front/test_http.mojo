"""test_http: the HTTP front over a real socket (`Server` on port 0), with FakeText and a scripted FakeEngine.

Per route and for stream: true: status, content type, PARSED body; the SSE event sequence ends in `data: [DONE]`.
Errors: bad JSON, engine DEAD, a stream with no terminal event, a disconnect mid-stream (the engine saw cancel).
This proves the module against fakes, not the front against a real engine.
Build: ./.venv/bin/mojo build serve/front/test_http.mojo -I . -I serve/front -o .work/test_http
"""
from api import (
    GenRequest, Ev, ApiError, ENG_IDLE, cancel_id, check_gen, fit_n, resolve_tool_choice, DEFAULT_MAX_TOKENS
)
from fakes import (
    FakeText, FakeEngine, Probe, FAKE_EOS, M_NORMAL, M_ERR_BEFORE, M_ERR_AFTER, M_NO_TERMINAL, M_SLOW, M_DEAD_MID,
    M_DEAD_IDLE, M_SUBMIT_FAIL, M_FAIL_WORD, P_SUBMITS, P_CANCELS, P_LAST_CANCEL, P_LAST_N
)
from http import Front
from bento.server import Server
from json import Json, parse, STR, ARR, OBJ
from posix import recv, send_all, close
from std.ffi import external_call
from std.memory import alloc, Layout


# ---- tally --------------------------------------------------------------------


struct Tally(Movable):
    var n: Int
    var ok: Int
    var failed: List[String]

    def __init__(out self):
        self.n = 0
        self.ok = 0
        self.failed = List[String]()

    def check(mut self, cond: Bool, name: String):
        self.n += 1
        if cond:
            self.ok += 1
        else:
            self.failed.append(name)
            print("FAIL", name)

    def eq(mut self, got: String, want: String, name: String):
        self.check(got == want, name + " (got `" + got + "`, want `" + want + "`)")

    def eqi(mut self, got: Int, want: Int, name: String):
        self.check(got == want, name + " (got " + String(got) + ", want " + String(want) + ")")


# ---- a tiny blocking HTTP/1.1 client ------------------------------------------


def connect_to(port: Int) raises -> Int32:
    var fd = external_call["socket", Int32](Int32(2), Int32(1 | 0o2000000), Int32(0))
    if fd < 0:
        raise Error("test: socket failed")
    var tv = SIMD[DType.int64, 2](10, 0)  # a 10 s receive timeout: a hang becomes a failure
    _ = external_call["setsockopt", Int32](fd, Int32(1), Int32(20), Pointer(to=tv), Int32(16))
    var addr = SIMD[DType.uint32, 4](UInt32(2) | (UInt32(port & 0xFF) << 24) | (UInt32(port >> 8) << 16), UInt32(0x0100007F), 0, 0)
    if external_call["connect", Int32](fd, Pointer(to=addr), Int32(16)) != 0:
        close(fd)
        raise Error("test: connect failed")
    return fd


def send_request(fd: Int32, method: String, path: String, body: String) -> Bool:
    var head = method + " " + path + " HTTP/1.1\r\nhost: t\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: " + String(body.byte_length()) + "\r\n\r\n"
    return send_all(fd, (head + body).as_bytes())


def read_to_eof(fd: Int32) -> List[UInt8]:
    var buf = List[UInt8]()
    while True:
        var n = recv(fd, buf, True)
        if n <= 0:
            break
    return buf^


struct Resp(Movable):
    var status: Int
    var names: List[String]
    var values: List[String]
    var body: String

    def __init__(out self):
        self.status = 0
        self.names = List[String]()
        self.values = List[String]()
        self.body = String()

    def header(self, name: String) -> String:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i]
        return ""

    def json(self) raises -> Json:
        return parse(self.body)


def _find(b: Span[UInt8, _], pat: String, from_: Int = 0) -> Int:
    var p = pat.as_bytes()
    var i = from_
    while i + len(p) <= len(b):
        var same = True
        for k in range(len(p)):
            if b[i + k] != p[k]:
                same = False
                break
        if same:
            return i
        i += 1
    return -1


def _hex(b: Span[UInt8, _]) -> Int:
    var n = 0
    for c in b:
        if c >= 48 and c <= 57:
            n = n * 16 + Int(c) - 48
        elif c >= 97 and c <= 102:
            n = n * 16 + Int(c) - 87
        elif c >= 65 and c <= 70:
            n = n * 16 + Int(c) - 55
    return n


def parse_response(raw: List[UInt8]) raises -> Resp:
    var r = Resp()
    var he = _find(Span(raw), "\r\n\r\n")
    if he < 0:
        raise Error("test: no response head (got " + String(len(raw)) + " bytes)")
    var head = String(unsafe_from_utf8=Span(raw)[0:he])
    var lines = head.split("\r\n")
    r.status = Int(lines[0].split(" ")[1])
    var chunked = False
    for i in range(1, len(lines)):
        var ln = String(lines[i])
        var c = ln.find(":")
        var lb = ln.as_bytes()
        var nm = String(unsafe_from_utf8=lb[0:c]).lower()
        var val = String(String(unsafe_from_utf8=lb[c + 1 :]).strip())
        r.names.append(nm)
        r.values.append(val)
        if nm == "transfer-encoding" and val == "chunked":
            chunked = True
    var rest = Span(raw)[he + 4 :]
    if not chunked:
        r.body = String(unsafe_from_utf8=rest)
        return r^
    var out = List[UInt8]()
    var i = 0
    while i < len(rest):
        var e = _find(rest, "\r\n", i)
        if e < 0:
            break
        var size = _hex(rest[i:e])
        if size == 0:
            break
        out.extend(rest[e + 2 : e + 2 + size])
        i = e + 2 + size + 2
    r.body = String(unsafe_from_utf8=Span(out))
    return r^


def call(port: Int, method: String, path: String, body: String) raises -> Resp:
    var fd = connect_to(port)
    if not send_request(fd, method, path, body):
        close(fd)
        raise Error("test: send failed")
    var raw = read_to_eof(fd)
    close(fd)
    return parse_response(raw)


# ---- SSE ----------------------------------------------------------------------


def sse_data(body: String) -> List[String]:
    """The `data:` payloads of a text/event-stream body, in order; comment lines are skipped."""
    var out = List[String]()
    for ev in body.split("\n\n"):
        for ln in ev.split("\n"):
            if ln.startswith("data: "):
                out.append(String(ln[byte=6:]))
    return out^


def sse_ok(mut t: Tally, events: List[String], name: String) -> Bool:
    var why = check_sse(events)
    t.eq(why, "", name)
    return why == ""


def check_sse(events: List[String]) -> String:
    """"" when the stream is well formed: JSON chunks, then `[DONE]` last and only once."""
    if len(events) == 0:
        return "no events"
    for i in range(len(events)):
        if events[i] == "[DONE]":
            if i != len(events) - 1:
                return "[DONE] before the end"
        else:
            try:
                _ = parse(events[i])
            except e:
                return "chunk " + String(i) + " is not JSON: " + String(e)
    if events[len(events) - 1] != "[DONE]":
        return "no [DONE] at the end"
    return ""


# ---- rig: Front + Server on a thread ------------------------------------------

comptime _Front = Front[FakeText, FakeEngine]
comptime _Srv = Server[_Front]


def _serve_main(arg: Int) abi("C") -> Int:
    var s = Pointer[_Srv, MutUntrackedOrigin](unsafe_from_address=arg)
    try:
        s[].serve(4)
    except:
        pass
    return 0


struct Rig(Movable):
    var sp: Pointer[_Srv, MutUntrackedOrigin]
    var tid: UInt64
    var port: Int
    var probe: Probe

    def __init__(out self, mode: Int = M_NORMAL, var script: List[Int32] = List[Int32](), max_inflight: Int = 64, keepalive_s: Float64 = 15.0, xml: Bool = True, model: String = "mojo-baro") raises:
        self.probe = Probe()
        var eng = FakeEngine(mode, script^, self.probe.addr())
        var front = _Front(FakeText(xml), eng^, model, max_inflight, keepalive_s)
        var srv = _Srv(front^, "127.0.0.1", 0)
        self.port = srv.port()
        self.sp = alloc(Layout[_Srv](count=1)).unsafe_leak()
        self.sp.unsafe_write(srv^)
        var tids = List[UInt64](length=1, fill=0)
        _ = external_call["pthread_create", Int32](tids.unsafe_ptr(), Int(0), _serve_main, Int(self.sp))
        self.tid = tids[0]
        _ = external_call["usleep", Int32](Int32(30000))  # the workers are up

    def close(deinit self):
        self.sp[].stop()
        _ = external_call["pthread_join", Int32](self.tid, Int(0))
        _ = self.sp.unsafe_take_pointee()
        self.sp.unsafe_free()


def ids(s: String) raises -> List[Int32]:
    return FakeText().encode(s, False)


def wait_for(probe: Probe, slot: Int, at_least: Int, ms: Int) -> Bool:
    for _ in range(ms // 10):
        if probe.get(slot) >= at_least:
            return True
        _ = external_call["usleep", Int32](Int32(10000))
    return probe.get(slot) >= at_least


def err_of(r: Resp) raises -> Json:
    return r.json().get("error")


def chat_body(extra: String = "", stream: Bool = False) -> String:
    return '{"messages":[{"role":"user","content":"hi"}]' + (',"stream":true' if stream else "") + extra + "}"


# ---- the cases ----------------------------------------------------------------


def case_get_routes(mut t: Tally) raises:
    var rig = Rig(M_NORMAL)
    var h = call(rig.port, "GET", "/health", "")
    t.eqi(h.status, 200, "health status")
    t.eq(h.header("content-type"), "application/json", "health content-type")
    var hj = h.json()
    t.eq(hj.text("status"), "ok", "health status field")
    t.eq(hj.text("engine_state"), "idle", "health engine_state")
    t.eqi(hj.get("limits").get("tmax").int(), 16384, "health tmax")
    t.eq(hj.text("pack"), "fake-pack", "health pack")
    t.check(hj.get("tokenizer").int() == 1, "health tokenizer")
    for p in ["/v1/models", "/models"]:
        var m = call(rig.port, "GET", p, "")
        t.eqi(m.status, 200, p + " status")
        var mj = m.json()
        t.eq(mj.text("object"), "list", p + " object")
        t.eq(mj.get("data").at(0).text("id"), "mojo-baro", p + " id")
        t.eq(mj.get("data").at(0).text("owned_by"), "mojo-baro", p + " owned_by")
    var pr = call(rig.port, "GET", "/props", "")
    var pj = pr.json()
    t.eqi(pj.get("default_generation_settings").get("n_ctx").int(), 16384, "props n_ctx")
    t.eqi(pj.get("default_generation_settings").get("n_predict").int(), -1, "props n_predict")
    t.check(pj.get("chat_template_caps").get("supports_tools").int() == 1, "props caps")
    t.eqi(call(rig.port, "GET", "/nope", "").status, 404, "404")
    t.eqi(call(rig.port, "GET", "/v1/chat/completions", "").status, 405, "405 get on post route")
    t.eqi(call(rig.port, "POST", "/health", "{}").status, 405, "405 post on get route")
    rig^.close()


def case_tokenize(mut t: Tally) raises:
    var rig = Rig(M_NORMAL)
    var a = call(rig.port, "POST", "/tokenize", '{"content":"hi","add_special":true}')
    t.eq(a.json().get("tokens").dumps(), "[1,107,108]", "tokenize add_special")
    var b = call(rig.port, "POST", "/tokenize", '{"content":"hi"}')
    t.eq(b.json().get("tokens").dumps(), "[107,108]", "tokenize default")
    var c = call(rig.port, "POST", "/detokenize", '{"tokens":[107,108]}')
    t.eq(c.json().text("content"), "hi", "detokenize")
    var d = call(rig.port, "POST", "/tokenize", "{}")
    t.eqi(d.status, 400, "tokenize missing content")
    var e = call(rig.port, "POST", "/detokenize", '{"tokens":[-1]}')
    t.eqi(e.status, 400, "detokenize bad id")
    rig^.close()


def case_completions(mut t: Tally) raises:
    var rig = Rig(M_NORMAL, ids("ok"))
    var r = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,107,108],"max_tokens":8,"model":"anything"}')
    t.eqi(r.status, 200, "completions status")
    t.eq(r.header("content-type"), "application/json", "completions content-type")
    var j = r.json()
    t.eq(j.text("object"), "text_completion", "object")
    t.check(j.text("id").startswith("cmpl-"), "id prefix")
    t.eq(j.text("model"), "anything", "model is echoed, as Rust")
    t.check(j.get("created").int() > 1700000000, "created is epoch seconds")
    var ch = j.get("choices").at(0)
    t.eq(ch.text("text"), "ok", "text")
    t.eq(ch.get("tokens").dumps(), "[114,110]", "tokens")
    t.eq(ch.text("finish_reason"), "length", "finish_reason")
    t.check(ch.get("logprobs").is_null(), "logprobs null")
    var u = j.get("usage")
    t.eqi(u.get("prompt_tokens").int(), 3, "prompt_tokens")
    t.eqi(u.get("completion_tokens").int(), 2, "completion_tokens")
    t.eqi(u.get("total_tokens").int(), 5, "total_tokens")
    t.eqi(u.get("baro").get("cached_tokens").int(), 0, "cached_tokens")
    t.eqi(u.get("baro").get("prefill_rows").int(), 7, "prefill_rows")
    t.eq(j.get("timings").get("tok_s_gen").dumps(), "4.0", "timings tok_s_gen")
    t.check(j.get("timings").get("restore_s").is_null(), "absent timing is null")
    # a string prompt goes through the tokenizer with specials
    var s = call(rig.port, "POST", "/v1/completions", '{"prompt":"hi","max_tokens":4}')
    t.eqi(s.json().get("usage").get("prompt_tokens").int(), 3, "string prompt: bos + 2 bytes")
    # the stop token cuts the text, not the token list
    rig^.close()
    var rig2 = Rig(M_NORMAL, ids("o") + [FAKE_EOS] + ids("k"))
    var q = call(rig2.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":8}')
    var qc = q.json().get("choices").at(0)
    t.eq(qc.text("text"), "o", "stop id cuts the text")
    t.eq(qc.get("tokens").dumps(), "[114,2,110]", "stop id stays in tokens")
    rig2^.close()


def case_completions_stream(mut t: Tally) raises:
    var rig = Rig(M_NORMAL, ids("ok"))
    var r = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,107],"max_tokens":8,"stream":true}')
    t.eqi(r.status, 200, "stream status")
    t.eq(r.header("content-type"), "text/event-stream", "stream content-type")
    t.eq(r.header("cache-control"), "no-cache", "stream cache-control")
    t.eq(r.header("transfer-encoding"), "chunked", "stream is chunked")
    var ev = sse_data(r.body)
    if not sse_ok(t, ev, "stream is well formed"):
        return
    t.eqi(len(ev), 4, "2 deltas + finish + [DONE]")
    var text = String()
    for i in range(2):
        var c = parse(ev[i])
        t.eq(c.text("object"), "text_completion", "delta object")
        t.check(c.get("choices").at(0).get("finish_reason").is_null(), "delta finish_reason null")
        t.eqi(c.get("choices").at(0).get("tokens").len(), 1, "delta carries its token")
        text += c.get("choices").at(0).text("text")
    t.eq(text, "ok", "deltas spell the text")
    var fin = parse(ev[2])
    t.eq(fin.get("choices").at(0).text("finish_reason"), "length", "finish chunk reason")
    t.eq(fin.get("choices").at(0).text("text"), "", "finish chunk text")
    t.eqi(fin.get("usage").get("completion_tokens").int(), 2, "finish usage")
    t.eq(fin.get("tokens").dumps(), "[114,110]", "finish chunk tokens")
    t.check(not fin.get("timings").is_null(), "finish timings")
    t.eq(ev[3], "[DONE]", "[DONE] last")
    rig^.close()


def case_chat(mut t: Tally) raises:
    var rig = Rig(M_NORMAL, ids("ok"))
    var n_prompt = len(FakeText().encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", True))
    var r = call(rig.port, "POST", "/v1/chat/completions", chat_body())
    t.eqi(r.status, 200, "chat status")
    var j = r.json()
    t.eq(j.text("object"), "chat.completion", "chat object")
    t.check(j.text("id").startswith("chatcmpl-"), "chat id prefix")
    t.check(j.get("checkpoint").is_null() and j.get("checkpoint_from").is_null(), "checkpoint fields null")
    var ch = j.get("choices").at(0)
    t.eq(ch.get("message").text("role"), "assistant", "role")
    t.eq(ch.get("message").text("content"), "ok", "content")
    t.check(not ch.get("message").has("tool_calls"), "no tool_calls")
    t.eqi(j.get("usage").get("prompt_tokens").int(), n_prompt, "chat prompt_tokens = rendered template")
    # the stream: content is held back by the tool-call scanner and ends in the finish delta
    var s = call(rig.port, "POST", "/chat/completions", chat_body("", True))
    var ev = sse_data(s.body)
    if not sse_ok(t, ev, "chat stream well formed"):
        return
    var first = parse(ev[0])
    t.eq(first.text("object"), "chat.completion.chunk", "chunk object")
    t.eq(first.get("choices").at(0).get("delta").text("role"), "assistant", "first delta has the role")
    var text = String()
    for i in range(len(ev) - 1):
        var c = parse(ev[i])
        if i > 0 and c.get("choices").at(0).get("delta").text("role").byte_length() > 0:
            t.check(False, "only the first delta has a role")
        text += c.get("choices").at(0).get("delta").text("content")
    t.eq(text, "ok", "chat stream content")
    var fin = parse(ev[len(ev) - 2])
    t.eq(fin.get("choices").at(0).text("finish_reason"), "length", "chat finish reason")
    t.check(fin.get("usage").has("prompt_tokens"), "chat finish usage")
    rig^.close()


def case_tools(mut t: Tally) raises:
    var call_text = "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    var rig = Rig(M_NORMAL, ids(call_text))
    var tools = ',"tools":[{"type":"function","function":{"name":"get_weather"}}]'
    var r = call(rig.port, "POST", "/v1/chat/completions", chat_body(tools))
    var ch = r.json().get("choices").at(0)
    t.eq(ch.text("finish_reason"), "tool_calls", "finish_reason tool_calls")
    t.eq(ch.get("message").text("content"), "Checking.", "content before the call")
    var tc = ch.get("message").get("tool_calls").at(0)
    t.eq(tc.text("id"), "call_0", "tool call id")
    t.eq(tc.text("type"), "function", "tool call type")
    t.eq(tc.get("function").text("name"), "get_weather", "tool call name")
    var args = parse(tc.get("function").text("arguments"))
    t.eq(args.text("city"), "Paris", "string parameter")
    t.eqi(args.get("days").int(), 3, "typed parameter")
    # streamed: a name delta, then argument pieces that join into the same object
    var s = call(rig.port, "POST", "/v1/chat/completions", chat_body(tools, True))
    var ev = sse_data(s.body)
    if not sse_ok(t, ev, "tool stream well formed"):
        return
    var name = String()
    var args_text = String()
    var content = String()
    var last_reason = String()
    for i in range(len(ev) - 1):
        var c = parse(ev[i]).get("choices").at(0)
        var d = c.get("delta")
        content += d.text("content")
        if d.has("tool_calls"):
            var f = d.get("tool_calls").at(0).get("function")
            if f.has("name"):
                name = f.text("name")
            args_text += f.text("arguments")
        if c.get("finish_reason").kind() == 4:
            last_reason = c.text("finish_reason")
    t.eq(name, "get_weather", "stream tool name")
    t.eq(parse(args_text).dumps(), '{"city":"Paris","days":3}', "stream arguments join to the object")
    t.eq(content, "Checking.\n", "stream content before the call")
    t.eq(last_reason, "tool_calls", "stream finish reason")
    rig^.close()
    # tool_choice
    var tj = parse('[{"type":"function","function":{"name":"get_weather"}}]')
    var named = parse('{"type":"function","function":{"name":"get_weather"}}')
    t.eq(force_of(tj, parse('"required"'), True), "<tool_call>\n<function=", "required, xml")
    t.eq(force_of(tj, named, True), "<tool_call>\n<function=get_weather>\n", "named, xml")
    t.eq(force_of(tj, named, False), '<tool_call>\n{"name": "get_weather", "arguments": ', "named, json")
    t.eq(force_of(tj, parse('"none"'), True), "none:" + "null", "none drops the tools")
    t.eq(force_of(tj, parse('{"type":"function","function":{"name":"other"}}'), True), "ERR 400", "tool_choice names a tool that is not there: 400")


def force_of(tools: Json, choice: Json, xml: Bool) -> String:
    """The forced text of `resolve_tool_choice`; `none:null` when the tools are dropped; `ERR <status>`."""
    try:
        var r = resolve_tool_choice(tools, choice, xml)
        if r[0].is_null():
            return "none:null"
        return r[1]
    except e:
        return "ERR " + String(e.status)


def cid(j: Json) -> Int:
    try:
        return cancel_id(j)
    except:
        return -1


def case_errors(mut t: Tally) raises:
    var rig = Rig(M_NORMAL, ids("ok"))
    var a = call(rig.port, "POST", "/v1/completions", "{not json")
    t.eqi(a.status, 400, "bad JSON status")
    t.eq(err_of(a).text("type"), "invalid_request_error", "bad JSON type")
    t.eqi(err_of(a).get("code").int(), 400, "bad JSON code")
    t.eqi(call(rig.port, "POST", "/v1/completions", "[1]").status, 400, "JSON that is not an object")
    t.eqi(call(rig.port, "POST", "/v1/completions", "{}").status, 400, "missing prompt")
    t.eqi(call(rig.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":-1}').status, 400, "negative max_tokens")
    t.eqi(call(rig.port, "POST", "/v1/completions", '{"prompt":{"a":1}}').status, 400, "prompt of the wrong type")
    t.eqi(call(rig.port, "POST", "/v1/chat/completions", '{"messages":[]}').status, 400, "empty messages")
    var u = call(rig.port, "POST", "/v1/chat/completions", chat_body(',"checkpoint":true'))
    t.eqi(u.status, 400, "unserved field is a named 400")
    t.check(err_of(u).text("message").find("checkpoint") >= 0, "the 400 names the field")
    t.eqi(rig.probe.get(P_SUBMITS), 0, "no rejected request reached the engine")
    # the length guard: one place, before the engine
    var big = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,2,3],"max_tokens":20000}')
    t.eqi(big.status, 400, "over the window: status")
    t.eq(err_of(big).text("type"), "exceed_context_size_error", "over the window: type")
    t.eqi(err_of(big).get("n_prompt_tokens").int(), 3, "over the window: n_prompt_tokens")
    t.eqi(err_of(big).get("n_ctx").int(), 16384, "over the window: n_ctx")
    t.eq(err_of(call(rig.port, "POST", "/v1/completions", '{"prompt":[],"max_tokens":4}')).text("message"), "prompt is empty", "empty prompt")
    t.eq(err_of(call(rig.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":0}')).text("message"), "max_tokens must be >= 1", "max_tokens 0")
    t.eqi(rig.probe.get(P_SUBMITS), 0, "the length guard ran before the engine")
    # an omitted max_tokens fills the window
    _ = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,2,3]}')
    t.eqi(rig.probe.get(P_LAST_N), 16384 - 3, "omitted max_tokens means up to the window")
    rig^.close()


def case_engine_failures(mut t: Tally) raises:
    # an error line before any token: 400 for JSON, an error chunk for SSE
    var r1 = Rig(M_ERR_BEFORE)
    var a = call(r1.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(a.status, 400, "error before a token: status")
    t.eq(err_of(a).text("message"), "engine: prompt rejected", "error before a token: message")
    var s = call(r1.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4,"stream":true}')
    var ev = sse_data(s.body)
    if not sse_ok(t, ev, "error stream well formed"):
        return
    t.eq(parse(ev[0]).get("error").text("type"), "engine_error", "error chunk type")
    t.eq(parse(ev[0]).get("error").text("message"), "engine: prompt rejected", "error chunk message")
    r1^.close()
    # an error line after tokens: 502
    var r2 = Rig(M_ERR_AFTER, ids("ok"))
    var b = call(r2.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(b.status, 502, "error after a token: status")
    t.eq(err_of(b).text("message"), "engine: boom", "error after a token: message")
    r2^.close()
    # the stream that ends with no terminal event (Rust: a partial answer as success)
    var r3 = Rig(M_NO_TERMINAL, ids("ok"))
    var c = call(r3.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(c.status, 502, "no terminal event: status")
    t.eq(err_of(c).text("message"), "engine stream ended without a terminal event", "no terminal event: message")
    var cs = call(r3.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4,"stream":true}')
    var cev = sse_data(cs.body)
    if not sse_ok(t, cev, "no terminal: the stream still ends with [DONE]"):
        return
    t.eqi(len(cev), 4, "no terminal: 2 deltas, an error chunk, [DONE]")
    t.eq(parse(cev[2]).get("error").text("message"), "engine: engine stream ended without a terminal event", "no terminal: error chunk before [DONE]")
    t.eq(parse(call(r3.port, "GET", "/health", "").body).text("engine_state"), "idle", "the front stays up")
    r3^.close()
    # the child dies mid-stream: a named error, the front stays up
    var r4 = Rig(M_DEAD_MID, ids("ok"))
    var d = call(r4.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(d.status, 502, "death mid-stream: status")
    t.eq(err_of(d).text("message"), "engine died: killed by signal 9", "death mid-stream: message")
    var h = parse(call(r4.port, "GET", "/health", "").body)
    t.eq(h.text("status"), "engine_dead", "health after death: status")
    t.eq(h.text("engine_reason"), "killed by signal 9", "health after death: reason")
    var e = call(r4.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(e.status, 503, "submit when dead: status")
    t.check(err_of(e).text("message").find("killed by signal 9") >= 0, "submit when dead names the reason")
    r4^.close()
    # a submit that fails while the engine stays up ends that request only
    var r6 = Rig(M_SUBMIT_FAIL)
    var f1 = call(r6.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(f1.status, 400, "submit failure, engine up: status")
    t.eq(err_of(f1).text("message"), "engine: request line too long", "submit failure, engine up: message")
    t.eq(parse(call(r6.port, "GET", "/health", "").body).text("status"), "ok", "submit failure does not kill the front")
    r6^.close()
    # a done line with a non-zero fail word: the tokens are invalid, never a 200 (serve/PROTOCOL.md, State words)
    var r7 = Rig(M_FAIL_WORD)
    var w1 = call(r7.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(w1.status, 502, "fail word: status")
    t.eq(err_of(w1).text("type"), "engine_invalid_tokens", "fail word: type")
    t.check(err_of(w1).text("message").find("fail word 2") >= 0, "fail word: message names the word")
    var w2 = call(r7.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4,"stream":true}')
    t.check(w2.body.find("engine_invalid_tokens") >= 0 and w2.body.find("[DONE]") >= 0, "fail word, stream: error chunk then [DONE]")
    t.check(w2.body.find("finish_reason\":\"length") < 0 and w2.body.find("finish_reason\":\"stop") < 0, "fail word, stream: no success finish chunk")
    r7^.close()
    # dead before any request
    var r5 = Rig(M_DEAD_IDLE)
    t.eq(parse(call(r5.port, "GET", "/health", "").body).text("status"), "engine_dead", "idle death: health")
    t.eqi(call(r5.port, "GET", "/health", "").status, 503, "idle death: /health is 503, so curl -sf and the router see it")
    t.eqi(call(r5.port, "POST", "/v1/chat/completions", chat_body()).status, 503, "idle death: 503")
    r5^.close()


def read_until_data(fd: Int32, mut buf: List[UInt8], n_events: Int) -> Bool:
    """Read from a live stream until `n_events` SSE `data:` lines arrived."""
    while True:
        var seen = 0
        var i = 0
        while True:
            var k = _find(Span(buf), "data: ", i)
            if k < 0:
                break
            seen += 1
            i = k + 6
        if seen >= n_events:
            return True
        if recv(fd, buf, True) <= 0:
            return False


def case_cancel(mut t: Tally) raises:
    # a client that disconnects mid-stream cancels the request
    var rig = Rig(M_SLOW)
    var fd = connect_to(rig.port)
    _ = send_request(fd, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":500,"stream":true}')
    var buf = List[UInt8]()
    t.check(read_until_data(fd, buf, 2), "slow stream delivers events")
    close(fd)
    t.check(wait_for(rig.probe, P_CANCELS, 1, 3000), "disconnect mid-stream: the engine saw cancel")
    t.eqi(rig.probe.get(P_LAST_CANCEL), 1, "disconnect mid-stream: cancel names the request")
    rig^.close()
    # /v1/cancel on a live request, from a second connection
    var r2 = Rig(M_SLOW)
    var fd2 = connect_to(r2.port)
    _ = send_request(fd2, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":500,"stream":true}')
    var buf2 = List[UInt8]()
    t.check(read_until_data(fd2, buf2, 1), "second stream delivers events")
    var c = call(r2.port, "POST", "/v1/cancel", '{"id":"cmpl-1"}')
    t.eqi(c.status, 200, "cancel status")
    t.eq(c.json().text("state"), "running", "cancel state of a streaming request")
    t.check(c.json().get("cancelled").int() == 1, "cancelled true")
    var rest = read_to_eof(fd2)
    close(fd2)
    for b in rest:
        buf2.append(b)
    var all = parse_response(buf2)
    var ev = sse_data(all.body)
    if not sse_ok(t, ev, "cancelled stream well formed"):
        return
    t.eq(parse(ev[len(ev) - 2]).get("choices").at(0).text("finish_reason"), "cancelled", "engine's own finish reason wins")
    t.eq(call(r2.port, "POST", "/v1/cancel", '{"id":"cmpl-99"}').json().text("state"), "unknown", "cancel state of an unknown id")
    t.eqi(call(r2.port, "POST", "/v1/cancel", '{"id":"nonumber"}').status, 400, "cancel id without a number")
    r2^.close()


def case_inflight_and_keepalive(mut t: Tally) raises:
    var rig = Rig(M_SLOW, List[Int32](), 1, 0.001)
    var fd = connect_to(rig.port)
    _ = send_request(fd, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":30,"stream":true}')
    var buf = List[UInt8]()
    t.check(read_until_data(fd, buf, 1), "first stream is live")
    var b = call(rig.port, "POST", "/v1/completions", '{"prompt":[1],"max_tokens":4}')
    t.eqi(b.status, 429, "over max_inflight: 429")
    t.eq(err_of(b).text("message"), "1 requests in flight (--max-inflight)", "429 message")
    var rest = read_to_eof(fd)
    close(fd)
    for x in rest:
        buf.append(x)
    var all = parse_response(buf)
    t.check(all.body.find(": keep-alive") >= 0, "keep-alive comments flow while the engine is quiet")
    if not sse_ok(t, sse_data(all.body), "keep-alive does not break the stream"):
        return
    rig^.close()


def gen_status(mut g: GenRequest, inflight: Int) -> Int:
    try:
        check_gen(g, 16384, inflight, 64, 4)
        return 0
    except e:
        return e.status


# ---- the Rust front as oracle -------------------------------------------------
# serve/front/testdata/http-oracle.jsonl: what baro-serve (Rust) answered for the same requests, captured by
# oracle-http/capture.py with a scripted engine behind it. PARSED values are compared (key order differs: Rust
# keeps insertion order, ours is the same, but nothing here relies on it); `id` and `created` are masked at the top.


def num_eq(a: Json, b: Json) -> Bool:
    return a.num() == b.num()


def json_eq(a: Json, b: Json, mask_top: Bool, subset: Bool) -> Bool:
    """Parsed equality; `subset`: every member of `a` is in `b` (b may carry more); mask_top skips id and created."""
    if a.is_num() and b.is_num():
        return num_eq(a, b)
    if a.kind() != b.kind():
        return False
    if a.kind() == ARR:
        if a.len() != b.len():
            return False
        for i in range(a.len()):
            if not json_eq(a.at(i), b.at(i), False, subset):
                return False
        return True
    if a.kind() == OBJ:
        if not subset and a.len() != b.len():
            return False
        for i in range(a.len()):
            var k = a.key_at(i)
            if mask_top and (k == "id" or k == "created"):
                if not b.has(k):
                    return False
                continue
            if not b.has(k) or not json_eq(a.at(i), b.get(k), False, subset):
                return False
        return True
    if a.kind() == STR:
        return a.str() == b.str()
    return a.dumps() == b.dumps()


def row_policy(name: String) -> String:
    """full: parsed bodies equal; subset: ours carries every Rust field; shape: both reject, 4xx status may differ.

    Every row not `full` is a DELIBERATE DIFFERENCE, listed in the report."""
    if name == "get-health" or name == "cancel-unknown":
        return "subset"  # we add engine_state / engine_reason / state
    if name == "err-bad-json" or name == "err-no-prompt" or name == "err-tokenize-no-content":
        return "shape"  # serde rejection text and status (422) are not ours
    return "full"


def case_oracle(mut t: Tally) raises:
    var text = String()
    with open("serve/front/testdata/http-oracle.jsonl", "r") as f:
        text = f.read()
    var n_rows = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var row = parse(String(line))
        n_rows += 1
        var name = row.text("name")
        var script = List[Int32]()
        for x in row.get("script").items():
            script.append(Int32(x.int()))
        var mode = M_NORMAL
        if row.text("mode") == "err_before":
            mode = M_ERR_BEFORE
        elif row.text("mode") == "err_after":
            mode = M_ERR_AFTER
        var rig = Rig(mode, script^, 64, 15.0, False, "pack")  # Rust names the model after the pack directory
        var got = call(rig.port, row.text("method"), row.text("path"), row.text("request"))
        rig^.close()
        var policy = row_policy(name)
        if policy == "shape":
            t.check(got.status >= 400 and got.status < 500, "oracle " + name + ": rejected with a 4xx (ours " + String(got.status) + ", Rust " + String(row.get("status").int()) + ")")
            t.check(got.json().get("error").has("message"), "oracle " + name + ": error shape")
            continue
        t.eqi(got.status, row.get("status").int(), "oracle " + name + ": status")
        var kind = row.text("kind")
        if kind == "json":
            t.eq(got.header("content-type"), row.text("content_type"), "oracle " + name + ": content-type")
            t.check(json_eq(row.get("body"), got.json(), True, policy == "subset"), "oracle " + name + ": parsed body (" + policy + ")\n  rust: " + row.get("body").dumps() + "\n  ours: " + got.body)
        elif kind == "sse":
            t.eq(got.header("content-type"), row.text("content_type"), "oracle " + name + ": content-type")
            var ev = sse_data(got.body)
            var want = row.get("sse")
            t.eqi(len(ev), want.len(), "oracle " + name + ": event count")
            if len(ev) == want.len():
                for i in range(len(ev)):
                    var w = want.at(i)
                    if w.kind() == STR:
                        t.eq(ev[i], w.str(), "oracle " + name + ": event " + String(i))
                    else:
                        t.check(json_eq(w, parse(ev[i]), True, False), "oracle " + name + ": event " + String(i) + "\n  rust: " + w.dumps() + "\n  ours: " + ev[i])
    t.check(n_rows >= 40, "oracle rows read: " + String(n_rows))


struct Job(Movable):
    var port: Int
    var body: String
    var status: Int
    var out: String

    def __init__(out self, port: Int, body: String):
        self.port = port
        self.body = body
        self.status = 0
        self.out = String()


def _client_main(arg: Int) abi("C") -> Int:
    var j = Pointer[Job, MutUntrackedOrigin](unsafe_from_address=arg)
    try:
        var r = call(j[].port, "POST", "/v1/completions", j[].body)
        j[].status = r.status
        j[].out = r.body
    except:
        j[].status = -1
    return 0


def case_concurrent(mut t: Tally) raises:
    """16 clients at once on 4 workers: every answer goes to its own request (prompt_tokens = its prompt length)."""
    var rig = Rig(M_NORMAL, ids("ok"))
    var n = 16
    var jobs = List[Job](capacity=n)
    for i in range(n):
        var prompt = String("[")
        for k in range(i + 1):
            prompt += ("," if k > 0 else "") + "1"
        jobs.append(Job(rig.port, '{"prompt":' + prompt + '],"max_tokens":8}'))
    var tids = List[UInt64](length=n, fill=0)
    for i in range(n):
        _ = external_call["pthread_create", Int32](tids.unsafe_ptr().unsafe_offset(i), Int(0), _client_main, Int(jobs.unsafe_ptr().unsafe_offset(i)))
    for i in range(n):
        _ = external_call["pthread_join", Int32](tids[i], Int(0))
    var ids_seen = List[String]()
    var routed = True
    var all_ok = True
    for i in range(n):
        if jobs[i].status != 200:
            all_ok = False
            continue
        var j = parse(jobs[i].out)
        if j.get("usage").get("prompt_tokens").int() != i + 1 or j.get("choices").at(0).text("text") != "ok":
            routed = False
        ids_seen.append(j.text("id"))
    t.check(all_ok, "16 concurrent requests all answered 200")
    t.check(routed, "each answer belongs to its request")
    var distinct = True
    for a in range(len(ids_seen)):
        for b in range(a + 1, len(ids_seen)):
            if ids_seen[a] == ids_seen[b]:
                distinct = False
    t.check(distinct, "request ids are distinct")
    t.eqi(rig.probe.get(P_SUBMITS), n, "the engine saw all 16")
    t.eq(parse(call(rig.port, "GET", "/health", "").body).get("queue").dumps(), "0", "nothing left in flight")
    rig^.close()


def rss_kb() raises -> Int:
    var text = String()
    with open("/proc/self/status", "r") as f:
        text = f.read()
    for ln in text.split("\n"):
        if ln.startswith("VmRSS:"):
            return Int(String(ln.split()[1]))
    return -1


def cpu_ticks() raises -> Int:
    var text = String()
    with open("/proc/self/stat", "r") as f:
        text = f.read()
    var f2 = text.split(") ")[1].split(" ")  # after the comm field: state is index 0, utime 11, stime 12
    return Int(String(f2[11])) + Int(String(f2[12]))


def case_footprint(mut t: Tally) raises:
    """No leak: RSS after 1000 requests on one rig stays within 4 MB of RSS after 200. Prints the idle CPU of
    the dispatcher poll (2 ms) and the per-request wall time, as numbers for the commit body."""
    var rig = Rig(M_NORMAL, ids("ok"))
    for _ in range(200):
        _ = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,107],"max_tokens":8}')
    var before = rss_kb()
    var t0 = cpu_ticks()
    for _ in range(1000):
        _ = call(rig.port, "POST", "/v1/completions", '{"prompt":[1,107],"max_tokens":8}')
    var after = rss_kb()
    var busy = cpu_ticks() - t0
    t.check(after - before < 4096, "no leak over 1000 requests: RSS " + String(before) + " -> " + String(after) + " KB")
    var i0 = cpu_ticks()
    _ = external_call["usleep", Int32](Int32(2000000))
    print("footprint: 1000 requests cost", busy * 10, "ms CPU (client + server, one process); idle 2 s cost", (cpu_ticks() - i0) * 10, "ms CPU; RSS", after, "KB")
    rig^.close()


def case_units(mut t: Tally) raises:
    t.eqi(cid(parse('{"id":"chatcmpl-12"}')), 12, "cancel_id chatcmpl")
    t.eqi(cid(parse('{"id":"chatcmpl-"}')), -1, "cancel_id without a number")
    t.eqi(fit_n(DEFAULT_MAX_TOKENS, 100, 16384), 16284, "fit_n window")
    t.eqi(fit_n(DEFAULT_MAX_TOKENS, 16384, 16384), 0, "fit_n full window")
    t.eqi(fit_n(500, 100, 16384), 500, "fit_n explicit")
    t.eqi(fit_n(20000, 100, 16384), 20000, "fit_n over")
    var g = GenRequest()
    g.prompt = [1, 2]
    t.eqi(gen_status(g, 64), 429, "check_gen inflight")
    t.eqi(gen_status(g, 0), 0, "check_gen ok")
    g.prompt = List[Int32]()
    t.eqi(gen_status(g, 0), 400, "check_gen empty prompt")
    # the SSE checker can fail: a stream without [DONE], with a bad chunk, with [DONE] early
    var good = List[String]()
    good.append('{"a":1}')
    good.append("[DONE]")
    if not sse_ok(t, good, "checker accepts a good stream"):
        return
    var no_done = List[String]()
    no_done.append('{"a":1}')
    t.check(check_sse(no_done) != "", "MUTANT: omit [DONE]: the checker fails")
    var early = List[String]()
    early.append("[DONE]")
    early.append('{"a":1}')
    t.check(check_sse(early) != "", "MUTANT: [DONE] first: the checker fails")
    var junk = List[String]()
    junk.append("not json")
    junk.append("[DONE]")
    t.check(check_sse(junk) != "", "MUTANT: a chunk that is not JSON: the checker fails")


def main() raises:
    var t = Tally()
    case_units(t)
    case_get_routes(t)
    case_tokenize(t)
    case_completions(t)
    case_completions_stream(t)
    case_chat(t)
    case_tools(t)
    case_errors(t)
    case_engine_failures(t)
    case_cancel(t)
    case_inflight_and_keepalive(t)
    case_concurrent(t)
    case_footprint(t)
    case_oracle(t)
    print("PASS " + String(t.ok) + "/" + String(t.n))
    if t.ok != t.n:
        raise Error("test_http: " + String(t.n - t.ok) + " checks failed")
