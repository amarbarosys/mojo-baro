"""Test_anthropic: Unit tests and socket end-to-end tests for the Anthropic Messages API.

Validates the full Messages event machine:
1. `parse_messages` and `count_tokens_response` request translation
2. Non-streaming `message_response` translation (text and tool use)
3. Streaming SSE event sequence (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`, `error`)
4. Socket-level end-to-end HTTP tests over a real socket (`Server` on port 0):
   - `POST /v1/messages` (stream=false)
   - `POST /v1/messages` (stream=true)
   - `POST /v1/messages/count_tokens`
   - Header auth acceptance (`x-api-key`, `Authorization: Bearer`)
   - Clean vision refusal (400 `invalid_request_error`)
   - Error cases (400 on malformed input)
5. Table-driven oracle tests from `serve/front/testdata/anthropic-oracle.jsonl`

Build: ./.venv/bin/mojo build serve/front/test_anthropic.mojo -I . -I serve/front -o .work/test_anthropic
"""
from anthropic import (
    parse_messages, _messages, count_tokens_response, message_response, MessageStream, messages_error_body,
    messages_stream_start, messages_stream_chunk, messages_stream_error
)
from api import (
    GenRequest, Ev, ApiError, ENG_IDLE, cancel_id, check_gen, fit_n, resolve_tool_choice, DEFAULT_MAX_TOKENS, TextLike, Call
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


# ---- tiny blocking HTTP client ------------------------------------------------


def connect_to(port: Int) raises -> Int32:
    var fd = external_call["socket", Int32](Int32(2), Int32(1 | 0o2000000), Int32(0))
    if fd < 0:
        raise Error("test: socket failed")
    var tv = SIMD[DType.int64, 2](10, 0)  # 10s receive timeout
    _ = external_call["setsockopt", Int32](fd, Int32(1), Int32(20), Pointer(to=tv), Int32(16))
    var addr = SIMD[DType.uint32, 4](UInt32(2) | (UInt32(port & 0xFF) << 24) | (UInt32(port >> 8) << 16), UInt32(0x0100007F), 0, 0)
    if external_call["connect", Int32](fd, Pointer(to=addr), Int32(16)) != 0:
        close(fd)
        raise Error("test: connect failed")
    return fd


def send_request(fd: Int32, method: String, path: String, body: String, extra_headers: String = "") -> Bool:
    var head = method + " " + path + " HTTP/1.1\r\nhost: t\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: " + String(body.byte_length()) + "\r\n" + extra_headers + "\r\n"
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


def _pos(r: Int) -> Int:
    return r


def _pos(r: Optional[Int]) -> Int:
    return r.or_else(-1)


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
        var c = _pos(ln.find(":"))
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


def call(port: Int, method: String, path: String, body: String, extra_headers: String = "") raises -> Resp:
    var fd = connect_to(port)
    if not send_request(fd, method, path, body, extra_headers):
        close(fd)
        raise Error("call: send failed")
    var raw = read_to_eof(fd)
    close(fd)
    return parse_response(raw)


# ---- rig: Front + Server on background thread ---------------------------------

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

    def __init__(out self, mode: Int = M_NORMAL, var script: List[Int32] = List[Int32](), max_inflight: Int = 64, keepalive_s: Float64 = 15.0, model: String = "pack") raises:
        self.probe = Probe()
        var eng = FakeEngine(mode, script^, self.probe.addr())
        var front = _Front(FakeText(True), eng^, model, max_inflight, keepalive_s)
        var srv = _Srv(front^, "127.0.0.1", 0)
        self.port = srv.port()
        self.sp = alloc(Layout[_Srv](count=1)).unsafe_leak()
        self.sp.unsafe_write(srv^)
        var tids = List[UInt64](length=1, fill=0)
        _ = external_call["pthread_create", Int32](tids.unsafe_ptr(), Int(0), _serve_main, Int(self.sp))
        self.tid = tids[0]
        _ = external_call["usleep", Int32](Int32(30000))  # wait for workers

    def close(deinit self):
        self.sp[].stop()
        _ = external_call["pthread_join", Int32](self.tid, Int(0))
        _ = self.sp.unsafe_take_pointee()
        self.sp.unsafe_free()


# ---- Helper wrappers for ApiError ---------------------------------------------


def _parse_messages[T: TextLike](text: T, j: Json, spec_k: Int) raises -> Call:
    try:
        return parse_messages(text, j, spec_k)
    except e:
        raise Error(e.body)


def _msgs(j: Json) raises -> Json:
    try:
        return _messages(j)
    except e:
        raise Error(e.body)


def _count_tokens_response[T: TextLike](text: T, j: Json, spec_k: Int) raises -> String:
    try:
        return count_tokens_response(text, j, spec_k)
    except e:
        raise Error(e.body)


def _message_response(chat_json: String, stop_sequences: Json) raises -> String:
    try:
        return message_response(chat_json, stop_sequences)
    except e:
        raise Error(e.body)


def _messages_stream_chunk(mut state: MessageStream, chat_chunk: String) raises -> String:
    try:
        return messages_stream_chunk(state, chat_chunk)
    except e:
        raise Error(e.body)


# ---- Unit Tests ---------------------------------------------------------------


def test_parse_messages(mut t: Tally) raises:
    var text = FakeText(True)
    
    # 1. basic user message
    var j1 = parse('{"messages":[{"role":"user","content":"hi"}]}')
    var c1 = _parse_messages(text, j1, 0)
    t.check(c1.anthropic, "parse_messages: anthropic flag set")
    t.check(c1.chat, "parse_messages: chat flag set")
    t.check(len(c1.gen.prompt) > 0, "parse_messages: prompt rendered")
    
    # 2. system prompt string
    var j2 = parse('{"system":"be brief","messages":[{"role":"user","content":"hi"}]}')
    var c2 = _parse_messages(text, j2, 0)
    t.check(len(c2.gen.prompt) > len(c1.gen.prompt), "parse_messages: system string included in prompt")
    
    # 3. system prompt array
    var j3 = parse('{"system":[{"type":"text","text":"be brief"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}')
    var c3 = _parse_messages(text, j3, 0)
    t.eqi(len(c3.gen.prompt), len(c2.gen.prompt), "parse_messages: system array matches system string")

    # 4. tools definitions
    var j4 = parse('{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"bash","description":"run cmd","input_schema":{"type":"object","properties":{"cmd":{"type":"string"}}}},{"name":"read","input_schema":{"type":"object"}}]}')
    var c4 = _parse_messages(text, j4, 0)
    t.check(len(c4.gen.prompt) > 0, "parse_messages: tools parsed")

    # 5. tool_choice: auto, any, tool, none
    var j5_auto = parse('{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"bash","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"}}')
    var c5_auto = _parse_messages(text, j5_auto, 0)
    t.check(len(c5_auto.gen.prompt) > 0, "parse_messages: tool_choice auto")

    var j5_any = parse('{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"bash","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}')
    var c5_any = _parse_messages(text, j5_any, 0)
    t.check(len(c5_any.gen.prompt) > 0, "parse_messages: tool_choice any")

    var j5_tool = parse('{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"bash","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"bash"}}')
    var c5_tool = _parse_messages(text, j5_tool, 0)
    t.check(len(c5_tool.gen.prompt) > 0, "parse_messages: tool_choice named tool")

    # 6. tool use and tool result in conversation
    var j6 = parse('{"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"123","name":"bash","input":{"cmd":"ls"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"123","content":"file.txt"}]}]}')
    var c6 = _parse_messages(text, j6, 0)
    t.check(len(c6.gen.prompt) > 0, "parse_messages: tool use & result roundtrip")

    # 7. thinking in message
    var j7 = parse('{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"let me see"}]},{"role":"user","content":"continue"}]}')
    var c7 = _parse_messages(text, j7, 0)
    t.check(len(c7.gen.prompt) > 0, "parse_messages: thinking block accepted")

    # 8. refusal: image input
    var j_img = parse('{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}]}')
    var img_refused = False
    try:
        _ = parse_messages(text, j_img, 0)
    except e:
        img_refused = True
        t.eqi(e.status, 400, "parse_messages: image refusal status is 400")
    t.check(img_refused, "parse_messages: image block refused cleanly")

    # 8b. golden: mid-conversation system (Claude Code hook output) becomes a user reminder in place
    var jsr = parse('{"system":"S","messages":[{"role":"user","content":"hi"},{"role":"system","content":[{"type":"text","text":"hook"}]}]}')
    t.eq(_msgs(jsr).dumps(),
        '[{"role":"system","content":"S"},{"role":"user","content":"hi"},{"role":"user","content":"<system-reminder>\\nhook\\n</system-reminder>"}]',
        "messages: mid system -> user reminder (golden)")
    var csr = _parse_messages(text, jsr, 0)
    t.check(len(csr.gen.prompt) > len(c2.gen.prompt), "parse_messages: mid system renders")

    # 9. refusal: empty messages
    var j_empty = parse('{"messages":[]}')
    var empty_refused = False
    try:
        _ = parse_messages(text, j_empty, 0)
    except:
        empty_refused = True
    t.check(empty_refused, "parse_messages: empty messages rejected")


def test_count_tokens(mut t: Tally) raises:
    var text = FakeText(True)
    var j = parse('{"messages":[{"role":"user","content":"hello world"}]}')
    var res_str = _count_tokens_response(text, j, 0)
    var res = parse(res_str)
    t.check(res.has("input_tokens"), "count_tokens: response has input_tokens")
    t.check(res.get("input_tokens").int() > 0, "count_tokens: input_tokens > 0")


def test_prompt_overflow(mut t: Tally) raises:
    var fits = GenRequest()
    fits.prompt = List[Int32](length=15360, fill=1)
    try:
        check_gen(fits, 16384, 0, 64, DEFAULT_MAX_TOKENS, True)
        t.eqi(fits.n, 1024, "Anthropic keeps 1024 generation tokens at the prompt limit")
    except:
        t.check(False, "Anthropic accepts a prompt with 1024 tokens remaining")

    var full = GenRequest()
    full.prompt = List[Int32](length=15361, fill=1)
    try:
        check_gen(full, 16384, 0, 64, DEFAULT_MAX_TOKENS, True)
        t.check(False, "Anthropic rejects a prompt below the remaining-window floor")
    except e:
        t.eqi(e.status, 400, "Anthropic prompt overflow status")
        var body = parse(messages_error_body(e.body))
        t.eq(body.get("type").str(), "error", "Anthropic prompt overflow top-level type")
        t.eq(body.get("error").text("type"), "invalid_request_error", "Anthropic prompt overflow error type")
        t.eq(body.get("error").text("message"), "prompt is too long: 15361 tokens > 15360 maximum", "Anthropic prompt overflow message")


def test_message_response(mut t: Tally) raises:
    # 1. text response translation
    var chat_json = '{"id":"test1","object":"chat.completion","created":1794256501,"model":"test-pack","choices":[{"index":0,"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4}}'
    var anth_str = _message_response(chat_json, Json())
    var anth = parse(anth_str)
    t.eq(anth.get("type").str(), "message", "message_response: type is message")
    t.eq(anth.get("role").str(), "assistant", "message_response: role is assistant")
    t.eq(anth.get("model").str(), "test-pack", "message_response: model preserved")
    t.eq(anth.get("stop_reason").str(), "end_turn", "message_response: finish_reason stop -> end_turn")
    var content = anth.get("content")
    t.eqi(content.len(), 1, "message_response: content block count is 1")
    t.eq(content.at(0).get("type").str(), "text", "message_response: block type is text")
    t.eq(content.at(0).get("text").str(), "hello there", "message_response: block text content")
    var usage = anth.get("usage")
    t.eqi(usage.get("input_tokens").int(), 12, "message_response: input_tokens")
    t.eqi(usage.get("output_tokens").int(), 4, "message_response: output_tokens")

    # 2. tool calls response translation
    var tool_json = '{"id":"test2","object":"chat.completion","created":1794256501,"model":"test-pack","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_99","type":"function","function":{"name":"search","arguments":"{\\"q\\":\\"mojo\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":8}}'
    var tool_anth_str = _message_response(tool_json, Json())
    var tool_anth = parse(tool_anth_str)
    t.eq(tool_anth.get("stop_reason").str(), "tool_use", "message_response: finish_reason tool_calls -> tool_use")
    var tool_content = tool_anth.get("content")
    t.eqi(tool_content.len(), 1, "message_response: tool_use block present")
    var block = tool_content.at(0)
    t.eq(block.get("type").str(), "tool_use", "message_response: block type is tool_use")
    t.eq(block.get("name").str(), "search", "message_response: tool name is search")
    t.check(block.has("input"), "message_response: tool input object exists")
    t.eq(block.get("input").get("q").str(), "mojo", "message_response: parsed tool input parameter")


def test_streaming(mut t: Tally) raises:
    # 1. message_start
    var start_ev = messages_stream_start("test-stream", "test-pack", 42)
    t.check("event: message_start\n" in start_ev, "stream_start: event name")
    t.check('"type":"message_start"' in start_ev, "stream_start: data type")
    t.check('"input_tokens":42' in start_ev, "stream_start: input_tokens in usage")

    # 2. message_chunk deltas
    var st = MessageStream()
    var chunk1 = '{"choices":[{"index":0,"delta":{"content":"Hello","role":"assistant"},"finish_reason":null}]}'
    var sse1 = _messages_stream_chunk(st, chunk1)
    t.check("content_block_start" in sse1, "stream_chunk: emits content_block_start on first text")
    t.check("content_block_delta" in sse1, "stream_chunk: emits content_block_delta")
    t.check('"text_delta"' in sse1, "stream_chunk: delta type is text_delta")

    var chunk2 = '{"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}'
    var sse2 = _messages_stream_chunk(st, chunk2)
    t.check("content_block_start" not in sse2, "stream_chunk: does not re-emit content_block_start")
    t.check("content_block_delta" in sse2, "stream_chunk: second text_delta")

    var chunk_finish = '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":2}}'
    var sse_fin = _messages_stream_chunk(st, chunk_finish)
    t.check("content_block_stop" in sse_fin, "stream_chunk: emits content_block_stop")
    t.check("message_delta" in sse_fin, "stream_chunk: emits message_delta")
    t.check('"end_turn"' in sse_fin, "stream_chunk: stop_reason end_turn")
    t.check("message_stop" in sse_fin, "stream_chunk: emits message_stop")

    # 3. tool_calls streaming
    var st_tool = MessageStream()
    var chunk_tool = '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_01","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\".work/anthropic-pilot-scratch.txt\\"}"}}]},"finish_reason":null}]}'
    var sse_tool = _messages_stream_chunk(st_tool, chunk_tool)
    t.check("content_block_start" in sse_tool, "stream_chunk: tool emits content_block_start")
    t.check('"type":"tool_use"' in sse_tool, "stream_chunk: tool block type is tool_use")
    t.check('"name":"Read"' in sse_tool, "stream_chunk: tool name is Read")
    t.check("content_block_delta" in sse_tool, "stream_chunk: tool emits content_block_delta")
    t.check('"type":"input_json_delta"' in sse_tool, "stream_chunk: tool delta is input_json_delta")
    t.check('"partial_json"' in sse_tool, "stream_chunk: tool has partial_json")

    var chunk_tool_fin = '{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}'
    var sse_tool_fin = _messages_stream_chunk(st_tool, chunk_tool_fin)
    t.check("content_block_stop" in sse_tool_fin, "stream_chunk: tool emits content_block_stop")
    t.check("message_delta" in sse_tool_fin, "stream_chunk: tool emits message_delta")
    t.check('"tool_use"' in sse_tool_fin, "stream_chunk: tool stop_reason is tool_use")
    t.check("message_stop" in sse_tool_fin, "stream_chunk: tool emits message_stop")

    # 4. stream error
    var err_ev = messages_stream_error("upstream failed")
    t.check("event: error\n" in err_ev, "stream_error: event name")
    t.check('"message":"upstream failed"' in err_ev, "stream_error: error message")


# ---- Socket End-to-End Tests --------------------------------------------------


def test_socket_nonstream(mut t: Tally) raises:
    var script = List[Int32]()
    script.append(114)
    script.append(FAKE_EOS)
    var rig = Rig(M_NORMAL, script^, 64, 15.0, "test-pack")
    var req = '{"messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
    var got = call(rig.port, "POST", "/v1/messages", req)
    rig^.close()

    t.eqi(got.status, 200, "socket nonstream: status 200")
    t.eq(got.header("content-type"), "application/json", "socket nonstream: content-type json")
    var j = got.json()
    t.eq(j.get("type").str(), "message", "socket nonstream: response type")
    t.eq(j.get("role").str(), "assistant", "socket nonstream: role assistant")
    t.eq(j.get("stop_reason").str(), "max_tokens", "socket nonstream: stop_reason max_tokens")
    var content = j.get("content")
    t.check(content.len() > 0, "socket nonstream: content not empty")
    t.eq(content.at(0).get("type").str(), "text", "socket nonstream: content block is text")


def test_socket_stream(mut t: Tally) raises:
    var script = List[Int32]()
    script.append(114)
    script.append(110)
    var rig = Rig(M_NORMAL, script^, 64, 15.0, "test-pack")
    var req = '{"messages":[{"role":"user","content":"hi"}],"max_tokens":10,"stream":true}'
    var got = call(rig.port, "POST", "/v1/messages", req)
    rig^.close()

    t.eqi(got.status, 200, "socket stream: status 200")
    t.eq(got.header("content-type"), "text/event-stream", "socket stream: content-type text/event-stream")
    t.check("event: message_start" in got.body, "socket stream: message_start event")
    t.check("event: content_block_start" in got.body, "socket stream: content_block_start event")
    t.check("event: content_block_delta" in got.body, "socket stream: content_block_delta event")
    t.check("event: content_block_stop" in got.body, "socket stream: content_block_stop event")
    t.check("event: message_delta" in got.body, "socket stream: message_delta event")
    t.check("event: message_stop" in got.body, "socket stream: message_stop event")


def test_socket_count_tokens(mut t: Tally) raises:
    var script = List[Int32]()
    var rig = Rig(M_NORMAL, script^, 64, 15.0, "test-pack")
    var req = '{"messages":[{"role":"user","content":"hi"}]}'
    var got = call(rig.port, "POST", "/v1/messages/count_tokens", req)
    rig^.close()

    t.eqi(got.status, 200, "socket count_tokens: status 200")
    var j = got.json()
    t.check(j.has("input_tokens"), "socket count_tokens: has input_tokens")
    t.check(j.get("input_tokens").int() > 0, "socket count_tokens: input_tokens > 0")


def test_socket_auth(mut t: Tally) raises:
    var script = List[Int32]()
    script.append(114)
    script.append(110)
    var rig = Rig(M_NORMAL, script^, 64, 15.0, "test-pack")
    var req = '{"messages":[{"role":"user","content":"hi"}]}'
    
    # 1. x-api-key
    var got1 = call(rig.port, "POST", "/v1/messages", req, "x-api-key: test-local-key\r\n")
    t.eqi(got1.status, 200, "socket auth: accepts x-api-key")

    # 2. Authorization Bearer
    var got2 = call(rig.port, "POST", "/v1/messages", req, "authorization: Bearer test-bearer-key\r\n")
    t.eqi(got2.status, 200, "socket auth: accepts Authorization Bearer")
    
    rig^.close()


def test_socket_errors(mut t: Tally) raises:
    var script = List[Int32]()
    var rig = Rig(M_NORMAL, script^, 64, 15.0, "test-pack")

    # 1. malformed json
    var got1 = call(rig.port, "POST", "/v1/messages", "{not-valid-json")
    t.eqi(got1.status, 400, "socket error: malformed JSON returns 400")

    # 2. missing messages
    var got2 = call(rig.port, "POST", "/v1/messages", '{"max_tokens":10}')
    t.eqi(got2.status, 400, "socket error: missing messages returns 400")

    # 3. image input refusal
    var got3 = call(rig.port, "POST", "/v1/messages", '{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}]}')
    t.eqi(got3.status, 400, "socket error: vision refusal returns 400")
    
    rig^.close()


def test_oracle_vectors(mut t: Tally) raises:
    var text: String
    with open("serve/front/testdata/anthropic-oracle.jsonl", "r") as f:
        text = f.read()
    var rows = text.split("\n")
    for r in rows:
        var r_str = String(r)
        if r_str == "":
            continue
        var row = parse(r_str)
        var name = row.get("name").str()
        var script = List[Int32]()
        for x in row.get("script").items():
            script.append(Int32(x.int()))
        var rig = Rig(M_NORMAL, script^, 64, 15.0, "pack")
        var got = call(rig.port, row.get("method").str(), row.get("path").str(), row.get("request").str())
        rig^.close()

        t.eqi(got.status, row.get("status").int(), "oracle " + name + ": status")
        var kind = row.get("kind").str()
        if kind == "json":
            var want_body = row.get("body")
            var got_body = got.json()
            if want_body.has("type"):
                t.eq(got_body.get("type").str(), want_body.get("type").str(), "oracle " + name + ": type")
            if want_body.has("role"):
                t.eq(got_body.get("role").str(), want_body.get("role").str(), "oracle " + name + ": role")
            if want_body.has("input_tokens"):
                t.eqi(got_body.get("input_tokens").int(), want_body.get("input_tokens").int(), "oracle " + name + ": input_tokens")
        elif kind == "error":
            t.eqi(got.status, 400, "oracle " + name + ": error status is 400")


def main() raises:
    var t = Tally()
    print("=== Testing Anthropic Messages Front ===")
    test_parse_messages(t)
    test_count_tokens(t)
    test_prompt_overflow(t)
    test_message_response(t)
    test_streaming(t)
    test_socket_nonstream(t)
    test_socket_stream(t)
    test_socket_count_tokens(t)
    test_socket_auth(t)
    test_socket_errors(t)
    test_oracle_vectors(t)

    print("PASS " + String(t.ok) + "/" + String(t.n))
    if len(t.failed) > 0:
        print("FAILURES:")
        for f in t.failed:
            print("  - " + f)
        raise Error("test_anthropic failed")
