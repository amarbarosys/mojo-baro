"""The OpenAI-shaped API of the Mojo front: request parsing and response building, pure functions over `Json`.

No socket, no thread, no engine here. `http.mojo` calls these; `test_http.mojo` and the fakes use the
same traits. Rust reference: serve/src/main.rs (`ApiError`, `check_and_submit`, `collect`, `sse_stream`,
`completions`, `chat_once`). Every field the Mojo front does not serve yet is a named 400, never ignored.
"""
from json import Json, parse, NULL, BOOL, INT, FLOAT, STR, ARR, OBJ
from std.os import getenv
from std.ffi import external_call

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

comptime DEFAULT_MAX_TOKENS = 1048576  # GEN_N in serve/registry.mojo; "up to the window"
comptime MAX_ID = 2147483647  # token ids are Int32 here


@fieldwise_init
struct Ev(Copyable, Movable):
    """One engine event. `text` = the error, or the raw JSON of the done line."""

    var kind: Int
    var id: Int
    var tok: Int32
    var text: String


struct GenRequest(Copyable, Movable):
    """Field names of serve/src/protocol.rs `Request` (the core subset); the boss adapter copies fields."""

    var id: Int
    var prompt: List[Int32]
    var n: Int
    var spec: Bool
    var stop: List[List[Int32]]
    var ckpt: List[Int32]  # role-boundary hints: not produced by this front (finding)
    var temperature: Optional[Float64]
    var top_p: Optional[Float64]
    var top_k: Optional[Int]
    var min_p: Optional[Float64]
    var seed: Optional[Int]
    var presence_penalty: Optional[Float64]
    var frequency_penalty: Optional[Float64]
    var schema: Optional[String]  # response_format.json_schema.schema as compact JSON
    var reasoning: Optional[Bool]

    def __init__(out self):
        self.id = 0
        self.prompt = List[Int32]()
        self.n = 0
        self.spec = False
        self.stop = List[List[Int32]]()
        self.ckpt = List[Int32]()
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.min_p = None
        self.seed = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.schema = None
        self.reasoning = None


trait DetokLike(Movable, Deinitable):
    def push(mut self, id: Int32) -> String:
        ...

    def flush(mut self) -> String:
        ...


trait TextLike(Movable, Deinitable):
    comptime D: DetokLike

    def encode(self, s: String, add_special: Bool) raises -> List[Int32]:
        ...

    def decode(self, ids: Span[Int32, _]) -> String:
        ...

    def apply_chat_template(self, messages: Json, tools: Json, kwargs: Json) raises -> String:
        ...

    def tool_xml(self) -> Bool:
        ...

    def stop_ids(self) -> List[Int32]:
        ...

    def detok(self) -> Self.D:
        ...


trait EngineLike(Movable, Deinitable):
    def submit(mut self, req: GenRequest) raises:
        ...

    def cancel(mut self, id: Int) -> Bool:
        ...

    def poll(mut self, mut out: List[Ev], timeout_ms: Int) raises -> Int:
        ...

    def state(self) -> Int:
        ...

    def state_name(self) -> String:
        ...

    def dead_reason(self) -> String:
        ...

    def shutdown(mut self, grace_ms: Int):
        ...

    def ready_info(self) -> Json:
        ...


# ---- errors -------------------------------------------------------------------


struct ApiError(Copyable, Movable):
    var status: Int
    var body: String  # JSON text

    def __init__(out self, status: Int, body: String):
        self.status = status
        self.body = body


def error_body(status: Int, msg: String, kind: String = "invalid_request_error") -> String:
    var e = Json.object()
    e.set("message", Json.of(msg))
    e.set("type", Json.of(kind))
    e.set("code", Json.of(status))
    var o = Json.object()
    o.set("error", e)
    return o.dumps()


def plain(status: Int, msg: String) -> ApiError:
    return ApiError(status, error_body(status, msg))


def bad(msg: String) -> ApiError:
    return plain(400, msg)


def exceed_context(n_prompt: Int, n_ctx: Int) -> ApiError:
    """llama.cpp's exceed_context_size_error shape, verbatim."""
    var e = Json.object()
    e.set("code", Json.of(400))
    e.set("message", Json.of("the request exceeds the available context size, try increasing it"))
    e.set("type", Json.of("exceed_context_size_error"))
    e.set("n_prompt_tokens", Json.of(n_prompt))
    e.set("n_ctx", Json.of(n_ctx))
    var o = Json.object()
    o.set("error", e)
    return ApiError(400, o.dumps())


def engine_error_chunk(msg: String) -> String:
    """The SSE error chunk: `{"error":{"message":"engine: ..","type":"engine_error"}}`."""
    var e = Json.object()
    e.set("message", Json.of("engine: " + msg))
    e.set("type", Json.of("engine_error"))
    var o = Json.object()
    o.set("error", e)
    return o.dumps()


def now_secs() -> Int:
    return Int(external_call["time", Int](Int(0)))


# ---- typed field readers (serde's strictness, named errors) -------------------


def _bad_field(key: String, want: String) -> ApiError:
    return bad("invalid field `" + key + "`: expected " + want)


def opt_int(j: Json, key: String) raises ApiError -> Optional[Int]:
    var v = j.get(key)
    if v.is_null():
        return None
    if v.kind() != INT or v.int() < 0:
        raise _bad_field(key, "a non-negative integer")
    return v.int()


def opt_signed(j: Json, key: String) raises ApiError -> Optional[Int]:
    var v = j.get(key)
    if v.is_null():
        return None
    if v.kind() != INT:
        raise _bad_field(key, "an integer")
    return v.int()


def opt_num(j: Json, key: String) raises ApiError -> Optional[Float64]:
    var v = j.get(key)
    if v.is_null():
        return None
    if not v.is_num():
        raise _bad_field(key, "a number")
    return v.num()


def opt_bool(j: Json, key: String) raises ApiError -> Optional[Bool]:
    var v = j.get(key)
    if v.is_null():
        return None
    if v.kind() != BOOL:
        raise _bad_field(key, "a boolean")
    return v.int() != 0


def flag(j: Json, key: String) raises ApiError -> Bool:
    var b = opt_bool(j, key)
    return b.value() if b else False


def _env_num(name: String) -> Optional[Float64]:
    var s = getenv(name)
    if s == "":
        return None
    try:
        return Float64(s)
    except:
        return None


def parse_body(body: Span[UInt8, _]) raises ApiError -> Json:
    var j: Json
    try:
        j = parse(body)
    except e:
        raise bad("Failed to parse the request body as JSON: " + String(e))
    if j.kind() != OBJ:
        raise bad("Failed to parse the request body as JSON: expected an object")
    return j^


# ---- shared plumbing: stop, sampler, the one submit check ---------------------


def fit_n(n: Int, n_prompt: Int, tmax: Int) -> Int:
    """An omitted max_tokens arrives as DEFAULT_MAX_TOKENS and means up to the window."""
    if n == DEFAULT_MAX_TOKENS:
        return min(max(tmax - n_prompt, 0), n)
    return n


def stop_list[T: TextLike](text: T, j: Json) raises ApiError -> List[List[Int32]]:
    """The tokenizer's own stop ids (length-1 sequences) plus the caller's `stop` strings, tokenized without specials."""
    var out = List[List[Int32]]()
    for id in text.stop_ids():
        out.append([id])
    var s = j.get("stop")
    var strs = List[String]()
    if s.kind() == STR:
        strs.append(s.str())
    elif s.kind() == ARR:
        for it in s.items():
            if it.kind() != STR:
                raise _bad_field("stop", "a string or an array of strings")
            strs.append(it.str())
    elif not s.is_null():
        raise _bad_field("stop", "a string or an array of strings")
    for st in strs:
        try:
            var ids = text.encode(st, False)
            if len(ids) > 0:
                out.append(ids^)
        except:
            pass  # Rust: an encode error drops that stop string
    return out^


def sampler(mut g: GenRequest, j: Json) raises ApiError:
    g.temperature = opt_num(j, "temperature")
    g.top_p = opt_num(j, "top_p")
    g.top_k = opt_signed(j, "top_k")
    g.min_p = opt_num(j, "min_p")
    g.seed = opt_int(j, "seed")
    g.presence_penalty = opt_num(j, "presence_penalty")
    g.frequency_penalty = opt_num(j, "frequency_penalty")
    # BARO_DEFAULT_*: a server-side default for a field the request leaves out (as llama-server).
    if not g.top_p:
        g.top_p = _env_num("BARO_DEFAULT_TOP_P")
    if not g.top_k:
        var k = _env_num("BARO_DEFAULT_TOP_K")
        if k:
            g.top_k = Int(k.value())
    if not g.min_p:
        g.min_p = _env_num("BARO_DEFAULT_MIN_P")


def reject_unserved(j: Json) raises ApiError:
    """Fields round 2 serves. A named 400, so a client never gets a silently different answer."""
    for k in ["checkpoint", "hidden"]:
        if flag(j, k):
            raise bad("`" + k + "` is not served by the Mojo front yet")
    for k in ["checkpoint_id", "steer", "capture", "logits_topk"]:
        if not j.get(k).is_null():
            raise bad("`" + k + "` is not served by the Mojo front yet")
    var lp = j.get("logprobs")
    if (lp.kind() == BOOL and lp.int() != 0) or (lp.kind() == INT and lp.int() > 0):
        raise bad("`logprobs` is not served by the Mojo front yet")
    var tl = j.get("top_logprobs")
    if tl.kind() == INT and tl.int() > 0:
        raise bad("`top_logprobs` is not served by the Mojo front yet")


def check_gen(mut g: GenRequest, tmax: Int, inflight: Int, max_inflight: Int, requested_n: Int) raises ApiError:
    """The ONE length and admission check, run at submit (Rust repeated it by hand in 4 places, report-A M4)."""
    if inflight >= max_inflight:
        raise plain(429, String(max_inflight) + " requests in flight (--max-inflight)")
    if len(g.prompt) == 0:
        raise bad("prompt is empty")
    var n = fit_n(requested_n, len(g.prompt), tmax)
    if n == 0 and requested_n == DEFAULT_MAX_TOKENS:
        raise exceed_context(len(g.prompt), tmax)
    if n == 0:
        raise bad("max_tokens must be >= 1")
    if len(g.prompt) + n > tmax:
        raise exceed_context(len(g.prompt), tmax)
    g.n = n


def spec_default(j: Json, spec_k: Int) raises ApiError -> Bool:
    var s = opt_bool(j, "spec")
    if s:
        return s.value()
    return getenv("BARO_SPEC") == "1" and spec_k > 0


# ---- what a handler needs after parsing ---------------------------------------


struct Call(Copyable, Movable):
    """A parsed completion or chat request: the engine request plus what the response needs."""

    var gen: GenRequest
    var requested_n: Int
    var model: String  # "" = the server's
    var stream: Bool
    var chat: Bool
    var force: String  # text forced after the generation prompt (tool_choice)
    var n_prompt: Int

    def __init__(out self):
        self.gen = GenRequest()
        self.requested_n = DEFAULT_MAX_TOKENS
        self.model = ""
        self.stream = False
        self.chat = False
        self.force = ""
        self.n_prompt = 0


def prompt_ids[T: TextLike](text: T, prompt: Json) raises ApiError -> List[Int32]:
    var out = List[Int32]()
    if prompt.kind() == STR:
        try:
            return text.encode(prompt.str(), True)
        except e:
            raise bad(String(e))
    if prompt.kind() == ARR:
        var items = prompt.items()
        var all_int = True
        for it in items:
            if it.kind() != INT or it.int() < 0:
                all_int = False
        if all_int:
            for it in items:
                if it.int() > MAX_ID:
                    raise bad("token id out of range")
                out.append(Int32(it.int()))
            return out^
        if len(items) == 1 and items[0].kind() == STR:
            try:
                return text.encode(items[0].str(), True)
            except e:
                raise bad(String(e))
    raise bad("prompt must be a string or an array of token ids")


def parse_completion[T: TextLike](text: T, j: Json, spec_k: Int) raises ApiError -> Call:
    if not j.has("prompt"):
        raise bad("missing field `prompt`")
    reject_unserved(j)
    var c = Call()
    sampler(c.gen, j)
    c.gen.prompt = prompt_ids(text, j.get("prompt"))
    var mt = opt_int(j, "max_tokens")
    c.requested_n = mt.value() if mt else DEFAULT_MAX_TOKENS
    c.gen.spec = spec_default(j, spec_k)
    c.stream = flag(j, "stream")
    c.gen.stop = stop_list(text, j)
    var m = j.get("model")
    c.model = m.str() if m.kind() == STR else ""
    c.n_prompt = len(c.gen.prompt)
    return c^


# ---- chat ---------------------------------------------------------------------


def content_text(v: Json) raises ApiError -> String:
    if v.kind() == STR:
        return v.str()
    if v.kind() == ARR:
        var buf = String()
        for p in v.items():
            if p.kind() == STR:
                buf += p.str()
            elif p.kind() == OBJ:
                if p.has("image_url") or p.has("image"):
                    raise bad("image parts are not served by the Mojo front yet")
                var t = p.get("text")
                if t.kind() == STR:
                    buf += t.str()
        return buf^
    if v.is_null():
        return String()
    raise bad("message content must be a string or an array of text parts")


def tool_calls_for_template(calls: Json) raises ApiError -> Json:
    """OpenAI `arguments` is a JSON string; the template reads a real object."""
    if calls.kind() != ARR:
        raise _bad_field("tool_calls", "an array")
    var out = Json.array()
    for c in calls.items():
        var f = c.get("function")
        if f.kind() != OBJ or f.get("name").kind() != STR:
            raise bad("invalid field `tool_calls`: each call needs function.name")
        var args = Json.object()
        var a = f.get("arguments")
        if a.kind() == STR:
            try:
                var p = parse(a.str())
                if p.kind() == OBJ:
                    args = p^
            except:
                pass
        var fn_ = Json.object()
        fn_.set("name", f.get("name"))
        fn_.set("arguments", args)
        var o = Json.object()
        o.set("function", fn_)
        out.push(o)
    return out^


def check_response_format(rf: Json) raises ApiError -> Json:
    if rf.kind() != OBJ:
        raise bad("response_format must be an object")
    var kind = rf.get("type")
    if kind.kind() != STR:
        raise bad("response_format.type is required")
    if kind.str() != "json_schema":
        raise bad("response_format.type \"" + kind.str() + "\" is not supported (only \"json_schema\")")
    var js = rf.get("json_schema")
    if js.kind() != OBJ:
        raise bad("response_format.json_schema is required for type \"json_schema\"")
    if not js.has("schema"):
        raise bad("response_format.json_schema.schema is required")
    return js.get("schema")


def resolve_tool_choice(tools: Json, choice: Json, xml: Bool) raises ApiError -> Tuple[Json, String]:
    """`tool_choice` -> (tools for the template, text forced after the generation prompt)."""
    var name = String()
    var named = False
    if choice.is_null():
        return (tools.copy(), String())
    if choice.kind() == STR:
        var s = choice.str()
        if s == "auto":
            return (tools.copy(), String())
        if s == "none":
            return (Json(), String())
        if s != "required":
            raise bad("tool_choice: \"auto\", \"none\", \"required\" or {\"type\":\"function\",\"function\":{\"name\":...}}")
    else:
        var n = choice.get("function").get("name")
        if n.kind() != STR:
            raise bad("tool_choice: \"auto\", \"none\", \"required\" or {\"type\":\"function\",\"function\":{\"name\":...}}")
        name = n.str()
        named = True
    var names = List[String]()
    if tools.kind() == ARR:
        for t in tools.items():
            var tn = t.get("function").get("name")
            if tn.kind() == STR:
                names.append(tn.str())
    if len(names) == 0:
        raise bad("tool_choice needs a non-empty tools array")
    if named:
        var found = False
        for n in names:
            if n == name:
                found = True
        if not found:
            raise bad("tool_choice names " + name + ", which is not in tools")
    var force: String
    if xml:
        force = "<tool_call>\n<function=" + (name + ">\n" if named else "")
    else:
        force = "<tool_call>\n{\"name\": \"" + (name + "\", \"arguments\": " if named else "")
    return (tools.copy(), force^)


def parse_chat[T: TextLike](text: T, j: Json, spec_k: Int) raises ApiError -> Call:
    var msgs_in = j.get("messages")
    if msgs_in.kind() != ARR:
        raise bad("missing field `messages`")
    if msgs_in.len() == 0:
        raise bad("messages is empty")
    reject_unserved(j)
    var c = Call()
    c.chat = True
    var schema = Json()
    var has_schema = False
    var rf = j.get("response_format")
    if not rf.is_null():
        schema = check_response_format(rf)
        has_schema = True
    var kwargs = j.get("chat_template_kwargs")
    if has_schema:
        var think = kwargs.get("enable_thinking")
        c.gen.reasoning = not (think.kind() == BOOL and think.int() == 0)
        c.gen.schema = schema.dumps()
    var msgs = Json.array()
    for m in msgs_in.items():
        var role = m.get("role")
        if m.kind() != OBJ or role.kind() != STR:
            raise bad("invalid field `messages`: each message needs a string role")
        var o = Json.object()
        o.set("role", role)
        o.set("content", Json.of(content_text(m.get("content"))))
        var tc = m.get("tool_calls")
        if not tc.is_null():
            o.set("tool_calls", tool_calls_for_template(tc))
        msgs.push(o)
    var resolved = resolve_tool_choice(j.get("tools"), j.get("tool_choice"), text.tool_xml())
    var rendered: String
    try:
        rendered = text.apply_chat_template(msgs, resolved[0], kwargs)
    except e:
        raise plain(500, String(e))
    rendered += resolved[1]
    c.force = resolved[1]
    try:
        c.gen.prompt = text.encode(rendered, True)
    except e:
        raise bad(String(e))
    sampler(c.gen, j)
    var mct = opt_int(j, "max_completion_tokens")
    var mt = opt_int(j, "max_tokens")
    c.requested_n = mct.value() if mct else (mt.value() if mt else DEFAULT_MAX_TOKENS)
    c.gen.spec = spec_default(j, spec_k)
    c.stream = flag(j, "stream")
    c.gen.stop = stop_list(text, j)
    var m = j.get("model")
    c.model = m.str() if m.kind() == STR else ""
    c.n_prompt = len(c.gen.prompt)
    return c^


# ---- tokenize / detokenize ----------------------------------------------------


def tokenize_response[T: TextLike](text: T, j: Json) raises ApiError -> String:
    var content = j.get("content")
    if content.kind() != STR:
        raise bad("missing field `content`")
    var add_special = flag(j, "add_special")
    var ids: List[Int32]
    try:
        ids = text.encode(content.str(), add_special)
    except e:
        raise bad(String(e))
    var arr = Json.array()
    for id in ids:
        arr.push(Json.of(Int(id)))
    var o = Json.object()
    o.set("tokens", arr)
    return o.dumps()


def detokenize_response[T: TextLike](text: T, j: Json) raises ApiError -> String:
    var toks = j.get("tokens")
    if toks.kind() != ARR:
        raise bad("missing field `tokens`")
    var ids = List[Int32]()
    for it in toks.items():
        if it.kind() != INT or it.int() < 0 or it.int() > MAX_ID:
            raise bad("invalid field `tokens`: expected token ids")
        ids.append(Int32(it.int()))
    var o = Json.object()
    o.set("content", Json.of(text.decode(Span(ids))))
    return o.dumps()


def cancel_id(j: Json) raises ApiError -> Int:
    """`{"id": "cmpl-7"}` -> 7."""
    var s = j.get("id")
    if s.kind() != STR:
        raise bad("missing field `id`")
    var raw = s.str()
    var b = raw.as_bytes()
    var i = len(b)
    while i > 0 and b[i - 1] >= 48 and b[i - 1] <= 57:
        i -= 1
    if i == len(b) or i == 0 or b[i - 1] != 45:
        raise bad("id \"" + raw + "\" has no trailing request number")
    var n = 0
    for k in range(i, len(b)):
        n = n * 10 + Int(b[k]) - 48
        if n > 1 << 53:
            raise bad("id \"" + raw + "\" has no trailing request number")
    return n


def cancel_response(state: String) -> String:
    """`/v1/cancel`: `cancelled` as Rust, plus the state the Rust front could not name."""
    var o = Json.object()
    o.set("cancelled", Json.of(state == "running" or state == "queued"))
    o.set("state", Json.of(state))
    return o.dumps()


# ---- accumulating one request's tokens ----------------------------------------


struct Acc[T: TextLike](Movable):
    var detok: Self.T.D
    var stop: List[Int32]
    var stopped: Bool
    var tokens: List[Int32]

    def __init__(out self, text: Self.T):
        self.detok = text.detok()
        self.stop = text.stop_ids()
        self.stopped = False
        self.tokens = List[Int32]()

    def take(mut self, tok: Int32) -> Optional[String]:
        """Text delta for one token; None once a stop token was seen."""
        self.tokens.append(tok)
        if self.stopped:
            return None
        for s in self.stop:
            if s == tok:
                self.stopped = True
                return None
        return self.detok.push(tok)

    def finish_reason(self, engine_finish: String) -> String:
        if engine_finish != "":
            return engine_finish
        return "stop" if self.stopped else "length"


def _opt_field(src: Json, key: String) -> Json:
    return src.get(key)


def stats_json(done: String) -> Json:
    """The Rust `stats_json` shape, from the engine's raw done line; an absent key is null."""
    var d: Json
    try:
        d = parse(done)
    except:
        d = Json.object()
    var o = Json.object()
    o.set("prefill_s", d.get("prefill_s"))
    o.set("decode_s", d.get("decode_s"))
    o.set("tok_s_gen", d.get("tok_s"))
    for k in ["drafted", "accepted", "cached", "prefill_rows", "restore_s", "load_s", "load_read_s", "load_parse_s", "load_pin_s", "load_host_s", "load_h2d_s", "wall_s", "finish"]:
        o.set(k, d.get(k))
    # wire seam words (serve/PROTOCOL.md, State words): present only when the engine sent them
    for k in ["fail", "path", "grammar", "gdraws", "gaccepted", "grejected", "captured", "load_tier"]:
        if d.has(k):
            o.set(k, d.get(k))
    if d.get("grejected").kind() == INT and d.get("grejected").int() > 0:
        var w = Json.array()
        w.push(Json.of("grammar_rejected_tokens"))
        o.set("warnings", w)
    return o^


def fail_word(stats: Json) -> Int:
    """The megakernel fail word of a done line; non-zero = the tokens of this request are invalid."""
    var f = stats.get("fail")
    return f.int() if f.kind() == INT else 0


def invalid_tokens_msg(fail: Int) -> String:
    return String("engine: the run produced invalid tokens (fail word ", fail, "); the output must not be used")


def invalid_tokens_error(fail: Int) -> ApiError:
    return ApiError(502, error_body(502, invalid_tokens_msg(fail), "engine_invalid_tokens"))


def invalid_tokens_chunk(fail: Int) -> String:
    var e = Json.object()
    e.set("message", Json.of(invalid_tokens_msg(fail)))
    e.set("type", Json.of("engine_invalid_tokens"))
    var o = Json.object()
    o.set("error", e)
    return o.dumps()


def usage_json(n_prompt: Int, n_completion: Int, stats: Json) -> Json:
    var u = Json.object()
    u.set("prompt_tokens", Json.of(n_prompt))
    u.set("completion_tokens", Json.of(n_completion))
    u.set("total_tokens", Json.of(n_prompt + n_completion))
    var b = Json.object()
    b.set("cached_tokens", stats.get("cached"))
    b.set("prefill_rows", stats.get("prefill_rows"))
    u.set("baro", b)
    return u^


def _ints(ids: List[Int32]) -> Json:
    var a = Json.array()
    for i in ids:
        a.push(Json.of(Int(i)))
    return a^


def _head(id: String, object: String, created: Int, model: String) -> Json:
    var o = Json.object()
    o.set("id", Json.of(id))
    o.set("object", Json.of(object))
    o.set("created", Json.of(created))
    o.set("model", Json.of(model))
    return o^


def _choices(one: Json) -> Json:
    var a = Json.array()
    a.push(one)
    return a^


# ---- completions responses ----------------------------------------------------


def completion_response(id: String, created: Int, model: String, text_out: String, tokens: List[Int32], reason: String, n_prompt: Int, stats: Json) -> String:
    var o = _head(id, "text_completion", created, model)
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("text", Json.of(text_out))
    ch.set("tokens", _ints(tokens))
    ch.set("finish_reason", Json.of(reason))
    ch.set("logprobs", Json())
    o.set("choices", _choices(ch))
    o.set("usage", usage_json(n_prompt, len(tokens), stats))
    o.set("timings", stats.copy())
    return o.dumps()


def completion_delta(id: String, created: Int, model: String, text_out: String, tok: Int32) -> String:
    var o = _head(id, "text_completion", created, model)
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("text", Json.of(text_out))
    var t = List[Int32]()
    t.append(tok)
    ch.set("tokens", _ints(t))
    ch.set("finish_reason", Json())
    ch.set("logprobs", Json())
    o.set("choices", _choices(ch))
    return o.dumps()


def completion_finish(id: String, created: Int, model: String, reason: String, n_prompt: Int, tokens: List[Int32], stats: Json) -> String:
    var o = _head(id, "text_completion", created, model)
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("text", Json.of(""))
    ch.set("finish_reason", Json.of(reason))
    o.set("choices", _choices(ch))
    o.set("usage", usage_json(n_prompt, len(tokens), stats))
    o.set("tokens", _ints(tokens))
    o.set("timings", stats.copy())
    return o.dumps()


# ---- tool calls ---------------------------------------------------------------

comptime _OPEN = "<tool_call>"
comptime _CLOSE = "</tool_call>"


def sub(s: String, a: Int, b: Int) -> String:
    """Bytes [a, b) of s."""
    return String(unsafe_from_utf8=s.as_bytes()[a:b])


def _find(s: String, needle: String, start: Int = 0) -> Int:
    if start > s.byte_length():
        return -1
    return s.find(needle, start)


def _trim(s: String) -> String:
    return String(s.strip())


struct ToolCall(Copyable, Movable):
    var name: String
    var args: Json  # an object

    def __init__(out self, name: String, var args: Json):
        self.name = name
        self.args = args^


def _value_or_str(s: String) -> Json:
    try:
        return parse(s)
    except:
        return Json.of(s)


def _parse_one_tool_call(body: String) -> Optional[ToolCall]:
    var t = _trim(body)
    # JSON form: {"name": .., "arguments": {..} or "<json string>"}
    var o = Json()
    var is_json = False
    try:
        o = parse(t)
        is_json = o.kind() == OBJ
    except:
        pass
    if is_json:
        if o.get("name").kind() != STR:
            return None
        var a = o.get("arguments")
        var args = Json.object()
        if a.kind() == OBJ:
            args = a^
        elif a.kind() == STR:
            try:
                args = parse(a.str())
            except:
                return None
            if args.kind() != OBJ:
                return None
        return ToolCall(o.get("name").str(), args^)
    if not t.startswith("<function="):
        return None
    var rest = sub(t, 10, t.byte_length())
    var gt = _find(rest, ">")
    if gt < 0:
        return None
    var name = sub(rest, 0, gt)
    var inner = sub(rest, gt + 1, rest.byte_length())
    if inner.endswith("</function>"):
        inner = sub(inner, 0, inner.byte_length() - 11)
    var args = Json.object()
    var r = inner
    while True:
        var p = _find(r, "<parameter=")
        if p < 0:
            break
        var after = sub(r, p + 11, r.byte_length())
        var g = _find(after, ">")
        if g < 0:
            break
        var pname = sub(after, 0, g)
        var after_name = sub(after, g + 1, after.byte_length())
        var e = _find(after_name, "</parameter>")
        if e < 0:
            break
        args.set(pname, _value_or_str(_trim(sub(after_name, 0, e))))
        r = sub(after_name, e + 12, after_name.byte_length())
    return ToolCall(name, args^)


def parse_tool_calls(text: String) -> Tuple[String, List[ToolCall]]:
    """(content before the first tool_call, calls); the whole trimmed text when there are none."""
    var calls = List[ToolCall]()
    var rest = text
    var first = -1
    while True:
        var start = _find(rest, _OPEN)
        if start < 0:
            break
        if first < 0:
            first = text.byte_length() - rest.byte_length() + start
        var after = sub(rest, start + 11, rest.byte_length())
        var end = _find(after, _CLOSE)
        if end < 0:
            break
        var one = _parse_one_tool_call(sub(after, 0, end))
        if one:
            calls.append(one.value().copy())
        rest = sub(after, end + 12, after.byte_length())
    var content = _trim(sub(text, 0, first)) if first >= 0 else _trim(text)
    return (content^, calls^)


def tool_calls_response_json(calls: List[ToolCall]) -> Json:
    var out = Json.array()
    for i in range(len(calls)):
        var f = Json.object()
        f.set("name", Json.of(calls[i].name))
        f.set("arguments", Json.of(calls[i].args.dumps()))
        var o = Json.object()
        o.set("id", Json.of("call_" + String(i)))
        o.set("type", Json.of("function"))
        o.set("function", f)
        out.push(o)
    return out^


struct ToolDelta(Copyable, Movable):
    var name: Optional[String]
    var arguments: String

    def __init__(out self, name: Optional[String], arguments: String):
        self.name = name
        self.arguments = arguments


def tool_delta_json(d: ToolDelta) -> Json:
    var f = Json.object()
    f.set("arguments", Json.of(d.arguments))
    var call = Json.object()
    call.set("index", Json.of(0))
    if d.name:
        call.set("id", Json.of("call_0"))
        call.set("type", Json.of("function"))
        f.set("name", Json.of(d.name.value()))
    call.set("function", f)
    var a = Json.array()
    a.push(call)
    return a^


struct ToolStream(Movable):
    """Incremental tool-call parser for a stream: content is held back until it cannot start `<tool_call>`."""

    var raw: String
    var content_sent: Int
    var marker: Int  # -1: none
    var function_start: Int
    var function_end: Int  # -1: none
    var scan: Int
    var params: Int
    var name_sent: Bool
    var args_open: Bool
    var closed: Bool

    def __init__(out self):
        self.raw = String()
        self.content_sent = 0
        self.marker = -1
        self.function_start = 0
        self.function_end = -1
        self.scan = 0
        self.params = 0
        self.name_sent = False
        self.args_open = False
        self.closed = False

    def has_tool(self) -> Bool:
        return self.marker >= 0

    def finish_content(mut self) -> String:
        if self.marker >= 0:
            return String()
        var c = sub(self.raw, self.content_sent, self.raw.byte_length())
        self.content_sent = self.raw.byte_length()
        return c^

    def push(mut self, text: String, mut delta: Optional[ToolDelta]) -> String:
        """The content now safe to send; a tool delta, when one is ready, goes to `delta`."""
        self.raw += text
        if self.marker < 0:
            var i = _find(self.raw, _OPEN)
            if i >= 0:
                self.marker = i
                var content = sub(self.raw, self.content_sent, i)
                self.content_sent = i
                delta = self._next_tool_delta()
                return content^
            var safe = max(self.raw.byte_length() - 10, 0)
            var b = self.raw.as_bytes()
            while safe > self.content_sent and safe < len(b) and (b[safe] & 0xC0) == 0x80:
                safe -= 1
            var content = sub(self.raw, self.content_sent, safe) if safe > self.content_sent else String()
            self.content_sent = max(safe, self.content_sent)
            return content^
        delta = self._next_tool_delta()
        return String()

    def _next_tool_delta(mut self) -> Optional[ToolDelta]:
        if self.closed or self.marker < 0:
            return None
        var start = self.marker + 11
        if self.function_end < 0:
            var i = _find(self.raw, "<function=", start)
            if i < 0:
                return None
            var end = _find(self.raw, ">", i)
            if end < 0:
                return None
            self.function_start = i + 10
            self.function_end = end + 1
        if not self.name_sent:
            var name = _trim(sub(self.raw, self.function_start, self.function_end - 1))
            self.name_sent = True
            self.scan = self.function_end
            self.args_open = True
            return ToolDelta(Optional[String](name), String())
        var p = _find(self.raw, "<parameter=", self.scan)
        if p >= 0:
            var name_end = _find(self.raw, ">", p)
            if name_end < 0:
                return None
            var value_start = name_end + 1
            var value_end = _find(self.raw, "</parameter>", value_start)
            if value_end < 0:
                return None
            var name = sub(self.raw, p + 11, name_end)
            var parsed = _value_or_str(_trim(sub(self.raw, value_start, value_end)))
            self.scan = value_end + 12
            var prefix = "{" if self.params == 0 else ","
            self.params += 1
            return ToolDelta(None, prefix + Json.of(name).dumps() + ":" + parsed.dumps())
        if _find(self.raw, "</function>", self.scan) >= 0:
            self.closed = True
            return ToolDelta(None, "}" if self.args_open else String())
        return None


# ---- chat responses -----------------------------------------------------------


def chat_response(id: String, created: Int, model: String, content: String, calls: List[ToolCall], tokens: List[Int32], reason: String, n_prompt: Int, stats: Json) -> String:
    var msg = Json.object()
    msg.set("role", Json.of("assistant"))
    msg.set("content", Json.of(content))
    var r = reason
    if len(calls) > 0:
        msg.set("tool_calls", tool_calls_response_json(calls))
        r = "tool_calls"
    var o = Json.object()
    o.set("checkpoint", Json())
    o.set("checkpoint_from", Json())
    o.set("id", Json.of(id))
    o.set("object", Json.of("chat.completion"))
    o.set("created", Json.of(created))
    o.set("model", Json.of(model))
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("message", msg)
    ch.set("tokens", _ints(tokens))
    ch.set("finish_reason", Json.of(r))
    ch.set("logprobs", Json())
    o.set("choices", _choices(ch))
    o.set("usage", usage_json(n_prompt, len(tokens), stats))
    o.set("timings", stats.copy())
    return o.dumps()


def chat_delta(id: String, created: Int, model: String, content: String, tool: Optional[ToolDelta], first: Bool, tok: Int32) -> String:
    var d = Json.object()
    d.set("content", Json.of(content))
    if first:
        d.set("role", Json.of("assistant"))
    if tool:
        d.set("tool_calls", tool_delta_json(tool.value()))
        if content == "":
            # Rust drops an empty `content` when a tool delta rides along: rebuild without it
            var d2 = Json.object()
            if first:
                d2.set("role", Json.of("assistant"))
            d2.set("tool_calls", tool_delta_json(tool.value()))
            d = d2^
    var o = _head(id, "chat.completion.chunk", created, model)
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("delta", d)
    var t = List[Int32]()
    t.append(tok)
    ch.set("tokens", _ints(t))
    ch.set("finish_reason", Json())
    ch.set("logprobs", Json())
    o.set("choices", _choices(ch))
    return o.dumps()


def chat_finish(id: String, created: Int, model: String, tail: String, reason: String, has_tool: Bool, n_prompt: Int, n_completion: Int, stats: Json) -> String:
    var o = _head(id, "chat.completion.chunk", created, model)
    var d = Json.object()
    d.set("content", Json.of(tail))
    var ch = Json.object()
    ch.set("index", Json.of(0))
    ch.set("delta", d)
    ch.set("finish_reason", Json.of("tool_calls" if has_tool else reason))
    o.set("choices", _choices(ch))
    o.set("usage", usage_json(n_prompt, n_completion, stats))
    o.set("timings", stats.copy())
    return o.dumps()


# ---- the small GET routes -----------------------------------------------------


def models_response(model: String) -> String:
    var m = Json.object()
    m.set("id", Json.of(model))
    m.set("object", Json.of("model"))
    m.set("created", Json.of(0))
    m.set("owned_by", Json.of("mojo-baro"))
    var data = Json.array()
    data.push(m)
    var o = Json.object()
    o.set("object", Json.of("list"))
    o.set("data", data)
    return o.dumps()


def props_response(model: String, tmax: Int) -> String:
    var g = Json.object()
    g.set("n_ctx", Json.of(tmax))
    g.set("n_predict", Json.of(-1))
    g.set("model", Json.of(model))
    var caps = Json.object()
    caps.set("supports_tools", Json.of(True))
    caps.set("supports_tool_calls", Json.of(True))
    caps.set("supports_parallel_tool_calls", Json.of(True))
    var o = Json.object()
    o.set("default_generation_settings", g)
    o.set("chat_template_caps", caps)
    return o.dumps()


def health_response(state: Int, state_name: String, reason: String, queue: Int, info: Json) -> String:
    """Rust's fields, plus `engine_state` and `engine_reason`. `status` is `ok` only while the engine can take work."""
    var alive = state == ENG_IDLE or state == ENG_BUSY
    var o = Json.object()
    o.set("status", Json.of("ok" if alive else ("engine_dead" if state == ENG_DEAD else state_name)))
    o.set("engine_state", Json.of(state_name))
    o.set("engine_reason", Json.of(reason))
    o.set("queue", Json.of(queue))
    var pool = Json.array()
    pool.push(Json.of(queue))
    o.set("pool", pool)
    o.set("tokenizer", Json.of(True))
    var lim = Json.object()
    for k in ["tmax", "mrows", "kmax", "spec_k"]:
        lim.set(k, info.get(k))
    o.set("limits", lim)
    o.set("pack", info.get("pack"))
    return o.dumps()
