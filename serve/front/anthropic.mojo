"""Anthropic Messages wire adapters for the Mojo front."""
from api import TextLike, Call, ApiError, bad, parse_chat, _pos
from json import Json, parse, STR, ARR, OBJ


def _anthropic_text(v: Json) raises ApiError -> String:
    if v.kind() == STR:
        return v.str()
    if v.kind() == ARR:
        var out = String()
        for b in v.items():
            if b.kind() != OBJ:
                raise bad("content blocks must be objects")
            var typ = b.get("type").str()
            if typ == "text":
                var t = b.get("text")
                if t.kind() != STR:
                    raise bad("text block needs a string `text`")
                out += t.str()
            elif typ == "image":
                raise bad("image input is not supported by this model front")
            elif typ == "thinking":
                out += "<think>" + b.get("thinking").str() + "</think>"
            elif typ == "redacted_thinking":
                continue
            else:
                raise bad("unsupported content block type: " + typ)
        return out^
    if v.is_null():
        return String()
    raise bad("content must be a string or an array of blocks")


def _tool_defs(tools: Json) raises ApiError -> Json:
    if tools.is_null():
        return Json()
    if tools.kind() != ARR:
        raise bad("`tools` must be an array")
    var out = Json.array()
    for t in tools.items():
        var name = t.get("name")
        var schema = t.get("input_schema")
        if t.kind() != OBJ or name.kind() != STR or schema.kind() != OBJ:
            raise bad("each tool needs a string `name` and object `input_schema`")
        var f = Json.object()
        f.set("name", name)
        f.set("description", t.get("description"))
        f.set("parameters", schema)
        var item = Json.object()
        item.set("type", Json.of("function"))
        item.set("function", f)
        out.push(item)
    return out^


def _tool_choice(choice: Json) raises ApiError -> Json:
    if choice.is_null():
        return Json()
    if choice.kind() == OBJ:
        var typ = choice.get("type").str()
        if typ == "auto":
            return Json.of("auto")
        if typ == "none":
            return Json.of("none")
        if typ == "any":
            return Json.of("required")
        if typ == "tool":
            var f = Json.object()
            f.set("name", choice.get("name"))
            var o = Json.object()
            o.set("type", Json.of("function"))
            o.set("function", f)
            return o^
    raise bad("`tool_choice` must be auto, none, any, or a named tool")


def _messages(j: Json) raises ApiError -> Json:
    var src = j.get("messages")
    if src.kind() != ARR or src.len() == 0:
        raise bad("`messages` must be a non-empty array")
    var out = Json.array()
    var system = j.get("system")
    if not system.is_null():
        var sm = Json.object()
        sm.set("role", Json.of("system"))
        sm.set("content", Json.of(_anthropic_text(system)))
        out.push(sm)
    for m in src.items():
        var role = m.get("role")
        if m.kind() != OBJ or role.kind() != STR:
            raise bad("each message needs a string `role`")
        var content = m.get("content")
        if role.str() == "system":
            # Claude Code puts hook output mid-conversation as role "system"; chat templates
            # accept system only first. Keep the position (KV prefix stays cached) as a user turn.
            var sr = Json.object()
            sr.set("role", Json.of("user"))
            sr.set("content", Json.of("<system-reminder>\n" + _anthropic_text(content) + "\n</system-reminder>"))
            out.push(sr)
            continue
        var plain = String()
        var calls = Json.array()
        var results = Json.array()
        if content.kind() == STR:
            plain = content.str()
        elif content.kind() == ARR:
            for b in content.items():
                var typ = b.get("type").str()
                if typ == "text":
                    plain += b.get("text").str()
                elif typ == "thinking":
                    plain += "<think>" + b.get("thinking").str() + "</think>"
                elif typ == "redacted_thinking":
                    continue
                elif typ == "image":
                    raise bad("image input is not supported by this model front")
                elif typ == "tool_use":
                    var id = b.get("id")
                    var name = b.get("name")
                    var input = b.get("input")
                    if id.kind() != STR or name.kind() != STR or input.kind() != OBJ:
                        raise bad("tool_use needs string `id`, string `name`, and object `input`")
                    var fun = Json.object()
                    fun.set("name", name)
                    fun.set("arguments", Json.of(input.dumps()))
                    var tc = Json.object()
                    tc.set("id", id)
                    tc.set("type", Json.of("function"))
                    tc.set("function", fun)
                    calls.push(tc)
                elif typ == "tool_result":
                    var result = Json.object()
                    result.set("role", Json.of("tool"))
                    result.set("tool_call_id", b.get("tool_use_id"))
                    result.set("content", Json.of(_anthropic_text(b.get("content"))))
                    results.push(result)
                else:
                    raise bad("unsupported content block type: " + typ)
        else:
            raise bad("message `content` must be a string or an array")
        var msg = Json.object()
        msg.set("role", role)
        msg.set("content", Json.of(plain))
        if calls.len() > 0:
            msg.set("tool_calls", calls)
        if plain != "" or calls.len() > 0 or (role.str() == "user" and results.len() == 0) or role.str() == "assistant":
            out.push(msg)
        for result in results.items():
            out.push(result)
    return out^


def parse_messages[T: TextLike](text: T, j: Json, spec_k: Int) raises ApiError -> Call:
    """Translate an Anthropic request to the existing chat-template request path."""
    var request = Json.object()
    request.set("messages", _messages(j))
    request.set("tools", _tool_defs(j.get("tools")))
    request.set("tool_choice", _tool_choice(j.get("tool_choice")))
    for key in ["model", "max_tokens", "temperature", "top_p", "stream"]:
        var value = j.get(key)
        if not value.is_null():
            request.set(key, value)
    var stops = j.get("stop_sequences")
    if not stops.is_null():
        request.set("stop", stops)
    var call = parse_chat(text, request, spec_k, True)
    call.anthropic = True
    call.stop_sequences = stops.copy()
    return call^


def count_tokens_response[T: TextLike](text: T, j: Json, spec_k: Int) raises ApiError -> String:
    var call = parse_messages(text, j, spec_k)
    var out = Json.object()
    out.set("input_tokens", Json.of(call.n_prompt))
    return out.dumps()


def _append_text_block(mut out: Json, text: String):
    if text == "":
        return
    var block = Json.object()
    block.set("type", Json.of("text"))
    block.set("text", Json.of(text))
    out.push(block)


def _append_thinking_block(mut out: Json, text: String):
    if text == "":
        return
    var block = Json.object()
    block.set("type", Json.of("thinking"))
    block.set("thinking", Json.of(text))
    block.set("signature", Json.of(""))
    out.push(block)


def _content_blocks(text: String) -> Json:
    var out = Json.array()
    var pos = 0
    while pos < text.byte_length():
        var open = _pos(text.find("<think>", pos))
        if open < 0:
            _append_text_block(out, String(text[byte=pos:]))
            break
        _append_text_block(out, String(text[byte=pos:open]))
        var start = open + 7
        var close = _pos(text.find("</think>", start))
        if close < 0:
            _append_text_block(out, String(text[byte=open:]))
            break
        _append_thinking_block(out, String(text[byte=start:close]))
        pos = close + 8
    return out^


def message_response(chat_json: String, stop_sequences: Json) raises ApiError -> String:
    var src = Json.object()
    try:
        src = parse(chat_json)
    except:
        raise bad("invalid chat response JSON")
    var first = Json()
    var found = False
    for item in src.get("choices").items():
        first = item.copy()
        found = True
        break
    if not found:
        raise bad("chat response has no choices")
    var msg = first.get("message")
    var text = msg.get("content")
    var content = _content_blocks(text.str() if text.kind() == STR else String())
    var tool_calls = msg.get("tool_calls")
    if tool_calls.kind() == ARR:
        var tool_index = 0
        for tc in tool_calls.items():
            var f = tc.get("function")
            var input = Json.object()
            try:
                input = parse(f.get("arguments").str())
            except:
                raise bad("tool call arguments are invalid JSON")
            var block = Json.object()
            block.set("type", Json.of("tool_use"))
            block.set("id", Json.of("toolu_" + String(tool_index)))
            block.set("name", f.get("name"))
            block.set("input", input)
            content.push(block)
            tool_index += 1
    var reason = first.get("finish_reason").str()
    var stop = "end_turn"
    var stop_sequence = Json()
    if reason == "length":
        stop = "max_tokens"
    elif reason == "tool_calls":
        stop = "tool_use"
    elif reason == "stop" and stop_sequences.kind() == ARR and text.kind() == STR:
        for seq in stop_sequences.items():
            if seq.kind() == STR and text.str().endswith(seq.str()):
                stop = "stop_sequence"
                stop_sequence = seq.copy()
                break
    var usage = src.get("usage")
    var u = Json.object()
    u.set("input_tokens", usage.get("prompt_tokens"))
    u.set("output_tokens", usage.get("completion_tokens"))
    var out = Json.object()
    out.set("id", Json.of("msg_" + src.get("id").str()))
    out.set("type", Json.of("message"))
    out.set("role", Json.of("assistant"))
    out.set("model", src.get("model"))
    out.set("content", content)
    out.set("stop_reason", Json.of(stop))
    out.set("stop_sequence", stop_sequence)
    out.set("usage", u)
    return out.dumps()


struct MessageStream(Movable):
    var block_index: Int
    var block_kind: Int
    var stop_sequences: Json
    var stop_sequence: Json
    var text: String
    var pending: String
    var thinking: Bool

    def __init__(out self, var stop_sequences: Json = Json()):
        self.block_index = -1
        self.block_kind = 0
        self.stop_sequences = stop_sequences^
        self.stop_sequence = Json()
        self.text = String()
        self.pending = String()
        self.thinking = False


def _sse(event: String, data: Json) -> String:
    return "event: " + event + "\ndata: " + data.dumps() + "\n\n"


def _stream_start_block(mut state: MessageStream, typ: String) -> String:
    state.block_index += 1
    state.block_kind = 1 if typ == "text" else 3
    var block = Json.object()
    block.set("type", Json.of(typ))
    if typ == "text":
        block.set("text", Json.of(""))
    else:
        block.set("thinking", Json.of(""))
        block.set("signature", Json.of(""))
    var event = Json.object()
    event.set("type", Json.of("content_block_start"))
    event.set("index", Json.of(state.block_index))
    event.set("content_block", block)
    return _sse("content_block_start", event)


def _stream_close_block(mut state: MessageStream) -> String:
    if state.block_kind == 0:
        return String()
    var out = String()
    if state.block_kind == 3:
        var d = Json.object()
        d.set("type", Json.of("signature_delta"))
        d.set("signature", Json.of(""))
        var event = Json.object()
        event.set("type", Json.of("content_block_delta"))
        event.set("index", Json.of(state.block_index))
        event.set("delta", d)
        out += _sse("content_block_delta", event)
    var event = Json.object()
    event.set("type", Json.of("content_block_stop"))
    event.set("index", Json.of(state.block_index))
    state.block_kind = 0
    out += _sse("content_block_stop", event)
    return out^


def _stream_content_delta(state: MessageStream, typ: String, text: String) -> String:
    var d = Json.object()
    d.set("type", Json.of("text_delta" if typ == "text" else "thinking_delta"))
    d.set("text" if typ == "text" else "thinking", Json.of(text))
    var event = Json.object()
    event.set("type", Json.of("content_block_delta"))
    event.set("index", Json.of(state.block_index))
    event.set("delta", d)
    return _sse("content_block_delta", event)


def _stream_emit(mut state: MessageStream, text: String, typ: String) -> String:
    if text == "":
        return String()
    var out = String()
    var kind = 1 if typ == "text" else 3
    if state.block_kind != kind:
        out += _stream_close_block(state)
        out += _stream_start_block(state, typ)
    out += _stream_content_delta(state, typ, text)
    return out^


def _stream_text(mut state: MessageStream, text: String, force: Bool = False) -> String:
    state.pending += text
    var out = String()
    while state.pending != "":
        if state.thinking:
            var close = _pos(state.pending.find("</think>"))
            if close >= 0:
                out += _stream_emit(state, String(state.pending[byte=0:close]), "thinking")
                out += _stream_close_block(state)
                var rest = String(state.pending[byte=close + 8:])
                state.pending = rest^
                state.thinking = False
                continue
            var safe = state.pending.byte_length()
            if not force:
                for n in range(1, 8):
                    if state.pending.endswith(String("</think>"[byte=0:n])):
                        safe -= n
            if safe > 0:
                out += _stream_emit(state, String(state.pending[byte=0:safe]), "thinking")
                var rest = String(state.pending[byte=safe:])
                state.pending = rest^
            break
        var open = _pos(state.pending.find("<think>"))
        if open >= 0:
            out += _stream_emit(state, String(state.pending[byte=0:open]), "text")
            out += _stream_close_block(state)
            var rest = String(state.pending[byte=open + 7:])
            state.pending = rest^
            state.thinking = True
            continue
        var safe = state.pending.byte_length()
        if not force:
            for n in range(1, 7):
                if state.pending.endswith(String("<think>"[byte=0:n])):
                    safe -= n
        if safe > 0:
            out += _stream_emit(state, String(state.pending[byte=0:safe]), "text")
            var rest = String(state.pending[byte=safe:])
            state.pending = rest^
        break
    return out^


def messages_stream_start(id: String, model: String, input_tokens: Int) -> String:
    var usage = Json.object()
    usage.set("input_tokens", Json.of(input_tokens))
    usage.set("output_tokens", Json.of(0))
    var msg = Json.object()
    msg.set("id", Json.of("msg_" + id))
    msg.set("type", Json.of("message"))
    msg.set("role", Json.of("assistant"))
    msg.set("model", Json.of(model))
    msg.set("content", Json.array())
    msg.set("stop_reason", Json())
    msg.set("stop_sequence", Json())
    msg.set("usage", usage)
    var data = Json.object()
    data.set("type", Json.of("message_start"))
    data.set("message", msg)
    return _sse("message_start", data)


def messages_stream_chunk(mut state: MessageStream, chat_chunk: String) raises ApiError -> String:
    var chunk = Json.object()
    try:
        chunk = parse(chat_chunk)
    except:
        return messages_stream_error("invalid chat stream JSON")
    var output = String()
    var choices = chunk.get("choices")
    if choices.is_null():
        var source_error = chunk.get("error")
        var err = Json.object()
        err.set("type", Json.of("error"))
        var detail = Json.object()
        detail.set("type", Json.of("api_error"))
        detail.set("message", source_error.get("message"))
        err.set("error", detail)
        return _sse("error", err)
    for choice in choices.items():
        var delta = choice.get("delta")
        var text = delta.get("content")
        if text.kind() == STR and text.str() != "":
            state.text += text.str()
        output += _stream_text(state, text.str() if text.kind() == STR else String())
        var tool_calls = delta.get("tool_calls")
        if tool_calls.kind() == ARR:
            for tc in tool_calls.items():
                if state.block_kind != 2:
                    output += _stream_text(state, String(), True)
                    output += _stream_close_block(state)
                    state.block_index += 1
                    state.block_kind = 2
                    var function = tc.get("function")
                    var block = Json.object()
                    block.set("type", Json.of("tool_use"))
                    block.set("id", Json.of("toolu_" + String(tc.get("index").int())))
                    block.set("name", function.get("name"))
                    block.set("input", Json.object())
                    var start = Json.object()
                    start.set("type", Json.of("content_block_start"))
                    start.set("index", Json.of(state.block_index))
                    start.set("content_block", block)
                    output += _sse("content_block_start", start)
                var arguments = tc.get("function").get("arguments")
                if arguments.kind() == STR and arguments.str() != "":
                    var d = Json.object()
                    d.set("type", Json.of("input_json_delta"))
                    d.set("partial_json", arguments)
                    var event = Json.object()
                    event.set("type", Json.of("content_block_delta"))
                    event.set("index", Json.of(state.block_index))
                    event.set("delta", d)
                    output += _sse("content_block_delta", event)
        var finish = choice.get("finish_reason")
        if finish.kind() == STR:
            output += _stream_text(state, String(), True)
            if state.block_kind != 0:
                output += _stream_close_block(state)
            var reason = "end_turn"
            if finish.str() == "length":
                reason = "max_tokens"
            elif finish.str() == "tool_calls":
                reason = "tool_use"
            elif state.stop_sequences.kind() == ARR:
                for seq in state.stop_sequences.items():
                    if seq.kind() == STR and state.text.endswith(seq.str()):
                        reason = "stop_sequence"
                        state.stop_sequence = seq.copy()
                        break
            var usage = chunk.get("usage")
            var delta = Json.object()
            delta.set("stop_reason", Json.of(reason))
            delta.set("stop_sequence", state.stop_sequence.copy())
            var event = Json.object()
            event.set("type", Json.of("message_delta"))
            event.set("delta", delta)
            var u = Json.object()
            u.set("output_tokens", usage.get("completion_tokens"))
            event.set("usage", u)
            output += _sse("message_delta", event)
            var end = Json.object()
            end.set("type", Json.of("message_stop"))
            output += _sse("message_stop", end)
    return output^


def messages_stream_error(message: String) -> String:
    var error = Json.object()
    error.set("type", Json.of("api_error"))
    error.set("message", Json.of(message))
    var data = Json.object()
    data.set("type", Json.of("error"))
    data.set("error", error)
    return _sse("error", data)


def messages_error_body(body: String) -> String:
    var message = body
    var kind = "invalid_request_error"
    try:
        var parsed = parse(body)
        var error = parsed.get("error")
        var msg = error.get("message")
        var typ = error.get("type")
        if msg.kind() == STR:
            message = msg.str()
        if typ.kind() == STR and typ.str() == "engine_error":
            kind = "api_error"
    except:
        pass
    var error = Json.object()
    error.set("type", Json.of(kind))
    error.set("message", Json.of(message))
    var out = Json.object()
    out.set("type", Json.of("error"))
    out.set("error", error)
    return out.dumps()
