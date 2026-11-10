"""Text in, text out for the Mojo front: the Rust `Text` (serve/src/text.rs) in Mojo, byte for byte.

Tokenizer: the engine's own (serve/tokenizer.mojo), which reads a GGUF. A first argument ending in `.gguf`
is loaded directly; any other path is a pack's `tokenizer.json`, whose sibling `tokenizer-meta.json` names
the GGUF (`source_gguf`) and carries the template, bos and eos. Missing sidecar, key or file is a named
error, never a fallback. Build with `-I . -I serve` (the tokenizer imports resolve from serve/).
"""
from std.memory import ArcPointer
from json import Json, parse, STR, OBJ, INT
from minja.render import Renderer
from minja.value import parse_json
from tokenizer import Tokenizer

comptime KEEP = 8


def _file_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _utf8_prefix(b: Span[UInt8, _], i: Int) -> Int:
    """Bytes of the valid UTF-8 sequence at i, or minus the bytes of the maximal invalid prefix (Rust from_utf8_lossy)."""
    var n = len(b)
    var b0 = Int(b[i])
    if b0 < 0x80:
        return 1
    var need: Int
    var lo = 0x80
    var hi = 0xBF
    if b0 >= 0xC2 and b0 <= 0xDF:
        need = 1
    elif b0 >= 0xE0 and b0 <= 0xEF:
        need = 2
        if b0 == 0xE0:
            lo = 0xA0
        elif b0 == 0xED:
            hi = 0x9F
    elif b0 >= 0xF0 and b0 <= 0xF4:
        need = 3
        if b0 == 0xF0:
            lo = 0x90
        elif b0 == 0xF4:
            hi = 0x8F
    else:
        return -1
    for k in range(1, need + 1):
        if i + k >= n:
            return -k
        var c = Int(b[i + k])
        if c < lo or c > hi:
            return -k
        lo = 0x80
        hi = 0xBF
    return need + 1


def lossy(s: String) -> String:
    """The text with each maximal invalid UTF-8 prefix replaced by one U+FFFD, as Rust's String::from_utf8_lossy."""
    var b = s.as_bytes()
    var i = 0
    var bad = False
    while i < len(b):
        var k = _utf8_prefix(b, i)
        if k < 0:
            bad = True
            break
        i += k
    if not bad:
        return s
    var out = List[UInt8](capacity=len(b) + 8)
    i = 0
    while i < len(b):
        var k = _utf8_prefix(b, i)
        if k < 0:
            out.append(0xEF)
            out.append(0xBF)
            out.append(0xBD)
            i += -k
        else:
            out.extend(b[i : i + k])
            i += k
    return String(unsafe_from_utf8=Span(out))


def _decode(tok: Tokenizer, ids: List[Int]) -> String:
    return lossy(tok.decode(ids, True))


def common_prefix(a: List[Int32], b: List[Int32]) -> Int:
    """Length of the longest common prefix of two token sequences."""
    var n = min(len(a), len(b))
    var i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _ends_with(b: Span[UInt8, _], t: Span[UInt8, _]) -> Bool:
    if len(t) > len(b):
        return False
    var off = len(b) - len(t)
    for i in range(len(t)):
        if b[off + i] != t[i]:
            return False
    return True


struct Detok(Movable):
    """Streaming detokenizer, one per request. Holds back a partial UTF-8 tail; decodes a window of recent
    ids, not the whole prefix. The tokenizer is shared (read only), so any thread may own a Detok."""

    var tok: ArcPointer[Tokenizer]
    var win: List[Int]
    var emitted: Int  # bytes of decode(win) already handed out

    def __init__(out self, var tok: ArcPointer[Tokenizer]):
        self.tok = tok^
        self.win = List[Int]()
        self.emitted = 0

    def _take(mut self, stable: Int, full: String) -> String:
        var b = full.as_bytes()
        var delta = String(unsafe_from_utf8=b[self.emitted : stable])
        self.emitted = stable
        return delta^

    def push(mut self, id: Int32) -> String:
        """The text printable now (may be empty while a codepoint is incomplete)."""
        self.win.append(Int(id))
        var full = _decode(self.tok[], self.win)
        var b = full.as_bytes()
        var stable = len(b)
        if stable >= 3 and b[stable - 3] == 0xEF and b[stable - 2] == 0xBF and b[stable - 1] == 0xBD:
            stable -= 3
        if stable <= self.emitted or (self.emitted < len(b) and (Int(b[self.emitted]) & 0xC0) == 0x80):
            return String()
        var delta = self._take(stable, full)
        if stable == len(b) and len(self.win) >= 2 * KEEP:
            var from_ = len(self.win) - KEEP
            var tail_ids = List[Int]()
            for i in range(from_, len(self.win)):
                tail_ids.append(self.win[i])
            var tail = _decode(self.tok[], tail_ids)
            if _ends_with(b, tail.as_bytes()):
                self.win = tail_ids^
                self.emitted = tail.byte_length()
        return delta^

    def flush(mut self) -> String:
        """End of stream: what push held back (a trailing U+FFFD from a cut codepoint)."""
        var full = _decode(self.tok[], self.win)
        var n = full.byte_length()
        if n <= self.emitted:
            return String()
        var delta = self._take(n, full)
        return delta^


struct Text(Movable):
    var tok: ArcPointer[Tokenizer]
    var template: String
    var has_template: Bool
    var bos_id: Int
    var add_bos: Bool
    var bos_token: String
    var eos_token: String
    var stops: List[Int32]

    def __init__(out self, tokenizer_json: String, template_file: String = "") raises:
        var gguf = tokenizer_json
        var meta = Json()
        var from_gguf = tokenizer_json.endswith(".gguf")
        if not from_gguf:
            var slash = tokenizer_json.rfind("/")
            var meta_path = (String(tokenizer_json[byte=: slash + 1]) if slash >= 0 else String("")) + "tokenizer-meta.json"
            try:
                meta = parse(_file_text(meta_path))
            except e:
                raise Error("tokenizer: source_gguf <unknown> not found: cannot read " + meta_path + " (" + String(e) + ")")
            gguf = meta.text("source_gguf")
            if gguf == "":
                raise Error("tokenizer: source_gguf <unknown> not found: key absent in " + meta_path)
        var tok: Tokenizer
        try:
            tok = Tokenizer(gguf)
        except e:
            raise Error("tokenizer: source_gguf " + gguf + " not found (" + String(e) + ")")
        var eos_id: Int
        if from_gguf:
            self.template = tok.chat_template
            self.bos_id = tok.bos_id
            self.add_bos = tok.add_bos
            self.bos_token = tok.token_str(tok.bos_id)
            self.eos_token = tok.token_str(tok.eos_id)
            eos_id = tok.eos_id
            self.has_template = self.template != ""
        else:
            self.template = meta.text("chat_template")
            self.has_template = meta.get("chat_template").kind() == STR
            self.bos_id = _first_int(meta, "bos_token_id", "bos_id")
            eos_id = _first_int(meta, "eos_token_id", "eos_id")
            self.add_bos = meta.get("add_bos").truthy()
            self.bos_token = meta.text("bos_token")
            self.eos_token = meta.text("eos_token")
        if template_file != "":
            self.template = _file_text(template_file)
            self.has_template = True
        self.stops = List[Int32]()
        if eos_id >= 0:
            self.stops.append(Int32(eos_id))
        for s in ["<|im_end|>", "<|endoftext|>"]:
            var id = tok.tok2id.get(String(s), -1)
            if id >= 0 and Int32(id) not in self.stops:
                self.stops.append(Int32(id))
        self.tok = ArcPointer(tok^)

    def encode(self, s: String, add_special: Bool) raises -> List[Int32]:
        var raw = self.tok[].encode(s, False)
        var out = List[Int32](capacity=len(raw) + 1)
        if add_special and self.add_bos and self.bos_id >= 0 and (len(raw) == 0 or raw[0] != self.bos_id):
            out.append(Int32(self.bos_id))
        for id in raw:
            out.append(Int32(id))
        return out^

    def decode(self, ids: Span[Int32, _]) -> String:
        var l = List[Int](capacity=len(ids))
        for id in ids:
            l.append(Int(id))
        return _decode(self.tok[], l)

    def detok(self) -> Detok:
        return Detok(self.tok.copy())

    def stop_ids(self) -> List[Int32]:
        return self.stops.copy()

    def tool_xml(self) -> Bool:
        return self.has_template and self.template.find("<function=") >= 0

    def apply_chat_template(self, messages: Json, tools: Json, kwargs: Json) raises -> String:
        return self._render(_norm(messages), True, tools, kwargs)

    def _render(self, messages: Json, gen: Bool, tools: Json, kwargs: Json) raises -> String:
        if not self.has_template:
            var s = String()
            for m in messages.items():
                s += "<|im_start|>" + m.text("role") + "\n" + m.text("content") + "<|im_end|>\n"
            if gen:
                s += "<|im_start|>assistant\n"
            return s^
        var r = Renderer(self.template)
        r.set_global("messages", parse_json(r.h, messages.dumps(False)))
        r.set_global("add_generation_prompt", r.h.boolean(gen))
        r.set_global("bos_token", r.h.string(self.bos_token))
        r.set_global("eos_token", r.h.string(self.eos_token))
        if not tools.is_null():
            r.set_global("tools", parse_json(r.h, tools.dumps(False)))
        if kwargs.kind() == OBJ:
            for i in range(kwargs.len()):
                r.set_global(kwargs.key_at(i), parse_json(r.h, kwargs.at(i).dumps(False)))
        try:
            return r.render()
        except e:
            raise Error("chat_template: " + String(e))

    def _ids_of(self, messages: Json, gen: Bool, tools: Json, kwargs: Json, mut out: List[Int32]) -> Bool:
        """Render and encode into out; False when either fails (a prefix that does not render is skipped)."""
        try:
            out = self.encode(self._render(messages, gen, tools, kwargs), True)
            return True
        except:
            return False

    def role_boundaries(self, messages: Json, tools: Json, kwargs: Json) -> List[Int]:
        """Token length after each message rendered with no generation prompt, the start of the last
        message's content, and where the next turn diverges: checkpoint hints, as Rust `role_boundaries`."""
        var msgs = _norm(messages)
        var items = msgs.items()
        var out = List[Int]()
        var full = List[Int32]()
        var have_full = False
        var ids = List[Int32]()
        for k in range(1, len(items) + 1):
            var part = Json.array()
            for i in range(k):
                part.push(items[i])
            if not self._ids_of(part, False, tools, kwargs, ids):
                continue
            out.append(len(ids))
            if k == len(items):
                full = ids.copy()
                have_full = True
        if have_full and len(items) > 0:
            var open = Json.array()
            for i in range(len(items) - 1):
                open.push(items[i])
            var last = items[len(items) - 1].copy()
            last.set("content", Json.of(String("")))
            open.push(last)
            if self._ids_of(open, False, tools, kwargs, ids):
                var start = common_prefix(ids, full)
                if start > 0 and start < len(full):
                    out.append(start)
        var nxt = msgs.copy()
        var reply = Json.object()
        reply.set("role", Json.of(String("assistant")))
        reply.set("content", Json.of(String("x")))
        nxt.push(reply)
        var a = List[Int32]()
        var b = List[Int32]()
        if self._ids_of(nxt, False, tools, kwargs, a) and self._ids_of(msgs, True, tools, kwargs, b):
            var at = common_prefix(a, b)
            if at > 0 and at < len(b):
                out.append(at)
        sort(out)
        var uniq = List[Int]()
        for v in out:
            if len(uniq) == 0 or uniq[len(uniq) - 1] != v:
                uniq.append(v)
        return uniq^


def _first_int(meta: Json, key: String, alias_: String) -> Int:
    var j = meta.get(key)
    if j.is_null():
        j = meta.get(alias_)
    return j.int() if j.kind() == INT else -1


def _norm(messages: Json) -> Json:
    """Each message as role, content and tool_calls only: what Rust hands the template."""
    var out = Json.array()
    for m in messages.items():
        var n = Json.object()
        n.set("role", Json.of(m.text("role")))
        n.set("content", m.get("content"))
        var tc = m.get("tool_calls")
        if not tc.is_null():
            n.set("tool_calls", tc)
        out.push(n)
    return out^
