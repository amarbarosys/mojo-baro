"""Text and Detok against the Rust oracle (exchange/receipts/front/text-oracle.jsonl, written by
`cargo test --release text_oracle`). Every row: prompt bytes, ids (add_special both ways), decode, Detok per id,
role_boundaries, stop ids. A mismatch prints the first differing byte; there is no tolerance.
Build: ./.venv/bin/mojo build serve/front/test_text.mojo -I . -I serve -o .work/test_text
Run from the repo root with MINJA_NOW set (the template has no strftime_now; the pin is for any that does).
TEXT_MUTANT=flip_add_special makes every encode call pass the opposite flag: the test must then FAIL."""
from std.collections import Dict
from std.os import getenv, stat, makedirs
from std.sys import exit
from std.time import perf_counter_ns
from json import Json, parse
from sha256 import sha256
from text import Text, Detok, common_prefix

comptime ORACLE = "exchange/receipts/front/text-oracle.jsonl"
comptime JINJA = "exchange/receipts/front/jinja2-oracle.jsonl"
comptime HEADER_MAX = 96 << 20


struct Counts(Movable):
    var n: Int
    var bad: Int
    var shown: Int
    var pushes: Int
    var push_ns: Int

    def __init__(out self):
        self.pushes = 0
        self.push_ns = 0
        self.n = 0
        self.bad = 0
        self.shown = 0


def check(mut c: Counts, ok: Bool, what: String):
    c.n += 1
    if not ok:
        c.bad += 1
        if c.shown < 25:
            c.shown += 1
            print("FAIL", what)


def file_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def hex(d: Array[UInt8, 32]) -> String:
    comptime digits = "0123456789abcdef"
    var s = String()
    for i in range(32):
        s += String(digits[byte=Int(d[i]) >> 4]) + String(digits[byte=Int(d[i]) & 15])
    return s^


def sha_file(path: String, limit: Int = -1) raises -> String:
    with open(path, "r") as f:
        var b = f.read_bytes(limit if limit >= 0 else Int(stat(path).st_size))
        return hex(sha256(Span(b)))


def first_diff(a: String, b: String) -> Int:
    """Byte index of the first difference, or -1 when equal."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return -1 if len(x) == len(y) else n


def ints(v: List[Int]) -> String:
    var s = String("[")
    for i in range(len(v)):
        s += (", " if i > 0 else "") + String(v[i])
    return s + "]"


def around(s: String, at: Int) -> String:
    """Up to 40 bytes of s from just before byte `at`, for a failure message."""
    var b = s.as_bytes()
    var lo = max(0, at - 12)
    return String(unsafe_from_utf8=b[lo : min(len(b), at + 28)])


def ids_diff(got: List[Int32], want: Json) -> Int:
    """Index of the first differing id, or -1 when equal."""
    var w = want.items()
    var n = min(len(got), len(w))
    for i in range(n):
        if Int(got[i]) != w[i].int():
            return i
    return -1 if len(got) == len(w) else n


def tool_calls(calls: Json) -> Json:
    """The conversion of main.rs tool_calls_for_template: `arguments` string back to an object."""
    var out = Json.array()
    for c in calls.items():
        var f = c.get("function")
        var args = Json.object()
        try:
            var a = f.text("arguments")
            if a != "":
                args = parse(a)
        except:
            args = Json.object()
        var fn_ = Json.object()
        fn_.set("name", Json.of(f.text("name")))
        fn_.set("arguments", args)
        var e = Json.object()
        e.set("function", fn_)
        out.push(e)
    return out^


def p0_messages(body: Json) -> Json:
    var out = Json.array()
    for m in body.get("messages").items():
        var n = Json.object()
        n.set("role", Json.of(m.text("role")))
        n.set("content", m.get("content"))
        var tc = m.get("tool_calls")
        if not tc.is_null():
            n.set("tool_calls", tool_calls(tc))
        out.push(n)
    return out^


def flip(mutant: Bool, add_special: Bool) -> Bool:
    return (not add_special) if mutant else add_special


def verify_pack(mut c: Counts, h: Json) raises:
    var pack = h.text("pack")
    var tj = h.text("tokenizer_json")
    var meta = h.text("meta")
    var gguf = h.text("gguf")
    for p in [tj, meta, gguf]:
        try:
            _ = stat(p)
        except:
            raise Error("pack " + pack + ": " + p + " is missing; make it with: BARO_WIRE_PACK=<pack dir with tokenizer.json> cargo test --release text_oracle")
    check(c, sha_file(tj) == h.text("tokenizer_sha256"), pack + ": tokenizer.json sha256 differs from the oracle's")
    check(c, sha_file(meta) == h.text("meta_sha256"), pack + ": tokenizer-meta.json sha256 differs from the oracle's")
    check(c, Int(stat(gguf).st_size) == h.get("gguf_size").int(), pack + ": gguf size differs from the oracle's")
    check(c, sha_file(gguf, HEADER_MAX) == h.text("gguf_header_sha256"), pack + ": gguf header sha256 differs from the oracle's")
    check(c, parse(file_text(meta)).text("source_gguf") == gguf, pack + ": sidecar names another gguf")


def stream(mut c: Counts, t: Text, ids: List[Int32], what: String) -> List[String]:
    """Push every id; with the flush, the pieces must equal decode(ids)."""
    var d = t.detok()
    var got = List[String]()
    var all = String()
    var t0 = perf_counter_ns()
    for id in ids:
        var s = d.push(id)
        all += s
        got.append(s^)
    c.push_ns += Int(perf_counter_ns() - t0)
    c.pushes += len(ids)
    all += d.flush()
    check(c, all == t.decode(Span(ids)), what + ": Detok pieces + flush differ from decode, first byte " + String(first_diff(all, t.decode(Span(ids)))))
    return got^


def row_checks(mut c: Counts, t: Text, r: Json, mutant: Bool) raises:
    """Ids, decode and Detok on the oracle's prompt text (the prompt bytes are checked by the caller)."""
    var what = r.text("pack") + "/" + r.text("kind") + "/" + r.text("id")
    var text = r.text("prompt")
    var ids = t.encode(text, flip(mutant, True))
    var di = ids_diff(ids, r.get("ids"))
    check(c, di < 0, what + ": ids (add_special) differ, first id index " + String(di))
    var plain = t.encode(text, flip(mutant, False))
    var pw = r.get("ids") if r.get("ids_plain").is_null() else r.get("ids_plain")
    di = ids_diff(plain, pw)
    check(c, di < 0, what + ": ids (plain) differ, first id index " + String(di))
    var dec = t.decode(Span(ids))
    var want_dec = text if r.get("decode_is_prompt").truthy() else r.text("decode")
    var d = first_diff(dec, want_dec)
    check(c, d < 0, what + ": decode differs, first byte " + String(d))
    var got = stream(c, t, ids, what)
    check(c, r.get("detok_concat_is_decode").truthy(), what + ": oracle's own Detok pieces differ from its decode")
    if r.has("detok"):
        var w = r.get("detok").items()
        var bad = -1
        for i in range(len(got)):
            if i >= len(w) or got[i] != w[i].str():
                bad = i
                break
        check(c, bad < 0 and len(got) == len(w), what + ": Detok piece differs from Rust at id index " + String(bad))


def main() raises:
    var t0 = perf_counter_ns()
    var mutant = getenv("TEXT_MUTANT", "") == "flip_add_special"
    var c = Counts()
    var lines = file_text(ORACLE).split("\n")
    var texts = List[Text]()
    var names = List[String]()
    var headers = List[Json]()
    var jinja = Dict[String, Json]()
    for jl in file_text(JINJA).split("\n"):
        if jl.byte_length() == 0:
            continue
        var jj = parse(String(jl))
        if jj.text("kind") != "header":
            jinja[jj.text("kind") + "/" + jj.text("id")] = jj.copy()
    var rb_rows = 0
    for ln in lines:
        if ln.byte_length() == 0:
            continue
        var r = parse(String(ln))
        var kind = r.text("kind")
        if kind == "header":
            verify_pack(c, r)
            names.append(r.text("pack"))
            texts.append(Text(r.text("tokenizer_json")))
            headers.append(r.copy())
            var ids = List[Int32]()
            for s in r.get("stop_ids").items():
                ids.append(Int32(s.int()))
            check(c, texts[len(texts) - 1].stop_ids() == ids, r.text("pack") + ": stop_ids differ")
            check(c, texts[len(texts) - 1].tool_xml() == r.get("tool_xml").truthy(), r.text("pack") + ": tool_xml differs")
            continue
        var pi = -1
        for i in range(len(names)):
            if names[i] == r.text("pack"):
                pi = i
        ref t = texts[pi]
        var msgs = Json.array()
        var tools = Json()
        var kwargs = Json()
        var prompt = r.text("prompt")
        var chat = True
        if kind == "raw":
            chat = False
        elif kind == "gate":
            var m = Json.object()
            m.set("role", Json.of(String("user")))
            m.set("content", Json.of(file_text("bench/mtp-prompts/" + r.text("id") + ".txt")))
            msgs.push(m)
        elif kind == "p0":
            var body = parse(file_text("bench/ailab/" + r.text("id") + ".json"))
            msgs = p0_messages(body)
            tools = body.get("tools")
            kwargs = body.get("chat_template_kwargs")
        else:
            var q = r.get("req")
            msgs = q.get("messages")
            tools = q.get("tools")
            kwargs = q.get("kwargs")
        if chat:
            var rendered = t.apply_chat_template(msgs, tools, kwargs)
            var what = r.text("pack") + "/" + kind + "/" + r.text("id")
            var jkey = kind + "/" + r.text("id")
            var jr = jinja[jkey].copy() if jkey in jinja else Json()
            var rust_ok = first_diff(rendered, prompt) < 0
            if jr.is_null():
                check(c, rust_ok, what + ": prompt bytes differ from Rust, first byte " + String(first_diff(rendered, prompt)) + " got [" + around(rendered, first_diff(rendered, prompt)) + "] want [" + around(prompt, first_diff(rendered, prompt)) + "]")
            elif kind == "chat":
                var jd = first_diff(rendered, jr.text("prompt"))
                check(c, jd < 0, what + ": prompt differs from jinja2, first byte " + String(jd) + " got [" + around(rendered, jd) + "] want [" + around(jr.text("prompt"), jd) + "]")
                # Rust equals jinja2 on this row, or the row is a named DELIBERATE DIFFERENCE and Mojo equals jinja2, not Rust
                check(c, rust_ok == jr.get("equals_rust").truthy(), what + ": Mojo/Rust equality does not match the jinja2 receipt's equals_rust")
                if not rust_ok:
                    print("DELIBERATE DIFFERENCE", what, "first differing byte vs Rust:", first_diff(rendered, prompt))
            else:
                check(c, hex(sha256(rendered.as_bytes())) == jr.text("sha256") and rendered.byte_length() == jr.get("len").int(), what + ": prompt differs from jinja2 (sha256)")
                check(c, rust_ok, what + ": prompt bytes differ from Rust, first byte " + String(first_diff(rendered, prompt)))
            row_checks(c, t, r, mutant)
            var rb = t.role_boundaries(msgs, tools, kwargs)
            var want = List[Int]()
            for v in r.get("role_boundaries").items():
                want.append(v.int())
            var rname = r.text("pack") + "/" + r.text("id")
            if rust_ok:
                check(c, rb == want, rname + ": role_boundaries differ, got " + ints(rb) + " want " + ints(want))
            else:
                # a DELIBERATE DIFFERENCE row: the float text has another token count, so each boundary is Rust's
                # (before the float) or Rust's minus the one delta of the full prompt (after it)
                var delta = len(t.encode(prompt, True)) - len(t.encode(rendered, True))
                var shaped = len(rb) == len(want) and delta > 0
                for i in range(min(len(rb), len(want))):
                    shaped = shaped and (rb[i] == want[i] or rb[i] + delta == want[i])
                check(c, shaped, rname + ": role_boundaries are not Rust's shifted by " + String(delta) + ", got " + ints(rb) + " want " + ints(want))
            rb_rows += 1
        else:
            row_checks(c, t, r, mutant)
    print("rows done:", len(lines) - 1, "role_boundaries rows:", rb_rows, "ms:", (perf_counter_ns() - t0) // 1000000)
    # the integer 10**22 prints as an int in jinja2 but no front can carry it (Rust and Json read it as a float)
    var pyint = jinja["chat/float-args-pyint"].text("prompt")
    check(c, pyint.find("\n10000000000000000000000\n</parameter>") >= 0 and pyint != jinja["chat/float-args"].text("prompt"), "jinja2 receipt: a Python int 10**22 must render as digits, unlike the float 1e+22")
    extra_checks(c, texts[0], headers[0])
    print("Detok.push:", c.pushes, "ids,", c.push_ns // max(1, c.pushes), "ns per id")
    if c.bad == 0:
        print("PASS", c.n - c.bad, "/", c.n)
    else:
        print("FAIL", c.n - c.bad, "/", c.n)
        exit(1)


def extra_checks(mut c: Counts, t: Text, h: Json) raises:
    """Truncated streams, the named load errors, and the .gguf direct load."""
    # a stream cut inside a 4-byte codepoint: the held tail comes out of flush, nothing is lost
    var cut = t.encode(String("\U0010ffff"), True)
    var cut1 = List[Int32]()
    for i in range(len(cut) - 1):
        cut1.append(cut[i])
    _ = stream(c, t, cut1, "truncated U+10FFFF")
    var d = t.detok()
    var held = True
    for id in cut1:
        held = held and d.push(id) == ""
    check(c, held, "Detok must hold back every piece of a cut codepoint")
    check(c, d.flush() == "\U0000FFFD", "flush after a cut codepoint is one U+FFFD")
    # named errors, never a fallback
    var dir = ".work/front-text/errs"
    makedirs(dir + "/nokey", exist_ok=True)
    makedirs(dir + "/badpath", exist_ok=True)
    with open(dir + "/nokey/tokenizer-meta.json", "w") as f:
        f.write("{}")
    with open(dir + "/badpath/tokenizer-meta.json", "w") as f:
        f.write('{"source_gguf": "/nonexistent/x.gguf"}')
    expect_error(c, "/nonexistent/tokenizer.json", "tokenizer: source_gguf", "missing sidecar")
    expect_error(c, dir + "/nokey/tokenizer.json", "tokenizer: source_gguf", "sidecar without source_gguf")
    expect_error(c, dir + "/badpath/tokenizer.json", "tokenizer: source_gguf /nonexistent/x.gguf not found", "gguf path not found")
    # the .gguf is loaded directly and encodes as the pack does
    var g = Text(h.text("gguf"))
    var a = g.encode(String("hello <|im_start|>user\nworld"), True)
    var b = t.encode(String("hello <|im_start|>user\nworld"), True)
    check(c, a == b, "a direct .gguf load encodes as the pack")
    # add_special must matter on the bos pack: the oracle rows for it fail under the mutant


def expect_error(mut c: Counts, path: String, needle: String, what: String):
    try:
        _ = Text(path)
        check(c, False, what + ": no error")
    except e:
        check(c, String(e).find(needle) >= 0, what + ": error text is '" + String(e) + "'")
