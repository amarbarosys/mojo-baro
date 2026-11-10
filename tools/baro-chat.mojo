from std.ffi import external_call
from std.sys import argv, exit, stderr
from grammar.json_value import JSONDoc, parse_json_bytes

comptime USAGE = """usage: baro chat [--url HOST:PORT] [--max-tokens N] [--temperature T] [--top-p P] [--seed S]
                 [--system TEXT] [--no-think] [-p PROMPT]

Terminal chat against baro-serve's /v1/chat/completions (stream: true), default 127.0.0.1:8080.
Tokens print as they arrive; after each reply one stats line from the server's own usage/timings
(completion tokens, tok/s, prefill seconds, prompt tokens restored from a checkpoint) goes to stderr.
-p sends one prompt, prints the reply and exits (exit 1 on any error). Interactive commands:
/reset (drop the history), /system TEXT (replace the system message), /exit (or Ctrl-D).
Sampler fields are sent only when given; the server's defaults apply otherwise."""


def die(msg: String):
    print("baro chat: " + msg, file=stderr)
    exit(1)


def json_str(s: String) -> String:
    comptime HEX = "0123456789abcdef"
    var o = List[UInt8]()
    o.append(34)
    for b in s.as_bytes():
        if b == 34 or b == 92:
            o.append(92)
            o.append(b)
        elif b == 10:
            o.extend("\\n".as_bytes())
        elif b == 13:
            o.extend("\\r".as_bytes())
        elif b == 9:
            o.extend("\\t".as_bytes())
        elif b < 32:
            o.extend("\\u00".as_bytes())
            o.append(HEX.as_bytes()[Int(b >> 4)])
            o.append(HEX.as_bytes()[Int(b & 15)])
        else:
            o.append(b)
    o.append(34)
    return String(unsafe_from_utf8=o^)


def fmt1(x: Float64) -> String:
    var t = Int(x * 10 + 0.5)
    return String(t // 10) + "." + String(t % 10)


struct Conn(Movable):
    var fd: Int32
    var buf: List[UInt8]
    var pos: Int

    def __init__(out self, fd: Int32):
        self.fd = fd
        self.buf = List[UInt8]()
        self.pos = 0

    def __init__(out self, host: String, port: Int) raises:
        self.fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0))
        if self.fd < 0:
            raise Error("socket() failed")
        var sa = List[UInt8](length=16, fill=0)
        sa[0] = 2
        sa[2] = UInt8(port >> 8)
        sa[3] = UInt8(port & 255)
        var hc = (String("127.0.0.1") if host == "localhost" else host) + "\0"
        if external_call["inet_pton", Int32](Int32(2), hc.unsafe_ptr(), sa.unsafe_ptr().unsafe_offset(4)) != 1:
            raise Error("not an IPv4 address: " + host)
        if external_call["connect", Int32](self.fd, sa.unsafe_ptr(), Int32(16)) != 0:
            _ = external_call["close", Int32](self.fd)
            raise Error("cannot connect to " + host + ":" + String(port) + " (is baro-serve running?)")
        self.buf = List[UInt8]()
        self.pos = 0

    def send(self, s: String) raises:
        var off = 0
        while off < s.byte_length():
            var w = external_call["__write", Int](Int(self.fd), s.unsafe_ptr().unsafe_offset(off).unsafe_bitcast[NoneType](), s.byte_length() - off)
            if w <= 0:
                raise Error("write failed")
            off += w

    def fill(mut self) -> Bool:
        if self.pos > 0:
            var keep = List[UInt8](self.buf[self.pos:])
            self.buf = keep^
            self.pos = 0
        var n0 = len(self.buf)
        self.buf.resize(n0 + 65536, 0)
        var r = external_call["__read", Int](Int(self.fd), self.buf.unsafe_ptr().unsafe_offset(n0).unsafe_bitcast[NoneType](), 65536)
        self.buf.resize(n0 + max(r, 0), 0)
        return r > 0

    def line(mut self) raises -> String:
        var i = self.pos
        while True:
            while i < len(self.buf):
                if self.buf[i] == 10:
                    var e = i - 1 if i > self.pos and self.buf[i - 1] == 13 else i
                    var s = String(unsafe_from_utf8=List[UInt8](self.buf[self.pos:e]))
                    self.pos = i + 1
                    return s
                i += 1
            i -= self.pos
            if not self.fill():
                raise Error("connection closed mid-line")

    def take(mut self, n: Int) raises -> List[UInt8]:
        while len(self.buf) - self.pos < n:
            if not self.fill():
                raise Error("connection closed mid-body")
        var out = List[UInt8](self.buf[self.pos : self.pos + n])
        self.pos += n
        return out^

    def rest(mut self) -> List[UInt8]:
        while self.fill():
            pass
        var out = List[UInt8](self.buf[self.pos:])
        self.pos = len(self.buf)
        return out^

    def close(self):
        _ = external_call["close", Int32](self.fd)


struct Reply(Movable):
    var text: String
    var stats: String
    # The server's finish_reason of the last choice ("" until one arrives) and whether `[DONE]` was seen:
    # a stream that closes without it is a cut reply, not a finished turn.
    var finish: String
    var done: Bool

    def __init__(out self):
        self.text = ""
        self.stats = ""
        self.finish = ""
        self.done = False


def num(doc: JSONDoc, idx: Int) -> Float64:
    return doc.get(idx).n if idx >= 0 and doc.get(idx).is_number() else -1.0



def on_event(data: String, mut rep: Reply) raises -> Bool:
    if data == "[DONE]":
        rep.done = True
        return False
    var doc = parse_json_bytes(List[UInt8](data.as_bytes()))
    var r = doc.root
    var err = doc.get_field(r, "error")
    if err >= 0:
        var m = doc.get_field(err, "message")
        raise Error("server: " + (doc.get(m).s if m >= 0 else data))
    var ch = doc.get_field(r, "choices")
    if ch >= 0 and len(doc.get(ch).arr) > 0:
        var fr = doc.get_field(doc.get(ch).arr[0], "finish_reason")
        if fr >= 0 and doc.get(fr).is_string():
            rep.finish = doc.get(fr).s
        var c = doc.get_field(doc.get_field(doc.get(ch).arr[0], "delta"), "content")
        if c >= 0 and doc.get(c).is_string():
            var t = doc.get(c).s
            print(t, end="", flush=True)
            rep.text += t
    var u = doc.get_field(r, "usage")
    var tm = doc.get_field(r, "timings")
    if u >= 0 and tm >= 0:
        var cached = num(doc, doc.get_field(doc.get_field(u, "baro"), "cached_tokens"))
        rep.stats = "[" + String(Int(num(doc, doc.get_field(u, "completion_tokens")))) + " tok · " + fmt1(num(doc, doc.get_field(tm, "tok_s_gen"))) + " tok/s · prefill " + fmt1(num(doc, doc.get_field(tm, "prefill_s"))) + " s · cached " + String(Int(max(cached, 0.0))) + "]"
    return True


def ask(host: String, port: Int, body: String) raises -> Reply:
    var c = Conn(host, port)
    c.send("POST /v1/chat/completions HTTP/1.1\r\nHost: " + host + "\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: " + String(body.byte_length()) + "\r\nConnection: close\r\n\r\n" + body)
    var status = c.line()
    var chunked = False
    var clen = -1
    while True:
        var h = c.line().lower()
        if h.byte_length() == 0:
            break
        if h.startswith("transfer-encoding:") and "chunked" in h:
            chunked = True
        if h.startswith("content-length:"):
            clen = Int(atol(String(h[byte=15:]).strip()))
    var rep = Reply()
    if " 200 " not in status:
        var raw = c.take(clen) if clen >= 0 else c.rest()
        c.close()
        raise Error(status + ": " + String(unsafe_from_utf8=raw^))
    var pending = List[UInt8]()
    var live = True
    while live:
        if chunked:
            var sz = Int(atol(String(c.line().split(";")[0]).strip(), base=16))
            if sz == 0:
                break
            pending.extend(c.take(sz))
            _ = c.line()
        else:
            pending.extend(c.rest())
            live = False
        var start = 0
        for i in range(len(pending)):
            if pending[i] == 10:
                var e = i - 1 if i > start and pending[i - 1] == 13 else i
                var ln = String(unsafe_from_utf8=List[UInt8](pending[start:e]))
                start = i + 1
                if ln.startswith("data:") and not on_event(String(String(ln[byte=5:]).strip()), rep):
                    live = False
                    break
        var keep = List[UInt8](pending[start:])
        pending = keep^
    c.close()
    if not rep.done:
        raise Error("the stream ended without [DONE]: the reply is partial (" + String(rep.text.byte_length()) + " bytes received) and is not kept")
    return rep^


def main():
    var a = List[String]()
    for x in argv()[1:]:
        a.append(String(x))
    var host = String("127.0.0.1")
    var port = 8080
    var max_tokens = 2048
    var extra = String("")
    var system = String("")
    var once = String("")
    var one = False
    var i = 0
    try:
        while i < len(a):
            var k = a[i]
            if k == "-h" or k == "--help":
                print(USAGE)
                return
            if k == "--no-think":
                extra += ",\"chat_template_kwargs\":{\"enable_thinking\":false}"
                i += 1
                continue
            if i + 1 >= len(a):
                die("option " + k + " needs a value\n" + USAGE)
            var v = a[i + 1]
            if k == "--url":
                var hp = String(v.removeprefix("http://")).split(":")
                host = String(hp[0])
                if len(hp) > 1:
                    port = Int(atol(String(String(hp[1]).split("/")[0])))
            elif k == "--max-tokens":
                max_tokens = Int(atol(v))
            elif k == "--temperature":
                extra += ",\"temperature\":" + String(atof(v))
            elif k == "--top-p":
                extra += ",\"top_p\":" + String(atof(v))
            elif k == "--seed":
                extra += ",\"seed\":" + String(atol(v))
            elif k == "--system":
                system = v
            elif k == "-p":
                once = v
                one = True
            else:
                die("unknown option " + k + "\n" + USAGE)
            i += 2
    except e:
        die(String(e))
    var hist = List[String]()
    var tty = Conn(Int32(0))
    if not one:
        print("baro chat -> " + host + ":" + String(port) + "   (/reset, /system TEXT, /exit)", file=stderr)
    while True:
        var msg: String
        if one:
            msg = once
        else:
            print("\n\x1b[1;34m> \x1b[0m", end="", flush=True)
            try:
                msg = tty.line()
            except:
                print("", file=stderr)
                return
            if msg == "/exit":
                return
            if msg == "/reset":
                hist.clear()
                continue
            if msg.startswith("/system "):
                system = String(msg[byte=8:])
                continue
            if msg.strip().byte_length() == 0:
                continue
        hist.append("{\"role\":\"user\",\"content\":" + json_str(msg) + "}")
        var msgs = String("[")
        if system.byte_length() > 0:
            msgs += "{\"role\":\"system\",\"content\":" + json_str(system) + "},"
        for j in range(len(hist)):
            msgs += hist[j] + ("," if j + 1 < len(hist) else "")
        try:
            var rep = ask(host, port, "{\"messages\":" + msgs + "],\"stream\":true,\"max_tokens\":" + String(max_tokens) + extra + "}")
            print("")
            var info = rep.stats
            if rep.finish.byte_length() > 0:
                info += (" " if info.byte_length() > 0 else "") + "[finish " + rep.finish + ("; cut by max_tokens" if rep.finish == "length" else "") + "]"
            if info.byte_length() > 0:
                print("\x1b[2m" + info + "\x1b[0m", file=stderr)
            if rep.text.byte_length() == 0:
                raise Error("the reply is empty and is not kept")
            hist.append("{\"role\":\"assistant\",\"content\":" + json_str(rep.text) + "}")
        except e:
            _ = hist.pop()
            if one:
                die(String(e))
            print("\nbaro chat: " + String(e), file=stderr)
        if one:
            return
