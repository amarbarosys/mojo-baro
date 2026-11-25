"""test_net: the helpers the front's socket tests share: a pass/fail tally and a tiny blocking HTTP/1.1 client.

Used by test_http and test_anthropic (build with `-I serve/front`).
"""
from api import _pos
from json import Json, parse
from posix import recv, send_all, close
from std.ffi import external_call


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

