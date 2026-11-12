"""A minimal HTTP/1.1 GET to a local baro front (health endpoint). No TLS, no redirects, one request per
connection. The window thread never blocks longer than `timeout_ms` per read: a front that hangs shows as
`unreachable`, it does not freeze the frame loop."""
from std.ffi import external_call
from imkit.window import readable


struct Reply(Copyable, Movable):
    var status: Int  # 0: no reply (named in `why`)
    var body: String
    var why: String

    def __init__(out self, status: Int, var body: String, var why: String):
        self.status, self.body, self.why = status, body^, why^


def _ipv4(host: String) raises -> UInt32:
    var parts = ("127.0.0.1" if host == "localhost" else host).split(".")
    var ip = UInt32(0)
    for i in range(len(parts)):
        var v = Int(parts[i])
        if len(parts) != 4 or v < 0 or v > 255:
            raise Error("face: not an IPv4 address: ", host)
        ip |= UInt32(v) << UInt32(8 * i)
    return ip


def tcp_connect(host: String, port: Int) raises -> Int32:
    """A connected blocking socket, or -1 (refused, unreachable)."""
    var fd = external_call["socket", Int32](Int32(2), Int32(1 | 0x80000), Int32(0))  # AF_INET, SOCK_STREAM | SOCK_CLOEXEC
    if fd < 0:
        raise Error("face: socket() failed")
    var addr = SIMD[DType.uint32, 4](UInt32(2) | (UInt32(port & 0xFF) << 24) | (UInt32(port >> 8) << 16), _ipv4(host), 0, 0)
    if external_call["connect", Int32](fd, Pointer(to=addr), Int32(16)) != 0:
        _ = external_call["close", Int32](fd)
        return -1
    return fd


def send_get(fd: Int32, host: String, path: String) -> Bool:
    var req = "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    var n = req.byte_length()
    return Int(external_call["send", Int](fd, req.unsafe_ptr(), UInt(n), Int32(0x4000))) == n  # MSG_NOSIGNAL


def read_reply(fd: Int32, timeout_ms: Int) -> Reply:
    """Read until the peer closes or the timeout passes with nothing new; closes fd."""
    var raw = List[UInt8]()
    var buf = List[UInt8](length=4096, fill=0)
    var why = String("")
    while True:
        if not readable(Int(fd), timeout_ms):
            why = "timeout"
            break
        var n = Int(external_call["recv", Int](fd, buf.unsafe_ptr(), UInt(4096), Int32(0)))
        if n <= 0:
            break
        raw.extend(Span(buf)[:n])
    _ = external_call["close", Int32](fd)
    var s = String(unsafe_from_utf8=Span(raw))
    var sep = s.find("\r\n\r\n")
    if sep < 0 or s.byte_length() < 12 or not s.startswith("HTTP/"):
        return Reply(0, String(""), why if why else String("no HTTP reply"))
    try:
        return Reply(Int(String(s[byte = 9:12])), String(s[byte = sep + 4 :]), String(""))
    except:
        return Reply(0, String(""), String("bad status line"))


def http_get(host: String, port: Int, path: String, timeout_ms: Int = 300) raises -> Reply:
    var fd = tcp_connect(host, port)
    if fd < 0:
        return Reply(0, String(""), String("connection refused"))
    if not send_get(fd, host, path):
        _ = external_call["close", Int32](fd)
        return Reply(0, String(""), String("send failed"))
    return read_reply(fd, timeout_ms)
