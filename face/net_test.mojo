"""Tests net.mojo against an in-process listener: the canned 200 body is read back, a dead port reads as refused, a
silent peer times out. No engine, no GPU, no window.  usage: mojo build -I . -I face face/net_test.mojo"""
from std.ffi import external_call
from aio import tcp_listen, local_port, accept
from net import http_get, tcp_connect, send_get, read_reply


def serve_once(lfd: Int32, status_line: String, body: String) raises:
    var c = Int32(-1)
    for _ in range(200):
        c = accept(lfd)
        if c >= 0:
            break
        _ = external_call["usleep", Int32](Int32(5000))
    if c < 0:
        raise Error("FAIL net_test: no client reached the listener")
    var buf = List[UInt8](length=1024, fill=0)
    _ = external_call["recv", Int](c, buf.unsafe_ptr(), UInt(1024), Int32(0))
    var out = status_line + "\r\nContent-Length: " + String(body.byte_length()) + "\r\nConnection: close\r\n\r\n" + body
    _ = external_call["send", Int](c, out.unsafe_ptr(), UInt(out.byte_length()), Int32(0))
    _ = external_call["close", Int32](c)


def main() raises:
    var lfd = tcp_listen(0)
    var port = local_port(lfd)
    # connect + send first (the kernel queues them on the backlog), then answer, then read
    var fd = tcp_connect("127.0.0.1", port)
    if fd < 0 or not send_get(fd, "127.0.0.1", "/health"):
        raise Error("FAIL net_test: connect or send")
    serve_once(lfd, "HTTP/1.1 200 OK", '{"status":"ok","engine_state":"idle"}')
    var r = read_reply(fd, 500)
    if r.status != 200 or r.body != '{"status":"ok","engine_state":"idle"}':
        raise Error("FAIL net_test: 200 reply read back as " + String(r.status) + " " + r.body)
    fd = tcp_connect("127.0.0.1", port)
    _ = send_get(fd, "127.0.0.1", "/health")
    serve_once(lfd, "HTTP/1.1 503 Service Unavailable", '{"status":"engine_dead"}')
    r = read_reply(fd, 500)
    if r.status != 503 or r.body != '{"status":"engine_dead"}':
        raise Error("FAIL net_test: 503 reply read back as " + String(r.status))
    # silent peer: connect, send, nobody answers -> timeout, status 0
    fd = tcp_connect("127.0.0.1", port)
    _ = send_get(fd, "127.0.0.1", "/health")
    r = read_reply(fd, 100)
    if r.status != 0 or r.why != "timeout":
        raise Error("FAIL net_test: silent peer gave status " + String(r.status) + " why " + r.why)
    # nothing listening: refused
    var dead = http_get("127.0.0.1", 1, "/health")
    if dead.status != 0 or dead.why != "connection refused":
        raise Error("FAIL net_test: dead port gave " + dead.why)
    print("PASS net_test: 200, 503, timeout, refused")
