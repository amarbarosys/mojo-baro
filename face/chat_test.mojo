"""Tests chat.mojo against an in-process listener that speaks the front SSE shape: events cut at the worst byte (inside a
blank-line pair, inside a UTF-8 code point), keep-alives, an error event, an HTTP error status, a peer that closes with
no [DONE], a refused connection, Stop closing the socket. No engine, no GPU, no window.
usage: mojo build -I . -I face face/chat_test.mojo"""
from std.ffi import external_call
from aio import tcp_listen, local_port, accept
from chat import Chat, json_escape, sse_decode, sse_split


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL chat_test: " + what)


def chunk(text: String) -> String:
    return 'data: {"choices":[{"index":0,"delta":{"content":"' + text + '"},"finish_reason":null}]}\n\n'


def accept_one(lfd: Int32) raises -> Int32:
    for _ in range(400):
        var c = accept(lfd)
        if c >= 0:
            return c
        _ = external_call["usleep", Int32](Int32(5000))
    raise Error("FAIL chat_test: no client reached the listener")


def serve_bytes(c: Int32, data: String):
    """Send `out` one byte at a time (a pause every 7 bytes), then close: every event is cut at every possible byte."""
    var bytes = data.as_bytes()
    for i in range(len(bytes)):
        _ = external_call["send", Int](c, bytes.unsafe_ptr().unsafe_offset(i), UInt(1), Int32(0x4000))
        if i % 7 == 0:
            _ = external_call["usleep", Int32](Int32(300))
    _ = external_call["close", Int32](c)


def run_case(reply: String, expect_text: String, expect_note: String) raises:
    var lfd = tcp_listen(0)
    var chat = Chat("127.0.0.1", Int(local_port(lfd)))
    chat.input = 'say "hi"'
    chat.submit()
    check(chat.busy(), "no stream opened: " + chat.note)
    check(len(chat.msgs) == 2 and chat.input == "", "submit did not move the input into the list")
    var c = accept_one(lfd)
    var buf = List[UInt8](length=4096, fill=0)
    var n = Int(external_call["recv", Int](c, buf.unsafe_ptr(), UInt(4096), Int32(0)))
    var req = String(unsafe_from_utf8=Span(buf)[:n])
    check(req.startswith("POST /v1/chat/completions HTTP/1.0"), "request line: " + req)
    check(req.find('"stream":true') >= 0 and req.find('"max_tokens":16000') >= 0, "request lacks stream or the 16000 floor")
    check(req.find('"content":"say \\"hi\\""') >= 0, "user text not JSON-escaped: " + req)
    serve_bytes(c, reply)  # one thread: the kernel buffers it all, then the client drains
    for _ in range(3000):
        chat.pump()
        if not chat.busy():
            break
        _ = external_call["usleep", Int32](Int32(1000))
    check(not chat.busy(), "the stream did not end")
    check(chat.msgs[1].text == expect_text, "answer '" + chat.msgs[1].text + "' != '" + expect_text + "'")
    check(chat.note.startswith(expect_note), "note '" + chat.note + "' does not start with '" + expect_note + "'")
    _ = external_call["close", Int32](lfd)


def main() raises:
    check(json_escape('a"b\\c\nd') == 'a\\"b\\\\c\\nd', "json_escape")
    var p = List[UInt8]()
    p.extend("data: x\n\ndata: y\n".as_bytes())
    var ev = sse_split(p)
    check(len(ev) == 1 and ev[0] == "data: x" and len(p) == 8, "sse_split keeps the partial tail")
    check(sse_decode(": keep-alive").kind == 0, "keep-alive is nothing")
    check(sse_decode("data: [DONE]").kind == 2, "[DONE]")
    check(sse_decode(chunk("hé")).text == "hé", "utf-8 delta")
    check(sse_decode('data: {"error":{"message":"boom","type":"engine_error"}}').kind == 3, "error event")
    var ok = String("HTTP/1.0 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
    run_case(ok + chunk("Hel") + ": keep-alive\n\n" + chunk("lo é world") + "data: [DONE]\n\n", "Hello é world", "done, ")
    run_case(ok + chunk("par") + 'data: {"error":{"message":"engine: kv pool full","type":"engine_error"}}\n\n' + "data: [DONE]\n\n", "par", "front: engine: kv pool full")
    run_case(String("HTTP/1.0 503 Service Unavailable\r\n\r\nengine starting"), "", "front: HTTP 503 engine starting")
    run_case(ok + chunk("cut"), "cut", "front closed the stream without [DONE]")
    run_case(String("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n"), "", "front sent a chunked body")
    # refused connection: a note, not a crash
    var lfd = tcp_listen(0)
    var dead = Int(local_port(lfd))
    _ = external_call["close", Int32](lfd)
    var chat = Chat("127.0.0.1", dead)
    chat.input = "hi"
    chat.submit()
    check(not chat.busy() and chat.note.startswith("front unreachable"), "refused: " + chat.note)
    # Stop closes the socket (the cancel): the peer reads EOF after the request
    var l2 = tcp_listen(0)
    var c2 = Chat("127.0.0.1", Int(local_port(l2)))
    c2.input = "hi"
    c2.submit()
    var srv = accept_one(l2)
    check(c2.busy(), "stop test: no stream")
    c2.stop()
    var buf = List[UInt8](length=4096, fill=0)
    _ = external_call["recv", Int](srv, buf.unsafe_ptr(), UInt(4096), Int32(0))  # the request
    var n = Int(external_call["recv", Int](srv, buf.unsafe_ptr(), UInt(4096), Int32(0)))
    check(n == 0 and not c2.busy() and c2.note == "stopped", "stop did not close the stream (recv " + String(n) + ")")
    print("PASS chat_test: sse parse, 5 stream shapes, refused, stop closes the socket")
