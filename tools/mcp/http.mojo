"""HTTP/1.1 client for the MCP tools: GET and POST to a local baro front. Connection and reply parsing come from
face/net.mojo (tcp_connect, send_get, read_reply, Reply); this file adds the POST sender and the GET wrapper.
One request per connection, Connection: close. A front that does not answer gives status 0 and `why`."""
from std.ffi import external_call
from net import tcp_connect, read_reply, Reply, http_get  # http_get: face/net.mojo, the GET the front already has


def _send(fd: Int32, req: String) -> Bool:
    var n = req.byte_length()
    return Int(external_call["send", Int](fd, req.unsafe_ptr(), UInt(n), Int32(0x4000))) == n  # MSG_NOSIGNAL


def http_post(host: String, port: Int, path: String, body: String, timeout_ms: Int) raises -> Reply:
    var fd = tcp_connect(host, port)
    if fd < 0:
        return Reply(0, String(""), String("connection refused"))
    var req = (
        "POST " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\nContent-Type: application/json\r\n"
        + "Content-Length: " + String(body.byte_length()) + "\r\n\r\n" + body
    )
    if not _send(fd, req):
        _ = external_call["close", Int32](fd)
        return Reply(0, String(""), String("send failed"))
    return read_reply(fd, timeout_ms)
