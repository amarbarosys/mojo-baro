"""Line reader for fd 0 shared by `nx chat` and the MCP server."""
from std.ffi import external_call


def read_line(mut eof: Bool) -> String:
    """One line from fd 0 with read(2). `input()` gave EOF after the first line of a pipe and /dev/stdin cannot be opened when the
    client hands over a socket (Claude Code does), so read the descriptor itself, byte by byte."""
    var b = List[UInt8]()
    var c = List[UInt8](length=1, fill=0)
    while True:
        var n = external_call["read", Int](Int32(0), c.unsafe_ptr(), 1)
        if n <= 0:
            eof = True
            break
        if c[0] == 10:
            break
        b.append(c[0])
    return String(unsafe_from_utf8=Span(b))
