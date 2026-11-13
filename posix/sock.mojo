# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-posix/src/posix/sock.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from std.ffi import external_call, c_ssize_t
from posix._sys import CLOEXEC, NONBLOCK, EINTR, EAGAIN, errno, error, cstr
from posix.poll import PollSet, POLLOUT

comptime _ADDR = 110  # sizeof(struct sockaddr_un)
comptime _MAX_PATH = 108  # sun_path size; Linux and Python accept a full, unterminated path
comptime _DONTWAIT = 0x40
comptime _NOSIGNAL = 0x4000
comptime _STALLS = 50  # send_all waits 100 ms for room, 5 s in all


def _addr(path: String) raises -> List[UInt8]:
    if path.byte_length() > _MAX_PATH:
        raise Error("posix: unix socket path longer than ", _MAX_PATH, " bytes: ", path)
    var a = List[UInt8](length=_ADDR, fill=0)
    a[0] = 1  # AF_UNIX
    for i in range(path.byte_length()):
        a[2 + i] = path.as_bytes()[i]
    return a^


def unix_listen(path: String) raises -> Int32:
    """A listening stream socket at path: non-blocking, close-on-exec, mode 0600. A stale file at
    path is replaced. Raises `posix: ...` when the path cannot be bound."""
    var c = cstr(path)
    var a = _addr(path)
    _ = external_call["unlink", Int32](c.unsafe_ptr())
    var fd = external_call["socket", Int32](Int32(1), Int32(1 | CLOEXEC | NONBLOCK), Int32(0))
    if fd < 0:
        raise error("socket")
    var rc = external_call["bind", Int32](fd, a.unsafe_ptr(), Int32(_ADDR))
    var what = "bind '" + path + "'"
    if rc == 0:
        rc = external_call["listen", Int32](fd, Int32(128))
        what = "listen"
    if rc == 0:
        rc = external_call["chmod", Int32](c.unsafe_ptr(), Int32(0o600))
        what = "chmod '" + path + "'"
    _ = (a[0], c[0])
    if rc != 0:
        var e = error(what)
        _ = external_call["close", Int32](fd)
        raise e^
    return fd


def unix_connect(path: String) raises -> Int32:
    """A connected stream socket to the listener at path (blocking, close-on-exec). Raises
    `posix: connect ...` with errno 2 (no such file) or 111 (nobody listening)."""
    var a = _addr(path)
    var fd = external_call["socket", Int32](Int32(1), Int32(1 | CLOEXEC), Int32(0))
    if fd < 0:
        raise error("socket")
    var rc = external_call["connect", Int32](fd, a.unsafe_ptr(), Int32(_ADDR))
    _ = a[0]
    if rc != 0:
        var e = error("connect '" + path + "'")
        _ = external_call["close", Int32](fd)
        raise e^
    return fd


def accept(fd: Int32) raises -> Int32:
    """The next waiting connection of a unix_listen() socket (non-blocking, close-on-exec), or -1
    when none is waiting."""
    while True:
        var c = external_call["accept4", Int32](fd, Int(0), Int(0), Int32(CLOEXEC | NONBLOCK))
        if c >= 0:
            return c
        var e = errno()
        if e == EAGAIN:
            return -1
        if e != EINTR:
            raise error("accept")


def recv(fd: Int32, mut buf: List[UInt8], block: Bool = False) -> Int:
    """Append what the socket holds (up to 64 KiB) to buf. Returns the bytes added; 0 at end of
    stream; -1 when nothing is readable yet (block=False) or on an error."""
    var at = len(buf)
    buf.resize(unsafe_uninit_length=at + 65536)
    var n = Int(external_call["recv", c_ssize_t](Int(fd), buf.unsafe_ptr().unsafe_offset(at), 65536, Int32(0 if block else _DONTWAIT)))
    buf.shrink(at + (n if n > 0 else 0))
    return n


def send_all(fd: Int32, data: Span[UInt8, _]) -> Bool:
    """Write all of data. On a non-blocking socket it waits up to 5 s for room. False on an error
    or a closed peer; never raises SIGPIPE."""
    var off = 0
    var stalls = 0
    var ps = PollSet()
    _ = ps.add(fd, POLLOUT)
    while off < len(data):
        var w = Int(external_call["send", c_ssize_t](Int(fd), data.unsafe_ptr().unsafe_offset(off), len(data) - off, Int32(_NOSIGNAL)))
        if w > 0:
            off += w
            stalls = 0
        elif w < 0 and errno() == EINTR:
            continue
        elif w < 0 and errno() == EAGAIN and stalls < _STALLS:
            stalls += 1
            _ = ps.wait(100)
        else:
            return False
    return True


def shutdown_write(fd: Int32):
    """Half-close: the peer reads end of stream after the bytes already sent."""
    _ = external_call["shutdown", Int32](fd, Int32(1))


def close(fd: Int32):
    _ = external_call["close", Int32](fd)
