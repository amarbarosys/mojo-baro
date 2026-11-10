# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-aio/src/aio/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""epoll event loop: readiness events, one-shot timers, non-blocking TCP. Callback-free: wait()
fills a caller-owned list of (token, events), the caller dispatches."""
from std.ffi import external_call, c_ssize_t
from std.time import perf_counter_ns

comptime IN: UInt32 = 0x001
comptime OUT: UInt32 = 0x004
comptime ERR: UInt32 = 0x008
comptime HUP: UInt32 = 0x010
comptime RDHUP: UInt32 = 0x2000
comptime ONESHOT: UInt32 = 1 << 30
comptime ET: UInt32 = 1 << 31
comptime TIMER: UInt32 = 1 << 29  # set on events produced by after()
comptime _NB = 0o4000
comptime _CLOEXEC = 0o2000000
comptime _EINTR = 4
comptime _EAGAIN = 11


def _errno() -> Int:
    return Int(external_call["__errno_location", Pointer[Int32, MutAnyOrigin]]()[])


def _fail(what: StaticString) -> Error:
    return Error("aio: ", what, " failed (errno ", _errno(), ")")


@fieldwise_init
struct Event(TrivialRegisterPassable, Writable):
    var token: UInt64
    var events: UInt32


@fieldwise_init
struct _Timer(TrivialRegisterPassable):
    var at: Int
    var token: UInt64


def _ready(epfd: Int32, mut out: List[Event], mut buf: List[UInt32], timeout_ms: Int) raises -> Int:
    var n = Int(external_call["epoll_wait", Int32](epfd, buf.unsafe_ptr(), Int32(len(buf) // 3), Int32(timeout_ms)))
    if n < 0 and _errno() != _EINTR:
        raise _fail("epoll_wait")
    for i in range(max(n, 0)):
        out.append(Event(UInt64(buf[3 * i + 1]) | (UInt64(buf[3 * i + 2]) << 32), buf[3 * i]))
    return max(n, 0)


struct Loop(Movable):
    var epfd: Int32
    var buf: List[UInt32]  # packed struct epoll_event: events, data lo, data hi
    var heap: List[_Timer]  # min-heap on deadline (ns, monotonic)

    def __init__(out self, max_events: Int = 1024) raises:
        self.epfd = external_call["epoll_create1", Int32](Int32(_CLOEXEC))
        if self.epfd < 0:
            raise _fail("epoll_create1")
        self.buf = List[UInt32](length=3 * max_events, fill=0)
        self.heap = List[_Timer]()

    def __deinit__(deinit self):
        _ = external_call["close", Int32](self.epfd)

    def _ctl(self, op: Int32, fd: Int32, events: UInt32, token: UInt64) raises:
        var ev = SIMD[DType.uint32, 4](events, UInt32(token & 0xFFFFFFFF), UInt32(token >> 32), 0)
        if external_call["epoll_ctl", Int32](self.epfd, op, fd, Pointer(to=ev)) != 0:
            raise _fail("epoll_ctl")

    def add(self, fd: Int32, events: UInt32, token: UInt64) raises:
        self._ctl(1, fd, events, token)

    def modify(self, fd: Int32, events: UInt32, token: UInt64) raises:
        self._ctl(3, fd, events, token)

    def remove(self, fd: Int32) raises:
        self._ctl(2, fd, 0, 0)

    def after(mut self, ms: Int, token: UInt64):
        """One TIMER event carrying token, no earlier than ms from now."""
        self.heap.append(_Timer(Int(perf_counter_ns()) + ms * 1_000_000, token))
        var i = len(self.heap) - 1
        while i > 0 and self.heap[(i - 1) // 2].at > self.heap[i].at:
            self.heap.swap_elements(i, (i - 1) // 2)
            i = (i - 1) // 2

    def _pop(mut self) -> _Timer:
        var top = self.heap[0]
        var last = self.heap.pop()
        var n = len(self.heap)
        if n > 0:
            self.heap[0] = last
            var i = 0
            while True:
                var m = i
                for c in range(2 * i + 1, min(2 * i + 3, n)):
                    if self.heap[c].at < self.heap[m].at:
                        m = c
                if m == i:
                    break
                self.heap.swap_elements(i, m)
                i = m
        return top

    def ready(self, mut out: List[Event], mut buf: List[UInt32], timeout_ms: Int) raises -> Int:
        """Append ready descriptors to out, no timers. Safe from many threads at once when each passes
        its own buf (3 UInt32 per event slot); pair with ONESHOT so one thread gets each event."""
        return _ready(self.epfd, out, buf, timeout_ms)

    def wait(mut self, mut out: List[Event], timeout_ms: Int = -1) raises -> Int:
        """Block until a descriptor is ready, a timer is due, or timeout_ms passes (-1: no limit).
        Replaces out's contents with what fired; returns len(out). A signal yields 0 events."""
        out.clear()
        var t = timeout_ms
        if len(self.heap) > 0:
            var due = max(0, (self.heap[0].at - Int(perf_counter_ns()) + 999_999) // 1_000_000)
            t = due if t < 0 else min(t, due)
        _ = _ready(self.epfd, out, self.buf, t)
        var now = Int(perf_counter_ns())
        while len(self.heap) > 0 and self.heap[0].at <= now:
            out.append(Event(self._pop().token, TIMER))
        return len(out)


def _ipv4(host: String) raises -> UInt32:
    """Network-order IPv4 address: a dotted quad, "localhost", or "" / "*" for every interface."""
    if host == "" or host == "*":
        return 0
    var parts = ("127.0.0.1" if host == "localhost" else host).split(".")
    var ip = UInt32(0)
    for i in range(len(parts)):
        var v = Int(parts[i])
        if len(parts) != 4 or v < 0 or v > 255:
            raise Error("aio: not an IPv4 address: ", host)
        ip |= UInt32(v) << UInt32(8 * i)
    return ip


def tcp_listen(port: Int, host: String = "127.0.0.1", backlog: Int = 4096) raises -> Int32:
    """A non-blocking listening TCP socket on host:port (port 0 picks a free one, see local_port)."""
    var fd = external_call["socket", Int32](Int32(2), Int32(1 | _NB | _CLOEXEC), Int32(0))
    if fd < 0:
        raise _fail("socket")
    var one = Int32(1)
    _ = external_call["setsockopt", Int32](fd, Int32(1), Int32(2), Pointer(to=one), Int32(4))  # SO_REUSEADDR
    # sockaddr_in: AF_INET, port big-endian, 127.0.0.1, zero padding
    var addr = SIMD[DType.uint32, 4](UInt32(2) | (UInt32(port & 0xFF) << 24) | (UInt32(port >> 8) << 16), _ipv4(host), 0, 0)
    if external_call["bind", Int32](fd, Pointer(to=addr), Int32(16)) != 0 or external_call["listen", Int32](fd, Int32(backlog)) != 0:
        var e = _fail("bind/listen")
        close(fd)
        raise e^
    return fd


def local_port(fd: Int32) raises -> Int:
    var addr = SIMD[DType.uint32, 4](0)
    var len = Int32(16)
    if external_call["getsockname", Int32](fd, Pointer(to=addr), Pointer(to=len)) != 0:
        raise _fail("getsockname")
    return Int(((addr[0] >> 24) & 0xFF) | ((addr[0] >> 8) & 0xFF00))


def accept(fd: Int32) raises -> Int32:
    """The next pending connection (non-blocking, TCP_NODELAY); -1 when none is waiting, or when it
    was aborted or the process is out of descriptors (keep serving, retry on the next event)."""
    while True:
        var c = external_call["accept4", Int32](fd, Int(0), Int(0), Int32(_NB | _CLOEXEC))
        if c >= 0:
            var one = Int32(1)
            _ = external_call["setsockopt", Int32](c, Int32(6), Int32(1), Pointer(to=one), Int32(4))
            return c
        var e = _errno()
        if e == _EAGAIN or e == 103 or e == 24 or e == 23:  # ECONNABORTED, EMFILE, ENFILE
            return -1
        if e != _EINTR:
            raise _fail("accept4")


def read(fd: Int32, mut buf: List[UInt8]) -> Int:
    """Fill buf (its full length) from fd. Bytes read; 0 at end of stream; -1 not ready; -2 error."""
    while True:
        var n = Int(external_call["read", c_ssize_t](fd, buf.unsafe_ptr(), len(buf)))
        if n >= 0:
            return n
        var e = _errno()
        if e != _EINTR:
            return -1 if e == _EAGAIN else -2


def write(fd: Int32, data: Span[UInt8, _]) -> Int:
    """Write what fits now, no SIGPIPE. Bytes written; -1 not ready; -2 error (peer gone)."""
    while True:
        var n = Int(external_call["send", c_ssize_t](fd, data.unsafe_ptr(), len(data), Int32(0x4000)))
        if n >= 0:
            return n
        var e = _errno()
        if e != _EINTR:
            return -1 if e == _EAGAIN else -2


def close(fd: Int32):
    _ = external_call["close", Int32](fd)
