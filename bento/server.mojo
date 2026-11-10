# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-bento/src/bento/server.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""B1: an HTTP/1.1 server. Every worker thread waits on one epoll set; EPOLLONESHOT hands each ready
connection to exactly one worker, which reads, parses, runs the handler (it may block: a Batcher.submit
parks only that worker) and writes the responses before the connection is re-armed. Pipelined requests
are answered in order, in one write. No queue, no lock: the kernel is the scheduler."""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import alloc
from std.time import perf_counter_ns
from systemd import ready, status, stopping, watchdog as notify_watchdog, watchdog_usec
from posix import PollSet, POLLIN, SIGINT, SIGTERM, close, recv, send_all, signal_fd, read_signals
from bento.http import Request, Response, Limits, Parsed, parse_request, reason
import aio
from aio import IN, ERR, HUP, RDHUP, ONESHOT

comptime _MutConn = Pointer[Conn, MutUntrackedOrigin]
comptime _MutInt = Pointer[Int32, MutUntrackedOrigin]
comptime _MutBytes = Pointer[List[UInt8], MutUntrackedOrigin]
comptime _WAKE_MS = 100  # how often an idle worker looks at the stop flag


trait Handler(Movable, Deinitable):
    """What the server runs per request. Called from many threads at once on one shared value:
    keep it read-only, or lock what it mutates."""

    def handle(self, req: Request, mut resp: Response) raises:
        ...

    def draining(self):
        """Called once when SIGINT/SIGTERM arrives, `drain_ms` before the workers stop: turn /readyz to 503."""
        ...


struct Conn(Movable):
    var inbuf: List[UInt8]
    var pos: Int  # start of the unparsed bytes in inbuf
    var told: Bool  # a 100-continue went out for the request being read
    var open: Bool  # the fd is accepted and not yet closed: serve() closes what is still open on return

    def __init__(out self):
        self.inbuf = List[UInt8]()
        self.pos = 0
        self.told = False
        self.open = False

    def reset(mut self):
        self.inbuf = List[UInt8]()
        self.pos = 0
        self.told = False
        self.open = False


struct Server[H: Handler](Movable):
    var handler: Self.H
    var limits: Limits
    var listen_fd: Int32
    var loop: aio.Loop
    var sig: Int32
    var table: List[Conn]
    var ctl: List[Int32]  # [0] stop flag
    var stop_p: _MutInt
    var conn_p: _MutConn

    def __init__(out self, var handler: Self.H, host: String = "127.0.0.1", port: Int = 0, var limits: Limits = Limits(), max_conns: Int = 16384) raises:
        self.handler = handler^
        self.limits = limits^
        self.listen_fd = aio.tcp_listen(port, host, 1024)
        self.loop = aio.Loop(1)
        self.sig = -1
        self.table = List[Conn]()
        for _ in range(max_conns):
            self.table.append(Conn())
        self.ctl = List[Int32](length=16, fill=0)
        self.stop_p = _MutInt(unsafe_from_address=Int(self.ctl.unsafe_ptr()))
        self.conn_p = _MutConn(unsafe_from_address=Int(self.table.unsafe_ptr()))

    def port(self) -> Int:
        try:
            return aio.local_port(self.listen_fd)
        except:
            return -1

    def stop(self):
        """Ask every worker to finish its request and return. Safe from any thread."""
        Atomic[Int32].store[ordering=Ordering.RELEASE](self.stop_p, 1)

    def _drain(self):
        """A signal arrived: ask the main thread to run the drain, unless a stop is already under way."""
        var running = Int32(0)
        _ = Atomic[Int32].compare_exchange(self.stop_p, running, Int32(2))

    def stopped(self) -> Bool:
        return Atomic[Int32].load[ordering=Ordering.ACQUIRE](self.stop_p) == 1

    def serve(mut self, workers: Int = 4, signals: Bool = False, drain_ms: Int = 0) raises:
        """Run until stop() (or SIGINT/SIGTERM when signals; call posix.block_signals first in main).
        A signal first calls handler.draining(), keeps serving for drain_ms, then stops. Blocks the caller;
        spawns `workers` threads. Open connections are closed on return."""
        self.loop.add(self.listen_fd, IN | ONESHOT, UInt64(self.listen_fd))
        if signals:
            self.sig = signal_fd([SIGINT, SIGTERM])
            self.loop.add(self.sig, IN, UInt64(self.sig))
        var tids = List[UInt64](length=workers, fill=0)
        var me = Int(Pointer(to=self))
        for i in range(workers):
            var rc = external_call["pthread_create", Int32](tids.unsafe_ptr().unsafe_offset(i), Int(0), _worker_main[Self.H], me)
            if rc != 0:
                self.stop()
                raise Error("bento: pthread_create rc ", rc)
        try:
            _ = ready("listening on " + String(self.port()))
        except:
            pass  # a bad NOTIFY_SOCKET must not stop a server that works
        var wd = watchdog_usec() * 1000 // 2  # ping at half the interval systemd asked for
        var last = perf_counter_ns()
        while not self.stopped():
            _ = external_call["usleep", Int32](Int32(20000))
            if Atomic[Int32].load[ordering=Ordering.ACQUIRE](self.stop_p) == 2:
                self.handler.draining()
                try:
                    _ = status("draining")
                except:
                    pass
                _ = external_call["usleep", Int32](Int32(drain_ms * 1000))
                self.stop()
            if wd > 0 and Int(perf_counter_ns() - last) >= wd:
                last = perf_counter_ns()
                try:
                    _ = notify_watchdog()
                except:
                    pass
        try:
            _ = stopping()
        except:
            pass
        for i in range(workers):
            _ = external_call["pthread_join", Int32](tids[i], Int(0))
        for fd in range(len(self.table)):
            if self.table[fd].open:
                close(Int32(fd))
            self.table[fd].reset()
        close(self.listen_fd)
        if self.sig >= 0:
            close(self.sig)

    def _loop(self):
        var evs = List[aio.Event]()
        var buf = List[UInt32](length=3, fill=0)
        while not self.stopped():
            evs.clear()
            try:
                if self.loop.ready(evs, buf, _WAKE_MS) == 0:
                    continue
            except:
                continue
            var fd = Int32(evs[0].token)
            if fd == self.listen_fd:
                self._accept()
            elif fd == self.sig:
                if len(read_signals(fd)) > 0:
                    self._drain()
            else:
                var keep = self._serve(fd, Int(evs[0].events))
                try:
                    if keep:
                        self.loop.modify(fd, IN | RDHUP | ONESHOT, UInt64(fd))
                        continue
                    self.loop.remove(fd)
                except:
                    pass
                self.conn_p.unsafe_offset(Int(fd))[].reset()
                close(fd)

    def _accept(self):
        try:
            for _ in range(128):
                var c = aio.accept(self.listen_fd)
                if c < 0:
                    break
                if Int(c) >= len(self.table):
                    close(c)
                    continue
                self.conn_p.unsafe_offset(Int(c))[].open = True
                self.loop.add(c, IN | RDHUP | ONESHOT, UInt64(c))
        except:
            pass
        try:
            self.loop.modify(self.listen_fd, IN | ONESHOT, UInt64(self.listen_fd))
        except:
            pass

    def _serve(self, fd: Int32, events: Int) -> Bool:
        """Read what arrived on fd, answer every complete request. False: close the connection."""
        ref c = self.conn_p.unsafe_offset(Int(fd))[]
        var n = recv(fd, c.inbuf)
        if n == 0:
            return False
        if n < 0:
            return (events & Int(ERR | HUP)) == 0
        var out = List[UInt8]()
        var keep = True
        var req = Request()
        while keep and c.pos < len(c.inbuf):
            var p = parse_request(Span(c.inbuf)[c.pos :], req, self.limits)
            if p.status != 0:
                var r = Response.text(p.status, reason(p.status) + "\n")
                r.encode(1, False, False, out)
                keep = False
                break
            if p.used == 0:
                if p.cont and not c.told:
                    c.told = True
                    out.extend("HTTP/1.1 100 Continue\r\n\r\n".as_bytes())
                break
            c.pos += p.used
            c.told = False
            var resp = Response()
            keep = req.keep_alive() and not self.stopped()
            resp.attach(fd, req.minor, keep, Int(Pointer(to=out)))
            try:
                self.handler.handle(req, resp)
            except e:
                if resp.state == 0:
                    resp = Response.text(500, String(e) + "\n")
                else:
                    resp.end_stream()  # an error after the head went out can only end the stream
            if resp.state != 0:  # the handler streamed: the answer is already on the wire
                keep = resp.state == 2 and resp.keep
                continue
            resp.encode(req.minor, keep, req.method == "HEAD", out)
        if c.pos == len(c.inbuf):
            c.inbuf.clear()
            c.pos = 0
        elif c.pos > 0:
            c.inbuf = List[UInt8](Span(c.inbuf)[c.pos :])
            c.pos = 0
        if len(out) > 0 and not send_all(fd, out):
            return False
        return keep


def _worker_main[H: Handler](arg: Int) abi("C") -> Int:
    var s = Pointer[Server[H], MutUntrackedOrigin](unsafe_from_address=arg)
    s[]._loop()
    return 0
