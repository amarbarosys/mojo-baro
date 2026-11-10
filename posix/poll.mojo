# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-posix/src/posix/poll.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from std.ffi import external_call

comptime POLLIN = 1
comptime POLLOUT = 4
comptime POLLERR = 8
comptime POLLHUP = 16
comptime POLLNVAL = 32


struct PollSet(Movable, Sized):
    """A set of descriptors for poll(2), kept as packed `struct pollfd` records
    (fd:i32, events:i16, revents:i16 in one Int64), so wait() hands the kernel the list as is."""

    var fds: List[Int64]

    def __init__(out self):
        self.fds = List[Int64]()

    def add(mut self, fd: Int32, events: Int = POLLIN) -> Int:
        """Watch fd for events (POLLIN, POLLOUT, ...); returns its index for revents()."""
        self.fds.append((Int64(fd) & 0xFFFFFFFF) | (Int64(events) << 32))
        return len(self.fds) - 1

    def clear(mut self):
        self.fds.clear()

    def __len__(self) -> Int:
        return len(self.fds)

    def wait(self, timeout_ms: Int) -> Int:
        """Call poll(2). Returns the number of ready descriptors, 0 on timeout (-1 waits forever, 0 only checks),
        -1 when interrupted by a signal or on error. An empty set just sleeps."""
        return Int(external_call["poll", Int32](self.fds.unsafe_ptr(), UInt64(len(self.fds)), Int32(timeout_ms)))

    def revents(self, i: Int) -> Int:
        """The events that fired on descriptor i at the last wait()."""
        return Int((self.fds[i] >> 48) & 0xFFFF)
