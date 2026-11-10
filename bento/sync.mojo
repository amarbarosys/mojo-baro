# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-bento/src/bento/sync.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""pthread mutex and condition variable (glibc: all-zero bytes are the static initializers).

Both own a small heap block, so a struct holding them can move while threads wait on them."""
from std.ffi import external_call
from std.memory import alloc, Layout


struct Mutex(Movable):
    var p: MutPointer[UInt8, MutUntrackedOrigin]

    def __init__(out self):
        self.p = alloc(Layout[UInt8](count=64)).unsafe_leak()
        for i in range(64):
            self.p.unsafe_offset(i)[] = 0

    def lock(self):
        _ = external_call["pthread_mutex_lock", Int32](self.p)

    def unlock(self):
        _ = external_call["pthread_mutex_unlock", Int32](self.p)

    def __deinit__(deinit self):
        self.p.unsafe_free()


struct Cond(Movable):
    var p: MutPointer[UInt8, MutUntrackedOrigin]

    def __init__(out self):
        self.p = alloc(Layout[UInt8](count=64)).unsafe_leak()
        for i in range(64):
            self.p.unsafe_offset(i)[] = 0

    def wait(self, mu: Mutex):
        """Release `mu`, sleep until a broadcast, relock. Callers loop on their own predicate."""
        _ = external_call["pthread_cond_wait", Int32](self.p, mu.p)

    def timedwait(self, mu: Mutex, seconds: Float64) -> Bool:
        """wait() for at most `seconds`; False when it timed out."""
        var ts = alloc(Layout[Int](count=2)).unsafe_leak()
        _ = external_call["clock_gettime", Int32](Int32(0), ts)  # CLOCK_REALTIME, the clock pthread_cond_timedwait uses
        var ns = ts.unsafe_offset(1)[] + Int(seconds * 1e9)
        ts.unsafe_offset(0)[] += ns // 1_000_000_000
        ts.unsafe_offset(1)[] = ns % 1_000_000_000
        var rc = external_call["pthread_cond_timedwait", Int32](self.p, mu.p, ts)
        ts.unsafe_free()
        return rc == 0

    def broadcast(self):
        _ = external_call["pthread_cond_broadcast", Int32](self.p)

    def __deinit__(deinit self):
        self.p.unsafe_free()
