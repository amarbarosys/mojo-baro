# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-systemd/src/systemd/notify.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""The sd_notify(3) protocol: one datagram per call to the socket named in $NOTIFY_SOCKET."""
from std.ffi import external_call
from std.os import getenv

comptime _ADDR = 110  # sizeof(struct sockaddr_un)


def notify(state: String) raises -> Bool:
    """Sends state (for example "READY=1\\nSTATUS=up") to the service manager. False when
    $NOTIFY_SOCKET is unset (not run by systemd). Raises when the socket is set but unusable."""
    var path = getenv("NOTIFY_SOCKET")
    if path == "":
        return False
    var n = path.byte_length()
    var c0 = path.as_bytes()[0]
    if n >= _ADDR - 2 or (c0 != 47 and c0 != 64):  # "/" or "@"
        raise Error("systemd: bad NOTIFY_SOCKET: ", path)
    var a = List[UInt8](length=_ADDR, fill=0)
    a[0] = 1  # AF_UNIX
    for i in range(n):
        a[2 + i] = path.as_bytes()[i]
    if a[2] == 64:  # '@' = abstract namespace, no trailing NUL in the length
        a[2] = 0
    else:
        n += 1
    var fd = external_call["socket", Int32](Int32(1), Int32(2 | 0o2000000), Int32(0))  # DGRAM, CLOEXEC
    if fd < 0:
        raise Error("systemd: socket failed")
    var sent = external_call["sendto", Int](
        fd, state.unsafe_ptr(), state.byte_length(), Int32(0x4000), a.unsafe_ptr(), Int32(2 + n)
    )
    _ = external_call["close", Int32](fd)
    if sent != state.byte_length():
        raise Error("systemd: send to ", path, " failed")
    return True


def ready(msg: String = "") raises -> Bool:
    """READY=1, plus STATUS=msg when msg is set. For Type=notify units."""
    return notify("READY=1" if msg == "" else "READY=1\nSTATUS=" + msg)


def status(msg: String) raises -> Bool:
    """STATUS=msg, shown by `systemctl status`."""
    return notify("STATUS=" + msg)


def watchdog() raises -> Bool:
    """WATCHDOG=1, the keep-alive ping; send more often than watchdog_usec()."""
    return notify("WATCHDOG=1")


def stopping() raises -> Bool:
    return notify("STOPPING=1")


def reloading() raises -> Bool:
    """RELOADING=1 with the MONOTONIC_USEC systemd 253+ wants; send ready() when done."""
    var ts = List[Int64](length=2, fill=0)
    _ = external_call["clock_gettime", Int32](Int32(1), ts.unsafe_ptr())
    return notify("RELOADING=1\nMONOTONIC_USEC=" + String(Int(ts[0]) * 1_000_000 + Int(ts[1]) // 1000))


def watchdog_usec() -> Int:
    """The watchdog interval systemd asked for, in microseconds; 0 when off or meant for another
    process ($WATCHDOG_PID differs from ours). Ping at about half of it."""
    var us = getenv("WATCHDOG_USEC")
    var pid = getenv("WATCHDOG_PID")
    try:
        if pid != "" and Int(pid) != Int(external_call["getpid", Int32]()):
            return 0
        return Int(us) if us != "" else 0
    except:
        return 0
