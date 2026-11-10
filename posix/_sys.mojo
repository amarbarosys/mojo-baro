# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-posix/src/posix/_sys.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from std.ffi import external_call

comptime CLOEXEC = 0o2000000
comptime NONBLOCK = 0o4000
comptime EINTR = 4
comptime EAGAIN = 11


def errno() -> Int:
    """The calling thread's errno."""
    return Int(external_call["__errno_location", Pointer[Int32, MutAnyOrigin]]()[])


def error_code(what: String, code: Int) -> Error:
    var p = external_call["strerror", Pointer[UInt8, MutAnyOrigin]](Int32(code))
    var msg = StringSlice(unsafe_from_utf8=Span(unsafe_ptr=p, length=external_call["strlen", Int](p)))
    return Error("posix: ", what, ": ", msg, " (errno ", code, ")")


def error(what: String) -> Error:
    """The error `posix: <what>: <strerror> (errno N)` from the current errno."""
    return error_code(what, errno())


def cstr(s: String) -> List[UInt8]:
    """The bytes of s followed by a NUL, for a C `char *`."""
    var out = List[UInt8](capacity=s.byte_length() + 1)
    out.extend(s.as_bytes())
    out.append(0)
    return out^
