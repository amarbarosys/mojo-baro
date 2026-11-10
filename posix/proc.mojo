# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-posix/src/posix/proc.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from std.ffi import external_call, c_ssize_t
from std.os import listdir
from std.os.path import basename, realpath
from std.pathlib import Path
from std.sys import argv
from posix._sys import CLOEXEC, NONBLOCK, EINTR, errno, error, error_code, cstr

comptime SIGHUP = 1
comptime SIGINT = 2
comptime SIGKILL = 9
comptime SIGUSR1 = 10
comptime SIGUSR2 = 12
comptime SIGTERM = 15
comptime SIGCHLD = 17
comptime SIGCONT = 18
comptime SIGSTOP = 19

comptime _WNOHANG = 1
comptime _SIGSET = 128  # sizeof(sigset_t)
comptime _FILE_ACTIONS = 80  # sizeof(posix_spawn_file_actions_t), glibc
comptime _SPAWNATTR = 336  # sizeof(posix_spawnattr_t), glibc
comptime _SETSIGMASK = 0x8
comptime _SETSIGDEF = 0x4
comptime _SETSID = 0x80
comptime _APPEND = 0x441  # O_WRONLY | O_CREAT | O_APPEND
comptime _TRUNC = 0x241  # O_WRONLY | O_CREAT | O_TRUNC


def getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def getppid() -> Int:
    return Int(external_call["getppid", Int32]())


def getpgid(pid: Int) -> Int:
    """The process group of pid (0 is this process); -1 when there is no such process."""
    return Int(external_call["getpgid", Int32](Int32(pid)))


def kill(pid: Int, sig: Int) -> Bool:
    """True when the signal was sent. Signal 0 only tests that the process exists."""
    return external_call["kill", Int32](Int32(pid), Int32(sig)) == 0


def killpg(pgid: Int, sig: Int) -> Bool:
    """Like kill(), for every process in the group."""
    return external_call["killpg", Int32](Int32(pgid), Int32(sig)) == 0


def reap(pid: Int, block: Bool = False) raises -> Optional[Int]:
    """Collect a finished child. Its exit code, or minus the signal number when a signal killed it
    (Popen's returncode); None while it still runs and block is False. Each child is reaped once:
    a second call raises `No child processes` (errno 10)."""
    var st = Int32(0)
    while True:
        var r = external_call["waitpid", Int32](Int32(pid), Pointer(to=st), Int32(0 if block else _WNOHANG))
        if r > 0:
            var s = Int(st)
            return -(s & 0x7F) if (s & 0x7F) != 0 else (s >> 8) & 0xFF
        if r == 0:
            return None
        if errno() != EINTR:
            raise error("waitpid")


def environ() raises -> List[String]:
    """This process's environment as `NAME=value` strings."""
    var env = List[String]()
    for e in Path("/proc/self/environ").read_text().split("\x00"):
        if e.byte_length() > 0:
            env.append(String(e))
    return env^


def _environ() -> Int:
    """libc's `environ`: the live `char **` envp, setenv changes included."""
    var name = cstr("environ")
    var sym = external_call["dlsym", Int](Int(0), name.unsafe_ptr())
    _ = name[0]
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=sym)[] if sym != 0 else 0


struct _CStrings(Movable):
    """A NULL-terminated `char *` array (argv, envp) over one buffer holding all the strings."""

    var data: List[UInt8]
    var ptrs: List[Int]

    def __init__(out self, items: List[String]):
        var n = 0
        for ref s in items:
            n += s.byte_length() + 1
        self.data = List[UInt8](capacity=n)
        self.ptrs = List[Int](capacity=len(items) + 1)
        for ref s in items:
            self.ptrs.append(Int(self.data.unsafe_ptr()) + len(self.data))
            self.data.extend(s.as_bytes())
            self.data.append(0)
        self.ptrs.append(0)


def spawn(
    argv: List[String],
    env: List[String] = [],
    cwd: String = "",
    stdout: String = "",
    stderr: String = "",
    append: Bool = True,
    new_session: Bool = False,
) raises -> Int:
    """Run argv[0] with posix_spawnp (searched on this process's PATH) and return the child's pid.

    env: the child's environment as `NAME=value` strings; empty inherits this process's.
    cwd: its working directory ("" keeps ours). stdout, stderr: files the child writes to, created
    (mode 0666 minus umask) and appended to, or truncated when append is False; "" inherits ours, and
    the same path for both shares one descriptor. new_session: a new session, so pid == pgid == sid and
    killpg(pid, ...) reaches its whole tree. The child starts with no blocked signals and default
    handlers. Raises `posix: spawn ...` when the program cannot start (not found, bad cwd): use
    try_spawn() to get the errno as a number instead."""
    if len(argv) == 0:
        raise Error("posix: spawn: empty argv")
    var r = try_spawn(argv, env, cwd, stdout, stderr, append, new_session)
    if r[0] < 0:
        raise error_code("spawn '" + argv[0] + "'", r[1])
    return r[0]


def try_spawn(
    argv: List[String],
    env: List[String] = [],
    cwd: String = "",
    stdout: String = "",
    stderr: String = "",
    append: Bool = True,
    new_session: Bool = False,
) -> Tuple[Int, Int]:
    """spawn() without the exception: (pid, 0), or (-1, errno) when the program cannot start."""
    return _spawn(argv, env, cwd, stdout, stderr, append, new_session, -1)


def _spawn(
    argv: List[String],
    env: List[String],
    cwd: String,
    stdout: String,
    stderr: String,
    append: Bool,
    new_session: Bool,
    pair_end: Int,
) -> Tuple[Int, Int]:
    """The one posix_spawnp path: pair_end >= 0 is a descriptor the child gets as stdin and stdout."""
    if len(argv) == 0:
        return (-1, 22)  # EINVAL
    var st = List[UInt8](length=_FILE_ACTIONS + _SPAWNATTR + _SIGSET, fill=0)
    var fa = Int(st.unsafe_ptr())  # addresses: pointers into one buffer cannot be passed twice
    var at = fa + _FILE_ACTIONS
    var sg = at + _SPAWNATTR
    var o = cstr(stdout)
    var e = cstr(stderr)
    var d = cstr(cwd)
    var oflag = Int32(_APPEND if append else _TRUNC)
    _ = external_call["posix_spawn_file_actions_init", Int32](fa)
    if pair_end >= 0:
        _ = external_call["posix_spawn_file_actions_adddup2", Int32](fa, Int32(pair_end), Int32(0))
        _ = external_call["posix_spawn_file_actions_adddup2", Int32](fa, Int32(pair_end), Int32(1))
    if stdout.byte_length() > 0:
        _ = external_call["posix_spawn_file_actions_addopen", Int32](fa, Int32(1), o.unsafe_ptr(), oflag, Int32(0o666))
    if stderr.byte_length() > 0 and stderr == stdout:
        _ = external_call["posix_spawn_file_actions_adddup2", Int32](fa, Int32(1), Int32(2))
    elif stderr.byte_length() > 0:
        _ = external_call["posix_spawn_file_actions_addopen", Int32](fa, Int32(2), e.unsafe_ptr(), oflag, Int32(0o666))
    if cwd.byte_length() > 0:
        _ = external_call["posix_spawn_file_actions_addchdir_np", Int32](fa, d.unsafe_ptr())
    _ = external_call["posix_spawnattr_init", Int32](at)
    _ = external_call["sigemptyset", Int32](sg)
    _ = external_call["posix_spawnattr_setsigmask", Int32](at, sg)
    _ = external_call["sigfillset", Int32](sg)
    _ = external_call["posix_spawnattr_setsigdefault", Int32](at, sg)
    _ = external_call["posix_spawnattr_setflags", Int32](at, Int16(_SETSIGMASK | _SETSIGDEF | (_SETSID if new_session else 0)))
    var av = _CStrings(argv)
    var ev = _CStrings(env)
    var envp = Int(ev.ptrs.unsafe_ptr()) if len(env) > 0 else _environ()
    var prog = av.ptrs[0]
    var pid = Int32(0)
    var rc = external_call["posix_spawnp", Int32](Pointer(to=pid), prog, fa, at, Int(av.ptrs.unsafe_ptr()), envp)
    _ = external_call["posix_spawn_file_actions_destroy", Int32](fa)
    _ = external_call["posix_spawnattr_destroy", Int32](at)
    _ = (st[0], o[0], e[0], d[0], av.data[0], ev.ptrs[0])
    return (-1, Int(rc)) if rc != 0 else (Int(pid), 0)


def spawn_pair(argv: List[String], env: List[String] = [], cwd: String = "") raises -> Tuple[Int, Int32]:
    """Run argv[0] like spawn() with its stdin and stdout joined to one end of a unix socketpair, and
    return (pid, fd), fd being our end: non-blocking, close-on-exec, for recv() and send_all() (which
    never raises SIGPIPE) and PollSet. The child's end is blocking and its stderr is ours. The pair
    is half-closable: shutdown_write(fd) gives the child end of file on stdin while its output stays
    readable; end of stream on recv() means the child closed stdout or died, and reap() says which.
    Close fd when done. Raises `posix: spawn_pair ...` on an empty argv or when the program cannot
    start (not found, bad cwd); no descriptor is left open then."""
    if len(argv) == 0:
        raise Error("posix: spawn_pair: empty argv")
    var sv = List[Int32](length=2, fill=-1)
    if external_call["socketpair", Int32](Int32(1), Int32(1 | CLOEXEC), Int32(0), sv.unsafe_ptr()) != 0:
        raise error("socketpair")
    var ours = sv[0]
    var theirs = sv[1]
    _ = external_call["fcntl", Int32, num_fixed_args=2](ours, Int32(4), Int32(NONBLOCK))  # F_SETFL
    var r = _spawn(argv, env, cwd, "", "", True, False, Int(theirs))
    _ = external_call["close", Int32](theirs)
    if r[0] < 0:
        _ = external_call["close", Int32](ours)
        raise error_code("spawn_pair '" + argv[0] + "'", r[1])
    return (r[0], ours)


def exec(argv: List[String]) -> Int:
    """Replace this process with argv[0] (searched on PATH). Returns only on failure, with errno."""
    var av = _CStrings(argv)
    var prog = av.ptrs[0]
    _ = external_call["execvp", Int32](prog, Int(av.ptrs.unsafe_ptr()))
    _ = (av.data[0], av.ptrs[0])
    return errno()


def block_signals(sigs: List[Int] = [SIGINT, SIGTERM, SIGCHLD]) raises:
    """Block the signals in every thread of this process, so that signal_fd() receives them all.

    The Mojo runtime starts one worker thread per CPU before main() runs, each with an empty mask.
    The kernel may hand a process-wide signal to any thread that does not block it: SIGTERM then
    kills the program and SIGCHLD is lost. A mask survives exec, so when a thread has one of the
    signals open this blocks them here and re-runs the program through /proc/self/exe with the same
    arguments; every thread of the new process starts blocked. Raises under `mojo run`, which has no
    binary to re-run: use `mojo build`. Call it first in main(): everything
    before it runs twice. It returns at once when all threads already block the signals."""
    var want = UInt64(0)
    for s in sigs:
        want |= UInt64(1) << UInt64(s - 1)
    var open = False
    for tid in listdir("/proc/self/task"):
        for line in Path("/proc/self/task/" + tid + "/status").read_text().split("\n"):
            if line.startswith("SigBlk:"):
                var mask = UInt64(0)  # 16 hex digits: too wide for atol
                for c in line.split(":")[1].strip().as_bytes():
                    mask = mask * 16 + UInt64(Int(c) - 48 if c <= 57 else Int(c) - 87)
                open = open or (mask & want) != want
    if not open:
        return
    if basename(realpath("/proc/self/exe")) == "mojo":
        raise Error("posix: block_signals needs a built binary: under `mojo run` it would re-run the compiler")
    var set = List[UInt8](length=_SIGSET, fill=0)
    var p = Int(set.unsafe_ptr())
    _ = external_call["sigemptyset", Int32](p)
    for s in sigs:
        _ = external_call["sigaddset", Int32](p, Int32(s))
    _ = external_call["sigprocmask", Int32](Int32(0), p, Int(0))
    _ = set[0]
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var av = _CStrings(args)
    var path = cstr("/proc/self/exe")
    _ = external_call["execv", Int32](path.unsafe_ptr(), Int(av.ptrs.unsafe_ptr()))
    _ = (av.data[0], av.ptrs[0])  # the strings stay alive until the call has returned
    raise error("exec /proc/self/exe")


def signal_fd(sigs: List[Int] = [SIGINT, SIGTERM, SIGCHLD]) raises -> Int32:
    """Block the signals (default SIGINT, SIGTERM, SIGCHLD) in this thread and return a non-blocking,
    close-on-exec descriptor that becomes readable when one arrives; read_signals() drains it. Other
    threads can still take the signals: call block_signals() first in main()."""
    var set = List[UInt8](length=_SIGSET, fill=0)
    var p = Int(set.unsafe_ptr())
    _ = external_call["sigemptyset", Int32](p)
    for s in sigs:
        _ = external_call["sigaddset", Int32](p, Int32(s))
    _ = external_call["sigprocmask", Int32](Int32(0), p, Int(0))
    var fd = external_call["signalfd", Int32](Int32(-1), p, Int32(CLOEXEC | NONBLOCK))
    _ = set[0]
    if fd < 0:
        raise error("signalfd")
    return fd


def read_signals(fd: Int32) -> List[Int]:
    """The signal numbers queued on a signal_fd() descriptor, oldest first; empty when none."""
    var out = List[Int]()
    var rec = List[UInt8](length=128, fill=0)  # struct signalfd_siginfo; ssi_signo leads
    while Int(external_call["__read", c_ssize_t](Int(fd), rec.unsafe_ptr(), 128)) == 128:
        out.append(Int(rec[0]) | (Int(rec[1]) << 8))
    return out^
