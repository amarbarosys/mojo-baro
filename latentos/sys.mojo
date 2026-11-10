# CANONICAL. LatentOS lives here (mojo-baro latentos/); AMDHQ src/latentos is a
# pointer to this directory (plan 12 M1, 2026-09-24). Tests and the check run
# from AMDHQ tools/latent-os/latentos-check against this copy.
#
# sys.mojo — Low-level Linux / libc syscall surface for LatentOS in pure Mojo 1.0.
# All operations use modern Pointer, unsafe_offset, and Layout-free allocations with zero warnings.

from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call

# --- Socket & IPC constants
comptime AF_UNIX = 1
comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime SOCK_DGRAM = 2
comptime SOL_SOCKET = 1
comptime SCM_RIGHTS = 1
comptime SO_REUSEADDR = 2

# --- memfd and sealing constants
comptime MFD_CLOEXEC = 1
comptime MFD_ALLOW_SEALING = 2
comptime F_ADD_SEALS = 1033
comptime F_GET_SEALS = 1034
comptime F_SEAL_SEAL = 1
comptime F_SEAL_SHRINK = 2
comptime F_SEAL_GROW = 4
comptime F_SEAL_WRITE = 8

# --- mmap / memory protection constants
comptime PROT_READ = 1
comptime PROT_WRITE = 2
comptime MAP_SHARED = 1
comptime MADV_HUGEPAGE = 14
comptime MLOCK_ONFAULT = 1

# --- epoll and signals
comptime SIGTERM = 15

# --- Clock and architecture constants
comptime CLOCK_MONOTONIC = 1

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

# --- Memory and pointer helpers
def cstr(s: String) -> BytePtr:
    """Allocates a NUL-terminated C-compatible string on the heap."""
    var n = s.byte_length()
    var p = unsafe_alloc[UInt8](n + 1)
    var src = s.unsafe_ptr()
    for i in range(n):
        p[unsafe_offset=i] = src[unsafe_offset=i]
    p[unsafe_offset=n] = 0
    return p

def cstr_free(p: BytePtr):
    """Frees a heap-allocated C string."""
    p.unsafe_free()

def errno_now() -> Int32:
    """Returns current thread errno."""
    return external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]

def put_i64(p: BytePtr, off: Int, v: Int64):
    p.unsafe_offset(off).unsafe_bitcast[Int64]()[] = v

def get_u64(p: BytePtr, off: Int) -> UInt64:
    return p.unsafe_offset(off).unsafe_bitcast[UInt64]()[]

def get_u32(p: BytePtr, off: Int) -> UInt32:
    return p.unsafe_offset(off).unsafe_bitcast[UInt32]()[]

# --- Low-level Syscalls
def sys_memfd_create(name: String, flags: Int32 = MFD_CLOEXEC | MFD_ALLOW_SEALING) -> Int32:
    var p_name = cstr(name)
    var fd = external_call["memfd_create", Int32](p_name, flags)
    p_name.unsafe_free()
    return fd

def sys_ftruncate(fd: Int32, length: Int64) -> Int32:
    return external_call["ftruncate", Int32](fd, length)

def sys_fcntl_add_seals(fd: Int32, seals: Int32) -> Int32:
    return external_call["fcntl", Int32](fd, Int32(F_ADD_SEALS), seals)

def sys_fcntl_get_seals(fd: Int32) -> Int32:
    return external_call["fcntl", Int32](fd, Int32(F_GET_SEALS), Int32(0))

def sys_mmap(length: Int, prot: Int32, flags: Int32, fd: Int32, offset: Int64 = 0) -> BytePtr:
    return external_call["mmap", BytePtr](0, length, prot, flags, fd, offset)

def sys_munmap(addr: BytePtr, length: Int) -> Int32:
    return external_call["munmap", Int32](addr, length)

def sys_madvise(addr: BytePtr, length: Int, advice: Int32) -> Int32:
    return external_call["madvise", Int32](addr, length, advice)

def sys_mlock2(addr: BytePtr, length: Int, flags: Int32 = MLOCK_ONFAULT) -> Int32:
    return external_call["mlock2", Int32](addr, length, flags)

def sys_munlock(addr: BytePtr, length: Int) -> Int32:
    return external_call["munlock", Int32](addr, length)

def sys_mincore(addr: BytePtr, length: Int, vec: BytePtr) -> Int32:
    return external_call["mincore", Int32](addr, length, vec)

def sys_close(fd: Int32) -> Int32:
    return external_call["close", Int32](fd)

def sys_clock_monotonic_ns() -> Int64:
    var ts_arr = Array[Int64, 2](fill=0)
    var ts = ts_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var rc = external_call["clock_gettime", Int32](Int32(CLOCK_MONOTONIC), ts)
    var ns: Int64 = -1
    if rc == 0:
        ns = ts[unsafe_offset=0] * 1000000000 + ts[unsafe_offset=1]
    _ = ts_arr[0]
    return ns

def sys_clock_monotonic_us() -> Int64:
    return sys_clock_monotonic_ns() // 1000

def sys_clock_monotonic_s() -> Int64:
    return sys_clock_monotonic_ns() // 1000000000

def sys_posix_spawn(path: String, argv0: String, arg1: String = "") -> Int32:
    """Spawns an executable with 0 or 1 arguments and returns the pid, or -1 on error."""
    var p_path = cstr(path)
    var p_arg0 = cstr(argv0)
    var has_arg1 = arg1.byte_length() > 0
    var p_arg1 = cstr(arg1) if has_arg1 else p_arg0

    var argv_arr = Array[Int64, 4](fill=0)
    var argv = argv_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    argv[unsafe_offset=0] = Int64(Int(p_arg0))
    if has_arg1:
        argv[unsafe_offset=1] = Int64(Int(p_arg1))
        argv[unsafe_offset=2] = 0
    else:
        argv[unsafe_offset=1] = 0

    var envp_arr = Array[Int64, 1](fill=0)
    var envp = envp_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    envp[unsafe_offset=0] = 0

    var pid_arr = Array[Int32, 1](fill=-1)
    var pid_buf = pid_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    pid_buf[unsafe_offset=0] = -1

    var rc = external_call["posix_spawn", Int32](pid_buf, p_path, 0, 0, argv, envp)
    var res_pid: Int32 = -1
    if rc == 0:
        res_pid = pid_buf[unsafe_offset=0]
    _ = argv_arr[0]
    _ = envp_arr[0]
    _ = pid_arr[0]

    p_path.unsafe_free()
    p_arg0.unsafe_free()
    if has_arg1:
        p_arg1.unsafe_free()
    return res_pid

def sys_waitpid(pid: Int32, nohang: Bool = False) -> Tuple[Int32, Int32]:
    """Waits for child pid. Returns (waitpid_result, status)."""
    var status_arr = Array[Int32, 1](fill=-1)
    var status = status_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    status[unsafe_offset=0] = -1
    var flags: Int32 = 1 if nohang else 0 # WNOHANG = 1
    var w = external_call["waitpid", Int32](pid, status, flags)
    var res_status = status[unsafe_offset=0]
    _ = status_arr[0]
    return (w, res_status)

def sys_kill(pid: Int32, sig: Int32 = SIGTERM) -> Int32:
    return external_call["kill", Int32](pid, sig)

def sys_fork() -> Int32:
    """Forks current process. Returns 0 in child, child pid in parent, or -1 on error."""
    return external_call["fork", Int32]()

def sys_socket(domain: Int32, type: Int32, protocol: Int32 = 0) -> Int32:
    return external_call["socket", Int32](domain, type, protocol)

def sys_bind_unix(fd: Int32, path: String) -> Int32:
    var sun_arr = Array[UInt8, 110](fill=0)
    var sun = sun_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    sun.unsafe_offset(0).unsafe_bitcast[UInt16]()[] = UInt16(AF_UNIX)
    var n = path.byte_length()
    var src = path.unsafe_ptr()
    var max_len = 107
    if n < max_len:
        max_len = n
    for i in range(max_len):
        sun[unsafe_offset=2 + i] = src[unsafe_offset=i]
    sun[unsafe_offset=2 + max_len] = 0
    return external_call["bind", Int32](fd, sun, 110)

def sys_listen(fd: Int32, backlog: Int32 = 16) -> Int32:
    return external_call["listen", Int32](fd, backlog)

def sys_accept(fd: Int32) -> Int32:
    return external_call["accept", Int32](fd, 0, 0)

def sys_connect_unix(fd: Int32, path: String) -> Int32:
    var sun_arr = Array[UInt8, 110](fill=0)
    var sun = sun_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    sun.unsafe_offset(0).unsafe_bitcast[UInt16]()[] = UInt16(AF_UNIX)
    var n = path.byte_length()
    var src = path.unsafe_ptr()
    var max_len = 107
    if n < max_len:
        max_len = n
    for i in range(max_len):
        sun[unsafe_offset=2 + i] = src[unsafe_offset=i]
    sun[unsafe_offset=2 + max_len] = 0
    return external_call["connect", Int32](fd, sun, 110)

def sys_mkdir(path: String, mode: Int = 0o755) -> Int32:
    """mkdir(2). Returns 0 on success, or a negative-signalling -1 with errno
    set on failure (including EEXIST, which callers should tolerate: the S1
    mountpoints are idempotent-by-design, created fresh each boot). `mode`
    is `Int` (platform word), not `Int32`: the Mojo stdlib's own std.os
    module already declares external_call["mkdir", ...] with a native-`Int`
    mode argument, and a second declaration with any other type for that
    slot hits the same symbol-dedup collision sys_mount's docstring
    documents -- confirmed by trying Int32 first and hitting it as soon as
    a caller also imports anything from std.os (gpt.mojo does)."""
    var p = cstr(path)
    var rc = external_call["mkdir", Int32](p, mode)
    cstr_free(p)
    return rc

def sys_mount(source: String, target: String, fstype: String, flags: Int64 = 0, data: String = "") -> Int32:
    """mount(2): int mount(source, target, filesystemtype, mountflags, data).
    S1 ("act", 05-boot-path-mojo.md sec.4) is the first LatentOS code to
    issue a real mount syscall -- everything in gpt.mojo/S0 is read-only.
    Goes through `latentos_mount`, a tiny C shim
    (tools/latent-os/mount_shim.c, statically linked), not a direct
    `external_call["mount", ...]` or `external_call["syscall", ...]`:
    both those symbol names are already declared by the Mojo stdlib's own
    FFI layer with a conflicting signature, and external_call dedups by
    symbol name across the whole compilation (same class of collision
    sys_read's comment above documents for "read") -- confirmed by trying
    both and hitting "existing function with conflicting signature" both
    times before adding the shim. `data` empty passes a pointer to an
    empty C string rather than a true NULL: two call sites for the same
    external_call symbol with different argument *types* for that slot
    (a literal 0 vs. a BytePtr) hit the identical "conflicting signature"
    error against each other, so every call site must agree on the type;
    an empty string is the kernel-equivalent no-options value for every
    filesystem's mount(2) `data` argument that parses it as a string
    (hugetlbfs included)."""
    var p_source = cstr(source)
    var p_target = cstr(target)
    var p_fstype = cstr(fstype)
    var p_data = cstr(data)
    var rc = external_call["latentos_mount", Int32](p_source, p_target, p_fstype, flags, p_data)
    cstr_free(p_source)
    cstr_free(p_target)
    cstr_free(p_fstype)
    cstr_free(p_data)
    return rc

def sys_read(fd: Int32, buf: BytePtr, count: Int) -> Int64:
    # "recv" rather than libc "read": the stdlib already declares an external_call
    # signature for "read" with different pointer typing, and mojo dedups external_call
    # declarations by symbol name, so reusing "read" here fails to legalize. flags=0
    # makes recv() on a connected TCP socket behave exactly like read().
    return external_call["recv", Int64](fd, buf, count, Int32(0))

def sys_write(fd: Int32, buf: BytePtr, count: Int) -> Int64:
    return external_call["send", Int64](fd, buf, count, Int32(0))

def sys_usleep(microseconds: Int32) -> Int32:
    return external_call["usleep", Int32](microseconds)

def sys_setsockopt_reuseaddr(fd: Int32) -> Int32:
    var opt_arr = Array[Int32, 1](fill=0)
    var opt = opt_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    opt[unsafe_offset=0] = 1
    var rc = external_call["setsockopt", Int32](fd, Int32(SOL_SOCKET), Int32(SO_REUSEADDR), opt, Int32(4))
    _ = opt_arr[0]
    return rc

def parse_ipv4_octets(ip: String) -> Array[UInt8, 4]:
    """Parses a dotted-quad IPv4 string ('10.99.0.2') into its 4 bytes, MSB (first dotted
    component) first -- that ordering is already network byte order, so callers write these
    bytes directly into a sockaddr_in with no further swap."""
    var out = Array[UInt8, 4](fill=0)
    var b = ip.as_bytes()
    var idx = 0
    var cur = 0
    for i in range(len(b)):
        var c = b[i]
        if c == 46: # '.'
            out[idx] = UInt8(cur)
            idx += 1
            cur = 0
        else:
            cur = cur * 10 + Int(c) - 48
    out[idx] = UInt8(cur)
    return out^

def _build_sockaddr_in(buf: BytePtr, octets: Array[UInt8, 4], port: UInt16):
    unsafe_memset(buf, 0, 16)
    buf.unsafe_offset(0).unsafe_bitcast[UInt16]()[] = UInt16(AF_INET)
    buf[unsafe_offset=2] = UInt8((port >> 8) & 0xFF)
    buf[unsafe_offset=3] = UInt8(port & 0xFF)
    buf[unsafe_offset=4] = octets[0]
    buf[unsafe_offset=5] = octets[1]
    buf[unsafe_offset=6] = octets[2]
    buf[unsafe_offset=7] = octets[3]

def sys_bind_inet(fd: Int32, ip: String, port: UInt16) -> Int32:
    """Binds to ip:port, or 0.0.0.0:port if ip is empty."""
    var octets = Array[UInt8, 4](fill=0)
    if ip.byte_length() > 0:
        octets = parse_ipv4_octets(ip)
    var sa_arr = Array[UInt8, 16](fill=0)
    var sa = sa_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _build_sockaddr_in(sa, octets, port)
    var rc = external_call["bind", Int32](fd, sa, Int32(16))
    _ = sa_arr[0]
    return rc

def sys_connect_inet(fd: Int32, ip: String, port: UInt16) -> Int32:
    var octets = parse_ipv4_octets(ip)
    var sa_arr = Array[UInt8, 16](fill=0)
    var sa = sa_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _build_sockaddr_in(sa, octets, port)
    var rc = external_call["connect", Int32](fd, sa, Int32(16))
    _ = sa_arr[0]
    return rc

def sys_unlink(path: String) -> Int32:
    var p = cstr(path)
    var rc = external_call["unlink", Int32](p)
    p.unsafe_free()
    return rc

def sys_sd_notify(state: String) -> Bool:
    """Sends a state notification string to systemd via $NOTIFY_SOCKET."""
    var k_arr = Array[UInt8, 14](fill=0)
    var k = k_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var k_lit = "NOTIFY_SOCKET".as_bytes()
    for j in range(13):
        k[unsafe_offset=j] = k_lit[j]
    k[unsafe_offset=13] = 0
    var env = external_call["getenv", BytePtr](k)
    _ = k_arr[0]
    if Int(env) == 0:
        return False
    var fd = external_call["socket", Int32](AF_UNIX, SOCK_DGRAM, 0)
    if fd < 0:
        return False
    var sun_arr = Array[UInt8, 110](fill=0)
    var sun = sun_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    sun.unsafe_offset(0).unsafe_bitcast[UInt16]()[] = UInt16(AF_UNIX)
    var i = 0
    while env[unsafe_offset=i] != 0 and i < 107:
        var ch = env[unsafe_offset=i]
        if i == 0 and ch == 64: # '@' abstract socket prefix
            sun[unsafe_offset=2 + i] = 0
        else:
            sun[unsafe_offset=2 + i] = ch
        i += 1
    var addrlen = 2 + i
    var msg_p = state.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var n = external_call["sendto", Int64](fd, msg_p, state.byte_length(), 0, sun, addrlen)
    _ = sun_arr[0]
    _ = external_call["close", Int32](fd)
    return Int(n) == state.byte_length()

def sys_read_meminfo_hugepages() -> Tuple[Int, Int]:
    """Reads /proc/meminfo and returns (HugePages_Total, Hugepagesize_kB)."""
    var content: String
    try:
        with open("/proc/meminfo", "r") as f:
            content = f.read()
    except _:
        return (-1, -1)

    var b = content.as_bytes()
    var length = len(b)
    var total = 0
    var size_kb = 0
    var i = 0

    var p_total = "HugePages_Total:".as_bytes()
    var p_sz = "Hugepagesize:".as_bytes()

    while i < length:
        if i + 16 <= length:
            var is_total = True
            for k in range(16):
                if b[i + k] != p_total[k]:
                    is_total = False
                    break
            if is_total:
                var j = i + 16
                while j < length and (b[j] == 32 or b[j] == 9):
                    j += 1
                var v = 0
                while j < length and b[j] >= 48 and b[j] <= 57:
                    v = v * 10 + Int(b[j] - 48)
                    j += 1
                total = v

        if i + 13 <= length:
            var is_sz = True
            for k in range(13):
                if b[i + k] != p_sz[k]:
                    is_sz = False
                    break
            if is_sz:
                var j = i + 13
                while j < length and (b[j] == 32 or b[j] == 9):
                    j += 1
                var v = 0
                while j < length and b[j] >= 48 and b[j] <= 57:
                    v = v * 10 + Int(b[j] - 48)
                    j += 1
                size_kb = v

        while i < length and b[i] != 10:
            i += 1
        i += 1

    return (total, size_kb)
