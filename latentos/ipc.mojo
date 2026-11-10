# CANONICAL. LatentOS lives here (mojo-baro latentos/); AMDHQ src/latentos is a
# pointer to this directory (plan 12 M1, 2026-09-24). Tests and the check run
# from AMDHQ tools/latent-os/latentos-check against this copy.
#
# ipc.mojo — LatentOS IPC handle transport & memfd lifecycle (03 §4, §5).
# Provides sealed memfd creation, pure Mojo SCM_RIGHTS descriptor passing, and the agent handle table.

from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call

import latentos.sys as sys
import latentos.proto as proto

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

def _ptr[N: Int](mut arr: Array[UInt8, N]) -> BytePtr:
    return arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

def create_ipc_pair() raises -> Tuple[Int32, Int32]:
    """Creates a connected full-duplex UNIX stream socketpair for agent<->engine IPC."""
    var sv_arr = Array[Int32, 2](fill=-1)
    var sv = sv_arr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var rc = external_call["socketpair", Int32](sys.AF_UNIX, sys.SOCK_STREAM, 0, sv)
    var fd0 = sv[unsafe_offset=0]
    var fd1 = sv[unsafe_offset=1]
    _ = sv_arr[0]
    if rc != 0:
        raise Error("socketpair failed with errno " + String(sys.errno_now()))
    return (fd0, fd1)

def create_unix_listener(path: String) raises -> Int32:
    """Binds and listens on a UNIX domain socket at path."""
    _ = sys.sys_unlink(path)
    var fd = sys.sys_socket(sys.AF_UNIX, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var bind_rc = sys.sys_bind_unix(fd, path)
    if bind_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("bind() failed for " + path + " with errno " + String(sys.errno_now()))
    var listen_rc = sys.sys_listen(fd, 16)
    if listen_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("listen() failed with errno " + String(sys.errno_now()))
    return fd

def connect_unix_socket(path: String) raises -> Int32:
    """Connects to a UNIX domain socket at path."""
    var fd = sys.sys_socket(sys.AF_UNIX, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_connect_unix(fd, path)
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("connect() failed to " + path + " with errno " + String(sys.errno_now()))
    return fd

def mint_memfd_rw(name: String, size_bytes: Int) raises -> Tuple[Int32, BytePtr]:
    """Mints a sealable memfd and maps it writable.
    Caller fills the buffer, then calls seal_and_finalize().
    """
    var fd = sys.sys_memfd_create(name, sys.MFD_CLOEXEC | sys.MFD_ALLOW_SEALING)
    if fd < 0:
        raise Error("memfd_create failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_ftruncate(fd, Int64(size_bytes))
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("ftruncate failed with errno " + String(sys.errno_now()))
    var p = sys.sys_mmap(size_bytes, sys.PROT_READ | sys.PROT_WRITE, sys.MAP_SHARED, fd, 0)
    if Int(p) == -1:
        _ = sys.sys_close(fd)
        raise Error("mmap failed with errno " + String(sys.errno_now()))
    # 2 MiB shmem folios where the THP policy allows (shmem_enabled=advise): each fresh
    # 4 KiB page costs an allocation plus zeroing, about 9 ms per 20 MiB. Best effort.
    _ = sys.sys_madvise(p, size_bytes, sys.MADV_HUGEPAGE)
    return (fd, p)

def seal_and_finalize(fd: Int32, ptr: BytePtr, size_bytes: Int) -> Bool:
    """Unmaps the writable pointer and applies immutable seals.
    Mint order (03 §4): write -> munmap -> fcntl(F_ADD_SEALS) -> export.
    """
    var unmap_rc = sys.sys_munmap(ptr, size_bytes)
    if unmap_rc != 0:
        return False
    var seals = sys.F_SEAL_GROW | sys.F_SEAL_SHRINK | sys.F_SEAL_WRITE | sys.F_SEAL_SEAL
    var seal_rc = sys.sys_fcntl_add_seals(fd, Int32(seals))
    return seal_rc == 0

def is_sealed(fd: Int32) -> Bool:
    """True when the memfd carries all four immutable seals (the kernel keeps the state; this reads it)."""
    var want = sys.F_SEAL_GROW | sys.F_SEAL_SHRINK | sys.F_SEAL_WRITE | sys.F_SEAL_SEAL
    var got = Int(sys.sys_fcntl_get_seals(fd))
    return got >= 0 and (got & want) == want

def map_readonly(fd: Int32, size_bytes: Int) raises -> BytePtr:
    """Maps a sealed memfd read-only (zero-copy ingest). An unsealed fd is refused: STATE:unsealed."""
    if not is_sealed(fd):
        raise Error("STATE:unsealed")
    var p = sys.sys_mmap(size_bytes, sys.PROT_READ, sys.MAP_SHARED, fd, 0)
    if Int(p) == -1:
        raise Error("map_readonly failed with errno " + String(sys.errno_now()))
    _ = sys.sys_madvise(p, size_bytes, sys.MADV_HUGEPAGE)
    return p

# --- SCM_RIGHTS Handle Passing
def _build_msghdr(msg: BytePtr, iov: BytePtr, payload: BytePtr, payload_len: Int, ctrl: BytePtr, ctrl_len: Int):
    unsafe_memset(msg, 0, 56)
    iov.unsafe_offset(0).unsafe_bitcast[Int64]()[] = Int64(Int(payload))
    iov.unsafe_offset(8).unsafe_bitcast[Int64]()[] = Int64(payload_len)
    msg.unsafe_offset(16).unsafe_bitcast[Int64]()[] = Int64(Int(iov))
    msg.unsafe_offset(24).unsafe_bitcast[Int64]()[] = 1
    msg.unsafe_offset(32).unsafe_bitcast[Int64]()[] = Int64(Int(ctrl))
    msg.unsafe_offset(40).unsafe_bitcast[Int64]()[] = Int64(ctrl_len)

def send_handle(sock: Int32, header: proto.LatentHeader, fd: Int32) -> Bool:
    """Sends a 256-byte LatentHeader and open memfd atomically over a Unix socket via SCM_RIGHTS.
    Returns False for an fd that is not sealed: a writable memfd must not go out as immutable."""
    if not is_sealed(fd):
        return False
    var hbuf_arr = Array[UInt8, proto.LATENT_HEADER_SIZE](fill=0)
    var msg_arr = Array[UInt8, 56](fill=0)
    var iov_arr = Array[UInt8, 16](fill=0)
    var ctrl_arr = Array[UInt8, 24](fill=0)
    var hbuf = _ptr(hbuf_arr)
    var msg = _ptr(msg_arr)
    var iov = _ptr(iov_arr)
    var ctrl = _ptr(ctrl_arr)

    header.serialize(hbuf)

    # cmsghdr layout (x86-64 glibc):
    #   0..7: cmsg_len = 20 (CMSG_LEN(sizeof(int)))
    #   8..11: cmsg_level = SOL_SOCKET (1)
    #   12..15: cmsg_type = SCM_RIGHTS (1)
    #   16..19: fd
    ctrl.unsafe_offset(0).unsafe_bitcast[Int64]()[] = 20
    ctrl.unsafe_offset(8).unsafe_bitcast[Int32]()[] = sys.SOL_SOCKET
    ctrl.unsafe_offset(12).unsafe_bitcast[Int32]()[] = sys.SCM_RIGHTS
    ctrl.unsafe_offset(16).unsafe_bitcast[Int32]()[] = fd

    _build_msghdr(msg, iov, hbuf, proto.LATENT_HEADER_SIZE, ctrl, 24)
    var n = external_call["sendmsg", Int64](sock, msg, 0)
    return Int(n) == proto.LATENT_HEADER_SIZE

def recv_handle_why(sock: Int32) -> Tuple[proto.LatentHeader, Int32, String]:
    """Receives a 256-byte LatentHeader and its passed memfd atomically over a Unix socket. The third field is the
    reason when no handle came: "short_read", "peer_closed", "bad_header", "no_rights" or "bad_fd"; "" on success."""
    var hbuf_arr = Array[UInt8, proto.LATENT_HEADER_SIZE](fill=0)
    var msg_arr = Array[UInt8, 56](fill=0)
    var iov_arr = Array[UInt8, 16](fill=0)
    var ctrl_arr = Array[UInt8, 24](fill=0)
    var hbuf = _ptr(hbuf_arr)
    var msg = _ptr(msg_arr)
    var iov = _ptr(iov_arr)
    var ctrl = _ptr(ctrl_arr)

    _build_msghdr(msg, iov, hbuf, proto.LATENT_HEADER_SIZE, ctrl, 24)
    var n = external_call["recvmsg", Int64](sock, msg, 0)

    var header = proto.LatentHeader()
    var received_fd: Int32 = -1
    var why = String("")

    if Int(n) == 0:
        why = "peer_closed"
    elif Int(n) != proto.LATENT_HEADER_SIZE:
        why = "short_read"
    else:
        header = proto.LatentHeader.deserialize(hbuf)
        var sol = ctrl.unsafe_offset(8).unsafe_bitcast[Int32]()[]
        var scm = ctrl.unsafe_offset(12).unsafe_bitcast[Int32]()[]
        if sol == sys.SOL_SOCKET and scm == sys.SCM_RIGHTS:
            received_fd = ctrl.unsafe_offset(16).unsafe_bitcast[Int32]()[]
        if not header.is_valid():
            why = "bad_header"
        elif received_fd < 0:
            why = "no_rights" if not (sol == sys.SOL_SOCKET and scm == sys.SCM_RIGHTS) else "bad_fd"
        if why.byte_length() > 0:
            if received_fd >= 0:
                _ = sys.sys_close(received_fd)
            received_fd = -1
            header = proto.LatentHeader()
    return (header^, received_fd, why^)

def recv_handle(sock: Int32) -> Tuple[proto.LatentHeader, Int32]:
    """recv_handle_why without the reason (the signature serve/latent.mojo calls)."""
    var r = recv_handle_why(sock)
    return (r[0].copy(), r[1])

# --- TCP transport (03 sec.5, cross-host tier: raw payload, no shared kernel to
# hand an fd to, so this always copies bytes, never SCM_RIGHTS).
comptime FNV_OFFSET: UInt64 = 0xcbf29ce484222325
comptime FNV_PRIME: UInt64 = 0x100000001b3
comptime TCP_CHUNK = 4 * 1024 * 1024

def fnv1a64(ptr: BytePtr, n: Int) -> UInt64:
    """FNV-1a 64 over n bytes at ptr (8-byte UInt64 stride + byte tail) -- wire integrity
    check so cross-host framing stays self-contained (no crypto library to ship into a guest)."""
    var h: UInt64 = FNV_OFFSET
    var i = 0
    while i + 8 <= n:
        h = (h ^ ptr.unsafe_offset(i).unsafe_bitcast[UInt64]()[]) * FNV_PRIME
        i += 8
    while i < n:
        h = (h ^ UInt64(ptr[unsafe_offset=i])) * FNV_PRIME
        i += 1
    return h

def tcp_listen(bind_ip: String, port: UInt16) raises -> Int32:
    """Creates, binds (SO_REUSEADDR) and listens on a TCP socket at bind_ip:port."""
    var fd = sys.sys_socket(sys.AF_INET, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    _ = sys.sys_setsockopt_reuseaddr(fd)
    var bind_rc = sys.sys_bind_inet(fd, bind_ip, port)
    if bind_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("bind() failed for " + bind_ip + ":" + String(port) + " with errno " + String(sys.errno_now()))
    var listen_rc = sys.sys_listen(fd, 4)
    if listen_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("listen() failed with errno " + String(sys.errno_now()))
    return fd

def tcp_connect(ip: String, port: UInt16) raises -> Int32:
    var fd = sys.sys_socket(sys.AF_INET, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_connect_inet(fd, ip, port)
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("connect() failed to " + ip + ":" + String(port) + " with errno " + String(sys.errno_now()))
    return fd

def _tcp_write_all(fd: Int32, ptr: BytePtr, n: Int) raises:
    var sent = 0
    while sent < n:
        var rc = sys.sys_write(fd, ptr.unsafe_offset(sent), n - sent)
        if rc <= 0:
            raise Error("write() failed with errno " + String(sys.errno_now()))
        sent += Int(rc)

def _tcp_read_all(fd: Int32, ptr: BytePtr, n: Int) raises:
    var got = 0
    while got < n:
        var rc = sys.sys_read(fd, ptr.unsafe_offset(got), n - got)
        if rc <= 0:
            raise Error("read() failed or peer closed early with errno " + String(sys.errno_now()))
        got += Int(rc)

def send_tcp_latent(sock: Int32, header: proto.LatentHeader, payload: BytePtr, payload_len: Int) raises -> UInt64:
    """Sends a 256-byte LatentHeader, an 8-byte little-endian payload length, the payload itself
    (chunked), and a trailing 8-byte FNV-1a64 hash, over a connected TCP socket. Returns the hash
    sent, for the caller's own logging."""
    if payload_len <= 0 or not header.is_valid() or header.payload_len != UInt64(payload_len):
        raise Error("TCP:invalid_header_or_length")
    var hbuf_arr = Array[UInt8, proto.LATENT_HEADER_SIZE](fill=0)
    var hbuf = _ptr(hbuf_arr)
    header.serialize(hbuf)
    _tcp_write_all(sock, hbuf, proto.LATENT_HEADER_SIZE)

    var lenbuf_arr = Array[UInt8, 8](fill=0)
    var lenbuf = _ptr(lenbuf_arr)
    lenbuf.unsafe_bitcast[UInt64]()[] = UInt64(payload_len)
    _tcp_write_all(sock, lenbuf, 8)

    var sent = 0
    while sent < payload_len:
        var n = payload_len - sent
        if n > TCP_CHUNK:
            n = TCP_CHUNK
        _tcp_write_all(sock, payload.unsafe_offset(sent), n)
        sent += n

    var hash = fnv1a64(payload, payload_len)
    var hashbuf_arr = Array[UInt8, 8](fill=0)
    var hashbuf = _ptr(hashbuf_arr)
    hashbuf.unsafe_bitcast[UInt64]()[] = hash
    _tcp_write_all(sock, hashbuf, 8)
    return hash

def recv_tcp_latent(sock: Int32, dst: BytePtr, dst_cap: Int) raises -> Tuple[proto.LatentHeader, Int, UInt64, Bool]:
    """Receives a LatentHeader + length-prefixed payload (must fit in dst_cap) + trailing sender
    hash over a connected TCP socket. Returns (header, payload_len, local_hash, hash_matches)."""
    var hbuf_arr = Array[UInt8, proto.LATENT_HEADER_SIZE](fill=0)
    var hbuf = _ptr(hbuf_arr)
    _tcp_read_all(sock, hbuf, proto.LATENT_HEADER_SIZE)
    var header = proto.LatentHeader.deserialize(hbuf)
    if not header.is_valid():
        raise Error("TCP:invalid_header")

    var lenbuf_arr = Array[UInt8, 8](fill=0)
    var lenbuf = _ptr(lenbuf_arr)
    _tcp_read_all(sock, lenbuf, 8)
    var payload_len = Int(lenbuf.unsafe_bitcast[UInt64]()[])
    if payload_len <= 0 or header.payload_len != UInt64(payload_len):
        raise Error("TCP:length_mismatch")

    if payload_len > dst_cap:
        raise Error("payload_len " + String(payload_len) + " exceeds destination capacity " + String(dst_cap))

    var got = 0
    while got < payload_len:
        var n = payload_len - got
        if n > TCP_CHUNK:
            n = TCP_CHUNK
        _tcp_read_all(sock, dst.unsafe_offset(got), n)
        got += n

    var hashbuf_arr = Array[UInt8, 8](fill=0)
    var hashbuf = _ptr(hashbuf_arr)
    _tcp_read_all(sock, hashbuf, 8)
    var sender_hash = hashbuf.unsafe_bitcast[UInt64]()[]

    var local_hash = fnv1a64(dst, payload_len)
    if sender_hash != local_hash:
        raise Error("TCP:integrity")
    return (header^, payload_len, local_hash, True)

# --- Latent Store Handle Entry
struct HandleEntry(Copyable, Movable):
    var address: String
    var header: proto.LatentHeader
    var fd: Int32
    var bytes: Int
    var minted_at_s: Int64
    var expires_at_s: Int64
    var refcount: Int

    def __init__(out self, address: String, header: proto.LatentHeader, fd: Int32, now_s: Int64):
        self.address = address
        self.header = header.copy()
        self.fd = fd
        self.bytes = Int(header.payload_len)
        self.minted_at_s = now_s
        self.expires_at_s = now_s + Int64(header.ttl_s)
        self.refcount = 1

    def is_expired(self, now_s: Int64) -> Bool:
        return now_s >= self.expires_at_s


# --- Local Agent Handle Registry (03 §4)
struct LatentStore:
    var entries: List[HandleEntry]
    var retained_bytes: UInt64

    def __init__(out self):
        self.entries = List[HandleEntry]()
        self.retained_bytes = 0

    def put(mut self, header: proto.LatentHeader, fd: Int32) raises -> String:
        if fd < 0 or not header.is_valid():
            raise Error("STATE:invalid_handle")
        var now_s = sys.sys_clock_monotonic_s()
        var addr = header.content_address()
        for i in range(len(self.entries)):
            if self.entries[i].address == addr:
                # An expired entry is not revived by a second put: acquire() already refuses it with this word.
                if self.entries[i].is_expired(now_s):
                    if fd != self.entries[i].fd:
                        _ = sys.sys_close(fd)
                    raise Error("STATE:expired")
                if not proto.headers_equal(self.entries[i].header, header):
                    if fd != self.entries[i].fd:
                        _ = sys.sys_close(fd)
                    raise Error("STATE:content_collision")
                if fd != self.entries[i].fd:
                    _ = sys.sys_close(fd)
                self.entries[i].refcount += 1
                self.entries[i].expires_at_s = now_s + Int64(self.entries[i].header.ttl_s)
                return addr
        var entry = HandleEntry(addr, header, fd, now_s)
        var entry_bytes = UInt64(entry.bytes)
        self.entries.append(entry^)
        self.retained_bytes += entry_bytes
        return addr

    def get(mut self, address: String) raises -> Tuple[proto.LatentHeader, Int32]:
        """Same words as acquire (STATE:expired, STATE:stale_handle) instead of a (header, -1) that means both."""
        var now_s = sys.sys_clock_monotonic_s()
        for i in range(len(self.entries)):
            if self.entries[i].address == address:
                if self.entries[i].is_expired(now_s):
                    raise Error("STATE:expired")
                self.entries[i].expires_at_s = now_s + Int64(self.entries[i].header.ttl_s)
                return (self.entries[i].header.copy(), self.entries[i].fd)
        raise Error("STATE:stale_handle")

    def acquire(mut self, address: String) raises -> Tuple[proto.LatentHeader, Int32]:
        var now_s = sys.sys_clock_monotonic_s()
        for i in range(len(self.entries)):
            if self.entries[i].address == address:
                if self.entries[i].is_expired(now_s):
                    raise Error("STATE:expired")
                self.entries[i].refcount += 1
                self.entries[i].expires_at_s = now_s + Int64(self.entries[i].header.ttl_s)
                return (self.entries[i].header.copy(), self.entries[i].fd)
        raise Error("STATE:stale_handle")

    def release(mut self, address: String) raises -> Bool:
        for i in range(len(self.entries)):
            if self.entries[i].address == address:
                if self.entries[i].refcount <= 0:
                    return False
                self.entries[i].refcount -= 1
                if self.entries[i].refcount > 0:
                    return False
                var bytes = UInt64(self.entries[i].bytes)
                _ = sys.sys_close(self.entries[i].fd)
                _ = self.entries.pop(i)
                self.retained_bytes -= bytes
                return True
        return False

    def evict_expired(mut self, now_s: Int64) -> Int:
        var evicted = 0
        var i = len(self.entries) - 1
        while i >= 0:
            if self.entries[i].is_expired(now_s) and self.entries[i].refcount <= 1:
                var bytes = UInt64(self.entries[i].bytes)
                _ = sys.sys_close(self.entries[i].fd)
                _ = self.entries.pop(i)
                self.retained_bytes -= bytes
                evicted += 1
            i -= 1
        return evicted

    def count(self) -> Int:
        return len(self.entries)

    def bytes_retained(self) -> UInt64:
        return self.retained_bytes
