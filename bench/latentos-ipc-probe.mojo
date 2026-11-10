"""B3: same-GPU HIP IPC handoff probe.

Preregistered ~/AMDHQ/docs/design/latent-os/06-experiments.md ("E12-ipc"),
briefs/2026-09-15-c3-b3-lane.md. Two processes (fork, same shape as
~/AMDHQ/tools/latent-os/test_live_ipc.mojo) each own a DeviceContext on the
one GPU on this box. The child allocates a 2 GiB DeviceBuffer, fills it with
a deterministic pattern, exports a HIP IPC mem handle (hipIpcGetMemHandle)
and sends the raw 64-byte handle over a unix socket (latentos.ipc/sys -- a
HIP IPC handle is bytes, not a file descriptor, so plain sys_read/sys_write
replace SCM_RIGHTS). The parent opens the handle (hipIpcOpenMemHandle),
times a device-to-device hipMemcpy into its own buffer, and both sides
sha256 the transferred bytes.

hipIpcGetMemHandle/hipIpcOpenMemHandle have no precedent in either repo --
every existing external_call here is a libc function, always linked. A bare
external_call did NOT link (undefined reference to every hip* symbol at
link time, MAX's GPU runtime does not expose libamdhip64.so's symbols
globally); this build uses the E12-ipc prereg's fallback,
-Xlinker -lamdhip64 -Xlinker -L/opt/rocm/lib, see tools/ci-checks.sh below.

Gates (E12-ipc, frozen before this file existed): sha256 match on 3 runs,
D2D hipMemcpy alone under 10 ms for 2 GiB, both HIP calls return hipSuccess
on all 3 runs. A nonzero rc, an unresolved symbol, or a hash mismatch is
reported verbatim, not retried silently.
"""

# ci-checks: needs -Xlinker -lamdhip64 -Xlinker -L/opt/rocm/lib plus libamarbaro_shim (raw HIP IPC
# calls, not resolved by a bare external_call, see the docstring above) --
# skipped from tools/ci-checks.sh's generic bench-compile loop, same
# mechanism as its `^from grammar` skip for bench_latent_handoff.mojo.

from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

import latentos.sys as sys
import latentos.ipc as ipc

from sha256 import sha256

comptime u8 = DType.uint8

comptime SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB
comptime HANDLE_BYTES = 64  # HIP_IPC_HANDLE_SIZE
comptime CANARY_BYTES = 64
comptime BULK_FILL: UInt8 = 0xAB

comptime hipMemcpyDeviceToDevice: Int32 = 3
comptime hipIpcMemLazyEnablePeerAccess: UInt32 = 1




def hip_check(rc: Int32, what: String) raises:
    if rc != 0:
        raise Error("HIP call failed: " + what + " rc=" + String(rc))


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        print("FAIL:", msg)
        raise Error("Assertion failed: " + msg)
    print("PASS:", msg)


def sha256_device_buffer(ctx: DeviceContext, dbuf: DeviceBuffer[u8]) raises -> Array[UInt8, 32]:
    var hbuf = ctx.enqueue_create_host_buffer[u8](SIZE)
    ctx.enqueue_copy(dst_buf=hbuf, src_buf=dbuf)
    ctx.synchronize()
    var span = Span[Byte, _](unsafe_ptr=hbuf.unsafe_ptr(), length=SIZE)
    return sha256(span)


def send_all(fd: Int32, ptr: sys.BytePtr, n: Int) raises:
    var sent = 0
    while sent < n:
        var w = sys.sys_write(fd, ptr.unsafe_offset(sent), n - sent)
        if w <= 0:
            raise Error("sys_write failed at " + String(sent) + "/" + String(n) + " errno=" + String(sys.errno_now()))
        sent += Int(w)


def recv_all(fd: Int32, ptr: sys.BytePtr, n: Int) raises:
    var got = 0
    while got < n:
        var r = sys.sys_read(fd, ptr.unsafe_offset(got), n - got)
        if r <= 0:
            raise Error("sys_read failed at " + String(got) + "/" + String(n) + " errno=" + String(sys.errno_now()))
        got += Int(r)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var sock_path = ".work/latentos-ipc-probe.sock"
    var listen_fd = ipc.create_unix_listener(sock_path)
    assert_true(listen_fd > 0, "created and bound unix socket listener at " + sock_path)

    var t_start_ns = sys.sys_clock_monotonic_ns()
    var pid = sys.sys_fork()
    if pid < 0:
        _ = sys.sys_close(listen_fd)
        raise Error("sys_fork failed errno=" + String(sys.errno_now()))

    if pid == 0:
        # ---------------- child: owns the source 2 GiB buffer -------------
        var sock = ipc.connect_unix_socket(sock_path)
        if sock < 0:
            external_call["_exit", NoneType](Int32(10))

        var ctx = DeviceContext()
        var dbuf = ctx.enqueue_create_buffer[u8](SIZE)
        var dptr = dbuf.unsafe_ptr()

        # Bulk deterministic fill via hipMemset (fast, on-device), plus
        # head/tail canary regions distinct from the bulk value so a
        # transfer that silently truncates or offsets still shows up in
        # the hash, not just a same-value coincidence.
        var rc_memset = external_call["hipMemset", Int32](dptr, Int32(BULK_FILL), UInt64(SIZE))
        hip_check(rc_memset, "hipMemset bulk")

        var canary_host = ctx.enqueue_create_host_buffer[u8](CANARY_BYTES)
        var chp = canary_host.unsafe_ptr()
        for i in range(CANARY_BYTES):
            chp[unsafe_offset=i] = UInt8(i)
        var head = DeviceBuffer[u8](ctx, dptr, CANARY_BYTES, owning=False)
        ctx.enqueue_copy(dst_buf=head, src_buf=canary_host)
        for i in range(CANARY_BYTES):
            chp[unsafe_offset=i] = UInt8(255 - i)
        var tail = DeviceBuffer[u8](ctx, dptr.unsafe_offset(SIZE - CANARY_BYTES), CANARY_BYTES, owning=False)
        ctx.enqueue_copy(dst_buf=tail, src_buf=canary_host)
        ctx.synchronize()

        var handle_buf = unsafe_alloc[UInt8](HANDLE_BYTES)
        var rc_get = external_call["hipIpcGetMemHandle", Int32](handle_buf, dptr)
        hip_check(rc_get, "hipIpcGetMemHandle")

        send_all(sock, handle_buf, HANDLE_BYTES)
        print("child: sent handle,", HANDLE_BYTES, "bytes")

        var digest = sha256_device_buffer(ctx, dbuf)
        var digest_buf = unsafe_alloc[UInt8](32)
        for i in range(32):
            digest_buf[unsafe_offset=i] = digest[i]
        send_all(sock, digest_buf, 32)
        print("child: sent digest")

        # Block until the parent is done reading/copying via the handle --
        # the exported allocation must stay alive until then.
        var ack = unsafe_alloc[UInt8](1)
        recv_all(sock, ack, 1)

        _ = sys.sys_close(sock)
        external_call["_exit", NoneType](Int32(0))

    else:
        # ---------------- parent: opens the handle, times the D2D copy ----
        print("  -> spawned child pid " + String(pid))
        var conn = sys.sys_accept(listen_fd)
        assert_true(conn > 0, "accepted child connection")

        var handle_raw = unsafe_alloc[UInt8](HANDLE_BYTES)
        recv_all(conn, handle_raw, HANDLE_BYTES)
        var handoff_us = (sys.sys_clock_monotonic_ns() - t_start_ns) // 1000
        print("parent: received handle,", HANDLE_BYTES, "bytes, handoff", handoff_us, "us")

        var child_digest_buf = unsafe_alloc[UInt8](32)
        recv_all(conn, child_digest_buf, 32)

        var ctx = DeviceContext()
        var dst = ctx.enqueue_create_buffer[u8](SIZE)
        var dst_ptr = dst.unsafe_ptr()

        var opened_ptr_slot = unsafe_alloc[Int64](1)
        var rc_open = external_call["amarbaro_ipc_open", Int32](
            opened_ptr_slot, handle_raw, hipIpcMemLazyEnablePeerAccess
        )
        hip_check(rc_open, "hipIpcOpenMemHandle")
        var src_addr = opened_ptr_slot[unsafe_offset=0]

        var t0 = sys.sys_clock_monotonic_ns()
        var rc_copy = external_call["hipMemcpy", Int32](
            dst_ptr, src_addr, UInt64(SIZE), hipMemcpyDeviceToDevice
        )
        hip_check(rc_copy, "hipMemcpy D2D")
        var rc_sync = external_call["hipDeviceSynchronize", Int32]()
        hip_check(rc_sync, "hipDeviceSynchronize")
        var t1 = sys.sys_clock_monotonic_ns()
        var copy_us = (t1 - t0) // 1000
        print("parent: device-to-device copy took", copy_us, "us for", SIZE, "bytes")

        var rc_close = external_call["hipIpcCloseMemHandle", Int32](src_addr)
        hip_check(rc_close, "hipIpcCloseMemHandle")

        var parent_digest = sha256_device_buffer(ctx, dst)
        var hashes_match = True
        for i in range(32):
            if parent_digest[i] != child_digest_buf[unsafe_offset=i]:
                hashes_match = False
        assert_true(hashes_match, "sha256 of the transferred 2 GiB matches on both sides")
        assert_true(copy_us < 10000, "device-to-device copy under 10 ms for 2 GiB (" + String(copy_us) + " us)")

        var ack = unsafe_alloc[UInt8](1)
        ack[unsafe_offset=0] = 1
        send_all(conn, ack, 1)

        _ = sys.sys_close(conn)
        _ = sys.sys_close(listen_fd)
        _ = sys.sys_unlink(sock_path)

        var w = sys.sys_waitpid(pid)
        assert_true(w[0] == pid, "reaped child process")
        assert_true(w[1] == 0, "child process exited cleanly with status 0")

        print("LATENTOS IPC PROBE PASSED, D2D copy", copy_us, "us, handoff", handoff_us, "us")
