# Teleport KV T3a (bench/kv-ipc-protocol.md): KV pages A -> B over HIP IPC vs the memfd path.
# Build: mojo build -I . -I serve -I kernels -I latentos bench/kv_ipc_bench.mojo + shim and amdhip64 link flags
# (serve/latent_ipc.mojo docstring). Run under gpu-wait from the repo root.
# ci-checks: needs -Xlinker -lamarbaro_shim and -Xlinker -lamdhip64 (serve/latent_ipc.mojo docstring)
from std.sys import has_accelerator
from std.ffi import external_call
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc

from max.gpu.host import DeviceContext, DeviceBuffer

import latentos.ipc as ipc
import latentos.sys as sys
from registry import KVT
from latent import mint_kv_latent, ingest_kv_latent, PGSTR
from latent_ipc import KvIpcStage, ingest_kv_ipc, HANDLE_BYTES

comptime PAGES = 5
comptime ITERS = 8
comptime WARM = 2


def send_all(fd: Int32, ptr: sys.BytePtr, n: Int) raises:
    var sent = 0
    while sent < n:
        var w = sys.sys_write(fd, ptr.unsafe_offset(sent), n - sent)
        if w <= 0:
            raise Error("sys_write failed errno=" + String(sys.errno_now()))
        sent += Int(w)


def recv_all(fd: Int32, ptr: sys.BytePtr, n: Int) raises:
    var got = 0
    while got < n:
        var r = sys.sys_read(fd, ptr.unsafe_offset(got), n - got)
        if r <= 0:
            raise Error("sys_read failed errno=" + String(sys.errno_now()))
        got += Int(r)


def pattern(i: Int, it: Int) -> Float32:
    return Float32((i * 7 + it * 131) % 100003)


def fill_pattern(ctx: DeviceContext, buf: DeviceBuffer[KVT], n: Int, it: Int, off: Int) raises:
    var h = ctx.enqueue_create_host_buffer[KVT](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = pattern(i + off, it).cast[KVT]()
    ctx.enqueue_copy(dst_buf=buf, src_buf=h)
    ctx.synchronize()


def check(ctx: DeviceContext, buf: DeviceBuffer[KVT], n: Int, it: Int, off: Int) raises -> Int:
    var h = ctx.enqueue_create_host_buffer[KVT](n)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    var bad = 0
    for i in range(n):
        if h[i] != pattern(i + off, it).cast[KVT]():
            bad += 1
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var n = PAGES * PGSTR
    var sock_path = ".work/kv-ipc-bench.sock"
    var listen_fd = ipc.create_unix_listener(sock_path)
    if listen_fd <= 0:
        raise Error("FAIL listener")
    var pid = sys.sys_fork()
    if pid == 0:
        _ = sys.sys_close(listen_fd)
        var sock = ipc.connect_unix_socket(sock_path)
        var ctx = DeviceContext()
        var kA = ctx.enqueue_create_buffer[KVT](n)
        var vA = ctx.enqueue_create_buffer[KVT](n)
        var stage = KvIpcStage(PAGES)
        var ack = unsafe_alloc[UInt8](8)
        var tbuf = unsafe_alloc[Int64](1)
        for it in range(2 * ITERS):
            fill_pattern(ctx, kA, n, it, 0)
            fill_pattern(ctx, vA, n, it, n)
            var t = perf_counter_ns()
            if it >= ITERS:
                stage = KvIpcStage(PAGES)
            stage.fill(ctx, kA, vA, 0)
            tbuf[unsafe_offset=0] = Int64(perf_counter_ns() - t)
            send_all(sock, stage.handle, HANDLE_BYTES)
            send_all(sock, tbuf.unsafe_bitcast[UInt8](), 8)
            recv_all(sock, ack, 1)
        _ = sys.sys_close(sock)
        _ = stage^
        external_call["_exit", NoneType](Int32(0))
    var conn = sys.sys_accept(listen_fd)
    var ctx = DeviceContext()
    var kB = ctx.enqueue_create_buffer[KVT](n)
    var vB = ctx.enqueue_create_buffer[KVT](n)
    var handle = unsafe_alloc[UInt8](HANDLE_BYTES)
    var tbuf = unsafe_alloc[Int64](1)
    var ack = unsafe_alloc[UInt8](1)
    var fill_min = Float64(1e18)
    var ing_min = Float64(1e18)
    var hand_min = Float64(1e18)
    var cold_min = Float64(1e18)
    var cold_max = Float64(0)
    var bad_total = 0
    for it in range(2 * ITERS):
        ctx.enqueue_memset(kB, 0.0)
        ctx.enqueue_memset(vB, 0.0)
        ctx.synchronize()
        recv_all(conn, handle, HANDLE_BYTES)
        recv_all(conn, tbuf.unsafe_bitcast[UInt8](), 8)
        var t = perf_counter_ns()
        ingest_kv_ipc(ctx, kB, vB, 0, PAGES, handle)
        var ing = Float64(perf_counter_ns() - t) / 1e6
        var fill = Float64(tbuf[unsafe_offset=0]) / 1e6
        var bad = check(ctx, kB, n, it, 0) + check(ctx, vB, n, it, n)
        bad_total += bad
        ack[unsafe_offset=0] = 1
        send_all(conn, ack, 1)
        print("ipc", "cold" if it >= ITERS else "warm", "it", it, "fill_ms", fill, "ingest_ms", ing, "mismatched_floats", bad)
        if it >= ITERS:
            cold_min = min(cold_min, fill + ing)
            cold_max = max(cold_max, fill + ing)
        elif it >= WARM:
            fill_min = min(fill_min, fill)
            ing_min = min(ing_min, ing)
            hand_min = min(hand_min, fill + ing)
    var w = sys.sys_waitpid(pid)
    _ = sys.sys_close(conn)
    _ = sys.sys_close(listen_fd)
    _ = sys.sys_unlink(sock_path)

    var kA = ctx.enqueue_create_buffer[KVT](n)
    var vA = ctx.enqueue_create_buffer[KVT](n)
    ctx.enqueue_memset(kA, 1.0)
    ctx.enqueue_memset(vA, 2.0)
    ctx.synchronize()
    var memfd_min = Float64(1e18)
    for it in range(ITERS):
        var t = perf_counter_ns()
        var kv = mint_kv_latent(ctx, kA, vA, 0, PAGES, UInt64(it + 1))
        ingest_kv_latent(ctx, kB, vB, kv[0], kv[1])
        var ms = Float64(perf_counter_ns() - t) / 1e6
        if it >= WARM:
            memfd_min = min(memfd_min, ms)
    print("SUMMARY bytes", 2 * n * 4, "ipc fill_min_ms", fill_min, "ingest_min_ms", ing_min, "warm_handoff_min_ms", hand_min, "cold_handoff_min_ms", cold_min, "cold_max_ms", cold_max,
          "memfd_min_ms", memfd_min, "warm_ratio", memfd_min / hand_min, "cold_ratio", memfd_min / cold_min, "mismatched_floats", bad_total, "child_status", w[1])
    if bad_total != 0 or w[1] != 0:
        raise Error("FAIL kv-ipc: mismatched floats or child exit")
    print("PASS kv-ipc")
