"""P0 doorbell probe (Strata port, plan zany-napping-zebra): can a kernel and the host
hand off through pinned host memory with no synchronize, and how fast is the round trip
against the per-layer synchronize + copy that serve/expert_tier.mojo prepare() pays today?

One thread: for i in 1..N it stores i into bell[0], then spins on flag[0] >= i. The host
spins on bell[0] >= i and stores i into flag[0]. A host spin that passes TIMEOUT_NS raises
flag[0] past N so the kernel exits, and the test fails. The MAX host-buffer arm is non-coherent and only a
reference (3,949 us per round trip on 2026-10-09, never seen at all since 2026-10-10): it reports, it does not fail.
"""
from std.atomic import Atomic, Ordering
from std.memory import UnsafePointer
from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime N = 10000
comptime TIMEOUT_NS = 2_000_000_000


def pingpong(bell: UnsafePointer[Int32, MutAnyOrigin], flag: UnsafePointer[Int32, MutAnyOrigin], n: Int32):
    for i in range(1, Int(n) + 1):
        Atomic[Int32].store[ordering=Ordering.RELEASE](bell, Int32(i))
        while Atomic[Int32].load[ordering=Ordering.ACQUIRE](flag) < Int32(i):
            pass


def touch(out_d: UnsafePointer[Int32, MutAnyOrigin], v: Int32):
    out_d.unsafe_store(v)


def run(ctx: DeviceContext, bell: UnsafePointer[Int32, MutAnyOrigin], flag: UnsafePointer[Int32, MutAnyOrigin], label: String, required: Bool = True) raises -> Float64:
    bell.unsafe_store[volatile=True](Int32(0))
    flag.unsafe_store[volatile=True](Int32(0))
    ctx.enqueue_function[pingpong](bell, flag, Int32(N), grid_dim=1, block_dim=1)
    var t0 = perf_counter_ns()
    var ok = True
    for i in range(1, N + 1):
        var ts = perf_counter_ns()
        while bell.unsafe_load[volatile=True]() < Int32(i):
            if perf_counter_ns() - ts > TIMEOUT_NS:
                ok = False
                break
        if not ok:
            print("FAIL doorbell" if required else "NOTE doorbell (reference arm, not required)", label, ": host never saw bell", i)
            flag.unsafe_store[volatile=True](Int32(N + 1))
            break
        flag.unsafe_store[volatile=True](Int32(i))
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(N)
    ctx.synchronize()
    if not ok:
        if required:
            raise Error("doorbell probe failed")
        return Float64(TIMEOUT_NS) / 1e3
    print("doorbell", label, "round trip us:", us)
    return us


def hip_host(lib: OwnedDLHandle, flags: UInt32) raises -> UnsafePointer[Int32, MutAnyOrigin]:
    var addr = Int(0)
    var rc = lib.get_function[Int32]("hipHostMalloc")(UnsafePointer(to=addr), Int(64), flags)
    if rc != 0 or addr == 0:
        raise Error("hipHostMalloc failed rc " + String(rc))
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)


def main() raises:
    var ctx = DeviceContext()
    var bell_h = ctx.enqueue_create_host_buffer[DType.int32](16)
    var flag_h = ctx.enqueue_create_host_buffer[DType.int32](16)
    ctx.synchronize()
    var t_max = run(ctx, UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(bell_h.unsafe_ptr())), UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(flag_h.unsafe_ptr())), "max-host-buffer", required=False)
    var lib = OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")
    var t_coh = run(ctx, hip_host(lib, 0x40000002), hip_host(lib, 0x40000002), "hipHostMalloc-coherent")

    var dev = ctx.enqueue_create_buffer[DType.int32](16)
    var back = ctx.enqueue_create_host_buffer[DType.int32](16)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for i in range(N):
        ctx.enqueue_function[touch](dev.unsafe_ptr(), Int32(i), grid_dim=1, block_dim=1)
        ctx.enqueue_copy(dst_buf=back, src_buf=dev)
        ctx.synchronize()
    var t_sync = Float64(perf_counter_ns() - t1) / 1e3 / Float64(N)
    if back.unsafe_ptr()[0] != Int32(N - 1):
        raise Error("sync path read back the wrong value")
    print("sync+copy us:", t_sync, " max-buffer ratio:", t_sync / t_max, " coherent ratio:", t_sync / t_coh, " N:", N)
    print("PASS doorbell")
