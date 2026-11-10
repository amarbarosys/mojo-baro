"""P4 probe (Strata port): can a read-only MAP_SHARED file mapping be registered with
hipHostRegister so a kernel reads the page-cache pages directly (one RAM copy of the
experts instead of a pinned store plus the page cache)?

Gate: the GPU byte-word sum over the registered range equals the CPU sum, no HIP error,
and the GPU read rate is within 20% of the same read from a pinned HostBuffer.
Usage: test_hostregister [FILE] [LEN_BYTES]
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call, OwnedDLHandle
from std.memory import UnsafePointer
from std.sys import argv
from std.time import perf_counter_ns
from std.os.path import expanduser
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

comptime DEFAULT_FILE = "~/.cache/baro/qwen35moe-f2628b18234dbbb4/tier-e5c7fc0d/experts.bin"
comptime THREADS = 256
comptime BLOCKS = 4096
comptime ITERS = 9
comptime PROT_READ = Int32(1)
comptime MAP_SHARED = Int32(1)


def sum_words(
    src: UnsafePointer[UInt32, MutAnyOrigin],
    part: UnsafePointer[UInt64, MutAnyOrigin],
    nvec: Int64,
    stride: Int64,
):
    var g = Int(global_idx.x)
    var acc = UInt64(0)
    var i = g
    var nv = Int(nvec)
    var st = Int(stride)
    while i < nv:
        var v = src.load[width=4](i * 4)
        acc += UInt64(v[0]) + UInt64(v[1]) + UInt64(v[2]) + UInt64(v[3])
        i += st
    part[g] = acc


def spin_until(
    flag: UnsafePointer[Int32, MutAnyOrigin],
    done: UnsafePointer[Int32, MutAnyOrigin],
    cap: Int32,
):
    var n = Int32(0)
    while Atomic[Int32].load[ordering=Ordering.ACQUIRE](flag) == 0 and n < cap:
        n += 1
    done[0] = n


def hip_flag(lib: OwnedDLHandle) raises -> UnsafePointer[Int32, MutAnyOrigin]:
    var addr = Int(0)
    var rc = lib.get_function[Int32]("hipHostMalloc")(UnsafePointer(to=addr), Int(64), UInt32(0x40000002))
    if rc != 0 or addr == 0:
        raise Error("hipHostMalloc failed rc " + String(rc))
    var p = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)
    p.unsafe_store[volatile=True](Int32(0))
    return p


def copy_under_spin(
    ctx: DeviceContext, lib: OwnedDLHandle, src: HostBuffer[DType.uint8], off: Int, label: String
) raises -> Bool:
    """The doorbell leaves a kernel spinning on the main stream while expert_tier
    DMAs q6_k down pieces on a second stream and synchronizes it. A copy that
    cannot finish until that kernel exits deadlocks the engine."""
    comptime CAP = Int32(2_000_000)
    var flag = hip_flag(lib)
    var done = hip_flag(lib)
    var dst = ctx.enqueue_create_buffer[DType.uint8](8 * 860160)
    ctx.synchronize()
    _ = ctx.create_stream()
    var cctx = ctx.select_stream(ctx.num_streams() - 1)
    var part = ctx.enqueue_create_buffer[DType.uint64](BLOCKS * THREADS)
    var rd = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(src.unsafe_ptr()))
    for _ in range(4):
        ctx.enqueue_function[sum_words](
            rd, part.unsafe_ptr(), Int64((1 << 30) // 16), Int64(BLOCKS * THREADS), grid_dim=BLOCKS, block_dim=THREADS
        )
    ctx.enqueue_function[spin_until](flag, done, CAP, grid_dim=1, block_dim=1)
    var t0 = perf_counter_ns()
    for ci in range(8):
        cctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](ctx, dst.unsafe_ptr().unsafe_offset(ci * 860160), 860160, owning=False),
            src_buf=src.create_sub_buffer[DType.uint8](off + ci * 1720320, 860160),
        )
    cctx.stream().synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    flag.unsafe_store[volatile=True](Int32(1))
    ctx.synchronize()
    var spun = done.unsafe_load[volatile=True]()
    var blocked = spun >= CAP
    print("copy under spinning kernel,", label, "offset", off, ": copy+sync ms", ms, "spin iterations", spun, "blocked", blocked)
    return not blocked


def cpu_sum(p: UnsafePointer[UInt32, MutAnyOrigin], nwords: Int) -> UInt64:
    var acc = UInt64(0)
    for i in range(nwords):
        acc += UInt64(p[unsafe_offset=i])
    return acc


def timed_read(
    ctx: DeviceContext,
    src: UnsafePointer[UInt32, MutAnyOrigin],
    part: UnsafePointer[UInt64, MutAnyOrigin],
    nbytes: Int,
) raises -> Float64:
    var nvec = Int64(nbytes // 16)
    var stride = Int64(BLOCKS * THREADS)
    ctx.enqueue_function[sum_words](src, part, nvec, stride, grid_dim=BLOCKS, block_dim=THREADS)
    ctx.synchronize()
    var best = Float64(0)
    var rates = List[Float64]()
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[sum_words](src, part, nvec, stride, grid_dim=BLOCKS, block_dim=THREADS)
        ctx.synchronize()
        rates.append(Float64(nbytes) / (Float64(perf_counter_ns() - t0) / 1e9) / 1e9)
    sort(rates)
    return rates[len(rates) // 2]


def total(ctx: DeviceContext, out_h: UnsafePointer[UInt64, MutAnyOrigin]) -> UInt64:
    var acc = UInt64(0)
    for i in range(BLOCKS * THREADS):
        acc += out_h[i]
    return acc


def probe(path: String, length: Int, fd: Int) raises:
    var ctx = DeviceContext()
    var lib = OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")
    var addr = external_call["mmap", Int](Int(0), length, PROT_READ, MAP_SHARED, Int32(fd), Int64(0))
    if addr == -1:
        raise Error("FAIL mmap: errno-less MAP_FAILED for " + path)
    var base = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=addr)
    var nwords = length // 4
    var t_cpu = perf_counter_ns()
    var want = cpu_sum(base, nwords)
    print("cpu sum", want, "in s", Float64(perf_counter_ns() - t_cpu) / 1e9, "bytes", length)

    var out_d = ctx.enqueue_create_buffer[DType.uint64](BLOCKS * THREADS)
    var out_h = ctx.enqueue_create_host_buffer[DType.uint64](BLOCKS * THREADS)
    ctx.synchronize()
    var outp = UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(out_d.unsafe_ptr()))
    var outh = UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(out_h.unsafe_ptr()))

    var pinned = ctx.enqueue_create_host_buffer[DType.uint8](length)
    ctx.synchronize()
    _ = external_call["memcpy", Int](Int(pinned.unsafe_ptr()), addr, length)
    var pp = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(pinned.unsafe_ptr()))
    _ = copy_under_spin(ctx, lib, pinned, 0, "pinned HostBuffer")
    var gb_pin = timed_read(ctx, pp, outp, length)
    ctx.enqueue_copy(dst_buf=out_h, src_buf=out_d)
    ctx.synchronize()
    var got_pin = total(ctx, outh)
    print("pinned HostBuffer: GB/s", gb_pin, "sum", got_pin, "match", got_pin == want)

    var flag_sets = List[UInt32]()
    flag_sets.append(UInt32(0x0A))
    flag_sets.append(UInt32(0x02))
    var passed = False
    for fi in range(len(flag_sets)):
        var flags = flag_sets[fi]
        var rc = lib.get_function[Int32]("hipHostRegister")(addr, length, flags)
        print("hipHostRegister flags", flags, "rc", rc)
        if rc != 0:
            continue
        var dev = Int(0)
        var rc2 = lib.get_function[Int32]("hipHostGetDevicePointer")(UnsafePointer(to=dev), addr, UInt32(0))
        print("hipHostGetDevicePointer rc", rc2, "same address", dev == addr)
        if rc2 != 0 or dev == 0:
            _ = lib.get_function[Int32]("hipHostUnregister")(addr)
            continue
        var dp = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=dev)
        var gb_reg = timed_read(ctx, dp, outp, length)
        ctx.enqueue_copy(dst_buf=out_h, src_buf=out_d)
        ctx.synchronize()
        var got = total(ctx, outh)
        var ratio = gb_reg / gb_pin
        print("registered mmap flags", flags, ": GB/s", gb_reg, "sum", got, "match", got == want, "rate vs pinned", ratio)
        var view = HostBuffer[DType.uint8](
            ctx, UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr), length, owning=False
        )
        var copy_ok = True
        var offs = List[Int]()
        offs.append(0)
        offs.append(length // 4)
        offs.append(length // 2)
        offs.append(length - 8 * 1720320)
        for oi in range(len(offs)):
            if offs[oi] + 8 * 1720320 <= length and not copy_under_spin(ctx, lib, view, offs[oi], "registered mmap"):
                copy_ok = False
        var ok = got == want and ratio >= 0.8 and copy_ok
        _ = lib.get_function[Int32]("hipHostUnregister")(addr)
        if ok:
            print("PASS hostregister flags", flags)
            passed = True
            break
        print("FAIL hostregister flags", flags, ": sum match", got == want, "rate ratio", ratio, "copy under spin ok", copy_ok)
    _ = external_call["munmap", Int32](addr, length)
    if not passed:
        raise Error("FAIL hostregister: no flag set passed")


def main() raises:
    var args = argv()
    var path = expanduser(String(DEFAULT_FILE))
    var length = 1 << 30
    if len(args) > 1:
        path = String(args[1])
    if len(args) > 2:
        length = Int(String(args[2]))
    with open(path, "r") as f:
        probe(path, length, f._get_raw_fd())
