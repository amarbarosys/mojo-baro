# cpu-gemv: CPU weight-streaming rates for the big-model tier plan (P0, docs/BIG-MODELS.md).
# Mode "roof": every thread sums its share of a 2.3 GB buffer (64-byte SIMD loads), the RAM read ceiling.
# Mode "q4k": a q4_k GEMV over the same bytes through serve/expert_cpu.mojo's AVX-512 VNNI row dot (K = 4096 per row,
# 16 blocks of 144 B). Prints weight GB/s over the wall time; THREADS 1 and 16 bracket one core vs the socket.
# usage: cpu-gemv roof|q4k THREADS
from std.memory import Layout, alloc, dealloc
from std.sys import argv, exit, stderr
from std.time import perf_counter_ns
from max.algorithm import parallelize
from expert_cpu import q4k_row_dot_cpu

comptime K = 4096
comptime NB = K // 256
comptime ROW_BYTES = NB * 144
comptime ROWS = 1 << 20


def main() raises:
    var a = argv()
    if len(a) != 3:
        print("usage: cpu-gemv roof|q4k THREADS", file=stderr)
        exit(2)
    var mode = String(a[1])
    var nt = Int(atol(String(a[2])))
    var total = ROWS * ROW_BYTES
    var w = alloc(Layout[UInt8](count=total))
    var wp = w.unsafe_ptr()
    var seed = UInt64(88172645463325252)
    for i in range(0, total, 8):
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        wp.unsafe_offset(i).bitcast[UInt64]().unsafe_store(seed)
    var xq = alloc(Layout[Int8](count=K))
    var xd = alloc(Layout[Float32](count=K // 32))
    var xs = alloc(Layout[Float32](count=K // 32))
    for i in range(K):
        xq.unsafe_ptr().unsafe_offset(i).unsafe_store(Int8((i * 37) % 255 - 127))
    for g in range(K // 32):
        xd.unsafe_ptr().unsafe_offset(g).unsafe_store(Float32(0.01) * Float32(g % 7 + 1))
        xs.unsafe_ptr().unsafe_offset(g).unsafe_store(Float32(g % 5))
    var out = alloc(Layout[Float32](count=ROWS))
    var op = out.unsafe_ptr()
    var sink = alloc(Layout[UInt64](count=nt))
    var sp = sink.unsafe_ptr()
    var xqp = xq.unsafe_ptr()
    var xdp = xd.unsafe_ptr()
    var xsp = xs.unsafe_ptr()
    var roof = mode == "roof"
    var t0 = perf_counter_ns()

    def part(t: Int) {imm wp, imm op, imm sp, imm xqp, imm xdp, imm xsp, imm nt, imm total, imm roof}:
        if roof:
            var lo = total * t // nt // 64 * 64
            var hi = total * (t + 1) // nt // 64 * 64
            var acc = SIMD[DType.uint64, 8](0)
            for i in range(lo, hi, 64):
                acc += wp.unsafe_offset(i).bitcast[UInt64]().unsafe_load[width=8]()
            sp[unsafe_offset=t] = acc.reduce_add()
            return
        var r0 = ROWS * t // nt
        var r1 = ROWS * (t + 1) // nt
        for r in range(r0, r1):
            op[unsafe_offset=r] = q4k_row_dot_cpu(
                wp.unsafe_origin_cast[MutUntrackedOrigin](), r * ROW_BYTES, NB,
                xqp.unsafe_origin_cast[MutUntrackedOrigin](), xdp.unsafe_origin_cast[MutUntrackedOrigin](),
                xsp.unsafe_origin_cast[MutUntrackedOrigin](),
            )

    parallelize(part, nt, nt)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var chk = Float64(0)
    for r in range(0, ROWS, 4099):
        chk += Float64(op[unsafe_offset=r]) if not roof else 0.0
    print("mode", mode, "threads", nt, "GB", Float64(total) / 1e9, "s", secs, "GB/s", Float64(total) / secs / 1e9,
          "check", chk + Float64(sp[unsafe_offset=0] & 1) if roof else chk)
    dealloc(w^)
    dealloc(xq^)
    dealloc(xd^)
    dealloc(xs^)
    dealloc(out^)
    dealloc(sink^)
