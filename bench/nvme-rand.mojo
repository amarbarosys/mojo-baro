# nvme-rand: random O_DIRECT reads from an existing file at a fixed piece size and queue depth (P0 of the big-model
# tier plan, docs/BIG-MODELS.md). One thread per in-flight read, offsets 4 KiB-aligned and uniform over the file, so a
# real GGUF shard serves as incompressible data and nothing is written. Prints GB/s over the wall time plus mean and
# max per-piece latency: the max/mean ratio is the tail that gates a layer waiting on its slowest expert.
# usage: nvme-rand FILE PIECE_KB QD READS_PER_THREAD
from std.ffi import external_call, c_ssize_t
from std.memory import Layout, alloc, dealloc
from std.os import stat
from std.sys import argv, exit, stderr
from std.time import perf_counter_ns
from max.algorithm import parallelize

comptime O_DIRECT = 0o40000
comptime ALIGN = 4096


def main() raises:
    var a = argv()
    if len(a) != 5:
        print("usage: nvme-rand FILE PIECE_KB QD READS_PER_THREAD", file=stderr)
        exit(2)
    var path = String(a[1])
    var piece = Int(atol(String(a[2]))) * 1024
    var qd = Int(atol(String(a[3])))
    var reads = Int(atol(String(a[4])))
    var fsize = stat(path).st_size
    if piece % ALIGN != 0 or piece <= 0 or fsize < piece + ALIGN or qd < 1 or reads < 1:
        print("nvme-rand: PIECE_KB must be a positive multiple of 4 and smaller than the file", file=stderr)
        exit(2)
    var cpath = path
    var fd = external_call["open64", Int32](cpath.as_c_string_span().ptr(), Int32(O_DIRECT))
    if fd < 0:
        print("nvme-rand: open with O_DIRECT failed:", path, file=stderr)
        exit(1)
    var buf = alloc(Layout[UInt8](count=qd * piece + ALIGN))
    var base = (Int(buf.unsafe_ptr()) + ALIGN - 1) // ALIGN * ALIGN
    var lat = alloc(Layout[Int](count=qd * reads))
    var lp = lat.unsafe_ptr()
    var bad = alloc(Layout[Int](count=qd))
    var bp = bad.unsafe_ptr()
    var span = (fsize - piece) // ALIGN
    var t0 = perf_counter_ns()

    def worker(t: Int) {imm fd, imm base, imm piece, imm reads, imm span, imm lp, imm bp}:
        var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base + t * piece)
        var x = UInt64(0x9E3779B97F4A7C15) * UInt64(t + 1)
        bp[unsafe_offset=t] = 0
        for k in range(reads):
            x = x * 6364136223846793005 + 1442695040888963407
            var off = Int((x >> 17) % UInt64(span)) * ALIGN
            var s = perf_counter_ns()
            var got = 0
            while got < piece:
                var n = external_call["pread", c_ssize_t](fd, p.unsafe_offset(got), piece - got, Int64(off + got))
                if n <= 0:
                    bp[unsafe_offset=t] = 1
                    return
                got += Int(n)
            lp[unsafe_offset=t * reads + k] = Int(perf_counter_ns() - s)

    parallelize(worker, qd, qd)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    _ = external_call["close", Int32](fd)
    for t in range(qd):
        if bp[unsafe_offset=t] != 0:
            print("FAIL nvme-rand: short read in thread", t, file=stderr)
            exit(1)
    var total = 0
    var worst = 0
    for i in range(qd * reads):
        total += lp[unsafe_offset=i]
        worst = max(worst, lp[unsafe_offset=i])
    var mean_ms = Float64(total) / Float64(qd * reads) / 1e6
    var gbps = Float64(qd * reads * piece) / wall / 1e9
    print("piece_kb", piece // 1024, "qd", qd, "reads", qd * reads, "GB/s", gbps, "lat_mean_ms", mean_ms,
          "lat_max_ms", Float64(worst) / 1e6, "max/mean", Float64(worst) / 1e6 / mean_ms)
    dealloc(buf^)
    dealloc(lat^)
    dealloc(bad^)
