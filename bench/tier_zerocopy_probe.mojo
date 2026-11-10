"""Zero-copy probe for the expert tier: can a kernel read pinned host memory over
PCIe at the DMA rate? Arms per piece count NP (pieces of PIECE bytes at random
offsets in a 1 GiB store):

  dev    kernel reads the pieces from a VRAM mirror (VRAM bandwidth reference)
  host   kernel reads the pieces from the pinned host store (zero-copy, PCIe)
  dma    NP enqueue_copy host -> device, then synchronize (today's tier path)
  dma+k  dma then the dev kernel (today's serialized miss cost)

Checksums (xor over all pieces) must agree across dev, host and dma+k or the run
is VOID. Times are host wall clock around enqueue + synchronize, median of REPS.
Build: ./.venv/bin/mojo build bench/tier_zerocopy_probe.mojo -I kernels -o .work/tier-zc/probe
Run:   gpu-wait run --vram 4 --timeout 300 -- .work/tier-zc/probe
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

comptime STORE = 1 << 30
comptime PIECE = 589824  # gate/up q4_k expert projection bytes in the RegesCore pack
comptime THREADS = 256
comptime VEC = 16
comptime ITERS = 64
comptime CHUNK = THREADS * VEC * ITERS  # 262144 bytes per block
comptime BLOCKS_PER_PIECE = ceildiv(PIECE, CHUNK)
comptime REPS = 11
comptime NP_MAX = 24


def read_pieces(
    base: UnsafePointer[UInt8, MutAnyOrigin],
    offs: UnsafePointer[UInt64, MutAnyOrigin],
    piece_bytes: Int32,
    res: UnsafePointer[UInt64, MutAnyOrigin],
):
    var piece = Int(block_idx.x) // BLOCKS_PER_PIECE
    var chunk = Int(block_idx.x) % BLOCKS_PER_PIECE
    var start = Int(offs[piece]) + chunk * CHUNK
    var end = Int(offs[piece]) + Int(piece_bytes)
    var acc = SIMD[DType.uint64, 2](0)
    var p = start + Int(thread_idx.x) * VEC
    for it in range(ITERS):
        if p + VEC <= end:
            acc ^= base.unsafe_bitcast[UInt64]().load[width=2](p // 8)
        p += THREADS * VEC
    res[Int(block_idx.x) * THREADS + Int(thread_idx.x)] = acc[0] ^ acc[1]


def median(xs: List[Int]) -> Int:
    var v = xs.copy()
    for i in range(1, len(v)):
        var j = i
        while j > 0 and v[j - 1] > v[j]:
            var t = v[j - 1]
            v[j - 1] = v[j]
            v[j] = t
            j -= 1
    return v[len(v) // 2]


def checksum(
    ctx: DeviceContext, mut out_h: HostBuffer[DType.uint64], out_d: DeviceBuffer[DType.uint64], n_thr: Int
) raises -> UInt64:
    ctx.enqueue_copy(dst_buf=out_h, src_buf=out_d)
    ctx.synchronize()
    var c: UInt64 = 0
    for i in range(n_thr):
        c ^= out_h[i]
    return c


def main() raises:
    var ctx = DeviceContext()
    print("device", ctx.name(), "store", STORE, "piece", PIECE, "reps", REPS)
    var store_h = ctx.enqueue_create_host_buffer[DType.uint8](STORE)
    var store_d = ctx.enqueue_create_buffer[DType.uint8](STORE)
    var offs_h = ctx.enqueue_create_host_buffer[DType.uint64](NP_MAX)
    var offs_d = ctx.enqueue_create_buffer[DType.uint64](NP_MAX)
    comptime OUT_N = NP_MAX * BLOCKS_PER_PIECE * THREADS
    var out_h = ctx.enqueue_create_host_buffer[DType.uint64](OUT_N)
    var out_d = ctx.enqueue_create_buffer[DType.uint64](OUT_N)
    ctx.synchronize()
    var x: UInt64 = 0x9E3779B97F4A7C15
    var sp = store_h.unsafe_ptr()
    for i in range(0, STORE, 8):
        x = x * 6364136223846793005 + 1442695040888963407
        sp.unsafe_bitcast[UInt64]()[i // 8] = x
    var lcg: UInt64 = 12345
    for i in range(NP_MAX):
        lcg = lcg * 6364136223846793005 + 1442695040888963407
        offs_h[i] = ((lcg >> 20) % UInt64((STORE - PIECE) // 4096)) * 4096
    ctx.enqueue_copy(dst_buf=store_d, src_buf=store_h)
    ctx.enqueue_copy(dst_buf=offs_d, src_buf=offs_h)
    ctx.synchronize()

    var nps = List[Int]()
    nps.append(2)
    nps.append(8)
    nps.append(24)
    for k in range(len(nps)):
        var np = nps[k]
        var grid = np * BLOCKS_PER_PIECE
        var n_thr = grid * THREADS
        var bytes = np * PIECE
        var t_dev = List[Int]()
        var t_host = List[Int]()
        var t_dma = List[Int]()
        var t_dmak = List[Int]()
        var c_dev: UInt64 = 0
        var c_host: UInt64 = 0
        var c_dmak: UInt64 = 0
        for r in range(REPS + 2):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[read_pieces](
                store_d.unsafe_ptr(), offs_d.unsafe_ptr(), Int32(PIECE), out_d.unsafe_ptr(),
                grid_dim=grid, block_dim=THREADS,
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            if r >= 2:
                t_dev.append(Int(t1 - t0))
            c_dev = checksum(ctx, out_h, out_d, n_thr)

            t0 = perf_counter_ns()
            ctx.enqueue_function[read_pieces](
                store_h.unsafe_ptr(), offs_d.unsafe_ptr(), Int32(PIECE), out_d.unsafe_ptr(),
                grid_dim=grid, block_dim=THREADS,
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            if r >= 2:
                t_host.append(Int(t1 - t0))
            c_host = checksum(ctx, out_h, out_d, n_thr)

            # dma: today's tier path, one enqueue_copy per piece into a device region
            t0 = perf_counter_ns()
            for i in range(np):
                var off = Int(offs_h[i])
                ctx.enqueue_copy(
                    dst_buf=DeviceBuffer[DType.uint8](
                        ctx, store_d.unsafe_ptr().unsafe_offset(off), PIECE, owning=False
                    ),
                    src_buf=store_h.create_sub_buffer[DType.uint8](off, PIECE),
                )
            ctx.synchronize()
            t1 = perf_counter_ns()
            if r >= 2:
                t_dma.append(Int(t1 - t0))

            t0 = perf_counter_ns()
            for i in range(np):
                var off = Int(offs_h[i])
                ctx.enqueue_copy(
                    dst_buf=DeviceBuffer[DType.uint8](
                        ctx, store_d.unsafe_ptr().unsafe_offset(off), PIECE, owning=False
                    ),
                    src_buf=store_h.create_sub_buffer[DType.uint8](off, PIECE),
                )
            ctx.enqueue_function[read_pieces](
                store_d.unsafe_ptr(), offs_d.unsafe_ptr(), Int32(PIECE), out_d.unsafe_ptr(),
                grid_dim=grid, block_dim=THREADS,
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            if r >= 2:
                t_dmak.append(Int(t1 - t0))
            c_dmak = checksum(ctx, out_h, out_d, n_thr)
        var md = median(t_dev)
        var mh = median(t_host)
        var mc = median(t_dma)
        var mk = median(t_dmak)
        var ok = c_dev == c_host and c_dev == c_dmak
        print(
            "NP", np, "bytes", bytes,
            "dev_us", md // 1000, "dev_GBs", Float64(bytes) / Float64(md),
            "host_us", mh // 1000, "host_GBs", Float64(bytes) / Float64(mh),
            "dma_us", mc // 1000, "dma_GBs", Float64(bytes) / Float64(mc),
            "dma+k_us", mk // 1000,
            "checksum", "OK" if ok else "VOID",
        )
        if not ok:
            raise Error("VOID: checksum mismatch across arms")
