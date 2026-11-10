# LatentOS handoff cost, stage by stage (fix 1 of the 2026-09-11 weak-spot list).
# E12 measured ingest 37 ms and mint 60 ms against E7's 3.47 ms restore. Times each
# serve/latent.mojo call at E12's sizes (5 KV pages = 40 MiB, SSM slot 50.25 MiB), then
# the pieces underneath: pinned host alloc, D2H, H2D, and a memfd write on first touch
# vs already-faulted pages. Iterations 0-1 are warmup; mean and min over the rest.
from std.memory import unsafe_memcpy
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

import latentos.ipc as ipc
import latentos.sys as sys
from registry import CONV_SLOT, SSM_SLOT, KVT
from prefix import Chain, f32
from latent import mint_kv_latent, ingest_kv_latent, mint_chain_slot, ingest_into_chain, PGSTR

comptime PAGES = 5
comptime ITERS = 12
comptime WARM = 2
comptime NSTAGE = 11


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var names: List[String] = [
        "mint_kv_latent (40 MiB)", "Chain.save + sync (SSM D2H)", "mint_chain_slot (SSM -> memfd)",
        "ingest_kv_latent (40 MiB)", "ingest_into_chain (memfd -> SSM host)", "Chain.restore + sync (SSM H2D)",
        "  pinned alloc 2 x 20 MiB", "  D2H 40 MiB into pinned", "  H2D 40 MiB from pinned",
        "  memfd create + write 20 MiB, first touch", "  memfd write 20 MiB, pages faulted",
    ]
    var ctx = DeviceContext()
    var n = PAGES * PGSTR
    var kA = ctx.enqueue_create_buffer[KVT](n)
    var vA = ctx.enqueue_create_buffer[KVT](n)
    var kB = ctx.enqueue_create_buffer[KVT](n)
    var vB = ctx.enqueue_create_buffer[KVT](n)
    var cA = ctx.enqueue_create_buffer[f32](CONV_SLOT)
    var sA = ctx.enqueue_create_buffer[f32](SSM_SLOT)
    var cB = ctx.enqueue_create_buffer[f32](CONV_SLOT)
    var sB = ctx.enqueue_create_buffer[f32](SSM_SLOT)
    ctx.enqueue_memset(kA, 1.0)
    ctx.enqueue_memset(vA, 2.0)
    ctx.enqueue_memset(cA, 3.0)
    ctx.enqueue_memset(sA, 4.0)
    var chainA = Chain(ctx, 1, ".work/engine-pack-q4")
    var chainB = Chain(ctx, 1, ".work/engine-pack-q4")
    ctx.synchronize()
    var toks = List[Int]()
    for i in range(PAGES * 128):
        toks.append(i)

    var sums = List[Float64]()
    var mins = List[Float64]()
    for _ in range(NSTAGE):
        sums.append(0.0)
        mins.append(1e18)

    for it in range(ITERS):
        var dt = List[Float64]()
        var t = perf_counter_ns()
        var kv = mint_kv_latent(ctx, kA, vA, 0, PAGES, UInt64(it + 1))
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        chainA.save(ctx, cA, sA, 0, PAGES * 128, toks, False, False)
        ctx.synchronize()
        chainA.commit()
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        var ck = mint_chain_slot(chainA, 0)
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        ingest_kv_latent(ctx, kB, vB, kv[0], kv[1])
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        var slot = ingest_into_chain(chainB, ck[0], ck[1])
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        chainB.restore(ctx, cB, sB, 0, slot)
        ctx.synchronize()
        dt.append(Float64(perf_counter_ns() - t) / 1e6)

        t = perf_counter_ns()
        var h1 = ctx.enqueue_create_host_buffer[KVT](n)
        var h2 = ctx.enqueue_create_host_buffer[KVT](n)
        ctx.synchronize()
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=h1, src_buf=kA)
        ctx.enqueue_copy(dst_buf=h2, src_buf=vA)
        ctx.synchronize()
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=kB, src_buf=h1)
        ctx.enqueue_copy(dst_buf=vB, src_buf=h2)
        ctx.synchronize()
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        var nbytes = n * 4
        var src = h1.unsafe_ptr().unsafe_bitcast[UInt8]()
        t = perf_counter_ns()
        var m = ipc.mint_memfd_rw("probe", nbytes)
        unsafe_memcpy(dest=m[1], src=src, count=nbytes)
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        t = perf_counter_ns()
        unsafe_memcpy(dest=m[1], src=src, count=nbytes)
        dt.append(Float64(perf_counter_ns() - t) / 1e6)
        _ = ipc.seal_and_finalize(m[0], m[1], nbytes)
        _ = sys.sys_close(m[0])

        if it >= WARM:
            for s in range(NSTAGE):
                sums[s] += dt[s]
                if dt[s] < mins[s]:
                    mins[s] = dt[s]

    var reps = Float64(ITERS - WARM)
    print("stage | mean ms | min ms   (iterations", WARM, "to", ITERS - 1, ")")
    for s in range(NSTAGE):
        print(names[s], "|", sums[s] / reps, "|", mins[s])

    # PCIe link DPM probe: the same 40 MiB H2D back to back. If the first copies are
    # slow and the later ones reach E6's ~28 GB/s, the cost is link ramp-up, not the path.
    var hk = ctx.enqueue_create_host_buffer[KVT](n)
    var hv = ctx.enqueue_create_host_buffer[KVT](n)
    ctx.synchronize()
    print("back-to-back H2D 40 MiB: copy | ms | GB/s")
    for r in range(12):
        var t = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=kB, src_buf=hk)
        ctx.enqueue_copy(dst_buf=vB, src_buf=hv)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t) / 1e6
        print(r, "|", ms, "|", Float64(2 * n * 4) / (ms * 1e6))
