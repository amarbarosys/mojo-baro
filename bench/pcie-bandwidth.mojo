"""B4 stage 1: sustained PCIe bandwidth on this box, measured, not assumed.

briefs/2026-09-15-p3-b4-stage1-pcie.md. Times host-to-device, device-to-host,
and both enqueued back to back before one synchronize (see the caveat on
that third number in the report: DeviceContext.enqueue_copy has no stream
parameter in this API, so this is not a stream-isolated concurrent-transfer
number, it is "two async enqueues drained by one sync on the default queue").
Pinned host memory (ctx.enqueue_create_host_buffer, page-locked per its own
docstring) and pageable host memory (std.memory.alloc.alloc, a plain heap
allocation, not registered with the driver) are measured separately: a real
prefetcher uses pinned, a careless one gets pageable, and the gap between
them is part of the finding.

Read the PCIe link state back from the system next to these numbers
(/sys/class/drm/card1/device/current_link_speed and current_link_width, or
lspci -vv) before trusting them: a link trained down to x8 or 8 GT/s would
still produce clean, tight, wrong numbers (P1).

Usage: bench/pcie-bandwidth.mojo (build then run under gpu-wait; no args).
"""
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.memory.alloc import alloc, dealloc, Layout

from max.gpu.host import DeviceContext

comptime dtype = DType.uint8
comptime REPS = 5
comptime WARM = 1


def median_f64(vals: List[Float64]) -> Float64:
    var xs = List[Float64]()
    for v in vals:
        xs.append(v)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                var t = xs[i]
                xs[i] = xs[j]
                xs[j] = t
    return xs[len(xs) // 2]


def gbps(nbytes: Int, seconds: Float64) -> Float64:
    return Float64(nbytes) / seconds / 1e9


def report_row(label: String, nbytes: Int, times: List[Float64]) -> None:
    var mn = times[0]
    var mx = times[0]
    for t in times:
        if t < mn:
            mn = t
        if t > mx:
            mx = t
    var med = median_f64(times)
    print(
        label,
        " bytes:", nbytes,
        " median_GBps:", gbps(nbytes, med),
        " min_GBps:", gbps(nbytes, mx),  # max time -> min throughput
        " max_GBps:", gbps(nbytes, mn),  # min time -> max throughput
        " median_s:", med,
    )


def main() raises:
    comptime assert has_accelerator()
    var ctx = DeviceContext()
    var sizes_mib: List[Int] = [64, 256, 512, 1024, 2048]

    for mib in sizes_mib:
        var nbytes = mib * 1024 * 1024

        var dev = ctx.enqueue_create_buffer[dtype](nbytes)
        var pinned = ctx.enqueue_create_host_buffer[dtype](nbytes)
        var pageable_alloc = alloc(Layout[Scalar[dtype]](count=nbytes))
        var pageable_ptr = pageable_alloc.unsafe_ptr()
        try:
            ctx.enqueue_memset(dev, 0)
            for i in range(nbytes):
                pinned[i] = 0
                pageable_ptr[unsafe_offset=i] = 0
            ctx.synchronize()

            # warm-up, not timed: first transfer at a new size pays page-fault /
            # allocator cold-start cost that a steady-state prefetcher would not.
            for _ in range(WARM):
                ctx.enqueue_copy(dev, pinned.unsafe_ptr())
                ctx.enqueue_copy(pinned.unsafe_ptr(), dev)
            ctx.synchronize()

            # pinned H2D
            var t_h2d_pinned = List[Float64]()
            for _ in range(REPS):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(dev, pinned.unsafe_ptr())
                ctx.synchronize()
                t_h2d_pinned.append(Float64(perf_counter_ns() - t0) / 1e9)
            report_row("H2D pinned  ", nbytes, t_h2d_pinned)

            # pinned D2H
            var t_d2h_pinned = List[Float64]()
            for _ in range(REPS):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(pinned.unsafe_ptr(), dev)
                ctx.synchronize()
                t_d2h_pinned.append(Float64(perf_counter_ns() - t0) / 1e9)
            report_row("D2H pinned  ", nbytes, t_d2h_pinned)

            # pageable H2D
            var t_h2d_pageable = List[Float64]()
            for _ in range(REPS):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(dev, pageable_ptr)
                ctx.synchronize()
                t_h2d_pageable.append(Float64(perf_counter_ns() - t0) / 1e9)
            report_row("H2D pageable", nbytes, t_h2d_pageable)

            # pageable D2H
            var t_d2h_pageable = List[Float64]()
            for _ in range(REPS):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(pageable_ptr, dev)
                ctx.synchronize()
                t_d2h_pageable.append(Float64(perf_counter_ns() - t0) / 1e9)
            report_row("D2H pageable", nbytes, t_d2h_pageable)

            # "both at once": a separate device buffer for the D2H leg so the two
            # copies do not race on the same bytes; both enqueued before one
            # synchronize. Pinned only (what a real prefetcher would use). Caveat
            # in the module docstring: this API has no stream selector, so this
            # is two async enqueues drained together, not a proven concurrent
            # transfer -- report the number plainly, do not read it as 2x proof
            # of independent copy engines.
            var dev2 = ctx.enqueue_create_buffer[dtype](nbytes)
            var pinned2 = ctx.enqueue_create_host_buffer[dtype](nbytes)
            ctx.enqueue_memset(dev2, 0)
            for i in range(nbytes):
                pinned2[i] = 0
            ctx.synchronize()
            var t_both = List[Float64]()
            for _ in range(REPS):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(dev, pinned.unsafe_ptr())
                ctx.enqueue_copy(pinned2.unsafe_ptr(), dev2)
                ctx.synchronize()
                t_both.append(Float64(perf_counter_ns() - t0) / 1e9)
            report_row("both(H2D+D2H)", 2 * nbytes, t_both)
        finally:
            dealloc(pageable_alloc^)
        print("---")
