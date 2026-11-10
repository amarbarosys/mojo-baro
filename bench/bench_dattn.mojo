"""Arm O of bench/dattn-protocol.md: the generic decode attention, cache-cold.

Same rotation rule as bench/ggml-harness/op_bench.c: K/V pools (one per arm, identical
content from op_bench's fill formula) rotate over >= 4 x 96 MB so every iteration streams
from HBM. Every arm-defining parameter is echoed before timing (P1). The wall number is a
sanity receipt only; device time per iteration comes from rocprofv3 (bench/dattn-run.sh),
summed over the split and combine kernels the way the R arm's catalog sums ext_vec + combine.
usage: bench_dattn SHAPE T ITERS NS NLD ROT PATH   (SHAPE S0 (256/16/4 f32, the engine) | S1|S2|S3; NLD 2|4|8 loads per span; ROT 1 = per-block start rotation; PATH exact|split)
"""
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer

from dattn import f32, dspan, DTHREADS
from dattn_harness import fill_q, fill_kv, kv_pool_elems, eff_nsplit, launch_dattn

comptime f16 = DType.float16
comptime IC_BYTES = 96 * 1024 * 1024
comptime WARMUP = 20
comptime MAX_ARMS = 256


def run[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NLD: Int, ROT: Bool
](T: Int, iters: Int, ns_req: Int, exact: Bool) raises:
    var ctx = DeviceContext()
    var ns = 1 if exact else eff_nsplit[HD, NLD](T, ns_req)
    var pool = kv_pool_elems[HD, NKVH](T)
    var arm_bytes = 2 * pool * (2 if KVT == f16 else 4)
    var arms = (4 * IC_BYTES + arm_bytes - 1) // arm_bytes
    if arms < 1:
        arms = 1
    if arms > MAX_ARMS:
        arms = MAX_ARMS
    var q_h = ctx.enqueue_create_host_buffer[f32](NQH * HD)
    var kv_h = ctx.enqueue_create_host_buffer[KVT](pool)
    ctx.synchronize()
    fill_q(q_h, NQH * HD, 1)
    fill_kv[HD, NKVH, KVT](kv_h, T, 0)
    var q_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var o_d = ctx.enqueue_create_buffer[f32](NQH * HD)
    var p_d = ctx.enqueue_create_buffer[f32](NQH * ns * (HD + 2))
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    var ks = List[DeviceBuffer[KVT]]()
    var vs = List[DeviceBuffer[KVT]]()
    for _ in range(arms):
        var k = ctx.enqueue_create_buffer[KVT](pool)
        var v = ctx.enqueue_create_buffer[KVT](pool)
        ctx.enqueue_copy(dst_buf=k, src_buf=kv_h)
        ctx.enqueue_copy(dst_buf=v, src_buf=kv_h)
        ks.append(k^)
        vs.append(v^)
    ctx.synchronize()
    var rotated = Float64(arms * arm_bytes) / 1e6
    print(
        "arm dattn HD=" + String(HD) + " NQH=" + String(NQH) + " NKVH=" + String(NKVH)
        + " KVT=" + ("f16" if KVT == f16 else "f32") + " KV=" + String(T)
        + " path=" + ("exact" if exact else "split") + " nsplit=" + String(ns)
        + " nld=" + String(NLD) + " nw=8 rot=" + String(1 if ROT else 0) + " span=" + String(dspan[HD, NLD]())
        + " | grid " + ("(" + String(NQH) + ") block " + String(HD) if exact else "(" + String(NKVH) + "," + String(ns) + ") block 256")
        + (" | combine grid " + String(NQH) + " block " + String(DTHREADS) if (not exact and ns > 1) else " | no combine")
        + " | bytes/arm " + String(Float64(arm_bytes) / 1e6) + " MB | arms " + String(arms)
        + " | rotated " + String(rotated) + " MB ("
        + ("exceeds" if arms * arm_bytes >= 4 * IC_BYTES else "BELOW, cache-warm, INVALID")
        + " the 96 MB Infinity Cache x4)"
    )
    for i in range(WARMUP):
        launch_dattn[HD, NQH, NKVH, KVT, NLD, ROT](ctx, q_d, ks[i % arms], vs[i % arms], o_d, p_d, T, ns, exact)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for i in range(iters):
        launch_dattn[HD, NQH, NKVH, KVT, NLD, ROT](ctx, q_d, ks[i % arms], vs[i % arms], o_d, p_d, T, ns, exact)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(iters)
    var moved = Float64(arm_bytes + 2 * 4 * NQH * HD) / 1e6
    print(
        "wall " + String(us) + " us/iter  moved " + String(moved) + " MB  "
        + String(moved / us * 1e3) + " GB/s wall (device time: rocprofv3, bench/dattn-run.sh)"
    )


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var a = argv()
    if len(a) != 8:
        raise Error("usage: bench_dattn SHAPE T ITERS NS NLD ROT PATH")
    var shape = String(a[1])
    var T = Int(String(a[2]))
    var iters = Int(String(a[3]))
    var ns = Int(String(a[4]))
    var nld = Int(String(a[5]))
    var rot = String(a[6]) == "1"
    var exact = String(a[7]) == "exact"
    if shape == "S1":
        if nld == 2 and rot:
            run[256, 16, 4, f16, 2, True](T, iters, ns, exact)
        elif nld == 2 and not rot:
            run[256, 16, 4, f16, 2, False](T, iters, ns, exact)
        elif nld == 4 and rot:
            run[256, 16, 4, f16, 4, True](T, iters, ns, exact)
        elif nld == 4 and not rot:
            run[256, 16, 4, f16, 4, False](T, iters, ns, exact)
        elif nld == 8 and rot:
            run[256, 16, 4, f16, 8, True](T, iters, ns, exact)
        elif nld == 8 and not rot:
            run[256, 16, 4, f16, 8, False](T, iters, ns, exact)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S2":
        if nld == 2 and rot:
            run[64, 40, 8, f16, 2, True](T, iters, ns, exact)
        elif nld == 2 and not rot:
            run[64, 40, 8, f16, 2, False](T, iters, ns, exact)
        elif nld == 4 and rot:
            run[64, 40, 8, f16, 4, True](T, iters, ns, exact)
        elif nld == 4 and not rot:
            run[64, 40, 8, f16, 4, False](T, iters, ns, exact)
        elif nld == 8 and rot:
            run[64, 40, 8, f16, 8, True](T, iters, ns, exact)
        elif nld == 8 and not rot:
            run[64, 40, 8, f16, 8, False](T, iters, ns, exact)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S3":
        if nld == 2 and rot:
            run[128, 28, 4, f16, 2, True](T, iters, ns, exact)
        elif nld == 2 and not rot:
            run[128, 28, 4, f16, 2, False](T, iters, ns, exact)
        elif nld == 4 and rot:
            run[128, 28, 4, f16, 4, True](T, iters, ns, exact)
        elif nld == 4 and not rot:
            run[128, 28, 4, f16, 4, False](T, iters, ns, exact)
        elif nld == 8 and rot:
            run[128, 28, 4, f16, 8, True](T, iters, ns, exact)
        elif nld == 8 and not rot:
            run[128, 28, 4, f16, 8, False](T, iters, ns, exact)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S0":
        if nld == 2 and rot:
            run[256, 16, 4, f32, 2, True](T, iters, ns, exact)
        elif nld == 2 and not rot:
            run[256, 16, 4, f32, 2, False](T, iters, ns, exact)
        elif nld == 4 and rot:
            run[256, 16, 4, f32, 4, True](T, iters, ns, exact)
        elif nld == 4 and not rot:
            run[256, 16, 4, f32, 4, False](T, iters, ns, exact)
        elif nld == 8 and rot:
            run[256, 16, 4, f32, 8, True](T, iters, ns, exact)
        elif nld == 8 and not rot:
            run[256, 16, 4, f32, 8, False](T, iters, ns, exact)
        else:
            raise Error("NLD must be 2, 4 or 8")
    else:
        raise Error("unknown shape " + shape)
