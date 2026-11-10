"""Parity: amar_attn_decode_gqa (one block per KV head and row, the G query heads
of that KV head share every K/V load) against amar_attn_decode (one block per
query head and row), BIT-EXACT on every output.

Prefill attention runs the decode kernel once per (head, row) so that batched
prefill keeps the replay path's tokens; the GQA kernel has to keep that byte
for byte. KV pages are written by the real append kernel through a reversed
page table, the lengths cross tile (256) and page (128) boundaries and leave a
tail that is not a multiple of 8. Build with -D BARO_MODEL=qwen35moe for the
int8 KV of the MoE model, without for the dense model's f32 KV.
"""
from std.math import ceildiv
from std.sys import has_accelerator, exit

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from attn import amar_attn_decode, amar_attn_decode_gqa, amar_kv_append2, KVT, KVPAGE, KVHSTR, TCAP
from model import HD, NQH, NKVH

comptime f32 = DType.float32
comptime NAT = 3
comptime ATT_I = 1
comptime M = 6
comptime P0 = 2777
comptime T = P0 + M
comptime PAGES = ceildiv(T, KVPAGE)
comptime CH = 512

comptime cache_l = row_major[TCAP]()
comptime q_l = row_major[M * NQH, HD]()
comptime n_l = row_major[CH * NKVH, HD]()


def view[
    dt: DType, LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[dt], n: Int, lt: LT
) -> TileTensor[dt, LT, MutAnyOrigin]:
    var s = DeviceBuffer[dt](ctx, b.unsafe_ptr(), n, owning=False)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](TileTensor(s, lt))


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var seed = UInt64(0xA77E)
    comptime NKV = PAGES * NAT * NKVH * KVHSTR
    var kc = ctx.enqueue_create_buffer[KVT](NKV)
    var vc = ctx.enqueue_create_buffer[KVT](NKV)
    ctx.enqueue_memset(kc, 0)
    ctx.enqueue_memset(vc, 0)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](PAGES)
    var kn_h = ctx.enqueue_create_host_buffer[f32](CH * NKVH * HD)
    var vn_h = ctx.enqueue_create_host_buffer[f32](CH * NKVH * HD)
    var q_h = ctx.enqueue_create_host_buffer[f32](M * NQH * HD)
    var o1_h = ctx.enqueue_create_host_buffer[f32](M * NQH * HD)
    var o2_h = ctx.enqueue_create_host_buffer[f32](M * NQH * HD)
    ctx.synchronize()
    for p in range(PAGES):
        tab_h[p] = Int32(PAGES - 1 - p)
    var tab = ctx.enqueue_create_buffer[DType.int32](PAGES)
    ctx.enqueue_copy(dst_buf=tab, src_buf=tab_h)
    var kn = ctx.enqueue_create_buffer[f32](CH * NKVH * HD)
    var vn = ctx.enqueue_create_buffer[f32](CH * NKVH * HD)
    var q = ctx.enqueue_create_buffer[f32](M * NQH * HD)
    var o1 = ctx.enqueue_create_buffer[f32](M * NQH * HD)
    var o2 = ctx.enqueue_create_buffer[f32](M * NQH * HD)
    var Kc = view(ctx, kc, NKV, cache_l)
    var Vc = view(ctx, vc, NKV, cache_l)
    var t = 0
    while t < T:
        var rows = min(CH, T - t)
        for i in range(CH * NKVH * HD):
            seed = seed * 6364136223846793005 + 1442695040888963407
            kn_h[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 400.0
            seed = seed * 6364136223846793005 + 1442695040888963407
            vn_h[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 700.0
        ctx.enqueue_copy(dst_buf=kn, src_buf=kn_h)
        ctx.enqueue_copy(dst_buf=vn, src_buf=vn_h)
        ctx.enqueue_function[amar_kv_append2[type_of(cache_l), type_of(n_l), NAT]](
            Kc, Vc, view(ctx, kn, CH * NKVH * HD, n_l), view(ctx, vn, CH * NKVH * HD, n_l), tab.unsafe_ptr(),
            Int32(t), Int32(ATT_I), grid_dim=(NKVH, rows, 2), block_dim=HD)
        ctx.synchronize()
        t += rows
    for i in range(M * NQH * HD):
        seed = seed * 6364136223846793005 + 1442695040888963407
        q_h[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 300.0
    ctx.enqueue_copy(dst_buf=q, src_buf=q_h)
    var Q = view(ctx, q, M * NQH * HD, q_l)
    ctx.enqueue_function[amar_attn_decode[type_of(q_l), type_of(cache_l), type_of(q_l), NAT]](
        Q, Kc, Vc, view(ctx, o1, M * NQH * HD, q_l), tab.unsafe_ptr(), Int32(P0 + 1), Float32(0.0625), Int32(ATT_I),
        grid_dim=(NQH, M), block_dim=HD)
    ctx.enqueue_function[amar_attn_decode_gqa[type_of(q_l), type_of(cache_l), type_of(q_l), NAT]](
        Q, Kc, Vc, view(ctx, o2, M * NQH * HD, q_l), tab.unsafe_ptr(), Int32(P0 + 1), Float32(0.0625), Int32(ATT_I),
        grid_dim=(NKVH, M), block_dim=HD)
    ctx.enqueue_copy(dst_buf=o1_h, src_buf=o1)
    ctx.enqueue_copy(dst_buf=o2_h, src_buf=o2)
    ctx.synchronize()
    var diff = 0
    var finite = 0
    for i in range(M * NQH * HD):
        if o1_h[i].to_bits() != o2_h[i].to_bits():
            diff += 1
        if abs(o1_h[i]) < 1e30:
            finite += 1
    print("attn gqa vs decode:", M * NQH * HD, "outputs,", diff, "differ,", finite, "finite; KV", KVT, " G", NQH // NKVH, " T", T)
    if diff != 0 or finite != M * NQH * HD:
        print("FAIL: amar_attn_decode_gqa not byte-equal to amar_attn_decode")
        exit(1)
    print("PASS: amar_attn_decode_gqa byte-equal to amar_attn_decode")
