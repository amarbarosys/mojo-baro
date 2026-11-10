"""Numerics probe for the generic decode attention (bench/dattn-protocol.md gate 2).
deterministic inputs (op_bench's fill formula, V with a seed offset), one instantiation
per shape, output dumped for tools/dattn-ref.py's fp64 reference.
usage: test_dattn SHAPE T PATH NS NLD ROT M QSCALE OUT
  SHAPE S0 (shipped 256/16/4 f32) | S1 (256/16/4 f16) | S2 (64/40/8 f16) | S3 (128/28/4 f16)
  PATH exact | split; NS requested split count (clamped to the span count); NLD 2 | 4 | 8; ROT 0 | 1 (per-block start rotation); M query rows, row r attends to T + r tokens
"""
from std.sys import argv, has_accelerator

from max.gpu.host import DeviceContext, HostBuffer

from dattn import f32, dspan
from dattn_harness import fill_q, fill_kv, kv_pool_elems, eff_nsplit, launch_dattn

comptime f16 = DType.float16


def run[
    HD: Int, NQH: Int, NKVH: Int, KVT: DType, NLD: Int, ROT: Bool
](T: Int, exact: Bool, ns_req: Int, M: Int, qscale: Int, out_path: String) raises:
    var ctx = DeviceContext()
    var ns = 1 if exact else eff_nsplit[HD, NLD](T, ns_req)
    var TT = T + M - 1
    var pool = kv_pool_elems[HD, NKVH](TT)
    var q_h = ctx.enqueue_create_host_buffer[f32](M * NQH * HD)
    var k_h = ctx.enqueue_create_host_buffer[KVT](pool)
    var v_h = ctx.enqueue_create_host_buffer[KVT](pool)
    var o_h = ctx.enqueue_create_host_buffer[f32](M * NQH * HD)
    ctx.synchronize()
    fill_q(q_h, M * NQH * HD, qscale)
    fill_kv[HD, NKVH, KVT](k_h, TT, 0)
    fill_kv[HD, NKVH, KVT](v_h, TT, 1000003)
    var q_d = ctx.enqueue_create_buffer[f32](M * NQH * HD)
    var k_d = ctx.enqueue_create_buffer[KVT](pool)
    var v_d = ctx.enqueue_create_buffer[KVT](pool)
    var o_d = ctx.enqueue_create_buffer[f32](M * NQH * HD)
    var p_d = ctx.enqueue_create_buffer[f32](M * NQH * ns * (HD + 2))
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=k_d, src_buf=k_h)
    ctx.enqueue_copy(dst_buf=v_d, src_buf=v_h)
    o_d.enqueue_fill(0)
    launch_dattn[HD, NQH, NKVH, KVT, NLD, ROT](ctx, q_d, k_d, v_d, o_d, p_d, T, ns, exact, M)
    ctx.enqueue_copy(dst_buf=o_h, src_buf=o_d)
    ctx.synchronize()
    print(
        "echo HD", HD, "NQH", NQH, "NKVH", NKVH, "KVT", "f16" if KVT == f16 else "f32",
        "T", T, "path", "exact" if exact else "split", "nsplit", ns, "nld", NLD,
        "rot", 1 if ROT else 0, "m", M, "span", dspan[HD, NLD](), "qscale", qscale,
    )
    with open(out_path, "w") as f:
        var p = o_h.unsafe_ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=M * NQH * HD * 4))


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var a = argv()
    if len(a) != 10:
        raise Error("usage: test_dattn SHAPE T PATH NS NLD ROT M QSCALE OUT")
    var shape = String(a[1])
    var T = Int(String(a[2]))
    var exact = String(a[3]) == "exact"
    var ns = Int(String(a[4]))
    var nld = Int(String(a[5]))
    var rot = String(a[6]) == "1"
    var M = Int(String(a[7]))
    var qscale = Int(String(a[8]))
    var out_path = String(a[9])
    if shape == "S0":
        if nld == 2 and rot:
            run[256, 16, 4, f32, 2, True](T, exact, ns, M, qscale, out_path)
        elif nld == 2 and not rot:
            run[256, 16, 4, f32, 2, False](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and rot:
            run[256, 16, 4, f32, 4, True](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and not rot:
            run[256, 16, 4, f32, 4, False](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and rot:
            run[256, 16, 4, f32, 8, True](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and not rot:
            run[256, 16, 4, f32, 8, False](T, exact, ns, M, qscale, out_path)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S1":
        if nld == 2 and rot:
            run[256, 16, 4, f16, 2, True](T, exact, ns, M, qscale, out_path)
        elif nld == 2 and not rot:
            run[256, 16, 4, f16, 2, False](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and rot:
            run[256, 16, 4, f16, 4, True](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and not rot:
            run[256, 16, 4, f16, 4, False](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and rot:
            run[256, 16, 4, f16, 8, True](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and not rot:
            run[256, 16, 4, f16, 8, False](T, exact, ns, M, qscale, out_path)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S2":
        if nld == 2 and rot:
            run[64, 40, 8, f16, 2, True](T, exact, ns, M, qscale, out_path)
        elif nld == 2 and not rot:
            run[64, 40, 8, f16, 2, False](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and rot:
            run[64, 40, 8, f16, 4, True](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and not rot:
            run[64, 40, 8, f16, 4, False](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and rot:
            run[64, 40, 8, f16, 8, True](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and not rot:
            run[64, 40, 8, f16, 8, False](T, exact, ns, M, qscale, out_path)
        else:
            raise Error("NLD must be 2, 4 or 8")
    elif shape == "S3":
        if nld == 2 and rot:
            run[128, 28, 4, f16, 2, True](T, exact, ns, M, qscale, out_path)
        elif nld == 2 and not rot:
            run[128, 28, 4, f16, 2, False](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and rot:
            run[128, 28, 4, f16, 4, True](T, exact, ns, M, qscale, out_path)
        elif nld == 4 and not rot:
            run[128, 28, 4, f16, 4, False](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and rot:
            run[128, 28, 4, f16, 8, True](T, exact, ns, M, qscale, out_path)
        elif nld == 8 and not rot:
            run[128, 28, 4, f16, 8, False](T, exact, ns, M, qscale, out_path)
        else:
            raise Error("NLD must be 2, 4 or 8")
    else:
        raise Error("unknown shape " + shape)
