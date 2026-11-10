"""W3 gate 1: route one real MoE block from the loaded qwen35moe pack.

The router, routed experts, and shared expert all read the single Pack.wbuf
blob. Expert offsets are resolved by tensor name, never by the dense off list.
"""
from std.math import ceildiv
from std.memory import alloc, unsafe_memcpy
from std.os import getenv
from std.sys import has_accelerator

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from harness import load_pack
from moe_pack import parse_moe_index, resolve_expert, resolve_plain
from matmul_skinny import amar_matmul_skinny_m1_row, ROW_WAVES, ROW_THREADS
from ssm import amar_cast_bf16
from moe import (
    amar_moe_router_top8, amar_moe_sig_gate, moe_gate_up_q4k_pack, amar_moe_down_q4k,
    moe_gate_up_q8_0, moe_down_q8_0,
    N_EXP, TOPK, E_FFN, MOE_H, MOE_WAVES, MOE_THREADS, SH_FFN,
)

comptime H = MOE_H
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime x_1 = row_major[H]()
comptime x_2 = row_major[1, H]()
comptime idx_1 = row_major[TOPK]()
comptime wt_1 = row_major[TOPK]()
comptime one_1 = row_major[1]()
comptime h_1 = row_major[TOPK * E_FFN]()
comptime h_2 = row_major[TOPK, E_FFN]()
comptime hs_1 = row_major[SH_FFN]()
comptime hs_2 = row_major[1, SH_FFN]()
comptime o_1 = row_major[H]()
comptime router_w = row_major[N_EXP, H]()
comptime router_l = row_major[N_EXP]()


def load_into(path: String, dst: MutPointer[UInt8, MutUntrackedOrigin], size: Int) raises:
    with open(path, "r") as f:
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("size mismatch for " + path)
        unsafe_memcpy(dest=dst, src=data.unsafe_ptr(), count=size)


def check(name: String, got: MutPointer[Float32, MutUntrackedOrigin], want: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises:
    var worst = Float64(0)
    var wi = 0
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        var rel = e / (abs(Float64(want[unsafe_offset=i])) + 1e-2)
        if rel > worst:
            worst = rel
            wi = i
    print(name, "max_rel:", worst, "at", wi)
    if worst > 5e-3:
        raise Error("gate 1 parity failure: " + name)


def check_ids(got: MutPointer[Int32, MutUntrackedOrigin], want: MutPointer[Int32, MutUntrackedOrigin]) raises:
    for i in range(TOPK):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            raise Error("gate 1 expert id mismatch at " + String(i))
    print("gate 1 expert ids exact over", TOPK)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/moe-w1/pack")
    var pack = load_pack(ctx, packdir)
    var tensors = parse_moe_index(packdir + "/index.txt")
    var layer = atol(getenv("BARO_MOE_LAYER", "3"))
    var rg = resolve_expert(tensors, layer, "gate", False, 0, 0, E_FFN, E_FFN, H)
    var ru = resolve_expert(tensors, layer, "up", False, 0, 0, E_FFN, E_FFN, H)
    var rd = resolve_expert(tensors, layer, "down", False, 0, 0, H, H, E_FFN)
    var sg = resolve_expert(tensors, layer, "gate", True, 0, 0, SH_FFN, SH_FFN, H)
    var su = resolve_expert(tensors, layer, "up", True, 0, 0, SH_FFN, SH_FFN, H)
    var sd = resolve_expert(tensors, layer, "down", True, 0, 0, H, H, SH_FFN)
    var sgi = resolve_plain(tensors, "blk." + String(layer) + ".ffn_gate_inp_shexp.weight")
    var router = resolve_plain(tensors, "blk." + String(layer) + ".ffn_gate_inp.weight")

    var x_h = ctx.enqueue_create_host_buffer[f32](H)
    var idx_ref = alloc[Int32](TOPK)
    var wt_ref = alloc[Float32](TOPK)
    var routed_ref = alloc[Float32](H)
    var shared_ref = alloc[Float32](H)
    var y_ref = alloc[Float32](H)
    load_into(".work/gguf/moe_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), H * 4)
    load_into(".work/gguf/moe_idx_ref.bin", idx_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_into(".work/gguf/moe_w_ref.bin", wt_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_into(".work/gguf/moe_routed_ref.bin", routed_ref.unsafe_bitcast[UInt8](), H * 4)
    load_into(".work/gguf/moe_shared_ref.bin", shared_ref.unsafe_bitcast[UInt8](), H * 4)
    load_into(".work/gguf/moe_y_ref.bin", y_ref.unsafe_bitcast[UInt8](), H * 4)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var xb_d = ctx.enqueue_create_buffer[bf16](H)
    var logits_d = ctx.enqueue_create_buffer[f32](N_EXP)
    var idx_d = ctx.enqueue_create_buffer[i32](TOPK)
    var wt_d = ctx.enqueue_create_buffer[f32](TOPK)
    var xb = TileTensor(xb_d, x_2)
    var xb_cast = TileTensor(xb_d, x_1)
    var x = TileTensor(x_d, x_2)
    var idx = TileTensor(idx_d, idx_1)
    var wt = TileTensor(wt_d, wt_1)
    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.synchronize()
    comptime cast_x = amar_cast_bf16[type_of(x_1), type_of(x_1)]
    ctx.enqueue_function[cast_x](TileTensor(x_d, x_1), xb_cast, Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)

    var rw = DeviceBuffer[f32](ctx, (pack.wbuf.unsafe_ptr() + router.offset).unsafe_bitcast[Scalar[f32]](), N_EXP * H, owning=False)
    ctx.enqueue_function[amar_matmul_skinny_m1_row[f32, 2, type_of(x_2), type_of(router_w), type_of(router_l)]](
        x, TileTensor(rw, router_w), TileTensor(logits_d, router_l), Int32(N_EXP), Int32(H),
        grid_dim=ceildiv(N_EXP, ROW_WAVES), block_dim=ROW_THREADS,
    )
    ctx.enqueue_function[amar_moe_router_top8[type_of(router_l), type_of(idx_1), type_of(wt_1)]](
        TileTensor(logits_d, router_l), idx, wt, grid_dim=1, block_dim=N_EXP,
    )

    var h_d = ctx.enqueue_create_buffer[f32](TOPK * E_FFN)
    var hb_d = ctx.enqueue_create_buffer[bf16](TOPK * E_FFN)
    var routed_d = ctx.enqueue_create_buffer[f32](H)
    var hs_d = ctx.enqueue_create_buffer[f32](SH_FFN)
    var hsb_d = ctx.enqueue_create_buffer[bf16](SH_FFN)
    var shared_d = ctx.enqueue_create_buffer[f32](H)
    var sg_d = ctx.enqueue_create_buffer[f32](1)
    var idx0_d = ctx.enqueue_create_buffer[i32](1)
    var idx0_h = ctx.enqueue_create_host_buffer[i32](1)
    idx0_h[0] = 0
    ctx.enqueue_copy(dst_buf=idx0_d, src_buf=idx0_h)
    var idx0 = TileTensor(idx0_d, one_1)

    var qg = pack.wbuf.unsafe_ptr() + rg.byte_offset
    var qu = pack.wbuf.unsafe_ptr() + ru.byte_offset
    var qd = pack.wbuf.unsafe_ptr() + rd.byte_offset
    ctx.enqueue_function[moe_gate_up_q4k_pack[TOPK, E_FFN, type_of(x_2), type_of(idx_1), type_of(h_1)]](
        xb, qg, idx, TileTensor(h_d, h_1), Int32(H), Int32(ru.byte_offset - rg.byte_offset),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    comptime cast_h = amar_cast_bf16[type_of(h_1), type_of(h_1)]
    ctx.enqueue_function[cast_h](TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN), grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256)
    ctx.enqueue_function[amar_moe_down_q4k[TOPK, E_FFN, type_of(h_2), type_of(idx_1), type_of(wt_1), type_of(o_1)]](
        TileTensor(hb_d, h_2), qd, idx, wt, TileTensor(routed_d, o_1), Int32(H),
        grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS,
    )

    var qsg = pack.wbuf.unsafe_ptr() + sg.byte_offset
    var qsu = pack.wbuf.unsafe_ptr() + su.byte_offset
    var qsd = pack.wbuf.unsafe_ptr() + sd.byte_offset
    var sgw = DeviceBuffer[f32](ctx, (pack.wbuf.unsafe_ptr() + sgi.offset).unsafe_bitcast[Scalar[f32]](), H, owning=False)
    ctx.enqueue_function[amar_moe_sig_gate[type_of(x_1), type_of(x_1), type_of(one_1)]](
        TileTensor(x_d, x_1), TileTensor(sgw, x_1), TileTensor(sg_d, one_1), Int32(H), grid_dim=1, block_dim=32,
    )
    ctx.enqueue_function[moe_gate_up_q8_0[1, SH_FFN, type_of(x_2), type_of(one_1), type_of(hs_1)]](
        xb, qsg, idx0, TileTensor(hs_d, hs_1), Int32(H), Int32(su.row_bytes), Int32(su.byte_offset - sg.byte_offset),
        grid_dim=ceildiv(SH_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    comptime cast_hs = amar_cast_bf16[type_of(hs_1), type_of(hs_1)]
    ctx.enqueue_function[cast_hs](TileTensor(hs_d, hs_1), TileTensor(hsb_d, hs_1), Int32(SH_FFN), grid_dim=ceildiv(SH_FFN, 256), block_dim=256)
    ctx.enqueue_function[moe_down_q8_0[1, SH_FFN, type_of(hs_2), type_of(one_1), type_of(one_1), type_of(o_1)]](
        TileTensor(hsb_d, hs_2), qsd, idx0, TileTensor(sg_d, one_1), TileTensor(shared_d, o_1), Int32(H), Int32(sd.row_bytes),
        grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()

    var idx_got = ctx.enqueue_create_host_buffer[i32](TOPK)
    var wt_got = ctx.enqueue_create_host_buffer[f32](TOPK)
    var routed_got = ctx.enqueue_create_host_buffer[f32](H)
    var shared_got = ctx.enqueue_create_host_buffer[f32](H)
    ctx.enqueue_copy(dst_buf=idx_got, src_buf=idx_d)
    ctx.enqueue_copy(dst_buf=wt_got, src_buf=wt_d)
    ctx.enqueue_copy(dst_buf=routed_got, src_buf=routed_d)
    ctx.enqueue_copy(dst_buf=shared_got, src_buf=shared_d)
    ctx.synchronize()
    check_ids(idx_got.unsafe_ptr(), idx_ref)
    check("routing_weights", wt_got.unsafe_ptr(), wt_ref, TOPK)
    check("routed", routed_got.unsafe_ptr(), routed_ref, H)
    check("shared", shared_got.unsafe_ptr(), shared_ref, H)
    var y_got = alloc[Float32](H)
    for i in range(H):
        y_got[unsafe_offset=i] = routed_got[i] + shared_got[i]
    check("y", y_got, y_ref, H)
    print("PASS: W3 gate 1 loaded routed and shared Q4_K/Q8_0 experts from Pack.wbuf")
