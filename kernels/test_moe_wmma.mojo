"""MoE prefill WMMA arm (L5X-2 phase 2), two checks on random data at the engine's chunk shape.

1. moe_pairs_group (one thread per expert) against a host reference: GE, GP and S byte-equal
   at m = 5, 777, 1024 and group sizes 8 and 32, skewed routing (hot experts span many groups).
2. moe_wmma_q4k_grouped (gate+up+swiglu, down) against the exact grouped row kernels
   (moe_gate_up_q4k_grouped / moe_down_q4k_grouped, m=1 dot order). Not bit-exact by design:
   the weights are dequantized with the row kernel's own bf16 expression, so every product is
   equal and only the f32 summation order differs. Bar: max |diff| / max |ref| <= 2e-2 on the
   bf16 swiglu output and <= 1e-3 on the f32 down output, and >= 90% of bf16 outputs equal.
   Prints both kernels' times (median of 9, weights 453 MB > the 96 MB Infinity Cache).
"""
from std.math import ceildiv, abs, max
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from moe import N_EXP, TOPK, E_FFN, MOE_H, MOE_WAVES, MOE_THREADS
from moe_rows import moe_pairs_group, moe_gate_up_q4k_grouped, moe_down_q4k_grouped
from moe_wmma import moe_wmma_q4k_grouped, MW_THREADS

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime H = MOE_H
comptime CP = 1024
comptime NQ = CP * TOPK
comptime MR8 = 8
comptime MR32 = 32
comptime NG8 = NQ // MR8 + N_EXP
comptime NG32 = NQ // MR32 + N_EXP
comptime idx_layout = row_major[CP, TOPK]()
comptime ge8_layout = row_major[NG8]()
comptime gp8_layout = row_major[NG8 * MR8]()
comptime ge32_layout = row_major[NG32]()
comptime gp32_layout = row_major[NG32 * MR32]()
comptime gs_layout = row_major[2 * N_EXP]()
comptime x_layout = row_major[CP, H]()
comptime hh_layout = row_major[CP * TOPK, E_FFN]()
comptime dn_layout = row_major[CP * TOPK, H]()
comptime Q4_ROW = (H // 256) * 144
comptime Q4_DROW = (E_FFN // 256) * 144
comptime GU_BYTES = 2 * N_EXP * E_FFN * Q4_ROW
comptime DN_BYTES = N_EXP * H * Q4_DROW
comptime WM = 2
comptime WN = 4
comptime TM = 1
comptime TN = 2
comptime BN = WN * TN * 16


struct Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return self.s >> 33


def fill_q4k(mut g: Lcg, p: MutPointer[UInt8, MutUntrackedOrigin], n: Int):
    for i in range(n):
        p[unsafe_offset=i] = UInt8(Int(g.next() & 0xFF))
    for b in range(n // 144):
        p[unsafe_offset=b * 144 + 1] = UInt8(0x28 + Int(g.next() & 0x3))
        p[unsafe_offset=b * 144 + 3] = UInt8(0x24 + Int(g.next() & 0x3))


def route(mut g: Lcg, p: MutPointer[Int32, MutUntrackedOrigin], m: Int):
    for t in range(m):
        var j = 0
        while j < TOPK:
            var e = Int(g.next() % 256) * Int(g.next() % 256) // 256
            var dup = False
            for k in range(j):
                if Int(p[unsafe_offset=t * TOPK + k]) == e:
                    dup = True
            if not dup:
                p[unsafe_offset=t * TOPK + j] = Int32(e)
                j += 1


def ref_groups(idx: MutPointer[Int32, MutUntrackedOrigin], m: Int, mr: Int, ng: Int,
               mut ge: List[Int32], mut gp: List[Int32], mut s: List[Int32]):
    ge = List[Int32](length=ng, fill=-1)
    gp = List[Int32](length=ng * mr, fill=-1)
    s = List[Int32](length=2 * N_EXP, fill=0)
    var g = 0
    for e in range(N_EXP):
        var c = 0
        for q in range(m * TOPK):
            if Int(idx[unsafe_offset=q]) == e:
                gp[g * mr + c] = Int32(q)
                c += 1
        var n = (c + mr - 1) // mr
        for k in range(n):
            ge[g + k] = Int32(e)
        s[e] = Int32(c)
        s[N_EXP + e] = Int32(g * mr + c)
        g += n


def check_groups[MR: Int, NG: Int](ctx: DeviceContext, mut g: Lcg, m: Int) raises -> Int:
    comptime gel = row_major[NG]()
    comptime gpl = row_major[NG * MR]()
    var idx_h = ctx.enqueue_create_host_buffer[i32](CP * TOPK)
    ctx.synchronize()
    for i in range(CP * TOPK):
        idx_h[i] = 0
    route(g, idx_h.unsafe_ptr(), m)
    var idx_d = ctx.enqueue_create_buffer[i32](CP * TOPK)
    var ge_d = ctx.enqueue_create_buffer[i32](NG)
    var gp_d = ctx.enqueue_create_buffer[i32](NG * MR)
    var gs_d = ctx.enqueue_create_buffer[i32](2 * N_EXP)
    ctx.enqueue_copy(dst_buf=idx_d, src_buf=idx_h)
    var ng = m * TOPK // MR + N_EXP
    ctx.enqueue_function[moe_pairs_group[MR, NQ, type_of(idx_layout), type_of(gel), type_of(gpl), type_of(gs_layout)]](
        TileTensor(idx_d, idx_layout), TileTensor(ge_d, gel), TileTensor(gp_d, gpl), TileTensor(gs_d, gs_layout),
        Int32(m), Int32(ng), grid_dim=1, block_dim=N_EXP)
    var ge_h = ctx.enqueue_create_host_buffer[i32](NG)
    var gp_h = ctx.enqueue_create_host_buffer[i32](NG * MR)
    var gs_h = ctx.enqueue_create_host_buffer[i32](2 * N_EXP)
    ctx.enqueue_copy(dst_buf=ge_h, src_buf=ge_d)
    ctx.enqueue_copy(dst_buf=gp_h, src_buf=gp_d)
    ctx.enqueue_copy(dst_buf=gs_h, src_buf=gs_d)
    ctx.synchronize()
    var ge = List[Int32]()
    var gp = List[Int32]()
    var s = List[Int32]()
    ref_groups(idx_h.unsafe_ptr(), m, MR, ng, ge, gp, s)
    var bad = 0
    for i in range(ng):
        if ge_h[i] != ge[i]:
            bad += 1
    for i in range(ng * MR):
        if gp_h[i] != gp[i]:
            bad += 1
    for i in range(2 * N_EXP):
        if gs_h[i] != s[i]:
            bad += 1
    var used = 0
    for i in range(ng):
        if ge[i] >= 0:
            used += 1
    print("pairs_group m", m, "MR", MR, "groups", used, "of", ng, "differing entries", bad)
    return bad


def time_ms(ctx: DeviceContext, mut ts: List[Float64]) raises -> Float64:
    var a = ts.copy()
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[j] < a[i]:
                var t = a[i]
                a[i] = a[j]
                a[j] = t
    return a[len(a) // 2]


def compare_bf16(name: String, a: MutPointer[BFloat16, MutUntrackedOrigin], b: MutPointer[BFloat16, MutUntrackedOrigin], n: Int, bar: Float64) -> Bool:
    var mx = 0.0
    var md = 0.0
    var eq = 0
    var nz = 0
    for i in range(n):
        var x = Float64(a[unsafe_offset=i].cast[f32]())
        var y = Float64(b[unsafe_offset=i].cast[f32]())
        if x == y:
            eq += 1
        if y != 0:
            nz += 1
        mx = max(mx, abs(y))
        md = max(md, abs(x - y))
    var rel = md / mx if mx > 0 else 0.0
    var ok = rel <= bar and Float64(eq) >= 0.9 * Float64(n) and nz > n // 2
    print(name, "elements", n, "equal", eq, "nonzero ref", nz, "max|ref|", mx, "max|diff|", md, "rel", rel, "bar", bar, "PASS" if ok else "FAIL")
    return ok


def compare_f32(name: String, a: MutPointer[Float32, MutUntrackedOrigin], b: MutPointer[Float32, MutUntrackedOrigin], n: Int, bar: Float64) -> Bool:
    var mx = 0.0
    var md = 0.0
    var nz = 0
    for i in range(n):
        var x = Float64(a[unsafe_offset=i])
        var y = Float64(b[unsafe_offset=i])
        if y != 0:
            nz += 1
        mx = max(mx, abs(y))
        md = max(md, abs(x - y))
    var rel = md / mx if mx > 0 else 0.0
    var ok = rel <= bar and nz > n // 2
    print(name, "elements", n, "nonzero ref", nz, "max|ref|", mx, "max|diff|", md, "rel", rel, "bar", bar, "PASS" if ok else "FAIL")
    return ok


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var g = Lcg(0x5EED)
    var fails = 0
    fails += check_groups[MR8, NG8](ctx, g, 5)
    fails += check_groups[MR8, NG8](ctx, g, 777)
    fails += check_groups[MR8, NG8](ctx, g, CP)
    fails += check_groups[MR32, NG32](ctx, g, 5)
    fails += check_groups[MR32, NG32](ctx, g, CP)

    var m = CP
    var x_h = ctx.enqueue_create_host_buffer[bf16](CP * H)
    var idx_h = ctx.enqueue_create_host_buffer[i32](CP * TOPK)
    var gu_h = ctx.enqueue_create_host_buffer[u8](GU_BYTES)
    var dw_h = ctx.enqueue_create_host_buffer[u8](DN_BYTES)
    ctx.synchronize()
    for i in range(CP * H):
        x_h[i] = Scalar[bf16](Float32(Int(g.next() % 2001) - 1000) / 997.0)
    route(g, idx_h.unsafe_ptr(), m)
    fill_q4k(g, gu_h.unsafe_ptr(), GU_BYTES)
    fill_q4k(g, dw_h.unsafe_ptr(), DN_BYTES)
    var x_d = ctx.enqueue_create_buffer[bf16](CP * H)
    var idx_d = ctx.enqueue_create_buffer[i32](CP * TOPK)
    var gu_d = ctx.enqueue_create_buffer[u8](GU_BYTES)
    var dw_d = ctx.enqueue_create_buffer[u8](DN_BYTES)
    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.enqueue_copy(dst_buf=idx_d, src_buf=idx_h)
    ctx.enqueue_copy(dst_buf=gu_d, src_buf=gu_h)
    ctx.enqueue_copy(dst_buf=dw_d, src_buf=dw_h)
    var ge8_d = ctx.enqueue_create_buffer[i32](NG8)
    var gp8_d = ctx.enqueue_create_buffer[i32](NG8 * MR8)
    var ge32_d = ctx.enqueue_create_buffer[i32](NG32)
    var gp32_d = ctx.enqueue_create_buffer[i32](NG32 * MR32)
    var gs_d = ctx.enqueue_create_buffer[i32](2 * N_EXP)
    var ng8 = m * TOPK // MR8 + N_EXP
    var ng32 = m * TOPK // MR32 + N_EXP
    var Idx = TileTensor(idx_d, idx_layout)
    var Gs = TileTensor(gs_d, gs_layout)
    var Ge8 = TileTensor(ge8_d, ge8_layout)
    var Gp8 = TileTensor(gp8_d, gp8_layout)
    var Ge32 = TileTensor(ge32_d, ge32_layout)
    var Gp32 = TileTensor(gp32_d, gp32_layout)
    ctx.enqueue_function[moe_pairs_group[MR8, NQ, type_of(idx_layout), type_of(ge8_layout), type_of(gp8_layout), type_of(gs_layout)]](
        Idx, Ge8, Gp8, Gs, Int32(m), Int32(ng8), grid_dim=1, block_dim=N_EXP)
    ctx.enqueue_function[moe_pairs_group[MR32, NQ, type_of(idx_layout), type_of(ge32_layout), type_of(gp32_layout), type_of(gs_layout)]](
        Idx, Ge32, Gp32, Gs, Int32(m), Int32(ng32), grid_dim=1, block_dim=N_EXP)
    var X = TileTensor(x_d, x_layout)
    var hR_d = ctx.enqueue_create_buffer[bf16](CP * TOPK * E_FFN)
    var hW_d = ctx.enqueue_create_buffer[bf16](CP * TOPK * E_FFN)
    var dR_d = ctx.enqueue_create_buffer[f32](CP * TOPK * H)
    var dW_d = ctx.enqueue_create_buffer[f32](CP * TOPK * H)
    var HR = TileTensor(hR_d, hh_layout)
    var HW = TileTensor(hW_d, hh_layout)
    var DR = TileTensor(dR_d, dn_layout)
    var DW = TileTensor(dW_d, dn_layout)
    ctx.enqueue_memset(hR_d, 0)
    ctx.enqueue_memset(hW_d, 0)
    ctx.enqueue_memset(dR_d, 0)
    ctx.enqueue_memset(dW_d, 0)
    var up_off = Int32(N_EXP * E_FFN * Q4_ROW)
    comptime gu_row = moe_gate_up_q4k_grouped[MR8, E_FFN, type_of(x_layout), type_of(ge8_layout), type_of(gp8_layout), type_of(hh_layout)]
    comptime dn_row = moe_down_q4k_grouped[MR8, E_FFN, type_of(hh_layout), type_of(ge8_layout), type_of(gp8_layout), type_of(dn_layout)]
    comptime gu_w = moe_wmma_q4k_grouped[True, WM, WN, TM, TN, E_FFN, H, bf16, type_of(x_layout), type_of(ge32_layout), type_of(gp32_layout), type_of(hh_layout)]
    comptime dn_w = moe_wmma_q4k_grouped[False, WM, WN, TM, TN, H, E_FFN, f32, type_of(hh_layout), type_of(ge32_layout), type_of(gp32_layout), type_of(dn_layout)]
    var t_gr = List[Float64]()
    var t_gw = List[Float64]()
    var t_dr = List[Float64]()
    var t_dw = List[Float64]()
    for _ in range(9):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        ctx.enqueue_function[gu_row](X, gu_d.unsafe_ptr(), Ge8, Gp8, HR, Int32(H), up_off,
            grid_dim=(ceildiv(E_FFN, MOE_WAVES), ng8), block_dim=MOE_THREADS)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[gu_w](X, gu_d.unsafe_ptr(), Ge32, Gp32, HW, up_off,
            grid_dim=(E_FFN // BN, ng32), block_dim=MW_THREADS)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        ctx.enqueue_function[dn_row](HR, dw_d.unsafe_ptr(), Ge8, Gp8, DR, Int32(H),
            grid_dim=(ceildiv(H, MOE_WAVES), ng8), block_dim=MOE_THREADS)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        ctx.enqueue_function[dn_w](HR, dw_d.unsafe_ptr(), Ge32, Gp32, DW, Int32(0),
            grid_dim=(H // BN, ng32), block_dim=MW_THREADS)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        t_gr.append(Float64(t1 - t0) / 1e6)
        t_gw.append(Float64(t2 - t1) / 1e6)
        t_dr.append(Float64(t3 - t2) / 1e6)
        t_dw.append(Float64(t4 - t3) / 1e6)
    print("gate_up ms: rows", time_ms(ctx, t_gr), " wmma", time_ms(ctx, t_gw))
    print("down ms: rows", time_ms(ctx, t_dr), " wmma", time_ms(ctx, t_dw))
    var hr_h = ctx.enqueue_create_host_buffer[bf16](CP * TOPK * E_FFN)
    var hw_h = ctx.enqueue_create_host_buffer[bf16](CP * TOPK * E_FFN)
    var dr_h = ctx.enqueue_create_host_buffer[f32](CP * TOPK * H)
    var dw2_h = ctx.enqueue_create_host_buffer[f32](CP * TOPK * H)
    ctx.enqueue_copy(dst_buf=hr_h, src_buf=hR_d)
    ctx.enqueue_copy(dst_buf=hw_h, src_buf=hW_d)
    ctx.enqueue_copy(dst_buf=dr_h, src_buf=dR_d)
    ctx.enqueue_copy(dst_buf=dw2_h, src_buf=dW_d)
    ctx.synchronize()
    if not compare_bf16("gate_up swiglu wmma vs rows", hw_h.unsafe_ptr(), hr_h.unsafe_ptr(), m * TOPK * E_FFN, 2e-2):
        fails += 1
    if not compare_f32("down wmma vs rows", dw2_h.unsafe_ptr(), dr_h.unsafe_ptr(), m * TOPK * H, 1e-3):
        fails += 1
    if fails != 0:
        raise Error("FAIL test_moe_wmma: " + String(fails) + " checks")
    print("PASS test_moe_wmma")
