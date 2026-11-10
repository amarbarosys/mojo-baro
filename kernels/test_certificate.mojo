"""Item 1: per-token Freivalds certificate on the 7 GEMVs of layer 3 (launch path), CPU check from (x, y) only.

usage: test_certificate PACKDIR NSTEP KAPPA [SEED]     KAPPA 0 = measure only (no verdict)
For each step the GPU runs the attention layer chain; after every GEMV the host copies x (bf16) and the GEMV
partial y (f32) and cpu/certify.mojo checks r.y == (r^T W) x. Prints `cert NAME step z scale ratio`, per-matrix max ratio,
and exits non-zero if KAPPA > 0 and any ratio exceeds it. Honest runs: no kernel is touched.
"""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu.host import DeviceContext
from layout import TileTensor
from matmul_skinny import amar_skinny_reduce, amar_skinny_reduce_add, amar_skinny_reduce_swiglu_bf16
from elementwise import amar_rmsnorm_cast
from attn import amar_head_rmsnorm, amar_rope_yarn, amar_kv_append, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split
from model import H, FFN, QF, KV, HD, NQH, NKVH
from certify import Cert
from test_cpu_layer import *


def mk(w: Q4W, n: Int, k: Int, seed: UInt64) -> Cert:
    return Cert(w.qh.unsafe_ptr(), w.sh.unsafe_ptr().unsafe_bitcast[Float16](), n, k, seed)


def main() raises:
    var a = argv()
    if len(a) < 4:
        raise Error("usage: test_certificate PACKDIR NSTEP KAPPA [SEED]")
    var dir = String(a[1])
    var nstep = atol(String(a[2]))
    var kappa = Float64(atof(String(a[3])))
    var seed = UInt64(atol(String(a[4]))) if len(a) > 4 else UInt64(1)
    var idx: String
    with open(dir + "/index.txt", "r") as f:
        idx = f.read()
    var ctx = DeviceContext()
    var wq = Q4W(ctx, dir, idx, "blk.3.attn_q.weight")
    var wk = Q4W(ctx, dir, idx, "blk.3.attn_k.weight")
    var wv = Q4W(ctx, dir, idx, "blk.3.attn_v.weight")
    var wo = Q4W(ctx, dir, idx, "blk.3.attn_output.weight")
    var wg = Q4W(ctx, dir, idx, "blk.3.ffn_gate.weight")
    var wu = Q4W(ctx, dir, idx, "blk.3.ffn_up.weight")
    var wd = Q4W(ctx, dir, idx, "blk.3.ffn_down.weight")
    var names = ["q", "k", "v", "o", "gate", "up", "down"]
    var certs = List[Cert]()
    var ubytes = 0
    certs.append(mk(wq, QF, H, seed * 1000 + 0))
    certs.append(mk(wk, KV, H, seed * 1000 + 1))
    certs.append(mk(wv, KV, H, seed * 1000 + 2))
    certs.append(mk(wo, H, ATT, seed * 1000 + 3))
    certs.append(mk(wg, FFN, H, seed * 1000 + 4))
    certs.append(mk(wu, FFN, H, seed * 1000 + 5))
    certs.append(mk(wd, H, FFN, seed * 1000 + 6))
    for m in range(7):
        ubytes += certs[m].k * 8
    print("arm cert seed", seed, "nstep", nstep, "kappa", kappa, "u-vector bytes", ubytes, "matrices", 7, "pack", dir)
    var g_att = f32_at(dir, idx, "blk.3.attn_norm.weight", H)
    var g_post = f32_at(dir, idx, "blk.3.post_attention_norm.weight", H)
    var g_qn = f32_at(dir, idx, "blk.3.attn_q_norm.weight", HD)
    var g_kn = f32_at(dir, idx, "blk.3.attn_k_norm.weight", HD)
    var d_gatt = up_f32(ctx, g_att)
    var d_gpost = up_f32(ctx, g_post)
    var d_gqn = up_f32(ctx, g_qn)
    var d_gkn = up_f32(ctx, g_kn)
    var x_d = ctx.enqueue_create_buffer[f32](H)
    var curb = ctx.enqueue_create_buffer[bf16](H)
    var p_qf = ctx.enqueue_create_buffer[f32](QF)
    var p_kv = ctx.enqueue_create_buffer[f32](KV)
    var p_h = ctx.enqueue_create_buffer[f32](H)
    var p_g = ctx.enqueue_create_buffer[f32](FFN)
    var p_u = ctx.enqueue_create_buffer[f32](FFN)
    var qf = ctx.enqueue_create_buffer[f32](QF)
    var q_d = ctx.enqueue_create_buffer[f32](ATT)
    var gate_d = ctx.enqueue_create_buffer[f32](ATT)
    var k_d = ctx.enqueue_create_buffer[f32](KV)
    var v_d = ctx.enqueue_create_buffer[f32](KV)
    var ao_d = ctx.enqueue_create_buffer[f32](ATT)
    var aob = ctx.enqueue_create_buffer[bf16](ATT)
    var fgb = ctx.enqueue_create_buffer[bf16](FFN)
    var kc = ctx.enqueue_create_buffer[f32](CACHE)
    var vc = ctx.enqueue_create_buffer[f32](CACHE)
    var tab = ctx.enqueue_create_buffer[DType.int32](1 << 16)
    ctx.enqueue_memset(kc, 0)
    ctx.enqueue_memset(vc, 0)
    ctx.enqueue_memset(tab, 0)
    ctx.synchronize()
    var X = TileTensor(x_d, xm_l)
    var CurB = TileTensor(curb, xm_l)
    var Gatt = TileTensor(d_gatt, h_l)
    var Gpost = TileTensor(d_gpost, h_l)
    var Gqn = TileTensor(d_gqn, hd_l)
    var Gkn = TileTensor(d_gkn, hd_l)
    var Qfm = TileTensor(qf, qfm_l)
    var Q = TileTensor(q_d, qm_l)
    var Gate = TileTensor(gate_d, attf_l)
    var Kf = TileTensor(k_d, kvf_l)
    var Khd = TileTensor(k_d, kvm_l)
    var Vf = TileTensor(v_d, kvf_l)
    var Vhd = TileTensor(v_d, kvm_l)
    var Kc = TileTensor(kc, cache_l)
    var Vc = TileTensor(vc, cache_l)
    var Ao = TileTensor(ao_d, qm_l)
    var Aof = TileTensor(ao_d, attf_l)
    var Aob = TileTensor(aob, attf_l)
    var Fgb = TileTensor(fgb, ffnm_l)
    var Pqf = TileTensor(p_qf, p_qf_l)
    var Pkv = TileTensor(p_kv, p_kv_l)
    var Ph = TileTensor(p_h, p_h_l)
    var Pg = TileTensor(p_g, p_ffn_l)
    var Pu = TileTensor(p_u, p_ffn_l)
    var worst = List[Float64](length=7, fill=0)
    var fails = 0
    var check_ns = 0
    var rng = seed * 2654435761 + 12345
    for st in range(nstep):
        rng = rng * 6364136223846793005 + 1442695040888963407
        var tok = Int((rng >> 33) % 240000)
        var pos = st
        var eb = read_at(dir + "/pack.bin", tok * H * 2, H * 2)
        var x0 = List[Float32](length=H, fill=0)
        for i in range(H):
            x0[i] = eb.unsafe_ptr().unsafe_bitcast[BFloat16]().unsafe_offset(i).unsafe_load().cast[f32]()
        ctx.enqueue_copy(x_d, x0.unsafe_ptr())
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gatt, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur_g = dn_bf16(ctx, curb, H)
        gemv[QF, H](ctx, curb, wq, p_qf)
        var y0 = dn_f32(ctx, p_qf, QF)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_qf_l), type_of(qfm_l), 1]](Pqf, Qfm, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
        gemv[KV, H](ctx, curb, wk, p_kv)
        var y1 = dn_f32(ctx, p_kv, KV)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Kf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        gemv[KV, H](ctx, curb, wv, p_kv)
        var y2 = dn_f32(ctx, p_kv, KV)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Vf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        ctx.enqueue_function[amar_qgate_split[type_of(qfm_l), type_of(qm_l), type_of(attf_l)]](Qfm, Q, Gate, grid_dim=(NQH, 1), block_dim=HD)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(qm_l), type_of(hd_l)]](Q, Gqn, Float32(1e-6), grid_dim=NQH, block_dim=HD)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(kvm_l), type_of(hd_l)]](Khd, Gkn, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
        ctx.enqueue_function[amar_rope_yarn[type_of(qm_l)]](Q, Int32(pos), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
        ctx.enqueue_function[amar_rope_yarn[type_of(kvm_l)]](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Kc, Khd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Vc, Vhd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_attn_decode[type_of(qm_l), type_of(cache_l), type_of(qm_l), 1]](Q, Kc, Vc, Ao, tab.unsafe_ptr(), Int32(pos + 1), Float32(0.0625), Int32(0), grid_dim=(NQH, 1), block_dim=HD)
        ctx.enqueue_function[amar_gate_mul_cast[type_of(attf_l), type_of(attf_l), type_of(attf_l)]](Aof, Gate, Aob, Int32(ATT), grid_dim=ceildiv(ATT, 256), block_dim=256)
        var aob_g = dn_bf16(ctx, aob, ATT)
        gemv[H, ATT](ctx, aob, wo, p_h)
        var y3 = dn_f32(ctx, p_h, H)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gpost, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur2 = dn_bf16(ctx, curb, H)
        gemv[FFN, H](ctx, curb, wg, p_g)
        gemv[FFN, H](ctx, curb, wu, p_u)
        var y4 = dn_f32(ctx, p_g, FFN)
        var y5 = dn_f32(ctx, p_u, FFN)
        ctx.enqueue_function[amar_skinny_reduce_swiglu_bf16[type_of(p_ffn_l), type_of(ffnm_l), 1]](Pg, Pu, Fgb, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
        var fgb_g = dn_bf16(ctx, fgb, FFN)
        gemv[H, FFN](ctx, fgb, wd, p_h)
        var y6 = dn_f32(ctx, p_h, H)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.synchronize()
        var xs = [cur_g.copy(), cur_g.copy(), cur_g.copy(), aob_g.copy(), cur2.copy(), cur2.copy(), fgb_g.copy()]
        var ys = [y0.copy(), y1.copy(), y2.copy(), y3.copy(), y4.copy(), y5.copy(), y6.copy()]
        for m in range(7):
            var t0 = perf_counter_ns()
            var r = certs[m].check(xs[m], ys[m])
            check_ns += Int(perf_counter_ns() - t0)
            var ratio = r[0] / r[1]
            if ratio > worst[m]:
                worst[m] = ratio
            if kappa > 0 and ratio > kappa:
                fails += 1
            print("cert", names[m], "step", st, "z", r[0], "scale", r[1], "ratio", ratio)
    print("check-us-per-matrix", Float64(check_ns) / 1e3 / Float64(7 * nstep))
    for m in range(7):
        print("max-ratio", names[m], worst[m])
    if kappa > 0:
        if fails != 0:
            raise Error("CERT FAIL " + String(fails) + " of " + String(7 * nstep))
        print("CERT PASS " + String(7 * nstep) + " of " + String(7 * nstep) + " checks <= kappa")
