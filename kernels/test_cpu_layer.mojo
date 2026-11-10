"""Step 1: one attention layer of the launch path (serve/window.mojo blk32_forward shape), layer 3 of the Qwythos pack.

usage: test_cpu_layer PACKDIR
Every stage runs the GPU kernel, then the CPU form of cpu/ops.mojo on the GPU's own input, and compares bits.
Exit 0 only if every stage of every step is bit-equal. Prints one line per stage and a final table.
"""
from std.memory import bitcast
from std.sys import argv
from std.math import ceildiv
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from matmul_skinny import amar_matmul_skinny_q4rowb, amar_skinny_reduce, amar_skinny_reduce_add, amar_skinny_reduce_swiglu_bf16, ROW_WAVES, ROW_THREADS
from elementwise import amar_rmsnorm_cast
from attn import amar_head_rmsnorm, amar_rope_yarn, amar_kv_append, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, KVT, KVPAGE
from model import H, FFN, QF, KV, HD, NQH, NKVH, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE
from ops import q4_row_dot, rmsnorm_cast, head_rmsnorm, rope_yarn, attn_head
from bexp import exp

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32
comptime ATT = NQH * HD
comptime NSTEP = 8
comptime CACHE = NKVH * KVPAGE * HD
comptime h_l = row_major[H]()
comptime xm_l = row_major[1, H]()
comptime qfm_l = row_major[1, QF]()
comptime qm_l = row_major[NQH, HD]()
comptime kvm_l = row_major[NKVH, HD]()
comptime kvf_l = row_major[1, KV]()
comptime attf_l = row_major[ATT]()
comptime attm_l = row_major[1, ATT]()
comptime ffnm_l = row_major[1, FFN]()
comptime hd_l = row_major[HD]()
comptime cache_l = row_major[CACHE]()
comptime p_h_l = row_major[1, 1, H]()
comptime p_qf_l = row_major[1, 1, QF]()
comptime p_kv_l = row_major[1, 1, KV]()
comptime p_ffn_l = row_major[1, 1, FFN]()


struct Q4W(Movable):
    var q: DeviceBuffer[DType.uint8]
    var s: DeviceBuffer[DType.float16]
    var qh: List[UInt8]
    var sh: List[UInt8]

    def __init__(out self, ctx: DeviceContext, dir: String, idx: String, name: String) raises:
        var off = 0
        var n = 0
        for l in idx.split("\n"):
            var p = l.split(" ")
            if len(p) == 4 and String(p[0]) == name:
                off = atol(String(p[2]))
                n = atol(String(p[3]))
        if n == 0:
            raise Error("FAIL index: no tensor " + name)
        self.qh = read_at(dir + "/pack.bin", off, n // 2)
        self.sh = read_at(dir + "/pack.bin", off + n // 2, n // 16)
        self.q = ctx.enqueue_create_buffer[DType.uint8](n // 2)
        self.s = ctx.enqueue_create_buffer[DType.float16](n // 32)
        ctx.enqueue_copy(self.q, self.qh.unsafe_ptr())
        ctx.enqueue_copy(self.s, self.sh.unsafe_ptr().unsafe_bitcast[Float16]())


def read_at(path: String, off: Int, n: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        _ = f.seek(off)
        return f.read_bytes(n)


def f32_at(dir: String, idx: String, name: String, n: Int) raises -> List[Float32]:
    for l in idx.split("\n"):
        var p = l.split(" ")
        if len(p) == 4 and String(p[0]) == name:
            var b = read_at(dir + "/pack.bin", atol(String(p[2])), n * 4)
            var out = List[Float32](length=n, fill=0)
            for i in range(n):
                out[i] = b.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(i).unsafe_load()
            return out^
    raise Error("FAIL index: no tensor " + name)


def up_f32(ctx: DeviceContext, v: List[Float32]) raises -> DeviceBuffer[f32]:
    var d = ctx.enqueue_create_buffer[f32](len(v))
    ctx.enqueue_copy(d, v.unsafe_ptr())
    return d^


def dn_f32(ctx: DeviceContext, d: DeviceBuffer[f32], n: Int) raises -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(v.unsafe_ptr(), d)
    ctx.synchronize()
    return v^


def dn_bf16(ctx: DeviceContext, d: DeviceBuffer[bf16], n: Int) raises -> List[BFloat16]:
    var v = List[BFloat16](length=n, fill=0)
    ctx.enqueue_copy(v.unsafe_ptr(), d)
    ctx.synchronize()
    return v^


def bits32(x: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](SIMD[f32, 1](x))[0]


def digest(v: List[Float32], h: UInt64) -> UInt64:
    var r = h
    for x in v:
        r = (r ^ UInt64(bits32(x))) * 1099511628211
    return r


def diff_f32(name: String, step: Int, a: List[Float32], b: List[Float32], mut bad: Dict[String, Int]):
    var c = 0
    for i in range(len(a)):
        if bits32(a[i]) != bits32(b[i]):
            if c < 2 and bad.get(name, 0) < 2:
                print("  mismatch", name, "step", step, "i", i, "cpu", a[i], "gpu", b[i])
            c += 1
    bad[name] = bad.get(name, 0) + c


def diff_bf16(name: String, step: Int, a: List[BFloat16], b: List[BFloat16], mut bad: Dict[String, Int]):
    var c = 0
    for i in range(len(a)):
        if bitcast[DType.uint16, 1](SIMD[bf16, 1](a[i]))[0] != bitcast[DType.uint16, 1](SIMD[bf16, 1](b[i]))[0]:
            if c < 2 and bad.get(name, 0) < 2:
                print("  mismatch", name, "step", step, "i", i, "cpu", a[i], "gpu", b[i])
            c += 1
    bad[name] = bad.get(name, 0) + c


def gemv[N: Int, K: Int](ctx: DeviceContext, a: DeviceBuffer[bf16], w: Q4W, p: DeviceBuffer[f32]) raises:
    comptime a_l = row_major[1, K]()
    comptime q_l = row_major[N, K // 2]()
    comptime s_l = row_major[N, K // 32]()
    comptime p_l = row_major[1, 1, N]()
    ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 1, type_of(a_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        TileTensor(a, a_l), TileTensor(w.q, q_l), TileTensor(w.s, s_l), TileTensor(p, p_l),
        Int32(1), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS,
    )


def cpu_gemv(a: List[BFloat16], w: Q4W, n: Int, k: Int, r0: Int = 0) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for r in range(n):
        out[r] = q4_row_dot(
            a.unsafe_ptr(), w.qh.unsafe_ptr().unsafe_offset((r0 + r) * (k // 2)),
            w.sh.unsafe_ptr().unsafe_bitcast[Float16]().unsafe_offset((r0 + r) * (k // 32)), k // 32,
        )
    return out^


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: test_cpu_layer PACKDIR")
    var dir = String(a[1])
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
    comptime assert KVT == f32
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
    var Aobm = TileTensor(aob, attm_l)
    var Fgb = TileTensor(fgb, ffnm_l)
    var Pqf = TileTensor(p_qf, p_qf_l)
    var Pkv = TileTensor(p_kv, p_kv_l)
    var Ph = TileTensor(p_h, p_h_l)
    var Pg = TileTensor(p_g, p_ffn_l)
    var Pu = TileTensor(p_u, p_ffn_l)
    var toks = [1000, 5000, 20000, 100000, 300, 77, 4242, 99999]
    var poss = [0, 1, 2, 3, 4, 5, 300, 5000]
    var bad = Dict[String, Int]()
    var dg = UInt64(14695981039346656037)
    var emb_cache = Dict[Int, Int]()
    print("arm gpu=launch-path kernels (amar_rmsnorm_cast, q4rowb, head_rmsnorm, rope_yarn, attn_decode, reduce*) cpu=cpu/ops.mojo pack=" + dir)
    for st in range(NSTEP):
        var eb = read_at(dir + "/pack.bin", toks[st] * H * 2, H * 2)
        var x0 = List[Float32](length=H, fill=0)
        for i in range(H):
            x0[i] = eb.unsafe_ptr().unsafe_bitcast[BFloat16]().unsafe_offset(i).unsafe_load().cast[f32]()
        ctx.enqueue_copy(x_d, x0.unsafe_ptr())
        var pos = poss[st]
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gatt, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur_g = dn_bf16(ctx, curb, H)
        diff_bf16("rms1", st, rmsnorm_cast(x0, g_att, Float32(1e-6)), cur_g, bad)
        gemv[QF, H](ctx, curb, wq, p_qf)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_qf_l), type_of(qfm_l), 1]](Pqf, Qfm, Int32(1), Int32(QF), grid_dim=ceildiv(QF, 256), block_dim=256)
        var qf_g = dn_f32(ctx, qf, QF)
        diff_f32("gemv_q", st, cpu_gemv(cur_g, wq, QF, H), qf_g, bad)
        gemv[KV, H](ctx, curb, wk, p_kv)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Kf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        var k_g = dn_f32(ctx, k_d, KV)
        diff_f32("gemv_k", st, cpu_gemv(cur_g, wk, KV, H), k_g, bad)
        gemv[KV, H](ctx, curb, wv, p_kv)
        ctx.enqueue_function[amar_skinny_reduce[type_of(p_kv_l), type_of(kvf_l), 1]](Pkv, Vf, Int32(1), Int32(KV), grid_dim=ceildiv(KV, 256), block_dim=256)
        var v_g = dn_f32(ctx, v_d, KV)
        diff_f32("gemv_v", st, cpu_gemv(cur_g, wv, KV, H), v_g, bad)
        ctx.enqueue_function[amar_qgate_split[type_of(qfm_l), type_of(qm_l), type_of(attf_l)]](Qfm, Q, Gate, grid_dim=(NQH, 1), block_dim=HD)
        var q_in = dn_f32(ctx, q_d, ATT)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(qm_l), type_of(hd_l)]](Q, Gqn, Float32(1e-6), grid_dim=NQH, block_dim=HD)
        ctx.enqueue_function[amar_head_rmsnorm[type_of(kvm_l), type_of(hd_l)]](Khd, Gkn, Float32(1e-6), grid_dim=NKVH, block_dim=HD)
        var q_n = dn_f32(ctx, q_d, ATT)
        var k_n = dn_f32(ctx, k_d, KV)
        var q_c = q_in.copy()
        for h in range(NQH):
            head_rmsnorm(q_c, h, HD, g_qn, Float32(1e-6))
        diff_f32("hrms_q", st, q_c, q_n, bad)
        var k_c = k_g.copy()
        for h in range(NKVH):
            head_rmsnorm(k_c, h, HD, g_kn, Float32(1e-6))
        diff_f32("hrms_k", st, k_c, k_n, bad)
        ctx.enqueue_function[amar_rope_yarn[type_of(qm_l)]](Q, Int32(pos), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
        ctx.enqueue_function[amar_rope_yarn[type_of(kvm_l)]](Khd, Int32(pos), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
        var q_r = dn_f32(ctx, q_d, ATT)
        var k_r = dn_f32(ctx, k_d, KV)
        var q_rc = q_n.copy()
        for h in range(NQH):
            rope_yarn(q_rc, h, HD, pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
        diff_f32("rope_q", st, q_rc, q_r, bad)
        var k_rc = k_n.copy()
        for h in range(NKVH):
            rope_yarn(k_rc, h, HD, pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
        diff_f32("rope_k", st, k_rc, k_r, bad)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Kc, Khd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_kv_append[type_of(cache_l), type_of(kvm_l), 1]](Vc, Vhd, tab.unsafe_ptr(), Int32(pos), Int32(0), grid_dim=(NKVH, 1), block_dim=HD)
        ctx.enqueue_function[amar_attn_decode[type_of(qm_l), type_of(cache_l), type_of(qm_l), 1]](Q, Kc, Vc, Ao, tab.unsafe_ptr(), Int32(pos + 1), Float32(0.0625), Int32(0), grid_dim=(NQH, 1), block_dim=HD)
        var kc_h = dn_f32(ctx, kc, CACHE)
        var vc_h = dn_f32(ctx, vc, CACHE)
        var ao_g = dn_f32(ctx, ao_d, ATT)
        var ao_c = List[Float32](length=ATT, fill=0)
        for h in range(NQH):
            var o = attn_head(q_r, h * HD, kc_h, vc_h, h // (NQH // NKVH), pos + 1, HD, Float32(0.0625))
            for d in range(HD):
                ao_c[h * HD + d] = o[d]
        diff_f32("attn", st, ao_c, ao_g, bad)
        ctx.enqueue_function[amar_gate_mul_cast[type_of(attf_l), type_of(attf_l), type_of(attf_l)]](Aof, Gate, Aob, Int32(ATT), grid_dim=ceildiv(ATT, 256), block_dim=256)
        var gate_g = dn_f32(ctx, gate_d, ATT)
        var aob_g = dn_bf16(ctx, aob, ATT)
        var aob_c = List[BFloat16](length=ATT, fill=0)
        for i in range(ATT):
            aob_c[i] = (ao_g[i] * (Float32(1) / (1 + exp(-gate_g[i])))).cast[bf16]()
        diff_bf16("gate_mul", st, aob_c, aob_g, bad)
        gemv[H, ATT](ctx, aob, wo, p_h)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        var x1_g = dn_f32(ctx, x_d, H)
        var o_c = cpu_gemv(aob_g, wo, H, ATT)
        var x1_c = List[Float32](length=H, fill=0)
        for i in range(H):
            x1_c[i] = x0[i] + (Float32(0) + o_c[i])
        diff_f32("o_add", st, x1_c, x1_g, bad)
        ctx.enqueue_function[amar_rmsnorm_cast[type_of(xm_l), type_of(h_l), type_of(xm_l)]](X, Gpost, CurB, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
        var cur2 = dn_bf16(ctx, curb, H)
        diff_bf16("rms2", st, rmsnorm_cast(x1_g, g_post, Float32(1e-6)), cur2, bad)
        gemv[FFN, H](ctx, curb, wg, p_g)
        gemv[FFN, H](ctx, curb, wu, p_u)
        ctx.enqueue_function[amar_skinny_reduce_swiglu_bf16[type_of(p_ffn_l), type_of(ffnm_l), 1]](Pg, Pu, Fgb, Int32(1), Int32(FFN), grid_dim=ceildiv(FFN, 256), block_dim=256)
        var fgb_g = dn_bf16(ctx, fgb, FFN)
        var gc = cpu_gemv(cur2, wg, FFN, H)
        var uc = cpu_gemv(cur2, wu, FFN, H)
        var fgb_c = List[BFloat16](length=FFN, fill=0)
        for i in range(FFN):
            var gg = Float32(0) + gc[i]
            var uu = Float32(0) + uc[i]
            fgb_c[i] = ((gg / (1 + exp(-gg))) * uu).cast[bf16]()
        diff_bf16("gateup_swiglu", st, fgb_c, fgb_g, bad)
        gemv[H, FFN](ctx, fgb, wd, p_h)
        ctx.enqueue_function[amar_skinny_reduce_add[type_of(p_h_l), type_of(xm_l), 1]](Ph, X, Int32(1), Int32(H), grid_dim=ceildiv(H, 256), block_dim=256)
        var x2_g = dn_f32(ctx, x_d, H)
        var d_c = cpu_gemv(fgb_g, wd, H, FFN)
        var x2_c = List[Float32](length=H, fill=0)
        for i in range(H):
            x2_c[i] = x1_g[i] + (Float32(0) + d_c[i])
        diff_f32("down_add", st, x2_c, x2_g, bad)
        dg = digest(x2_g, digest(ao_g, digest(k_r, digest(q_r, dg))))
        print("step", st, "pos", pos, "ao", ao_g[0], ao_g[1], ao_g[300], "x", x2_g[0], x2_g[1])
    print("gpu-digest", dg)
    var total = 0
    for name in ["rms1", "gemv_q", "gemv_k", "gemv_v", "hrms_q", "hrms_k", "rope_q", "rope_k", "attn", "gate_mul", "o_add", "rms2", "gateup_swiglu", "down_add"]:
        print("stage", name, "mismatches", bad.get(name, 0))
        total += bad.get(name, 0)
    if total != 0:
        raise Error("MISMATCH " + String(total))
    print("PASS test_cpu_layer: cpu == gpu bitwise on 14 stages x " + String(NSTEP) + " steps")
