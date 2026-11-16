"""gemma4 decode attention (kernels/gemma4_attn.mojo: amar_attn_decode_gemma4, amar_attn_merge_gemma4)
against a CPU brute-force oracle with f64 accumulation. Mojo only, no Python twin.

usage: test_gemma4_attn                      correctness gate, exit 1 on any miss
       test_gemma4_attn bench [HD WIN T NS]  microseconds per launch, cold (rotated KV) per arm; 0 = all,
                                             WIN -1 = both; enq_us = host enqueue time alone

Gate: HD 256 and 512, WINDOW 512 and 0, T in {1, 2, 511, 512, 513, 4096, 16384}, split counts
1, 2, 3, 7, 16, 64 and the shipped rule g4_nsplit. Bar, fixed before the first run: max over heads of max|o - ref| / max|ref| <= 1e-4.
Reverse arms (the pass needs the code path): the page table is reversed, the data sit in KV slot 1 of 2
and every float outside the attended rows of that slot is NaN, so a kernel that ignores the table, the
slot, the window start or the context end reads NaN.
Mutants that must miss the bar: the kernel built with WINDOW - 1 and WINDOW + 1, and a launch with
t_len - 1 (the oldest and the newest attended rows are planted on heads 0 and 2), and a launch whose q is
multiplied by 1 / sqrt(HD), which is the output of a kernel with the usual attention scale (T >= 2).
qs 1 and 4 scale q: random scores of std 5 to 30, planted scores 43 to 341 (exp far from 0).
Split counts above the span count leave empty blocks (ns 64 at T 1). two_pools keeps an S pool (HD 256,
window 512) and an F pool (HD 512, full) live and interleaves their launches before one sync.
Bench reference arm: amar_attn_decode_swa_gated and amar_attn_split_part + amar_attn_split_merge
(kernels/spark_kernels.mojo) at the same shape. -D G4_GB=n -D G4_NLD=n -D G4_BT=n override the tuned picks.
"""
from std.math import exp, sqrt
from std.memory import bitcast
from std.sys import argv, has_accelerator, get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major
from dattn import dkv_off
from gemma4_attn import amar_attn_decode_gemma4, amar_attn_merge_gemma4, g4_gb, g4_nld, g4_bt, g4_nsplit, G4_PHDR, G4_NSMAX
from spark_kernels import amar_attn_decode_swa_gated, amar_attn_split_part, amar_attn_split_merge, att_span_len, ATT_NSMAX

comptime f32 = DType.float32
comptime i32 = DType.int32
comptime f64 = DType.float64
comptime NQH = 8
comptime NAT = 2
comptime SLOT = 1
comptime PAGE = 128
comptime BAR = 1.0e-4
comptime BPOOL = 1 << 25
comptime flat = row_major[1 << 27]()
comptime FL = type_of(flat)
comptime GB_D = get_defined_int["G4_GB", 0]()
comptime NLD_D = get_defined_int["G4_NLD", 0]()
comptime BT_D = get_defined_int["G4_BT", 0]()


def cfg_gb(hd: Int, window: Int) -> Int:
    return GB_D if GB_D > 0 else g4_gb(hd, window)


def cfg_bt(hd: Int, window: Int) -> Int:
    return BT_D if BT_D > 0 else g4_bt(hd, window)


def cfg_nld(hd: Int, window: Int) -> Int:
    return NLD_D if NLD_D > 0 else g4_nld(hd, window)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int(st >> 40)) / Float32(1 << 23) - 1


def launch[HD_: Int, WINDOW: Int](
    ctx: DeviceContext, mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32], mut v_d: DeviceBuffer[f32],
    mut o_d: DeviceBuffer[f32], mut p_d: DeviceBuffer[f32], mut tab_d: DeviceBuffer[i32], tab_off: Int,
    T: Int, ns: Int, slot: Int,
) raises:
    comptime GB = cfg_gb(HD_, WINDOW)
    comptime k_part = amar_attn_decode_gemma4[HD_, WINDOW, NAT, GB, cfg_nld(HD_, WINDOW), cfg_bt(HD_, WINDOW), FL, FL, FL, FL]
    comptime k_merge = amar_attn_merge_gemma4[HD_, FL, FL]
    ctx.enqueue_function[k_part](
        TileTensor(q_d, flat), TileTensor(k_d, flat), TileTensor(v_d, flat), TileTensor(o_d, flat), TileTensor(p_d, flat),
        tab_d.unsafe_ptr().unsafe_offset(tab_off), Int32(T), Int32(ns), Int32(slot), grid_dim=(NQH // GB, ns), block_dim=cfg_bt(HD_, WINDOW),
    )
    if ns > 1:
        ctx.enqueue_function[k_merge](TileTensor(p_d, flat), TileTensor(o_d, flat), Int32(ns), grid_dim=NQH, block_dim=HD_ // 8)


def launch_ref[HD_: Int, SPLIT: Bool](
    ctx: DeviceContext, mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32], mut v_d: DeviceBuffer[f32],
    mut o_d: DeviceBuffer[f32], mut p_d: DeviceBuffer[f32], mut g_d: DeviceBuffer[f32], mut tab_d: DeviceBuffer[i32], tab_off: Int,
    T: Int, window: Int, slot: Int,
) raises:
    comptime q_l = row_major[NQH, HD_]()
    comptime g_l = row_major[NQH]()
    var Q = TileTensor(q_d, q_l)
    var O = TileTensor(o_d, q_l)
    var G = TileTensor(g_d, g_l)
    var tab = tab_d.unsafe_ptr().unsafe_offset(tab_off)
    comptime if SPLIT:
        comptime k_part = amar_attn_split_part[type_of(q_l), FL, NAT, HD_, NQH, 1]
        comptime k_merge = amar_attn_split_merge[type_of(g_l), type_of(q_l), HD_, False, f32]
        var span = att_span_len(T, window)
        var n = min(T, window) if window > 0 else T
        var ns = (n + span - 1) // span
        var pg = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(p_d.unsafe_ptr()))
        ctx.enqueue_function[k_part](
            Q, TileTensor(k_d, flat), TileTensor(v_d, flat), pg, tab, Int32(T), Int32(window), Int32(span), Float32(1), Int32(slot),
            grid_dim=(NQH, ns), block_dim=HD_,
        )
        ctx.enqueue_function[k_merge](pg, G, O, Int32(ns), grid_dim=NQH, block_dim=HD_)
    else:
        comptime k_one = amar_attn_decode_swa_gated[type_of(q_l), FL, type_of(g_l), type_of(q_l), NAT, HD_, NQH, 1, False, f32]
        ctx.enqueue_function[k_one](
            Q, TileTensor(k_d, flat), TileTensor(v_d, flat), G, O, tab, Int32(T), Int32(window), Float32(1), Int32(slot),
            grid_dim=(NQH, 1), block_dim=HD_,
        )


def oracle[HD_: Int](
    q_h: HostBuffer[f32], k_h: HostBuffer[f32], v_h: HostBuffer[f32], tab_h: HostBuffer[i32], t_lo: Int, T: Int
) -> List[Float64]:
    var qp = q_h.unsafe_ptr()
    var kp = k_h.unsafe_ptr()
    var vp = v_h.unsafe_ptr()
    var tp = tab_h.unsafe_ptr()
    var n = T - t_lo
    var ref_o = List[Float64](length=NQH * HD_, fill=0)
    var s = List[Float64](length=n, fill=0)
    for h in range(NQH):
        var mx = Float64(-1.0e300)
        for i in range(n):
            var t = t_lo + i
            var b = dkv_off[HD_, 1, NAT](Int(tp[unsafe_offset=t >> 7]) * PAGE + (t & (PAGE - 1)), SLOT, 0)
            var a = Float64(0)
            for d in range(HD_):
                a += qp[unsafe_offset=h * HD_ + d].cast[f64]() * kp[unsafe_offset=b + d].cast[f64]()
            s[i] = a
            mx = max(mx, a)
        var l = Float64(0)
        for i in range(n):
            var t = t_lo + i
            var b = dkv_off[HD_, 1, NAT](Int(tp[unsafe_offset=t >> 7]) * PAGE + (t & (PAGE - 1)), SLOT, 0)
            var w = exp(s[i] - mx)
            l += w
            for d in range(HD_):
                ref_o[h * HD_ + d] += w * vp[unsafe_offset=b + d].cast[f64]()
        for d in range(HD_):
            ref_o[h * HD_ + d] /= l
    return ref_o^


def relerr[HD_: Int](o_h: HostBuffer[f32], ref_o: List[Float64]) -> Float64:
    var op = o_h.unsafe_ptr()
    var worst = Float64(0)
    for h in range(NQH):
        var den = Float64(0)
        var num = Float64(0)
        for d in range(HD_):
            var r = ref_o[h * HD_ + d]
            var x = op[unsafe_offset=h * HD_ + d].cast[f64]()
            if x != x:
                return 1.0e30
            den = max(den, abs(r))
            num = max(num, abs(x - r))
        worst = max(worst, num / den)
    return worst


struct Case[HD_: Int, WINDOW: Int]:
    var q_h: HostBuffer[f32]
    var o_h: HostBuffer[f32]
    var q_d: DeviceBuffer[f32]
    var qm_d: DeviceBuffer[f32]
    var o_d: DeviceBuffer[f32]
    var k_d: DeviceBuffer[f32]
    var v_d: DeviceBuffer[f32]
    var p_d: DeviceBuffer[f32]
    var tab_d: DeviceBuffer[i32]
    var ref_o: List[Float64]
    var T: Int
    var qs: Int

    def __init__(out self, ctx: DeviceContext, T: Int, qs: Int) raises:
        var t_lo = max(T - Self.WINDOW, 0) if Self.WINDOW > 0 else 0
        var npage = (T + PAGE - 1) // PAGE
        var pool = npage * NAT * PAGE * Self.HD_
        var q_h = ctx.enqueue_create_host_buffer[f32](NQH * Self.HD_)
        var k_h = ctx.enqueue_create_host_buffer[f32](pool)
        var v_h = ctx.enqueue_create_host_buffer[f32](pool)
        var tab_h = ctx.enqueue_create_host_buffer[i32](npage)
        self.o_h = ctx.enqueue_create_host_buffer[f32](NQH * Self.HD_)
        ctx.synchronize()
        var st = UInt64(Self.HD_ * 131 + Self.WINDOW * 7 + T + qs * 1000003)
        var nan = bitcast[f32, 1](SIMD[DType.uint32, 1](0x7FC00000))[0]
        for i in range(NQH * Self.HD_):
            q_h[i] = lcg(st) * Float32(qs)
        for i in range(pool):
            k_h[i] = nan
            v_h[i] = nan
        for p in range(npage):
            tab_h[p] = Int32(npage - 1 - p)
        for t in range(t_lo, T):
            var b = dkv_off[Self.HD_, 1, NAT](Int(tab_h[t >> 7]) * PAGE + (t & (PAGE - 1)), SLOT, 0)
            for d in range(Self.HD_):
                k_h[b + d] = lcg(st)
                v_h[b + d] = lcg(st)
                if t == t_lo:
                    k_h[b + d] = q_h[d] * (0.5 / Float32(qs))
                if t == T - 1:
                    k_h[b + d] = q_h[2 * Self.HD_ + d] * (0.5 / Float32(qs))
        var qm_h = ctx.enqueue_create_host_buffer[f32](NQH * Self.HD_)
        ctx.synchronize()
        for i in range(NQH * Self.HD_):
            qm_h[i] = q_h[i] / sqrt(Float32(Self.HD_))
        self.q_d = ctx.enqueue_create_buffer[f32](NQH * Self.HD_)
        self.qm_d = ctx.enqueue_create_buffer[f32](NQH * Self.HD_)
        ctx.enqueue_copy(dst_buf=self.qm_d, src_buf=qm_h)
        self.o_d = ctx.enqueue_create_buffer[f32](NQH * Self.HD_)
        self.k_d = ctx.enqueue_create_buffer[f32](pool)
        self.v_d = ctx.enqueue_create_buffer[f32](pool)
        self.p_d = ctx.enqueue_create_buffer[f32](NQH * G4_NSMAX * (Self.HD_ + G4_PHDR))
        self.tab_d = ctx.enqueue_create_buffer[i32](npage)
        ctx.enqueue_copy(dst_buf=self.q_d, src_buf=q_h)
        ctx.enqueue_copy(dst_buf=self.k_d, src_buf=k_h)
        ctx.enqueue_copy(dst_buf=self.v_d, src_buf=v_h)
        ctx.enqueue_copy(dst_buf=self.tab_d, src_buf=tab_h)
        ctx.synchronize()
        self.ref_o = oracle[Self.HD_](q_h, k_h, v_h, tab_h, t_lo, T)
        self.q_h = q_h^
        self.T = T
        self.qs = qs

    def go[WIN: Int](mut self, ctx: DeviceContext, t_len: Int, ns: Int, scaled: Bool = False) raises:
        self.o_d.enqueue_fill(3.0e38)
        if scaled:
            launch[Self.HD_, WIN](ctx, self.qm_d, self.k_d, self.v_d, self.o_d, self.p_d, self.tab_d, 0, t_len, ns, SLOT)
        else:
            launch[Self.HD_, WIN](ctx, self.q_d, self.k_d, self.v_d, self.o_d, self.p_d, self.tab_d, 0, t_len, ns, SLOT)

    def check(mut self, ctx: DeviceContext, what: String, want_miss: Bool) raises -> Int:
        ctx.enqueue_copy(dst_buf=self.o_h, src_buf=self.o_d)
        ctx.synchronize()
        var e = relerr[Self.HD_](self.o_h, self.ref_o)
        var ok = (e <= BAR) != want_miss
        print("hd", Self.HD_, "win", Self.WINDOW, "T", self.T, "qs", self.qs, what, "rel", e,
            ("CAUGHT" if want_miss else "PASS") if ok else "FAIL")
        return 0 if ok else 1


def run_case[HD_: Int, WINDOW: Int](ctx: DeviceContext, T: Int, qs: Int) raises -> Int:
    var c = Case[HD_, WINDOW](ctx, T, qs)
    var fails = 0
    var nss: List[Int] = [1, 2, 3, 7, 16, G4_NSMAX, g4_nsplit(WINDOW, T)]
    for ns in nss:
        c.go[WINDOW](ctx, T, ns)
        fails += c.check(ctx, "ns " + String(ns), False)
    if T >= 2:
        c.go[WINDOW](ctx, T - 1, 1)
        fails += c.check(ctx, "mutant t_len-1", True)
        c.go[WINDOW](ctx, T, 1, True)
        fails += c.check(ctx, "mutant scale 1/sqrt(HD)", True)
    comptime if WINDOW > 0 and HD_ == 256:
        if T >= WINDOW:
            c.go[WINDOW - 1](ctx, T, 1)
            fails += c.check(ctx, "mutant window-1", True)
        if T > WINDOW:
            c.go[WINDOW + 1](ctx, T, 1)
            fails += c.check(ctx, "mutant window+1", True)
    return fails


def two_pools(ctx: DeviceContext, T: Int) raises -> Int:
    var s = Case[256, 512](ctx, T, 1)
    var f = Case[512, 0](ctx, T, 1)
    var fails = 0
    var nss: List[Int] = [1, 5]
    for ns in nss:
        s.go[512](ctx, T, ns)
        f.go[0](ctx, T, 6 - ns)
        s.go[512](ctx, T, 6 - ns)
        f.go[0](ctx, T, ns)
        fails += s.check(ctx, "two pools live, S ns " + String(6 - ns), False)
        fails += f.check(ctx, "two pools live, F ns " + String(ns), False)
    return fails


def time_arm[HD_: Int, WINDOW: Int, ARM: Int](
    ctx: DeviceContext, mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32], mut v_d: DeviceBuffer[f32],
    mut o_d: DeviceBuffer[f32], mut p_d: DeviceBuffer[f32], mut g_d: DeviceBuffer[f32], mut tab_d: DeviceBuffer[i32],
    T: Int, ns: Int, rot: Int, npage: Int, warm_ns: Int,
) raises -> SIMD[f64, 8]:
    comptime BLOCKS = 11
    var r = 0
    var enq = Float64(1.0e30)
    var iters = 0
    var ts = List[Float64]()
    for blk in range(BLOCKS + 1):
        var n = 0
        var t0 = perf_counter_ns()
        while (n < iters) if blk > 0 else (perf_counter_ns() - t0 < warm_ns):
            for _ in range(4):
                var rr = r % rot
                comptime if ARM == 0:
                    launch[HD_, WINDOW](ctx, q_d, k_d, v_d, o_d, p_d, tab_d, rr * npage, T, ns, rr % 2)
                elif ARM == 1:
                    launch_ref[HD_, False](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, rr * npage, T, WINDOW, rr % 2)
                else:
                    launch_ref[HD_, True](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, rr * npage, T, WINDOW, rr % 2)
                r += 1
            n += 4
            if blk == 0:
                ctx.synchronize()
        var te = perf_counter_ns()
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(n)
        if blk == 0:
            iters = max(8, min(200, Int(30000.0 / us) // 4 * 4))
        else:
            enq = min(enq, Float64(te - t0) / 1.0e3 / Float64(n))
            var j = len(ts)
            ts.append(us)
            while j > 0 and ts[j - 1] > us:
                ts[j] = ts[j - 1]
                j -= 1
            ts[j] = us
    return SIMD[f64, 8](ts[BLOCKS // 2], ts[0], ts[BLOCKS - 1], Float64(iters), enq, 0, 0, 0)


def bench_case[HD_: Int, WINDOW: Int](
    ctx: DeviceContext, mut q_d: DeviceBuffer[f32], mut k_d: DeviceBuffer[f32], mut v_d: DeviceBuffer[f32],
    mut o_d: DeviceBuffer[f32], mut p_d: DeviceBuffer[f32], mut g_d: DeviceBuffer[f32], T: Int, nssel: Int,
) raises:
    comptime GB = cfg_gb(HD_, WINDOW)
    comptime NLD = cfg_nld(HD_, WINDOW)
    var t_lo = max(T - WINDOW, 0) if WINDOW > 0 else 0
    var npage = (T + PAGE - 1) // PAGE
    var p_lo = t_lo >> 7
    var npg = npage - p_lo
    var rot = BPOOL // (PAGE * HD_) // npg
    var tab_h = ctx.enqueue_create_host_buffer[i32](rot * npage)
    ctx.synchronize()
    for r in range(rot):
        for p in range(npage):
            tab_h[r * npage + p] = Int32((r // 2) * npg + max(p - p_lo, 0))
    var tab_d = ctx.enqueue_create_buffer[i32](rot * npage)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    ctx.synchronize()
    var spans = (T + NLD - 1) // NLD - t_lo // NLD
    var mb = Float64(rot * npg * PAGE * HD_ * 8) / 1.0e6
    var nss: List[Int] = [1, 2, 4, 8, 12, 16, 24, 32, 48, G4_NSMAX]
    var first = True
    print("RULE hd", HD_, "win", WINDOW, "T", T, "g4_nsplit", g4_nsplit(WINDOW, T))
    for ns in nss:
        if (ns > spans and ns > 1) or (nssel > 0 and ns != nssel):
            continue
        var w = 1500000000 if first else 400000000
        first = False
        var c = time_arm[HD_, WINDOW, 0](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, ns, rot, npage, w)
        var h = time_arm[HD_, WINDOW, 0](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, ns, 1, npage, 400000000)
        print("BENCH hd", HD_, "win", WINDOW, "T", T, "arm g4 gb", GB, "nld", NLD, "bt", cfg_bt(HD_, WINDOW), "ns", ns, "blocks", ns * NQH // GB,
            "cold_us", c[0], "min", c[1], "max", c[2], "iters", Int(c[3]), "hot_us", h[0], "enq_us", c[4], "rot", rot, "rot_MB", mb)
    var c1 = time_arm[HD_, WINDOW, 1](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, 1, rot, npage, 400000000)
    var h1 = time_arm[HD_, WINDOW, 1](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, 1, 1, npage, 400000000)
    print("BENCH hd", HD_, "win", WINDOW, "T", T, "arm ref-one blocks", NQH,
        "cold_us", c1[0], "min", c1[1], "max", c1[2], "iters", Int(c1[3]), "hot_us", h1[0], "rot", rot, "rot_MB", mb)
    var span = att_span_len(T, WINDOW)
    var rs = ((min(T, WINDOW) if WINDOW > 0 else T) + span - 1) // span
    var c2 = time_arm[HD_, WINDOW, 2](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, 1, rot, npage, 400000000)
    var h2 = time_arm[HD_, WINDOW, 2](ctx, q_d, k_d, v_d, o_d, p_d, g_d, tab_d, T, 1, 1, npage, 400000000)
    print("BENCH hd", HD_, "win", WINDOW, "T", T, "arm ref-split span", span, "blocks", rs * NQH,
        "cold_us", c2[0], "min", c2[1], "max", c2[2], "iters", Int(c2[3]), "hot_us", h2[0], "rot", rot, "rot_MB", mb)
    _ = tab_d.unsafe_ptr()


def bench_hd[HD_: Int](ctx: DeviceContext, win: Int, tsel: Int, nssel: Int) raises:
    var k_h = ctx.enqueue_create_host_buffer[f32](BPOOL)
    var v_h = ctx.enqueue_create_host_buffer[f32](BPOOL)
    var q_h = ctx.enqueue_create_host_buffer[f32](NQH * HD_)
    ctx.synchronize()
    var st = UInt64(HD_)
    for i in range(BPOOL):
        k_h[i] = lcg(st)
        v_h[i] = lcg(st)
    for i in range(NQH * HD_):
        q_h[i] = lcg(st)
    var k_d = ctx.enqueue_create_buffer[f32](BPOOL)
    var v_d = ctx.enqueue_create_buffer[f32](BPOOL)
    var q_d = ctx.enqueue_create_buffer[f32](NQH * HD_)
    var o_d = ctx.enqueue_create_buffer[f32](NQH * HD_)
    var g_d = ctx.enqueue_create_buffer[f32](NQH)
    var p_d = ctx.enqueue_create_buffer[f32](NQH * G4_NSMAX * (HD_ + G4_PHDR))
    ctx.enqueue_copy(dst_buf=k_d, src_buf=k_h)
    ctx.enqueue_copy(dst_buf=v_d, src_buf=v_h)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    g_d.enqueue_fill(0)
    ctx.synchronize()
    var ts: List[Int] = [512, 4096, 16384]
    if tsel > 0:
        ts = [tsel]
    for t in ts:
        if win != 0:
            bench_case[HD_, 512](ctx, q_d, k_d, v_d, o_d, p_d, g_d, t, nssel)
        if win <= 0:
            bench_case[HD_, 0](ctx, q_d, k_d, v_d, o_d, p_d, g_d, t, nssel)
    _ = (q_d.unsafe_ptr(), k_d.unsafe_ptr(), v_d.unsafe_ptr(), o_d.unsafe_ptr(), p_d.unsafe_ptr(), g_d.unsafe_ptr())


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var a = argv()
    var ctx = DeviceContext()
    if len(a) > 1 and String(a[1]) == "bench":
        var hd = Int(String(a[2])) if len(a) > 2 else 0
        var win = Int(String(a[3])) if len(a) > 3 else -1
        var tsel = Int(String(a[4])) if len(a) > 4 else 0
        var nssel = Int(String(a[5])) if len(a) > 5 else 0
        if hd == 0 or hd == 256:
            bench_hd[256](ctx, win, tsel, nssel)
        if hd == 0 or hd == 512:
            bench_hd[512](ctx, win, tsel, nssel)
        return
    print("bar: max over heads of max|o - ref| / max|ref| <=", BAR, "against the f64 oracle")
    var ts: List[Int] = [1, 2, 511, 512, 513, 4096, 16384]
    var fails = 0
    for t in ts:
        for qs in range(1, 5, 3):
            fails += run_case[256, 512](ctx, t, qs)
            fails += run_case[256, 0](ctx, t, qs)
            fails += run_case[512, 512](ctx, t, qs)
            fails += run_case[512, 0](ctx, t, qs)
    fails += two_pools(ctx, 513)
    fails += two_pools(ctx, 4096)
    if fails > 0:
        raise Error("FAIL test_gemma4_attn: " + String(fails) + " miss(es)")
    print("PASS test_gemma4_attn")
