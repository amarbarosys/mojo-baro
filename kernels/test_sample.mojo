"""Device sampler checks for kernels/sample.mojo (KSAMP, bench/chat-protocol.md P-K1..P-K10).

Philox4x32-10 known-answer vectors; temperature 0 equal to amar_argmax_row at the real
vocabulary; chi-square on 10k draws per sampling configuration against the exact float64
distribution; reproducibility; exact speculative acceptance against a mismatched draft;
fast path and forced general path (CAP 16) draw identical tokens; time per call at
V = 248320 on a Gaussian and an LM-like peaked row.
"""
from std.math import cos, exp, log, sin, sqrt
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.utils.numerics import nan, neg_inf

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major

from elementwise import amar_argmax_row, EW_THREADS
from sample import amar_sample_row, amar_sample_probs, amar_spec_accept, philox4x32, SAMP_THREADS, SAMP_CAP

comptime VR = 248320
comptime VS = 64
comptime ND = 10000
comptime RG = 13
comptime f32 = DType.float32
comptime i32 = DType.int32
comptime u32 = DType.uint32
comptime FMAX = 3.4028234663852886e38

comptime xs_l = row_major[ND, VS]()
comptime ts_l = row_major[ND]()
comptime x1_l = row_major[1, VS]()
comptime xg_l = row_major[RG, VR]()
comptime tg_l = row_major[RG]()
comptime xr_l = row_major[1, VR]()
comptime t1_l = row_major[1]()


@fieldwise_init
struct Cfg(Copyable, Movable):
    var name: String
    var t: Float32
    var k: Int
    var p: Float32
    var mp: Float32


def fail(msg: String) raises:
    print("FAIL:", msg)
    raise Error("sampler check failed: " + msg)


def hu(seed: UInt32, i: Int) -> Float64:
    var w = philox4x32(SIMD[u32, 4](UInt32(i), 0, 0, 0), seed, 0x5EED)
    return (Float64(Int(w[0])) + 0.5) / 4294967296.0


def gauss(seed: UInt32, i: Int) -> Float64:
    var u1 = hu(seed, 2 * i)
    var u2 = hu(seed, 2 * i + 1)
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


def is_valid(v: Float32) -> Bool:
    return v == v and Float64(v) >= -FMAX and Float64(v) <= FMAX


def ref_q(l: List[Float32], c: Cfg) raises -> List[Float64]:
    var n = len(l)
    var order = List[Int]()
    for i in range(n):
        if is_valid(l[i]):
            order.append(i)
    for a in range(1, len(order)):
        var j = a
        while j > 0 and l[order[j]] > l[order[j - 1]]:
            order.swap_elements(j, j - 1)
            j -= 1
    var q = List[Float64]()
    for _ in range(n):
        q.append(0.0)
    var m = len(order)
    if m == 0:
        return q^
    var keff = c.k if c.k > 0 and c.k < m else m
    var lmax = Float64(l[order[0]])
    var zk = 0.0
    for j in range(keff):
        zk += exp(Float64(l[order[j]]) - lmax)
    var cnt = keff
    if c.p < 1.0:
        var cum = 0.0
        var p = Float64(c.p)
        for j in range(keff):
            var w = exp(Float64(l[order[j]]) - lmax)
            cum += w
            if cum >= p * zk:
                if abs(cum / zk - p) < 1e-4 or abs((cum - w) / zk - p) < 1e-4:
                    fail("ambiguous top-p fixture " + c.name)
                cnt = min(cnt, j + 1)
                break
    if c.mp > 0:
        var mc = 0
        for j in range(keff):
            var w = exp(Float64(l[order[j]]) - lmax)
            if abs(w - Float64(c.mp)) < 1e-5:
                fail("ambiguous min-p fixture " + c.name)
            if w >= Float64(c.mp):
                mc = j + 1
        cnt = min(cnt, mc)
    var z = 0.0
    for j in range(cnt):
        z += exp((Float64(l[order[j]]) - lmax) / Float64(c.t))
    for j in range(cnt):
        q[order[j]] = exp((Float64(l[order[j]]) - lmax) / Float64(c.t)) / z
    return q^


def crit(df: Int) -> Float64:
    if df <= 0:
        return 1e-9
    var d = Float64(df)
    var a = 2.0 / (9.0 * d)
    var c = 1.0 - a + 3.090232 * sqrt(a)
    return d * c * c * c


def chi2(name: String, counts: List[Int], q: List[Float64], n: Int) raises:
    var outside = 0
    var stat = 0.0
    var bins = 0
    var po = 0.0
    var pe = 0.0
    var lo = 0.0
    var le = 0.0
    for i in range(len(q)):
        if q[i] == 0.0:
            outside += counts[i]
            continue
        var e = q[i] * Float64(n)
        if e >= 5.0:
            stat += (Float64(counts[i]) - e) ** 2 / e
            bins += 1
            lo = Float64(counts[i])
            le = e
        else:
            po += Float64(counts[i])
            pe += e
    if pe >= 5.0:
        stat += (po - pe) ** 2 / pe
        bins += 1
    elif pe > 0.0:
        stat -= (lo - le) ** 2 / le
        stat += (lo + po - le - pe) ** 2 / (le + pe)
    var cv = crit(bins - 1)
    print("  ", name, ": chi2", stat, "df", bins - 1, "crit(p=0.001)", cv, "outside", outside)
    if outside != 0:
        fail(name + ": draws outside the truncated set")
    if bins > 1 and stat >= cv:
        fail(name + ": chi-square above critical value")


def chi2_two(name: String, a: List[Int], b: List[Int]) raises:
    var stat = 0.0
    var bins = 0
    var pa = 0.0
    var pb = 0.0
    for i in range(len(a)):
        var s = Float64(a[i] + b[i])
        if s >= 10.0:
            stat += (Float64(a[i]) - Float64(b[i])) ** 2 / s
            bins += 1
        else:
            pa += Float64(a[i])
            pb += Float64(b[i])
    if pa + pb > 0.0:
        stat += (pa - pb) ** 2 / (pa + pb)
        bins += 1
    var cv = crit(bins - 1)
    print("  ", name, ": two-sample chi2", stat, "df", bins - 1, "crit(p=0.001)", cv)
    if bins > 1 and stat >= cv:
        fail(name + ": two-sample chi-square above critical value")


def hist_of(h: List[Int32]) -> List[Int]:
    var c = List[Int]()
    for _ in range(VS):
        c.append(0)
    for i in range(len(h)):
        var t = Int(h[i])
        if t >= 0 and t < VS:
            c[t] += 1
    return c^


def draw[
    CAP: Int = SAMP_CAP
](
    ctx: DeviceContext, mut x: DeviceBuffer[f32], mut t: DeviceBuffer[i32], mut p: DeviceBuffer[f32],
    c: Cfg, seed: UInt64, counter: UInt64,
) raises -> Tuple[List[Int32], List[Float32]]:
    comptime k = amar_sample_row[type_of(xs_l), type_of(ts_l), type_of(ts_l), CAP]
    ctx.enqueue_function[k](
        TileTensor(x, xs_l), TileTensor(t, ts_l), TileTensor(p, ts_l), Int32(VS),
        c.t, Int32(c.k), c.p, c.mp, seed, counter, grid_dim=ND, block_dim=SAMP_THREADS,
    )
    var th = ctx.enqueue_create_host_buffer[i32](ND)
    var ph = ctx.enqueue_create_host_buffer[f32](ND)
    ctx.enqueue_copy(dst_buf=th, src_buf=t)
    ctx.enqueue_copy(dst_buf=ph, src_buf=p)
    ctx.synchronize()
    var toks = List[Int32]()
    var probs = List[Float32]()
    for i in range(ND):
        toks.append(th[i])
        probs.append(ph[i])
    return (toks^, probs^)


def probs_row[CAP: Int = SAMP_CAP](ctx: DeviceContext, l: List[Float32], c: Cfg) raises -> List[Float32]:
    var xh = ctx.enqueue_create_host_buffer[f32](VS)
    ctx.synchronize()
    for i in range(VS):
        xh[i] = l[i]
    var xd = ctx.enqueue_create_buffer[f32](VS)
    var pd = ctx.enqueue_create_buffer[f32](VS)
    ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
    comptime k = amar_sample_probs[type_of(x1_l), type_of(x1_l), CAP]
    ctx.enqueue_function[k](
        TileTensor(xd, x1_l), TileTensor(pd, x1_l), Int32(VS), c.t, Int32(c.k), c.p, c.mp,
        grid_dim=1, block_dim=SAMP_THREADS,
    )
    var ph = ctx.enqueue_create_host_buffer[f32](VS)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(VS):
        out.append(ph[i])
    return out^


def fill_rows(ctx: DeviceContext, mut d: DeviceBuffer[f32], l: List[Float32]) raises:
    var h = ctx.enqueue_create_host_buffer[f32](ND * VS)
    ctx.synchronize()
    for r in range(ND):
        for i in range(VS):
            h[r * VS + i] = l[i]
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()


def path_identity(
    ctx: DeviceContext, mut xd: DeviceBuffer[f32], mut td: DeviceBuffer[i32], mut pd: DeviceBuffer[f32],
    l: List[Float32], c: Cfg, toks: List[Int32], probs: List[Float32], prow: List[Float32],
    seed: UInt64, counter: UInt64,
) raises:
    var rf = draw[16](ctx, xd, td, pd, c, seed, counter)
    var prf = probs_row[16](ctx, l, c)
    var same = 0
    var pe = 0.0
    for i in range(ND):
        if rf[0][i] == toks[i]:
            same += 1
        pe = max(pe, abs(Float64(rf[1][i]) - Float64(probs[i])))
    for i in range(VS):
        pe = max(pe, abs(Float64(prf[i]) - Float64(prow[i])))
    print("P-K7", c.name, ": fast vs general path (CAP 16) tokens equal", same, "of", ND, "prob max diff", pe)
    if same != ND or pe > 1e-6:
        fail(c.name + ": fast and general path disagree")


def med3(xs: List[Float64]) -> Tuple[Float64, Float64, Float64]:
    var s = xs.copy()
    for a in range(1, len(s)):
        var j = a
        while j > 0 and s[j] < s[j - 1]:
            s.swap_elements(j, j - 1)
            j -= 1
    return (s[0], s[len(s) // 2], s[len(s) - 1])


def time_arm(
    ctx: DeviceContext, mut rd: DeviceBuffer[f32], mut r1: DeviceBuffer[i32], mut p1: DeviceBuffer[f32],
    row: String, c: Cfg,
) raises:
    comptime sm1 = amar_sample_row[type_of(xr_l), type_of(t1_l), type_of(t1_l)]
    var XR = TileTensor(rd, xr_l)
    var TR = TileTensor(r1, t1_l)
    var PR = TileTensor(p1, t1_l)
    for w in range(200):
        ctx.enqueue_function[sm1](
            XR, TR, PR, Int32(VR), c.t, Int32(c.k), c.p, c.mp, UInt64(9), UInt64(w),
            grid_dim=1, block_dim=SAMP_THREADS,
        )
    ctx.synchronize()
    var blocks = List[Float64]()
    for b in range(11):
        var t0 = perf_counter_ns()
        for w in range(100):
            ctx.enqueue_function[sm1](
                XR, TR, PR, Int32(VR), c.t, Int32(c.k), c.p, c.mp, UInt64(9), UInt64(b * 100 + w),
                grid_dim=1, block_dim=SAMP_THREADS,
            )
        ctx.synchronize()
        blocks.append(Float64(perf_counter_ns() - t0) / 1e3 / 100.0)
    var s = med3(blocks)
    print(
        "P-K8 arm amar_sample_row", row, c.name, "| V", VR, "R 1 grid 1 block", SAMP_THREADS,
        "CAP", SAMP_CAP, "T", c.t, "k", c.k, "p", c.p, "minp", c.mp,
        ": median", s[1], "us/call, min", s[0], "max", s[2], "(11 blocks x 100)",
    )


def kat() raises:
    var z = philox4x32(SIMD[u32, 4](0, 0, 0, 0), 0, 0)
    var o = philox4x32(
        SIMD[u32, 4](0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF, 0xFFFFFFFF
    )
    var p = philox4x32(
        SIMD[u32, 4](0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), 0xA4093822, 0x299F31D0
    )
    print("P-K1 philox zero", z, "ones", o, "pi", p)
    if z != SIMD[u32, 4](0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8):
        fail("philox KAT zero")
    if o != SIMD[u32, 4](0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD):
        fail("philox KAT ones")
    if p != SIMD[u32, 4](0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1):
        fail("philox KAT pi")
    print("P-K1 PASS")


def greedy_identity(ctx: DeviceContext) raises:
    var xh = ctx.enqueue_create_host_buffer[f32](RG * VR)
    ctx.synchronize()
    for r in range(8):
        for i in range(VR):
            xh[r * VR + i] = Float32(2.5 * gauss(UInt32(100 + r), i))
    for i in range(VR):
        xh[8 * VR + i] = Float32(Int(hu(7, i) * 20.0))
        xh[9 * VR + i] = Float32(0.0) if i % 2 == 1 else Float32(-0.0)
        xh[10 * VR + i] = neg_inf[f32]()
        xh[11 * VR + i] = nan[f32]() if i % 3 == 0 else Float32(2.5 * gauss(9, i))
        xh[12 * VR + i] = neg_inf[f32]()
    for j in range(100):
        xh[10 * VR + 1 + Int(hu(11, j) * Float64(VR - 1))] = Float32(gauss(12, j))
    var xd = ctx.enqueue_create_buffer[f32](RG * VR)
    var ta = ctx.enqueue_create_buffer[i32](RG)
    var tb = ctx.enqueue_create_buffer[i32](RG)
    var pb = ctx.enqueue_create_buffer[f32](RG)
    ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
    var X = TileTensor(xd, xg_l)
    comptime am = amar_argmax_row[type_of(xg_l), type_of(tg_l)]
    comptime sm = amar_sample_row[type_of(xg_l), type_of(tg_l), type_of(tg_l)]
    ctx.enqueue_function[am](X, TileTensor(ta, tg_l), Int32(VR), grid_dim=RG, block_dim=EW_THREADS)
    ctx.enqueue_function[sm](
        X, TileTensor(tb, tg_l), TileTensor(pb, tg_l), Int32(VR), Float32(0), Int32(20), Float32(0.8),
        Float32(0), UInt64(1), UInt64(2), grid_dim=RG, block_dim=SAMP_THREADS,
    )
    var ah = ctx.enqueue_create_host_buffer[i32](RG)
    var bh = ctx.enqueue_create_host_buffer[i32](RG)
    var ph = ctx.enqueue_create_host_buffer[f32](RG)
    ctx.enqueue_copy(dst_buf=ah, src_buf=ta)
    ctx.enqueue_copy(dst_buf=bh, src_buf=tb)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pb)
    ctx.synchronize()
    var same = 0
    for r in range(RG):
        if ah[r] == bh[r] and ph[r] == 1.0:
            same += 1
        else:
            print("  row", r, "argmax", ah[r], "sample T=0", bh[r], "prob", ph[r])
    print("P-K2 temperature 0 vs amar_argmax_row: rows equal", same, "of", RG, "(V =", VR, ")")
    if same != RG:
        fail("temperature 0 differs from amar_argmax_row")

    ctx.enqueue_function[sm](
        X, TileTensor(tb, tg_l), TileTensor(pb, tg_l), Int32(VR), Float32(0.7), Int32(20), Float32(0.8),
        Float32(0), UInt64(3), UInt64(4), grid_dim=RG, block_dim=SAMP_THREADS,
    )
    ctx.enqueue_copy(dst_buf=bh, src_buf=tb)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pb)
    ctx.synchronize()
    for r in range(RG):
        var t = Int(bh[r])
        if r == 12:
            if t != -1 or ph[r] != 0.0:
                fail("all -inf row must give token -1, prob 0")
            continue
        if t < 0 or t >= VR or not is_valid(xh[r * VR + t]) or not (ph[r] > 0.0 and ph[r] <= 1.0):
            fail("sampled token invalid on row " + String(r))
        var v = xh[r * VR + t]
        var rank = 0
        for i in range(VR):
            var w = xh[r * VR + i]
            if is_valid(w) and (w > v or (w == v and i < t)):
                rank += 1
        if rank >= 20:
            fail("sampled token outside top-20 on row " + String(r))
    print("P-K2 sampled rows at V: tokens valid and inside top-20 (ties, NaN, -inf rows included)")


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    kat()
    var ctx = DeviceContext()
    greedy_identity(ctx)

    var la = List[Float32]()
    var lb = List[Float32]()
    var ld = List[Float32]()
    for i in range(VS):
        var a = 2.5 * sin(0.37 * Float64(i) + 0.3) + 0.8 * cos(1.3 * Float64(i))
        la.append(Float32(a))
        lb.append(Float32(0.4) * Float32(i % 8))
        ld.append(Float32(a + 1.5 * sin(2.1 * Float64(i))))

    var cfgs = List[Cfg]()
    cfgs.append(Cfg("T1 k- p-", 1.0, 0, 1.0, 0.0))
    cfgs.append(Cfg("T0.7 k20 p0.8", 0.7, 20, 0.8, 0.0))
    cfgs.append(Cfg("T1.3 k12 p0.9 minp0.05", 1.3, 12, 0.9, 0.05))
    cfgs.append(Cfg("T0.5 k- p0.6", 0.5, 0, 0.6, 0.0))
    cfgs.append(Cfg("ties T1 k12", 1.0, 12, 1.0, 0.0))
    cfgs.append(Cfg("ties T0.9 p0.3", 0.9, 0, 0.3, 0.0))

    var xd = ctx.enqueue_create_buffer[f32](ND * VS)
    var td = ctx.enqueue_create_buffer[i32](ND)
    var pd = ctx.enqueue_create_buffer[f32](ND)
    print("P-K3 grid", ND, "block", SAMP_THREADS, "V", VS, "draws", ND)
    for ci in range(len(cfgs)):
        ref c = cfgs[ci]
        var l = lb.copy() if ci >= 4 else la.copy()
        var q = ref_q(l, c)
        fill_rows(ctx, xd, l)
        var r = draw(ctx, xd, td, pd, c, UInt64(1000 + ci), UInt64(7))
        chi2(c.name, hist_of(r[0]), q, ND)
        var worst = 0.0
        for i in range(ND):
            var e = abs(Float64(r[1][i]) - q[Int(r[0][i])])
            if e > worst:
                worst = e
        var pr = probs_row(ctx, l, c)
        var sum = 0.0
        var pw = 0.0
        for i in range(VS):
            sum += Float64(pr[i])
            var e = abs(Float64(pr[i]) - q[i])
            if e > pw:
                pw = e
        print("     prob max err", worst, "| probs row max err", pw, "sum", sum)
        if worst > 1e-5 or pw > 1e-5 or abs(sum - 1.0) > 1e-5:
            fail(c.name + ": probability mismatch")
        path_identity(ctx, xd, td, pd, l, c, r[0], r[1], pr, UInt64(1000 + ci), UInt64(7))
        if ci == 0 or ci == 1:
            var r2 = draw(ctx, xd, td, pd, c, UInt64(1000 + ci), UInt64(7))
            var r3 = draw(ctx, xd, td, pd, c, UInt64(5000 + ci), UInt64(7))
            var same = 0
            var diff = 0
            for i in range(ND):
                if r2[0][i] == r[0][i]:
                    same += 1
                if r3[0][i] != r[0][i]:
                    diff += 1
            print("P-K4", c.name, ": same seed equal", same, "of", ND, "| other seed differs", diff)
            if same != ND:
                fail("same (seed, counter) gave different tokens")
            if ci == 0 and diff < ND // 2:
                fail("different seed did not change the draws")
    print("P-K3 PASS, P-K4 PASS")

    var cs = Cfg("spec T0.9 k24 p0.95", 0.9, 24, 0.95, 0.0)
    var qt = ref_q(la, cs)
    var qd = ref_q(ld, cs)
    var alpha = 0.0
    for i in range(VS):
        alpha += min(qt[i], qd[i])
    var pt_row = probs_row(ctx, la, cs)
    var pd_row = probs_row(ctx, ld, cs)
    fill_rows(ctx, xd, ld)
    var dr = draw(ctx, xd, td, pd, cs, UInt64(11), UInt64(5))
    var ptd = ctx.enqueue_create_buffer[f32](ND * VS)
    var pdd = ctx.enqueue_create_buffer[f32](ND * VS)
    fill_rows(ctx, ptd, pt_row)
    fill_rows(ctx, pdd, pd_row)
    var oud = ctx.enqueue_create_buffer[i32](ND)
    var acd = ctx.enqueue_create_buffer[i32](ND)
    comptime sp = amar_spec_accept[type_of(xs_l), type_of(ts_l)]
    ctx.enqueue_function[sp](
        TileTensor(ptd, xs_l), TileTensor(pdd, xs_l), TileTensor(td, ts_l), TileTensor(oud, ts_l),
        TileTensor(acd, ts_l), Int32(VS), UInt64(12), UInt64(5), grid_dim=ND, block_dim=SAMP_THREADS,
    )
    var oh = ctx.enqueue_create_host_buffer[i32](ND)
    var ah = ctx.enqueue_create_host_buffer[i32](ND)
    ctx.enqueue_copy(dst_buf=oh, src_buf=oud)
    ctx.enqueue_copy(dst_buf=ah, src_buf=acd)
    ctx.synchronize()
    path_identity(ctx, xd, td, pd, ld, cs, dr[0], dr[1], pd_row, UInt64(11), UInt64(5))
    var ys = List[Int32]()
    var nacc = 0
    for i in range(ND):
        ys.append(oh[i])
        if ah[i] == 1:
            nacc += 1
            if oh[i] != dr[0][i]:
                fail("accepted row does not emit the draft token")
    var rate = Float64(nacc) / Float64(ND)
    var sd = sqrt(alpha * (1.0 - alpha) / Float64(ND))
    print("P-K5 acceptance", rate, "expected", alpha, "sigma", sd)
    if abs(rate - alpha) > 4.0 * sd:
        fail("acceptance rate off the exact value")
    chi2("spec accepted-or-resampled vs p_t", hist_of(ys), qt, ND)
    fill_rows(ctx, xd, la)
    var direct = draw(ctx, xd, td, pd, cs, UInt64(13), UInt64(5))
    chi2("direct p_t draws", hist_of(direct[0]), qt, ND)
    chi2_two("spec vs direct", hist_of(ys), hist_of(direct[0]))
    path_identity(ctx, xd, td, pd, la, cs, direct[0], direct[1], pt_row, UInt64(13), UInt64(5))
    print("P-K5 PASS, P-K7 PASS")

    var rh = ctx.enqueue_create_host_buffer[f32](VR)
    ctx.synchronize()
    for i in range(VR):
        rh[i] = Float32(2.5 * gauss(100, i))
    for j in range(5):
        rh[Int(hu(21, j) * Float64(VR))] = Float32(10.0 + Float64(j))
    var rd = ctx.enqueue_create_buffer[f32](VR)
    var r1 = ctx.enqueue_create_buffer[i32](1)
    var p1 = ctx.enqueue_create_buffer[f32](1)
    ctx.enqueue_copy(dst_buf=rd, src_buf=rh)
    ctx.synchronize()
    var XR = TileTensor(rd, xr_l)
    var TR = TileTensor(r1, t1_l)
    comptime am1 = amar_argmax_row[type_of(xr_l), type_of(t1_l)]
    for _ in range(200):
        ctx.enqueue_function[am1](XR, TR, Int32(VR), grid_dim=1, block_dim=EW_THREADS)
    ctx.synchronize()
    var blocks = List[Float64]()
    for _ in range(11):
        var t0 = perf_counter_ns()
        for _ in range(100):
            ctx.enqueue_function[am1](XR, TR, Int32(VR), grid_dim=1, block_dim=EW_THREADS)
        ctx.synchronize()
        blocks.append(Float64(perf_counter_ns() - t0) / 1e3 / 100.0)
    var s = med3(blocks)
    print(
        "P-K8 arm amar_argmax_row gaussian | V", VR, "R 1 grid 1 block", EW_THREADS,
        ": median", s[1], "us/call, min", s[0], "max", s[2], "(11 blocks x 100)",
    )
    var tcfg = List[Cfg]()
    tcfg.append(Cfg("greedy T0", 0.0, 0, 1.0, 0.0))
    tcfg.append(Cfg("qwen T0.7 k20 p0.8", 0.7, 20, 0.8, 0.0))
    tcfg.append(Cfg("llama T0.8 k40 p0.95 minp0.05", 0.8, 40, 0.95, 0.05))
    tcfg.append(Cfg("p-only T1 k- p0.95", 1.0, 0, 0.95, 0.0))
    tcfg.append(Cfg("plain T1 k- p-", 1.0, 0, 1.0, 0.0))
    for ci in range(len(tcfg)):
        time_arm(ctx, rd, r1, p1, "gaussian", tcfg[ci])
    for j in range(5):
        rh[Int(hu(21, j) * Float64(VR))] = Float32(20.0 + Float64(j))
    ctx.enqueue_copy(dst_buf=rd, src_buf=rh)
    ctx.synchronize()
    for ci in range(1, 4):
        time_arm(ctx, rd, r1, p1, "peaked", tcfg[ci])
    print("PASS: device sampler")
