"""Device sampler vs host reference, on real fixed logits rows (M5,
briefs/2026-09-15-wiring-lane.md gate 2, extended for the C3 fix round,
bench/chat-protocol.md "C3 fix round: host top-p mass target").

Two checks, both at real VOCAB width (not kernels/test_sample.mojo's VS=64
synthetic vocab):
  1. Per-token: amar_sample_row (device) vs sample_row_ref (host, C3-fixed)
     on the same (seed, counter) draws, every config shape, every row.
  2. Distribution: 20000 device draws per (row, config), binned by rank
     against an INDEPENDENT numpy oracle (tools/sample-nucleus-oracle.py;
     neither kernels/sample.mojo nor serve/sample_ref.mojo), chi-square at a
     preregistered p=0.001 critical value.

Not wired into run-tests.sh yet (bench/chat-protocol.md: "wire it into
run-tests.sh only once green") -- gate 2's T1_k0_p1 (no truncation) shape
currently fails on 2 of 3 rows; see the C3 Result in bench/chat-protocol.md.

Build: ./.venv/bin/mojo build kernels/test_sample_device.mojo -I kernels -I serve -o .work/test_sample_device
Needs .work/m5/logits-p01.bin, logits-p02.bin, logits-p03.bin (real decode
rows from three different prompts, and their oracle files
.work/m5/oracle-pNN-<slug>.txt); skips a row cleanly if its files are
absent. To regenerate:
  mkdir -p .work/m5
  gpu-wait run --vram 20 -- bash -c '
    BARO_PACK=.work/engine-pack-q4 BARO_PROMPT=bench/mtp-prompts/p01-water.tokens .work/engine
    cp .work/draft-logits.bin .work/m5/logits-p01.bin
    BARO_PACK=.work/engine-pack-q4 BARO_PROMPT=bench/mtp-prompts/p02-python-fib.tokens .work/engine
    cp .work/draft-logits.bin .work/m5/logits-p02.bin
    BARO_PACK=.work/engine-pack-q4 BARO_PROMPT=bench/mtp-prompts/p03-story.tokens .work/engine
    cp .work/draft-logits.bin .work/m5/logits-p03.bin
  '
  for p in p01 p02 p03; do ./.venv/bin/python3 tools/sample-nucleus-oracle.py .work/m5/logits-$p.bin .work/m5/oracle-$p; done
"""
from std.math import exp, log, sqrt
from std.memory import unsafe_memcpy
from std.sys import has_accelerator, exit
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from registry import *
from sample import amar_sample_row, amar_sample_row_masked, amar_sample_probs, amar_spec_accept, SAMP_THREADS, SAMP_CAP
from sample_ref import sample_row_ref, sample_probs_ref, spec_accept_ref, is_valid

comptime FMAX = Float32(3.4028234663852886e38)
comptime x_l = row_major[1, VOCAB]()
comptime o_l = row_major[1]()
comptime BATCH = 1000
comptime xb_l = row_major[BATCH, VOCAB]()
comptime tb_l = row_major[BATCH]()
comptime N_DRAWS = 20000
comptime N_BATCHES = N_DRAWS // BATCH
# Speculative gates (A1, bench/spec-sample-protocol.md). Smaller batch than the
# sampler gates because each spec draw needs three real-vocab rows resident per
# lane (target probs, draft probs, draft logits) instead of one: 200 x VOCAB x 4
# is about 200 MB per buffer, and the three together stay well inside the card
# while 1000 would not.
comptime SBATCH = 200
comptime S_BATCHES = N_DRAWS // SBATCH
comptime xs_l = row_major[SBATCH, VOCAB]()
comptime ts_l = row_major[SBATCH]()


@fieldwise_init
struct Cfg(Copyable, Movable):
    var slug: String
    var t: Float32
    var k: Int
    var p: Float32
    var mp: Float32


def configs() -> List[Cfg]:
    # Must match .work/m5/oracle2.py's CONFIGS exactly (name and values), so
    # the two sides read the same shape by the same slug.
    var c = List[Cfg]()
    c.append(Cfg("T1_k0_p1", Float32(1.0), 0, Float32(1.0), Float32(0.0)))
    c.append(Cfg("T0.8_k30_p1", Float32(0.8), 30, Float32(1.0), Float32(0.0)))
    c.append(Cfg("T0.7_k20_p0.8", Float32(0.7), 20, Float32(0.8), Float32(0.0)))
    c.append(Cfg("T1.3_k12_p0.9_minp0.05", Float32(1.3), 12, Float32(0.9), Float32(0.05)))
    c.append(Cfg("T0.5_k0_p0.6", Float32(0.5), 0, Float32(0.6), Float32(0.0)))
    return c^


def rows() -> List[Tuple[String, String]]:
    var r = List[Tuple[String, String]]()
    r.append(("p01-water", ".work/m5/logits-p01.bin"))
    r.append(("p02-python-fib", ".work/m5/logits-p02.bin"))
    r.append(("p03-story", ".work/m5/logits-p03.bin"))
    return r^


def load_logits(path: String) raises -> List[Float32]:
    with open(path, "r") as f:
        var data = f.read_bytes()
        var n = len(data) // 4
        var row = List[Float32](unsafe_uninit_length=n)
        var p = data.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n):
            row[i] = p[i]
        return row^


def device_sample(
    ctx: DeviceContext, mut xd: DeviceBuffer[f32], t: Float32, k: Int, p: Float32, mp: Float32, seed: UInt64, counter: UInt64,
) raises -> Tuple[Int, Float32]:
    var td = ctx.enqueue_create_buffer[DType.int32](1)
    var pd = ctx.enqueue_create_buffer[f32](1)
    comptime kern = amar_sample_row[type_of(x_l), type_of(o_l), type_of(o_l)]
    ctx.enqueue_function[kern](
        TileTensor(xd, x_l), TileTensor(td, o_l), TileTensor(pd, o_l), Int32(VOCAB),
        t, Int32(k), p, mp, seed, counter, grid_dim=1, block_dim=SAMP_THREADS,
    )
    var th = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ph = ctx.enqueue_create_host_buffer[f32](1)
    ctx.enqueue_copy(dst_buf=th, src_buf=td)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    return (Int(th[0]), Float32(ph[0]))


def device_sample_masked(
    ctx: DeviceContext, mut xd: DeviceBuffer[f32], mut md: DeviceBuffer[DType.uint64], t: Float32, k: Int, p: Float32, mp: Float32, seed: UInt64, counter: UInt64,
) raises -> Tuple[Int, Float32]:
    var td = ctx.enqueue_create_buffer[DType.int32](1)
    var pd = ctx.enqueue_create_buffer[f32](1)
    comptime kern = amar_sample_row_masked[type_of(x_l), type_of(o_l), type_of(o_l)]
    ctx.enqueue_function[kern](
        TileTensor(xd, x_l), TileTensor(td, o_l), TileTensor(pd, o_l), Int32(VOCAB),
        t, Int32(k), p, mp, seed, counter, md.unsafe_ptr(), Int32((VOCAB + 63) // 64), grid_dim=1, block_dim=SAMP_THREADS,
    )
    var th = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ph = ctx.enqueue_create_host_buffer[f32](1)
    ctx.enqueue_copy(dst_buf=th, src_buf=td)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    return (Int(th[0]), Float32(ph[0]))


def gate_mask_row(ctx: DeviceContext, mut xd: DeviceBuffer[f32], row_name: String, argmax: Int, mut fails: Int) raises:
    # A5 device half: a full mask reproduces the unmasked draw; a mask that
    # clears the argmax never returns it and still matches the host draw on
    # the row with that logit removed.
    comptime NW = (VOCAB + 63) // 64
    var mh = ctx.enqueue_create_host_buffer[DType.uint64](NW)
    var md = ctx.enqueue_create_buffer[DType.uint64](NW)
    ctx.synchronize()
    for i in range(NW):
        mh[i] = UInt64(0xFFFFFFFFFFFFFFFF)
    ctx.enqueue_copy(dst_buf=md, src_buf=mh)
    ctx.synchronize()
    var bad = 0
    for counter in range(16):
        var a = device_sample(ctx, xd, Float32(0.7), 20, Float32(0.8), Float32(0.0), UInt64(42), UInt64(counter))
        var b = device_sample_masked(ctx, xd, md, Float32(0.7), 20, Float32(0.8), Float32(0.0), UInt64(42), UInt64(counter))
        if a[0] != b[0] or a[1] != b[1]:
            bad += 1
    mh[argmax // 64] = mh[argmax // 64] & ~(UInt64(1) << UInt64(argmax % 64))
    ctx.enqueue_copy(dst_buf=md, src_buf=mh)
    ctx.synchronize()
    var hit_argmax = 0
    for counter in range(16):
        var b = device_sample_masked(ctx, xd, md, Float32(1.0), 0, Float32(1.0), Float32(0.0), UInt64(42), UInt64(counter))
        if b[0] == argmax:
            hit_argmax += 1
    if bad == 0 and hit_argmax == 0:
        print("PASS mask", row_name, ": full mask == unmasked on 16 draws; argmax masked out never drawn in 16 draws")
    else:
        print("FAIL mask", row_name, ": full-mask mismatches", bad, " masked argmax drawn", hit_argmax)
        fails += 1
    # T=0 masked greedy: argmax over the allowed set only, -1 on an empty mask.
    var xh = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=xh, src_buf=xd)
    ctx.synchronize()
    var second = -1
    for i in range(VOCAB):
        if i != argmax and (second < 0 or xh[i] > xh[second]):
            second = i
    var g_cleared = device_sample_masked(ctx, xd, md, Float32(0), 0, Float32(1), Float32(0), UInt64(42), UInt64(0))[0]
    var only = VOCAB - 7
    for i in range(NW):
        mh[i] = UInt64(0)
    mh[only // 64] = UInt64(1) << UInt64(only % 64)
    ctx.enqueue_copy(dst_buf=md, src_buf=mh)
    ctx.synchronize()
    var g_only = device_sample_masked(ctx, xd, md, Float32(0), 0, Float32(1), Float32(0), UInt64(42), UInt64(0))[0]
    mh[only // 64] = UInt64(0)
    ctx.enqueue_copy(dst_buf=md, src_buf=mh)
    ctx.synchronize()
    var g_empty = device_sample_masked(ctx, xd, md, Float32(0), 0, Float32(1), Float32(0), UInt64(42), UInt64(0))[0]
    for i in range(NW):
        mh[i] = UInt64(0xFFFFFFFFFFFFFFFF)
    ctx.enqueue_copy(dst_buf=md, src_buf=mh)
    ctx.synchronize()
    var g_full = device_sample_masked(ctx, xd, md, Float32(0), 0, Float32(1), Float32(0), UInt64(42), UInt64(0))[0]
    if g_cleared == second and g_only == only and g_empty == -1 and g_full == argmax:
        print("PASS mask T=0", row_name, ": argmax cleared ->", second, ", single bit ->", only, ", empty -> -1, full -> argmax")
    else:
        print("FAIL mask T=0", row_name, ": cleared", g_cleared, "want", second, " single", g_only, "want", only, " empty", g_empty, " full", g_full, "want", argmax)
        fails += 1


def gate1_row(ctx: DeviceContext, mut xd: DeviceBuffer[f32], row_name: String, row: List[Float32], mut fails: Int) raises:
    for cfg in configs():
        var mismatches = 0
        var checked = 0
        for counter in range(64):
            var d = device_sample(ctx, xd, cfg.t, cfg.k, cfg.p, cfg.mp, UInt64(42), UInt64(counter))
            var h = sample_row_ref(row, cfg.t, cfg.k, cfg.p, cfg.mp, UInt64(42), UInt64(counter), 0)
            checked += 1
            if d[0] != Int(h[0]) or abs(d[1] - h[1]) > Float32(1e-4):
                mismatches += 1
                if mismatches <= 3:
                    print("    mismatch counter", counter, ": device", d[0], d[1], "host", h[0], h[1])
        if mismatches == 0:
            print("  PASS gate1", row_name, cfg.slug, ": device == host on all", checked, "draws")
        else:
            print("  FAIL gate1", row_name, cfg.slug, ":", mismatches, "of", checked, "draws disagree")
            fails += 1


def path_exists(path: String) -> Bool:
    try:
        with open(path, "r"):
            return True
    except:
        return False


def load_oracle(path: String, mut ids: List[Int], mut probs: List[Float64], mut rest: Float64) raises:
    with open(path, "r") as f:
        var lines = f.read().splitlines()
        var head = lines[0].split(" ")
        var n = Int(head[0])
        rest = Float64(head[1])
        for i in range(1, n + 1):
            var parts = lines[i].split(" ")
            ids.append(Int(parts[0]))
            probs.append(Float64(parts[1]))


def crit(df: Int) -> Float64:
    if df <= 0:
        return 1e-9
    var d = Float64(df)
    var a = 2.0 / (9.0 * d)
    var c = 1.0 - a + 3.090232 * sqrt(a)
    return d * c * c * c


def gate2_row(ctx: DeviceContext, mut xd: DeviceBuffer[f32], row_name: String, mut fails: Int) raises:
    for cfg in configs():
        var oracle_path = ".work/m5/oracle-" + row_name.split("-")[0] + "-" + cfg.slug + ".txt"
        if not path_exists(oracle_path):
            print("  SKIP gate2", row_name, cfg.slug, ": no oracle file", oracle_path)
            continue
        var ids = List[Int]()
        var probs = List[Float64]()
        var rest = Float64(0)
        load_oracle(oracle_path, ids, probs, rest)
        var counts = List[Int](capacity=len(ids))
        for _ in range(len(ids)):
            counts.append(0)
        var rest_count = 0
        var t0 = perf_counter_ns()
        for b in range(N_BATCHES):
            var td = ctx.enqueue_create_buffer[DType.int32](BATCH)
            var pd = ctx.enqueue_create_buffer[f32](BATCH)
            comptime kern = amar_sample_row[type_of(xb_l), type_of(tb_l), type_of(tb_l)]
            ctx.enqueue_function[kern](
                TileTensor(xd, xb_l), TileTensor(td, tb_l), TileTensor(pd, tb_l), Int32(VOCAB),
                cfg.t, Int32(cfg.k), cfg.p, cfg.mp, UInt64(1000 + b), UInt64(b), grid_dim=BATCH, block_dim=SAMP_THREADS,
            )
            var th = ctx.enqueue_create_host_buffer[DType.int32](BATCH)
            ctx.enqueue_copy(dst_buf=th, src_buf=td)
            ctx.synchronize()
            for i in range(BATCH):
                var tok = Int(th[i])
                var found = False
                for j in range(len(ids)):
                    if ids[j] == tok:
                        counts[j] += 1
                        found = True
                        break
                if not found:
                    rest_count += 1
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        var n = N_BATCHES * BATCH
        var stat = 0.0
        var bins = 0
        var po = 0.0
        var pe = 0.0
        var lo = 0.0
        var le = 0.0
        for j in range(len(ids)):
            var e = probs[j] * Float64(n)
            if e >= 5.0:
                stat += (Float64(counts[j]) - e) ** 2 / e
                bins += 1
                lo = Float64(counts[j])
                le = e
            else:
                po += Float64(counts[j])
                pe += e
        if rest > 0:
            var er = rest * Float64(n)
            po += Float64(rest_count)
            pe += er
        if pe >= 5.0:
            stat += (po - pe) ** 2 / pe
            bins += 1
        elif pe > 0.0:
            stat -= (lo - le) ** 2 / le
            stat += (lo + po - le - pe) ** 2 / (le + pe)
        var cv = crit(bins - 1)
        var over = stat >= cv and bins > 1
        print("  ", "PASS" if not over else "FAIL", "gate2", row_name, cfg.slug, ": chi2", stat, "df", bins - 1, "crit(p=0.001)", cv, "n", n, "candidates", len(ids), "rest_count", rest_count, "elapsed_s", dt)
        if over:
            fails += 1


def spec_configs() -> List[Cfg]:
    # Both shapes have an oracle file already (the slugs match configs()), so
    # the spec gates compare against the same independent numpy distribution
    # the sampler gates use, not against our own reference.
    var c = List[Cfg]()
    c.append(Cfg("T1_k0_p1", Float32(1.0), 0, Float32(1.0), Float32(0.0)))
    c.append(Cfg("T0.7_k20_p0.8", Float32(0.7), 20, Float32(0.8), Float32(0.0)))
    return c^


def fill_rows(ctx: DeviceContext, mut dst: DeviceBuffer[f32], row: List[Float32], n: Int) raises:
    var h = ctx.enqueue_create_host_buffer[f32](n * VOCAB)
    ctx.synchronize()
    for r in range(n):
        unsafe_memcpy(
            dest=h.unsafe_ptr().unsafe_offset(r * VOCAB).unsafe_bitcast[UInt8](),
            src=row.unsafe_ptr().unsafe_bitcast[UInt8](),
            count=VOCAB * 4,
        )
    ctx.enqueue_copy(dst_buf=dst, src_buf=h)
    ctx.synchronize()


def probs_device(ctx: DeviceContext, mut xd: DeviceBuffer[f32], c: Cfg) raises -> List[Float32]:
    """One truncated probability row from the device kernel, read back."""
    var pd = ctx.enqueue_create_buffer[f32](VOCAB)
    comptime k = amar_sample_probs[type_of(x_l), type_of(x_l), SAMP_CAP]
    ctx.enqueue_function[k](
        TileTensor(xd, x_l), TileTensor(pd, x_l), Int32(VOCAB), c.t, Int32(c.k), c.p, c.mp,
        grid_dim=1, block_dim=SAMP_THREADS,
    )
    var ph = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    var out = List[Float32](unsafe_uninit_length=VOCAB)
    for i in range(VOCAB):
        out[i] = ph[i]
    return out^


def gate3_row(
    ctx: DeviceContext, row_name: String, label: String, target: List[Float32], draft: List[Float32],
    dtemp: Float32, mut fails: Int,
) raises:
    """Per-draw equality: device amar_spec_accept == spec_accept_ref.

    The draft distribution comes from a DIFFERENT prompt's logits row, so the
    accept branch is not the only one exercised: with q far from p, most draws
    reject and take the residual norm(max(0, p - q)) path, which is the part
    that carries the C3 tail fix.
    """
    var xt = ctx.enqueue_create_buffer[f32](VOCAB)
    var xq = ctx.enqueue_create_buffer[f32](VOCAB)
    fill_rows(ctx, xt, target, 1)
    fill_rows(ctx, xq, draft, 1)
    for cfg in spec_configs():
        var dc = Cfg(cfg.slug, cfg.t * dtemp, cfg.k, cfg.p, cfg.mp)
        var pt_h = sample_probs_ref(target, cfg.t, cfg.k, cfg.p, cfg.mp)
        var pd_h = sample_probs_ref(draft, dc.t, dc.k, dc.p, dc.mp)
        var pt_d_row = probs_device(ctx, xt, cfg)
        var pd_d_row = probs_device(ctx, xq, dc)
        var pmax = Float32(0)
        for i in range(VOCAB):
            pmax = max(pmax, abs(pt_d_row[i] - pt_h[i]))
            pmax = max(pmax, abs(pd_d_row[i] - pd_h[i]))
        var ptd = ctx.enqueue_create_buffer[f32](VOCAB)
        var pdd = ctx.enqueue_create_buffer[f32](VOCAB)
        fill_rows(ctx, ptd, pt_d_row, 1)
        fill_rows(ctx, pdd, pd_d_row, 1)
        var mismatches = 0
        var accepts = 0
        for counter in range(64):
            var dx = device_sample(ctx, xq, dc.t, dc.k, dc.p, dc.mp, UInt64(7), UInt64(counter))
            var x = dx[0]
            var td = ctx.enqueue_create_buffer[DType.int32](1)
            var oud = ctx.enqueue_create_buffer[DType.int32](1)
            var acd = ctx.enqueue_create_buffer[DType.int32](1)
            var th0 = ctx.enqueue_create_host_buffer[DType.int32](1)
            ctx.synchronize()
            th0[0] = Int32(x)
            ctx.enqueue_copy(dst_buf=td, src_buf=th0)
            comptime sp = amar_spec_accept[type_of(x_l), type_of(o_l)]
            ctx.enqueue_function[sp](
                TileTensor(ptd, x_l), TileTensor(pdd, x_l), TileTensor(td, o_l),
                TileTensor(oud, o_l), TileTensor(acd, o_l), Int32(VOCAB), UInt64(9), UInt64(counter),
                grid_dim=1, block_dim=SAMP_THREADS,
            )
            var oh = ctx.enqueue_create_host_buffer[DType.int32](1)
            var ah = ctx.enqueue_create_host_buffer[DType.int32](1)
            ctx.enqueue_copy(dst_buf=oh, src_buf=oud)
            ctx.enqueue_copy(dst_buf=ah, src_buf=acd)
            ctx.synchronize()
            var h = spec_accept_ref(pt_d_row, pd_d_row, x, UInt64(9), UInt64(counter), 0)
            if ah[0] == 1:
                accepts += 1
            if Int(oh[0]) != Int(h[0]) or (ah[0] == 1) != h[1]:
                mismatches += 1
                if mismatches <= 3:
                    print("    mismatch counter", counter, "x", x, ": device", oh[0], ah[0], "host", h[0], h[1])
        if mismatches == 0:
            print("  PASS gate3", row_name, label, cfg.slug, ": device == host on all 64 spec draws, accepted",
                  accepts, "of 64, max |probs device - host|", pmax)
        else:
            print("  FAIL gate3", row_name, label, cfg.slug, ":", mismatches, "of 64 spec draws disagree")
            fails += 1


def gate4_row(
    ctx: DeviceContext, row_name: String, label: String, target: List[Float32], draft: List[Float32],
    dtemp: Float32, mut fails: Int,
) raises:
    """Distribution: the tokens the speculative rule emits are distributed as p.

    That is the whole theorem (Leviathan 2211.17192), so the binning and the
    critical value are gate2's, against the same independent numpy oracle: if
    the rule is right, accepting from q and resampling the residual is
    indistinguishable from sampling p directly.
    """
    var xt = ctx.enqueue_create_buffer[f32](VOCAB)
    var xq1 = ctx.enqueue_create_buffer[f32](VOCAB)
    fill_rows(ctx, xt, target, 1)
    fill_rows(ctx, xq1, draft, 1)
    for cfg in spec_configs():
        var oracle_path = ".work/m5/oracle-" + row_name.split("-")[0] + "-" + cfg.slug + ".txt"
        if not path_exists(oracle_path):
            print("  SKIP gate4", row_name, label, cfg.slug, ": no oracle file", oracle_path)
            continue
        var ids = List[Int]()
        var probs = List[Float64]()
        var rest = Float64(0)
        load_oracle(oracle_path, ids, probs, rest)
        var dc = Cfg(cfg.slug, cfg.t * dtemp, cfg.k, cfg.p, cfg.mp)
        var pt_row = probs_device(ctx, xt, cfg)
        var pd_row = probs_device(ctx, xq1, dc)
        var ptd = ctx.enqueue_create_buffer[f32](SBATCH * VOCAB)
        var pdd = ctx.enqueue_create_buffer[f32](SBATCH * VOCAB)
        var xqb = ctx.enqueue_create_buffer[f32](SBATCH * VOCAB)
        fill_rows(ctx, ptd, pt_row, SBATCH)
        fill_rows(ctx, pdd, pd_row, SBATCH)
        fill_rows(ctx, xqb, draft, SBATCH)
        var counts = List[Int](capacity=len(ids))
        for _ in range(len(ids)):
            counts.append(0)
        var rest_count = 0
        var accepted = 0
        var t0 = perf_counter_ns()
        for b in range(S_BATCHES):
            var td = ctx.enqueue_create_buffer[DType.int32](SBATCH)
            var qp = ctx.enqueue_create_buffer[f32](SBATCH)
            comptime kern = amar_sample_row[type_of(xs_l), type_of(ts_l), type_of(ts_l)]
            ctx.enqueue_function[kern](
                TileTensor(xqb, xs_l), TileTensor(td, ts_l), TileTensor(qp, ts_l), Int32(VOCAB),
                dc.t, Int32(dc.k), dc.p, dc.mp, UInt64(3000 + b), UInt64(b), grid_dim=SBATCH, block_dim=SAMP_THREADS,
            )
            var oud = ctx.enqueue_create_buffer[DType.int32](SBATCH)
            var acd = ctx.enqueue_create_buffer[DType.int32](SBATCH)
            comptime sp = amar_spec_accept[type_of(xs_l), type_of(ts_l)]
            ctx.enqueue_function[sp](
                TileTensor(ptd, xs_l), TileTensor(pdd, xs_l), TileTensor(td, ts_l),
                TileTensor(oud, ts_l), TileTensor(acd, ts_l), Int32(VOCAB), UInt64(4000 + b), UInt64(b),
                grid_dim=SBATCH, block_dim=SAMP_THREADS,
            )
            var oh = ctx.enqueue_create_host_buffer[DType.int32](SBATCH)
            var ah = ctx.enqueue_create_host_buffer[DType.int32](SBATCH)
            ctx.enqueue_copy(dst_buf=oh, src_buf=oud)
            ctx.enqueue_copy(dst_buf=ah, src_buf=acd)
            ctx.synchronize()
            for i in range(SBATCH):
                if ah[i] == 1:
                    accepted += 1
                var tok = Int(oh[i])
                var found = False
                for j in range(len(ids)):
                    if ids[j] == tok:
                        counts[j] += 1
                        found = True
                        break
                if not found:
                    rest_count += 1
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        var n = S_BATCHES * SBATCH
        var stat = 0.0
        var bins = 0
        var po = 0.0
        var pe = 0.0
        var lo = 0.0
        var le = 0.0
        for j in range(len(ids)):
            var e = probs[j] * Float64(n)
            if e >= 5.0:
                stat += (Float64(counts[j]) - e) ** 2 / e
                bins += 1
                lo = Float64(counts[j])
                le = e
            else:
                po += Float64(counts[j])
                pe += e
        if rest > 0:
            var er = rest * Float64(n)
            po += Float64(rest_count)
            pe += er
        if pe >= 5.0:
            stat += (po - pe) ** 2 / pe
            bins += 1
        elif pe > 0.0:
            stat -= (lo - le) ** 2 / le
            stat += (lo + po - le - pe) ** 2 / (le + pe)
        var cv = crit(bins - 1)
        var over = stat >= cv and bins > 1
        print("  ", "PASS" if not over else "FAIL", "gate4", row_name, label, cfg.slug, ": chi2", stat, "df", bins - 1,
              "crit(p=0.001)", cv, "n", n, "accepted", accepted, "rest_count", rest_count, "elapsed_s", dt)
        if over:
            fails += 1


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    var ctx = DeviceContext()
    var fails = 0
    var ran_any = False
    # Every row is loaded up front, because the speculative gates need a
    # SECOND real row as the draft distribution (row i verifies against row
    # i+1's logits): a draft equal to the target accepts every draw and never
    # exercises the residual, which is the branch the C3 tail fix lives in.
    var names = List[String]()
    var rows_l = List[List[Float32]]()
    for row_spec in rows():
        var row: List[Float32]
        try:
            row = load_logits(row_spec[1])
        except:
            print("SKIP", row_spec[0], ": ", row_spec[1], "not present")
            continue
        if len(row) != VOCAB:
            print("SKIP", row_spec[0], ": ", len(row), "entries, this build's VOCAB is", VOCAB)
            continue
        names.append(row_spec[0])
        rows_l.append(row^)
    for ri in range(len(names)):
        var row_name = names[ri]
        ref row = rows_l[ri]
        ran_any = True
        print("== row", row_name)

        var xh = ctx.enqueue_create_host_buffer[f32](VOCAB)
        ctx.synchronize()
        for i in range(VOCAB):
            xh[i] = row[i]
        var xd = ctx.enqueue_create_buffer[f32](VOCAB)
        ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
        ctx.synchronize()

        # Greedy sanity check (temperature = 0): device must match host.
        var dg = device_sample(ctx, xd, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0))
        var hg = sample_row_ref(row, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0), 0)
        gate_mask_row(ctx, xd, row_name, Int(hg[0]), fails)
        if dg[0] == Int(hg[0]) and dg[1] == hg[1]:
            print("  PASS greedy (T=0): device token", dg[0], "prob", dg[1], "== host")
        else:
            print("  FAIL greedy (T=0): device", dg[0], dg[1], "!= host", hg[0], hg[1])
            fails += 1

        gate1_row(ctx, xd, row_name, row, fails)

        # Batch buffer for gate 2: BATCH copies of the same row, filled once,
        # reused across every config (the logits do not change per config).
        var xbh = ctx.enqueue_create_host_buffer[f32](BATCH * VOCAB)
        ctx.synchronize()
        for r in range(BATCH):
            unsafe_memcpy(
                dest=xbh.unsafe_ptr().unsafe_offset(r * VOCAB).unsafe_bitcast[UInt8](),
                src=row.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=VOCAB * 4,
            )
        var xbd = ctx.enqueue_create_buffer[f32](BATCH * VOCAB)
        ctx.enqueue_copy(dst_buf=xbd, src_buf=xbh)
        ctx.synchronize()
        gate2_row(ctx, xbd, row_name, fails)

        # Speculative gates (A1): draft = the next row's logits, wrapping.
        # Two draft arms, because one branch each is not coverage. "near" is
        # the same row at 1.3x temperature: q is close to p, most draws accept,
        # and the accept branch carries the weight. "far" is the next row's
        # logits: q is unrelated to p, almost nothing accepts, and every draw
        # goes through the residual norm(max(0, p - q)), which is where the C3
        # tail fix lives. The far arm alone accepted 0 of 64 on the first run,
        # which is why the near arm exists.
        gate3_row(ctx, row_name, "near", row, row, Float32(1.3), fails)
        gate4_row(ctx, row_name, "near", row, row, Float32(1.3), fails)
        if len(names) > 1:
            ref draft = rows_l[(ri + 1) % len(names)]
            gate3_row(ctx, row_name, "far", row, draft, Float32(1.0), fails)
            gate4_row(ctx, row_name, "far", row, draft, Float32(1.0), fails)
        else:
            print("  SKIP gate3/gate4 far", row_name, ": need a second row for the draft distribution")

    if not ran_any:
        print("SKIP: no rows present (see .work/m5/logits-*.bin)")
        return
    if fails == 0:
        print("PASS: device sampler and speculative accept match serve/sample_ref.mojo (C3-fixed) at real vocab")
    else:
        print("FAIL:", fails, "check(s) failed")
        exit(1)
