"""Device sampler vs host reference, at a Spark-family profile's own VOCAB
(briefs/2026-09-16-sampling-all-models-lane.md item 2). Same shape as
kernels/test_sample_device.mojo's gate1/gate2 (per-token exact match, then
a chi-square distribution check against an independent numpy oracle), but
parameterized on `profile` (the Spark build's `-I <profile-dir>` module)
instead of `registry`/`model`'s qwen35/qwen35moe VOCAB, and with no
speculative gates (Spark has no draft head). Lives in bench/, not kernels/,
because kernels/*.mojo is off limits to this lane.

Build (one profile dir per model, same -I the model's own spark-engine uses):
  ./.venv/bin/mojo build bench/sample-spark-device.mojo -I . -I kernels -I serve -I <PROFILE_DIR> -o OUT
Needs ROW (a real VOCAB-width f32 logits row, from spark.mojo's
BARO_DUMP_LOGITS) and its oracle prefix (tools/sample-nucleus-oracle.py ROW
OUT_PREFIX) at fixed paths, both passed as argv:
  OUT ROW.bin ORACLE_PREFIX
"""
# ci-checks: needs -I <a generated profile dir> (a per-model `profile.mojo`
# from tools/gen-profile.mojo), which ci-checks' generic bench-compile loop
# does not have; build it as documented above instead.
from std.math import sqrt
from std.sys import argv, has_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from sample import amar_sample_row, SAMP_THREADS
from sample_ref import sample_row_ref
from profile import VOCAB

comptime f32 = DType.float32
comptime x_l = row_major[1, VOCAB]()
comptime o_l = row_major[1]()
comptime BATCH = 1000
comptime xb_l = row_major[BATCH, VOCAB]()
comptime tb_l = row_major[BATCH]()
comptime N_DRAWS = 20000
comptime N_BATCHES = N_DRAWS // BATCH


@fieldwise_init
struct Cfg(Copyable, Movable):
    var slug: String
    var t: Float32
    var k: Int
    var p: Float32
    var mp: Float32


def configs() -> List[Cfg]:
    var c = List[Cfg]()
    c.append(Cfg("T1_k0_p1", Float32(1.0), 0, Float32(1.0), Float32(0.0)))
    c.append(Cfg("T0.8_k30_p1", Float32(0.8), 30, Float32(1.0), Float32(0.0)))
    c.append(Cfg("T0.7_k20_p0.8", Float32(0.7), 20, Float32(0.8), Float32(0.0)))
    return c^


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


def gate2(ctx: DeviceContext, mut xd: DeviceBuffer[f32], oracle_prefix: String, mut fails: Int) raises:
    for cfg in configs():
        var oracle_path = oracle_prefix + "-" + cfg.slug + ".txt"
        if not path_exists(oracle_path):
            print("  SKIP gate2", cfg.slug, ": no oracle file", oracle_path)
            continue
        var ids = List[Int]()
        var probs = List[Float64]()
        var rest = Float64(0)
        load_oracle(oracle_path, ids, probs, rest)
        var counts = List[Int](capacity=len(ids))
        for _ in range(len(ids)):
            counts.append(0)
        var rest_count = 0
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
        print("  ", "PASS" if not over else "FAIL", "gate2", cfg.slug, ": chi2", stat, "df", bins - 1, "crit(p=0.001)", cv, "n", n, "candidates", len(ids), "rest_count", rest_count)
        if over:
            fails += 1


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    var args = argv()
    if len(args) < 3:
        print("usage: OUT ROW.bin ORACLE_PREFIX")
        return
    var row_path = args[1]
    var oracle_prefix = args[2]
    var ctx = DeviceContext()
    var fails = 0
    var row = load_logits(row_path)
    if len(row) != VOCAB:
        print("SKIP:", len(row), "entries, this build's VOCAB is", VOCAB)
        return
    var xh = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.synchronize()
    for i in range(VOCAB):
        xh[i] = row[i]
    var xd = ctx.enqueue_create_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
    ctx.synchronize()

    # Greedy sanity: T=0 must match the argmax the running path already gates.
    var dg = device_sample(ctx, xd, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0))
    var hg = sample_row_ref(row, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0), 0)
    if dg[0] == Int(hg[0]) and dg[1] == hg[1]:
        print("PASS greedy (T=0): device token", dg[0], "prob", dg[1], "== host")
    else:
        print("FAIL greedy (T=0): device", dg[0], dg[1], "!= host", hg[0], hg[1])
        fails += 1

    # gate1: device == host per draw, 64 counters per config.
    for cfg in configs():
        var mismatches = 0
        for counter in range(64):
            var d = device_sample(ctx, xd, cfg.t, cfg.k, cfg.p, cfg.mp, UInt64(42), UInt64(counter))
            var h = sample_row_ref(row, cfg.t, cfg.k, cfg.p, cfg.mp, UInt64(42), UInt64(counter), 0)
            if d[0] != Int(h[0]) or abs(d[1] - h[1]) > Float32(1e-4):
                mismatches += 1
        if mismatches == 0:
            print("PASS gate1", cfg.slug, ": device == host on all 64 draws")
        else:
            print("FAIL gate1", cfg.slug, ":", mismatches, "of 64 draws disagree")
            fails += 1

    # Same-seed reproduces / different-seed diverges, at real vocab.
    var same_a = device_sample(ctx, xd, Float32(1.0), 0, Float32(1.0), Float32(0.0), UInt64(7), UInt64(3))
    var same_b = device_sample(ctx, xd, Float32(1.0), 0, Float32(1.0), Float32(0.0), UInt64(7), UInt64(3))
    var diff_c = device_sample(ctx, xd, Float32(1.0), 0, Float32(1.0), Float32(0.0), UInt64(8), UInt64(3))
    if same_a[0] == same_b[0] and abs(same_a[1] - same_b[1]) < Float32(1e-6):
        print("PASS same seed reproduces: seed 7 counter 3 gave", same_a[0], "twice")
    else:
        print("FAIL same seed did not reproduce:", same_a[0], "vs", same_b[0])
        fails += 1
    if diff_c[0] != same_a[0]:
        print("PASS different seed diverges: seed 8 gave", diff_c[0], "vs seed 7's", same_a[0])
    else:
        print("  (seed 8 happened to also draw", diff_c[0], "-- not a failure, but not evidence either)")

    # Batch buffer for gate2, filled once.
    var xbh = ctx.enqueue_create_host_buffer[f32](BATCH * VOCAB)
    ctx.synchronize()
    for r in range(BATCH):
        for i in range(VOCAB):
            xbh[r * VOCAB + i] = row[i]
    var xbd = ctx.enqueue_create_buffer[f32](BATCH * VOCAB)
    ctx.enqueue_copy(dst_buf=xbd, src_buf=xbh)
    ctx.synchronize()
    gate2(ctx, xbd, oracle_prefix, fails)

    if fails == 0:
        print("PASS: device sampler matches serve/sample_ref.mojo at real vocab", VOCAB)
    else:
        print("FAIL:", fails, "check(s) failed")
