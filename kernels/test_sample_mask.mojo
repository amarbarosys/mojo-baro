"""Grammar-masked sampling: device kernels vs the host reference at real
VOCAB width (briefs/2026-09-16-fable-masked-sampler.md).

The mask is applied before every truncating sampler on both sides: the host
reference is sample_row_ref / sample_probs_ref on mask_logits(row, words),
where masked tokens are NaN (invalid).

Gates, on 3 real decode rows (.work/m5/logits-p0{1,2,3}.bin) or synthetic:
  A. T = 0 masked argmax: device token == host argmax on the masked copy, for a
     mask that clears the top 3 tokens and for a sparse mask of 5 allowed ids.
  B. T > 0 masked draws: device token and prob == host on the same
     (seed, counter), 16 counters x 3 configs x 2 masks, including the
     all-allowed-outside-the-cut mask (5 low-logit ids with top_k 20): never -1.
  C. chi-square: 20000 device draws under the top-3-cleared mask against the
     host masked distribution (p = 0.001 critical), ranks 0..9 plus rest; the
     three forbidden tokens are never drawn.
  D. masked probs rows: amar_sample_probs_masked vs sample_probs_ref on the
     masked copy to 1e-5, masked tokens exactly 0; T = 0 one-hot on the masked
     argmax.
  E. unmasked identity: amar_sample_row at T = 0 == host argmax, and the
     full mask == unmasked draws on 16 counters (the kernel path is the same
     code with MASK = False).

Build: ./.venv/bin/mojo build kernels/test_sample_mask.mojo -I kernels -I serve -o .work/test_sample_mask
"""
from std.math import exp, sqrt
from std.sys import has_accelerator, exit

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from registry import *
from sample import amar_sample_row, amar_sample_row_masked, amar_sample_probs, amar_sample_probs_masked, SAMP_THREADS
from sample_ref import sample_row_ref, sample_probs_ref, mask_logits, is_valid

comptime NW = (VOCAB + 63) // 64
comptime x_l = row_major[1, VOCAB]()
comptime o_l = row_major[1]()
comptime BATCH = 500
comptime xb_l = row_major[BATCH, VOCAB]()
comptime tb_l = row_major[BATCH]()
comptime N_DRAWS = 20000


def path_exists(path: String) -> Bool:
    try:
        with open(path, "r") as f:
            _ = f.read_bytes(1)
        return True
    except:
        return False


def load_logits(path: String) raises -> List[Float32]:
    with open(path, "r") as f:
        var data = f.read_bytes()
        var n = len(data) // 4
        var row = List[Float32](unsafe_uninit_length=n)
        var p = data.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n):
            row[i] = p[i]
        return row^


def synthetic_row(seed: UInt64) -> List[Float32]:
    var row = List[Float32](unsafe_uninit_length=VOCAB)
    var s = seed
    for i in range(VOCAB):
        s = s * 6364136223846793005 + 1442695040888963407
        var u = Float32((s >> 40) & 0xFFFFFF) / Float32(16777216.0)
        row[i] = u * 12.0 - 8.0
    for k in range(40):
        s = s * 6364136223846793005 + 1442695040888963407
        var idx = Int((s >> 33) % UInt64(VOCAB))
        row[idx] = 6.0 + Float32(k) * 0.35
    return row^


def upload_row(ctx: DeviceContext, row: List[Float32]) raises -> DeviceBuffer[f32]:
    var h = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.synchronize()
    for i in range(VOCAB):
        h[i] = row[i]
    var d = ctx.enqueue_create_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def upload_mask(ctx: DeviceContext, words: List[UInt64]) raises -> DeviceBuffer[DType.uint64]:
    var h = ctx.enqueue_create_host_buffer[DType.uint64](NW)
    ctx.synchronize()
    for i in range(NW):
        h[i] = words[i]
    var d = ctx.enqueue_create_buffer[DType.uint64](NW)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def full_mask() -> List[UInt64]:
    var w = List[UInt64](unsafe_uninit_length=NW)
    for i in range(NW):
        w[i] = UInt64(0xFFFFFFFFFFFFFFFF)
    return w^


def empty_mask() -> List[UInt64]:
    var w = List[UInt64](unsafe_uninit_length=NW)
    for i in range(NW):
        w[i] = 0
    return w^


def set_bit(mut w: List[UInt64], i: Int, on: Bool):
    if on:
        w[i // 64] = w[i // 64] | (UInt64(1) << UInt64(i % 64))
    else:
        w[i // 64] = w[i // 64] & ~(UInt64(1) << UInt64(i % 64))


def top_ids(row: List[Float32], k: Int) -> List[Int]:
    var taken = List[Bool](unsafe_uninit_length=len(row))
    for i in range(len(row)):
        taken[i] = False
    var out = List[Int]()
    for _ in range(k):
        var bi = -1
        var bv = Float32(-3.4028234663852886e38)
        for i in range(len(row)):
            if not taken[i] and is_valid(row[i]) and row[i] > bv:
                bv = row[i]
                bi = i
        taken[bi] = True
        out.append(bi)
    return out^


def host_argmax(row: List[Float32]) -> Int:
    var bi = -1
    var bv = Float32(-3.4028234663852886e38)
    for i in range(len(row)):
        if is_valid(row[i]) and row[i] > bv:
            bv = row[i]
            bi = i
    return bi


def dev_masked(ctx: DeviceContext, mut xd: DeviceBuffer[f32], mut md: DeviceBuffer[DType.uint64], t: Float32, k: Int, p: Float32, mp: Float32, seed: UInt64, counter: UInt64) raises -> Tuple[Int, Float32]:
    var td = ctx.enqueue_create_buffer[DType.int32](1)
    var pd = ctx.enqueue_create_buffer[f32](1)
    comptime kern = amar_sample_row_masked[type_of(x_l), type_of(o_l), type_of(o_l)]
    ctx.enqueue_function[kern](
        TileTensor(xd, x_l), TileTensor(td, o_l), TileTensor(pd, o_l), Int32(VOCAB),
        t, Int32(k), p, mp, seed, counter, md.unsafe_ptr(), Int32(NW), grid_dim=1, block_dim=SAMP_THREADS,
    )
    var th = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ph = ctx.enqueue_create_host_buffer[f32](1)
    ctx.enqueue_copy(dst_buf=th, src_buf=td)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    return (Int(th[0]), Float32(ph[0]))


def dev_plain(ctx: DeviceContext, mut xd: DeviceBuffer[f32], t: Float32, k: Int, p: Float32, mp: Float32, seed: UInt64, counter: UInt64) raises -> Tuple[Int, Float32]:
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


def dev_probs_masked(ctx: DeviceContext, mut xd: DeviceBuffer[f32], mut md: DeviceBuffer[DType.uint64], t: Float32, k: Int, p: Float32, mp: Float32) raises -> List[Float32]:
    var pd = ctx.enqueue_create_buffer[f32](VOCAB)
    comptime kern = amar_sample_probs_masked[type_of(x_l), type_of(x_l)]
    ctx.enqueue_function[kern](
        TileTensor(xd, x_l), TileTensor(pd, x_l), Int32(VOCAB), t, Int32(k), p, mp,
        md.unsafe_ptr(), Int32(NW), grid_dim=1, block_dim=SAMP_THREADS,
    )
    var ph = ctx.enqueue_create_host_buffer[f32](VOCAB)
    ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
    ctx.synchronize()
    var out = List[Float32](unsafe_uninit_length=VOCAB)
    for i in range(VOCAB):
        out[i] = ph[i]
    return out^


def crit(df: Int) -> Float64:
    if df <= 0:
        return 1e-9
    var d = Float64(df)
    var a = 2.0 / (9.0 * d)
    var c = 1.0 - a + 3.090232 * sqrt(a)
    return d * c * c * c


def masks_for(row: List[Float32]) -> Tuple[List[UInt64], List[UInt64], List[Int]]:
    var top = top_ids(row, 3)
    var m1 = full_mask()
    for t in top:
        set_bit(m1, t, False)
    var m2 = empty_mask()
    var allowed = List[Int]()
    var s = UInt64(99)
    for _ in range(5):
        s = s * 6364136223846793005 + 1442695040888963407
        var idx = Int((s >> 33) % UInt64(VOCAB))
        var again = False
        for a in allowed:
            if a == idx:
                again = True
        if again:
            continue
        allowed.append(idx)
        set_bit(m2, idx, True)
    return (m1^, m2^, allowed^)


def run_row(ctx: DeviceContext, name: String, row: List[Float32], mut fails: Int) raises:
    var xd = upload_row(ctx, row)
    var mk = masks_for(row)
    var m1 = mk[0].copy()
    var m2 = mk[1].copy()
    var allowed = mk[2].copy()
    var m1d = upload_mask(ctx, m1)
    var m2d = upload_mask(ctx, m2)
    var r1 = mask_logits(row, m1)
    var r2 = mask_logits(row, m2)

    var ga1 = dev_masked(ctx, xd, m1d, Float32(0), 0, Float32(1), Float32(0), UInt64(1), UInt64(0))
    var ga2 = dev_masked(ctx, xd, m2d, Float32(0), 0, Float32(1), Float32(0), UInt64(1), UInt64(0))
    var ha1 = host_argmax(r1)
    var ha2 = host_argmax(r2)
    var top3 = top_ids(row, 3)
    var hit_top = False
    for t in top3:
        if ga1[0] == t:
            hit_top = True
    if ga1[0] == ha1 and ga2[0] == ha2 and not hit_top and ga1[1] == 1 and ga2[1] == 1:
        print("PASS A", name, ": masked argmax", ga1[0], "(top-3 cleared) and", ga2[0], "(5 allowed) match host")
    else:
        print("FAIL A", name, ": device", ga1[0], ga2[0], "host", ha1, ha2, "hit_top", hit_top)
        fails += 1

    var ts: List[Float32] = [0.7, 1.0, 0.8]
    var ks: List[Int] = [20, 0, 40]
    var ps: List[Float32] = [0.8, 1.0, 0.95]
    var mps: List[Float32] = [0.0, 0.0, 0.05]
    var bad = 0
    var neg = 0
    var n = 0
    for c in range(3):
        for counter in range(16):
            var d1 = dev_masked(ctx, xd, m1d, ts[c], ks[c], ps[c], mps[c], UInt64(42), UInt64(counter))
            var h1 = sample_row_ref(r1, ts[c], ks[c], ps[c], mps[c], UInt64(42), UInt64(counter), 0)
            var d2 = dev_masked(ctx, xd, m2d, ts[c], ks[c], ps[c], mps[c], UInt64(42), UInt64(counter))
            var h2 = sample_row_ref(r2, ts[c], ks[c], ps[c], mps[c], UInt64(42), UInt64(counter), 0)
            n += 2
            if d1[0] != h1[0] or abs(d1[1] - h1[1]) > 1e-5:
                bad += 1
            if d2[0] != h2[0] or abs(d2[1] - h2[1]) > 1e-5:
                bad += 1
            if d1[0] < 0 or d2[0] < 0:
                neg += 1
            var inside = False
            for a in allowed:
                if d2[0] == a:
                    inside = True
            if not inside:
                bad += 1
    if bad == 0 and neg == 0:
        print("PASS B", name, ":", n, "masked draws match host token and prob; sparse mask (5 low ids, top_k up to 40) never returned -1")
    else:
        print("FAIL B", name, ": mismatches", bad, "negative draws", neg, "of", n)
        fails += 1

    var probs1 = sample_probs_ref(r1, Float32(0.9), 0, Float32(0.95), Float32(0))
    var bins = top_ids(r1, 10)
    var xb = ctx.enqueue_create_buffer[f32](BATCH * VOCAB)
    for b in range(BATCH):
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, xb.unsafe_ptr().unsafe_offset(b * VOCAB), VOCAB, owning=False), src_buf=xd)
    var mb = ctx.enqueue_create_host_buffer[DType.uint64](BATCH * NW)
    ctx.synchronize()
    for b in range(BATCH):
        for i in range(NW):
            mb[b * NW + i] = m1[i]
    var mbd = ctx.enqueue_create_buffer[DType.uint64](BATCH * NW)
    ctx.enqueue_copy(dst_buf=mbd, src_buf=mb)
    var tdb = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var pdb = ctx.enqueue_create_buffer[f32](BATCH)
    var thb = ctx.enqueue_create_host_buffer[DType.int32](BATCH)
    var counts = List[Int](unsafe_uninit_length=len(bins) + 1)
    for i in range(len(counts)):
        counts[i] = 0
    var top3b = top_ids(row, 3)
    var forbidden = 0
    comptime kb = amar_sample_row_masked[type_of(xb_l), type_of(tb_l), type_of(tb_l)]
    for batch in range(N_DRAWS // BATCH):
        ctx.enqueue_function[kb](
            TileTensor(xb, xb_l), TileTensor(tdb, tb_l), TileTensor(pdb, tb_l), Int32(VOCAB),
            Float32(0.9), Int32(0), Float32(0.95), Float32(0), UInt64(7), UInt64(batch), mbd.unsafe_ptr(), Int32(NW),
            grid_dim=BATCH, block_dim=SAMP_THREADS,
        )
        ctx.enqueue_copy(dst_buf=thb, src_buf=tdb)
        ctx.synchronize()
        for b in range(BATCH):
            var t = Int(thb[b])
            var slot = len(bins)
            for a in range(len(bins)):
                if bins[a] == t:
                    slot = a
            counts[slot] += 1
            for f in top3b:
                if t == f:
                    forbidden += 1
    var chi: Float64 = 0
    var df = 0
    var rest_e = Float64(N_DRAWS)
    for a in range(len(bins)):
        var e = Float64(probs1[bins[a]]) * Float64(N_DRAWS)
        rest_e -= e
        if e >= 5:
            var o = Float64(counts[a])
            chi += (o - e) * (o - e) / e
            df += 1
    if rest_e >= 5:
        var o = Float64(counts[len(bins)])
        chi += (o - rest_e) * (o - rest_e) / rest_e
        df += 1
    df -= 1
    if forbidden == 0 and chi <= crit(df):
        print("PASS C", name, ": chi-square", chi, "df", df, "crit", crit(df), "over", N_DRAWS, "draws under the top-3-cleared mask, forbidden drawn 0")
    else:
        print("FAIL C", name, ": chi-square", chi, "df", df, "crit", crit(df), "forbidden drawn", forbidden)
        fails += 1

    var dp = dev_probs_masked(ctx, xd, m1d, Float32(0.9), 40, Float32(0.95), Float32(0))
    var hp = sample_probs_ref(r1, Float32(0.9), 40, Float32(0.95), Float32(0))
    var maxd: Float32 = 0
    var leak = 0
    for i in range(VOCAB):
        var d = abs(dp[i] - hp[i])
        if d > maxd:
            maxd = d
        var allowed_i = ((m1[i // 64] >> UInt64(i % 64)) & 1) == 1
        if not allowed_i and dp[i] != 0:
            leak += 1
    var dp0 = dev_probs_masked(ctx, xd, m2d, Float32(0), 0, Float32(1), Float32(0))
    var onehot = 0
    var at_arg = dp0[ha2] == 1
    for i in range(VOCAB):
        if dp0[i] != 0:
            onehot += 1
    if maxd <= 1e-5 and leak == 0 and onehot == 1 and at_arg:
        print("PASS D", name, ": masked probs vs host max |dp|", maxd, ", masked tokens 0, T=0 one-hot on the masked argmax", ha2)
    else:
        print("FAIL D", name, ": max |dp|", maxd, "leaks", leak, "onehot count", onehot, "at argmax", at_arg)
        fails += 1

    var g = dev_plain(ctx, xd, Float32(0), 0, Float32(1), Float32(0), UInt64(1), UInt64(0))
    var full = full_mask()
    var fd = upload_mask(ctx, full)
    var same = 0
    for counter in range(16):
        var a = dev_plain(ctx, xd, Float32(0.7), 20, Float32(0.8), Float32(0), UInt64(42), UInt64(counter))
        var b = dev_masked(ctx, xd, fd, Float32(0.7), 20, Float32(0.8), Float32(0), UInt64(42), UInt64(counter))
        if a[0] == b[0] and a[1] == b[1]:
            same += 1
    if g[0] == host_argmax(row) and same == 16:
        print("PASS E", name, ": unmasked T=0 argmax", g[0], "== host; full mask == unmasked on 16/16 draws")
    else:
        print("FAIL E", name, ": T=0 device", g[0], "host", host_argmax(row), "full-mask matches", same)
        fails += 1


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var fails = 0
    var names: List[String] = [".work/m5/logits-p01.bin", ".work/m5/logits-p02.bin", ".work/m5/logits-p03.bin"]
    var used = 0
    for nm in names:
        if path_exists(nm):
            var r = load_logits(nm)
            if len(r) == VOCAB:
                run_row(ctx, nm, r, fails)
                used += 1
    if used == 0:
        run_row(ctx, "synthetic", synthetic_row(UInt64(1234)), fails)
    if fails > 0:
        print("FAIL:", fails, "checks failed")
        exit(1)
    print("PASS: masked sampling matches the host reference on", max(used, 1), "rows")
