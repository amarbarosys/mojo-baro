"""Penalties and top-N logprobs: device kernels vs the host reference at real
VOCAB width (briefs/2026-09-16-fable-sample-kernels.md).

Gate A: amar_apply_penalties (sparse per-row id/count lists, spec-window
rows carry drafts 0..j-1) leaves X bit-equal to sample_ref.apply_penalties
on a host copy, for presence-only, frequency-only and both; then
amar_sample_row on the penalized device row draws the same token as
sample_row_ref on the penalized host row.

Gate B: amar_topn_probs ids equal the host top-N by (logit desc, index asc)
and probs match sample_probs_ref's truncated distribution to 1e-5, for
four configs including temperature 0 (softmax at T=1, untruncated).

Rows: .work/m5/logits-p0{1,2,3}.bin when present (real decode rows), plus
one synthetic row so the test runs without fixtures.

Build: ./.venv/bin/mojo build kernels/test_sample_pen.mojo -I kernels -I serve -o .work/test_sample_pen
"""
from std.math import exp
from std.sys import has_accelerator, exit

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from registry import *
from sample import amar_sample_row, amar_apply_penalties, amar_topn_probs, SAMP_THREADS, SAMP_CAP, PEN_THREADS
from sample_ref import sample_row_ref, sample_probs_ref, apply_penalties, is_valid

comptime R = 3
comptime PCAP = 64
comptime NMAX = 20
comptime x_l = row_major[R, VOCAB]()
comptime id_l = row_major[R, PCAP]()
comptime np_l = row_major[R]()
comptime x1_l = row_major[1, VOCAB]()
comptime o_l = row_major[1]()
comptime top_l = row_major[R, NMAX]()
comptime topp_l = row_major[R, NMAX]()


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
    row[7] = row[7] + 0.0
    return row^


def upload_rows(ctx: DeviceContext, rows: List[List[Float32]]) raises -> DeviceBuffer[f32]:
    var h = ctx.enqueue_create_host_buffer[f32](R * VOCAB)
    ctx.synchronize()
    for r in range(R):
        for i in range(VOCAB):
            h[r * VOCAB + i] = rows[r][i]
    var d = ctx.enqueue_create_buffer[f32](R * VOCAB)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def download_rows(ctx: DeviceContext, d: DeviceBuffer[f32]) raises -> List[List[Float32]]:
    var h = ctx.enqueue_create_host_buffer[f32](R * VOCAB)
    ctx.enqueue_copy(dst_buf=h, src_buf=d)
    ctx.synchronize()
    var out = List[List[Float32]]()
    for r in range(R):
        var row = List[Float32](unsafe_uninit_length=VOCAB)
        for i in range(VOCAB):
            row[i] = h[r * VOCAB + i]
        out.append(row^)
    return out^


def history_for_row(base: List[Int], drafts: List[Int], j: Int) -> List[Int]:
    var h = List[Int]()
    for t in base:
        h.append(t)
    for d in range(j):
        h.append(drafts[d])
    return h^


def sparse_lists(hist: List[Int], mut ids: List[Int32], mut cnt: List[Int32]) -> Int:
    var m = 0
    for t in hist:
        var found = -1
        for k in range(m):
            if Int(ids[k]) == t:
                found = k
        if found >= 0:
            cnt[found] += 1
        else:
            ids[m] = Int32(t)
            cnt[m] = 1
            m += 1
    return m


def gate_a(ctx: DeviceContext, rows: List[List[Float32]], mut fails: Int) raises:
    var base: List[Int] = [5, 5, 7, 9, 5, 12, 12, 1000, 248000, 7, 5, 31, 31, 31]
    var drafts: List[Int] = [9, 5, 77]
    var pps: List[Float32] = [0.7, 1.5, 0.0]
    var fps: List[Float32] = [0.3, 0.0, 0.9]
    for c in range(len(pps)):
        var pp = pps[c]
        var fp = fps[c]
        var xd = upload_rows(ctx, rows)
        var ids_h = ctx.enqueue_create_host_buffer[DType.int32](R * PCAP)
        var cnt_h = ctx.enqueue_create_host_buffer[DType.int32](R * PCAP)
        var np_h = ctx.enqueue_create_host_buffer[DType.int32](R)
        ctx.synchronize()
        var expect = List[List[Float32]]()
        for j in range(R):
            var hist = history_for_row(base, drafts, j)
            var ids = List[Int32](unsafe_uninit_length=PCAP)
            var cnt = List[Int32](unsafe_uninit_length=PCAP)
            for k in range(PCAP):
                ids[k] = -1
                cnt[k] = 0
            var m = sparse_lists(hist, ids, cnt)
            np_h[j] = Int32(m)
            for k in range(PCAP):
                ids_h[j * PCAP + k] = ids[k]
                cnt_h[j * PCAP + k] = cnt[k]
            var lg = List[Float32]()
            for i in range(VOCAB):
                lg.append(rows[j][i])
            apply_penalties(lg, hist, pp, fp)
            expect.append(lg^)
        var ids_d = ctx.enqueue_create_buffer[DType.int32](R * PCAP)
        var cnt_d = ctx.enqueue_create_buffer[DType.int32](R * PCAP)
        var np_d = ctx.enqueue_create_buffer[DType.int32](R)
        ctx.enqueue_copy(dst_buf=ids_d, src_buf=ids_h)
        ctx.enqueue_copy(dst_buf=cnt_d, src_buf=cnt_h)
        ctx.enqueue_copy(dst_buf=np_d, src_buf=np_h)
        comptime kpen = amar_apply_penalties[type_of(x_l), type_of(id_l), type_of(id_l), type_of(np_l)]
        ctx.enqueue_function[kpen](
            TileTensor(xd, x_l), TileTensor(ids_d, id_l), TileTensor(cnt_d, id_l), TileTensor(np_d, np_l),
            Int32(VOCAB), pp, fp, grid_dim=R, block_dim=PEN_THREADS,
        )
        var got = download_rows(ctx, xd)
        var bad = 0
        var touched = 0
        for j in range(R):
            for i in range(VOCAB):
                if got[j][i] != expect[j][i]:
                    bad += 1
                if expect[j][i] != rows[j][i]:
                    touched += 1
        if bad == 0:
            print("PASS gate A penalties pp", pp, "fp", fp, ": device rows bit-equal to sample_ref, elements touched", touched)
        else:
            print("FAIL gate A penalties pp", pp, "fp", fp, ":", bad, "elements differ")
            fails += 1
        var td = ctx.enqueue_create_buffer[DType.int32](R)
        var pd = ctx.enqueue_create_buffer[f32](R)
        comptime ksamp = amar_sample_row[type_of(x_l), type_of(np_l), type_of(np_l)]
        var seed = UInt64(4242)
        var counter = UInt64(17)
        ctx.enqueue_function[ksamp](
            TileTensor(xd, x_l), TileTensor(td, np_l), TileTensor(pd, np_l), Int32(VOCAB),
            Float32(0.8), Int32(40), Float32(0.95), Float32(0), seed, counter, grid_dim=R, block_dim=SAMP_THREADS,
        )
        var th = ctx.enqueue_create_host_buffer[DType.int32](R)
        var ph = ctx.enqueue_create_host_buffer[f32](R)
        ctx.enqueue_copy(dst_buf=th, src_buf=td)
        ctx.enqueue_copy(dst_buf=ph, src_buf=pd)
        ctx.synchronize()
        for j in range(R):
            var hr = sample_row_ref(expect[j], Float32(0.8), 40, Float32(0.95), Float32(0), seed, counter, j)
            var dp = abs(Float32(ph[j]) - hr[1])
            if Int(th[j]) == hr[0] and dp <= 1e-5:
                print("PASS gate A draw row", j, "token", Int(th[j]), "prob", Float32(ph[j]), "matches sample_row_ref on the penalized row")
            else:
                print("FAIL gate A draw row", j, "device", Int(th[j]), Float32(ph[j]), "host", hr[0], hr[1])
                fails += 1


def host_topn(logits: List[Float32], probs: List[Float32], K: Int, mut ids: List[Int], mut ps: List[Float32]):
    var taken = List[Bool](unsafe_uninit_length=len(logits))
    for i in range(len(logits)):
        taken[i] = False
    for _ in range(K):
        var bi = -1
        var bv = Float32(-3.4028234663852886e38)
        for i in range(len(logits)):
            if not taken[i] and is_valid(logits[i]) and logits[i] > bv:
                bv = logits[i]
                bi = i
        if bi < 0:
            ids.append(-1)
            ps.append(0)
        else:
            taken[bi] = True
            ids.append(bi)
            ps.append(probs[bi])


def gate_b(ctx: DeviceContext, rows: List[List[Float32]], mut fails: Int) raises:
    var ts: List[Float32] = [0.7, 1.0, 0.8, 0.0]
    var ks: List[Int] = [40, 0, 0, 0]
    var pss: List[Float32] = [0.9, 1.0, 0.95, 1.0]
    var mps: List[Float32] = [0.0, 0.0, 0.05, 0.0]
    var Ks: List[Int] = [20, 20, 5, 20]
    var xd = upload_rows(ctx, rows)
    for c in range(len(ts)):
        var t = ts[c]
        var k = ks[c]
        var p = pss[c]
        var mp = mps[c]
        var K = Ks[c]
        var idd = ctx.enqueue_create_buffer[DType.int32](R * NMAX)
        var prd = ctx.enqueue_create_buffer[f32](R * NMAX)
        comptime ktop = amar_topn_probs[type_of(x_l), type_of(top_l), type_of(topp_l)]
        ctx.enqueue_function[ktop](
            TileTensor(xd, x_l), TileTensor(idd, top_l), TileTensor(prd, topp_l), Int32(VOCAB), Int32(K),
            t, Int32(k), p, mp, grid_dim=R, block_dim=SAMP_THREADS,
        )
        var idh = ctx.enqueue_create_host_buffer[DType.int32](R * NMAX)
        var prh = ctx.enqueue_create_host_buffer[f32](R * NMAX)
        ctx.enqueue_copy(dst_buf=idh, src_buf=idd)
        ctx.enqueue_copy(dst_buf=prh, src_buf=prd)
        ctx.synchronize()
        for j in range(R):
            var ht = t
            var hk = k
            var hp = p
            var hmp = mp
            if t <= 0:
                ht = 1.0
                hk = 0
                hp = 1.0
                hmp = 0.0
            var probs = sample_probs_ref(rows[j], ht, hk, hp, hmp)
            var ids = List[Int]()
            var ps = List[Float32]()
            host_topn(rows[j], probs, K, ids, ps)
            var bad = 0
            var maxd: Float32 = 0
            for q in range(NMAX):
                var di = Int(idh[j * NMAX + q])
                var dp = Float32(prh[j * NMAX + q])
                var ei = ids[q] if q < K else -1
                var ep = ps[q] if q < K else Float32(0)
                if di != ei:
                    bad += 1
                var d = abs(dp - ep)
                if d > maxd:
                    maxd = d
            if bad == 0 and maxd <= 1e-5:
                print("PASS gate B topn T", t, "k", k, "p", p, "min_p", mp, "N", K, "row", j, ": ids equal, max |dp|", maxd, "top id", ids[0], "p", ps[0])
            else:
                print("FAIL gate B topn T", t, "k", k, "p", p, "min_p", mp, "N", K, "row", j, ": id mismatches", bad, "max |dp|", maxd)
                fails += 1


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var rows = List[List[Float32]]()
    var names: List[String] = [".work/m5/logits-p01.bin", ".work/m5/logits-p02.bin", ".work/m5/logits-p03.bin"]
    var used = 0
    for nm in names:
        if len(rows) < R and path_exists(nm):
            var r = load_logits(nm)
            if len(r) == VOCAB:
                rows.append(r^)
                used += 1
    var s = 0
    while len(rows) < R:
        rows.append(synthetic_row(UInt64(1234 + s)))
        s += 1
    print("rows:", used, "real fixture rows,", R - used, "synthetic, VOCAB", VOCAB)
    var fails = 0
    gate_a(ctx, rows, fails)
    gate_b(ctx, rows, fails)
    if fails > 0:
        print("FAIL:", fails, "checks failed")
        exit(1)
    print("PASS: penalties and top-N probs match the host reference on", R, "rows")
