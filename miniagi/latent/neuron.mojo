"""Neuron hops: neuron ARM STEPS [B=256] [SEED=110] [k=v ...].

ARM: real | shuffled | none.
Task (subset made by tools/neuron-data): a start article x (bge vector fed through W_b on row 0) and H slot tokens s_1..s_H
(H = 2 or 3). Hop h follows the s_h-th in-subset link of the current article. Answer = k-means cluster (16) of the article
reached. Rows: ENT, TOK s_1, NEU, TOK s_2, NEU, [TOK s_3, NEU]; the last NEU row predicts the answer.
real: NEU row = soft lookup over one frozen memory row per link (key = unit(ent_i + c_slot), value = ent of the linked article),
      q = hidden @ W_a, input = value @ W_b, both trained.
shuffled: same, values permuted across memory rows (seeded), keys intact.
none: NEU rows replaced by learned pause rows, no lookup (same length, same capacity, ENT row kept).
k=v: data=PATH (neuron-data output), lr=1e-3, warmup=0, decay=1, wd=0.01, clip=1, hops=0 (0: mixed 2/3, else fixed),
     aux=0 (readout loss weight on the true link per hop, real and shuffled), evaln=4096, exact=0, ls=4 (lookup scale), every=0.
Prints RESULT arm seed acc2 acc3 mean readout2 readout3 (test starts, untouched stream seed 20000+SEED), loss curve rows.
"""
from std.sys import argv
from std.math import sqrt, ceildiv, log, cos
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload
from lmodel import *
from lops import CH, sumsq_k, clip_k, adamw_k
from nlook import Look, EB

comptime NH = 3
comptime NSLOT = 8


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))

    def normal(mut self) -> Float64:
        var u = (Float64(self.next() >> 11) + 1) / 9007199254740993.0
        var v = Float64(self.next() >> 11) / 9007199254740992.0
        return sqrt(-2 * log(u)) * cos(6.283185307179586 * v)


def rdf(path: String) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var h = List[Float32](length=len(raw) // 4, fill=0)
    for i in range(len(h)):
        h[i] = src[unsafe_offset=i]
    return h^


def rdi(path: String) raises -> List[Int32]:
    var raw = open(path, "r").read_bytes()
    var src = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var h = List[Int32](length=len(raw) // 4, fill=0)
    for i in range(len(h)):
        h[i] = src[unsafe_offset=i]
    return h^


struct World(Movable):
    var n: Int
    var R: Int
    var K: Int
    var ent: List[Float32]
    var nbr: List[Int32]
    var clu: List[Int32]
    var starts: List[List[Int]]

    def __init__(out self, dir: String) raises:
        var meta = open(dir + "/meta.txt", "r").read().split(" ")
        self.n = Int(String(meta[0]))
        self.R = Int(String(meta[1]))
        self.K = Int(String(meta[2]))
        if Int(String(meta[3]).strip()) != EB:
            raise Error("FAIL data: embedding width is not " + String(EB))
        self.ent = rdf(dir + "/ent.f32")
        self.nbr = rdi(dir + "/nbr.i32")
        self.clu = rdi(dir + "/clu.i32")
        var sp = rdi(dir + "/split.i32")
        self.starts = List[List[Int]]()
        for _ in range(3):
            self.starts.append(List[Int]())
        for i in range(self.n):
            self.starts[Int(sp[i])].append(i)


def layout(arm: String, H: Int) -> List[Int]:
    var k = List[Int]()
    k.append(ENT)
    for h in range(H):
        k.append(TOK)
        k.append(NEU if arm != "none" else PSE)
    return k^


struct Batch(Movable):
    var ib: DeviceBuffer[DType.int32]
    var tb: DeviceBuffer[DType.int32]
    var wb: DeviceBuffer[DType.float32]
    var ids: List[Int32]
    var tgt: List[Int32]
    var wts: List[Float32]
    var entv: List[Float32]
    var mt: List[Int32]
    var ans: List[Int]

    def __init__(out self, ctx: DeviceContext, b: Int) raises:
        var SM = 1 + 2 * NH
        self.ib = ctx.enqueue_create_buffer[DType.int32](SM * b)
        self.tb = ctx.enqueue_create_buffer[DType.int32](SM * b)
        self.wb = ctx.enqueue_create_buffer[DType.float32](SM * b)
        self.ids = List[Int32](length=SM * b, fill=-1)
        self.tgt = List[Int32](length=SM * b, fill=-1)
        self.wts = List[Float32](length=SM * b, fill=0)
        self.entv = List[Float32](length=b * EB, fill=0)
        self.mt = List[Int32](length=NH * b, fill=-1)
        self.ans = List[Int](length=b, fill=0)

    def fill(mut self, mut rng: Rng, w: World, pool: List[Int], H: Int, b: Int):
        var S = 1 + 2 * H
        for k in range(len(self.ids)):
            self.ids[k] = -1
            self.tgt[k] = -1
            self.wts[k] = 0
        for k in range(len(self.mt)):
            self.mt[k] = -1
        for i in range(b):
            var cur = pool[rng.below(len(pool))]
            for d in range(EB):
                self.entv[i * EB + d] = w.ent[cur * EB + d]
            for h in range(H):
                var s = rng.below(w.R)
                self.ids[(1 + 2 * h) * b + i] = Int32(s)
                self.mt[h * b + i] = Int32(cur * w.R + s)
                cur = Int(w.nbr[cur * w.R + s])
            self.ans[i] = Int(w.clu[cur])
            self.tgt[(S - 1) * b + i] = Int32(self.ans[i])
            self.wts[(S - 1) * b + i] = Float32(1) / Float32(b)

    def push(self, ctx: DeviceContext, m: Model, S: Int, b: Int) raises:
        upload(ctx, self.ib, self.ids)
        upload(ctx, self.tb, self.tgt)
        upload(ctx, self.wb, self.wts)
        upload(ctx, m.lk.ent, self.entv)
        upload(ctx, m.lk.mt, self.mt)


def argmax_row(z: List[Float32], o: Int) -> Int:
    var best = 0
    for v in range(1, V):
        if z[o + v] > z[o + best]:
            best = v
    return best


def evaluate[exact: Bool](ctx: DeviceContext, mut m: Model[exact], p: F, g: F, mut bt: Batch, w: World, pool: List[Int], arm: String, H: Int, b: Int, n: Int, seed: Int) raises -> List[Float64]:
    var rng = Rng(seed)
    var kinds = layout(arm, H)
    var S = len(kinds)
    var z = List[Float32](length=S * b * V, fill=0)
    var lo = List[Float32](length=NH * 2 * b, fill=0)
    var hit = 0
    var rd = 0
    var total = 0
    while total < n:
        bt.fill(rng, w, pool, H, b)
        bt.push(ctx, m, S, b)
        m.step(ctx, p, g, I(unsafe_from_address=Int(bt.ib.unsafe_ptr())), I(unsafe_from_address=Int(bt.tb.unsafe_ptr())), F(unsafe_from_address=Int(bt.wb.unsafe_ptr())), kinds, S, False)
        ctx.enqueue_copy(dst_ptr=z.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.z, S * b * V, owning=False))
        ctx.enqueue_copy(dst_ptr=lo.unsafe_ptr(), src_buf=m.lk.lo)
        ctx.synchronize()
        for i in range(b):
            hit += Int(argmax_row(z, ((S - 1) * b + i) * V) == bt.ans[i])
            if arm != "none":
                var all_ok = True
                for h in range(H):
                    all_ok = all_ok and lo[h * 2 * b + b + i] > 0.5
                rd += Int(all_ok)
        total += b
    var r = List[Float64]()
    r.append(Float64(hit) / Float64(total))
    r.append(Float64(rd) / Float64(total))
    return r^


def main() raises:
    var a = argv()
    var exact = False
    for i in range(1, len(a)):
        if String(a[i]) == "exact=1":
            exact = True
    if exact:
        go[True]()
    else:
        go[False]()


def go[exact: Bool]() raises:
    var a = argv()
    var arm = String(a[1])
    var steps = Int(String(a[2]))
    var b = Int(String(a[3])) if len(a) > 3 else 256
    var seed = Int(String(a[4])) if len(a) > 4 else 110
    if arm != "real" and arm != "shuffled" and arm != "none":
        raise Error("FAIL arm: " + arm)
    var lr: Float32 = 1e-3
    var wd: Float32 = 0.01
    var clipn: Float32 = 1
    var warmup = 0
    var decay = 1
    var hops = 0
    var aux: Float32 = 0
    var evaln = 4096
    var ls: Float32 = 4
    var every = 0
    var dir = String("")
    for i in range(5, len(a)):
        var kv = String(a[i]).split("=")
        var key = String(kv[0])
        var val = String(kv[1])
        if key == "lr":
            lr = Float32(Float64(val))
        elif key == "warmup":
            warmup = Int(val)
        elif key == "decay":
            decay = Int(val)
        elif key == "wd":
            wd = Float32(Float64(val))
        elif key == "clip":
            clipn = Float32(Float64(val))
        elif key == "hops":
            hops = Int(val)
        elif key == "aux":
            aux = Float32(Float64(val))
        elif key == "evaln":
            evaln = Int(val)
        elif key == "ls":
            ls = Float32(Float64(val))
        elif key == "every":
            every = Int(val)
        elif key == "data":
            dir = val
        elif key == "exact":
            pass
        else:
            raise Error("FAIL arg: " + String(a[i]))
    if dir == "":
        raise Error("FAIL arg: data=PATH is required")
    var w = World(dir)
    var NT = NP + 2 * D * EB
    print("params arm", arm, "steps", steps, "b", b, "seed", seed, "lr", lr, "warmup", warmup, "decay", decay, "wd", wd, "clip", clipn,
          "hops", hops, "aux", aux, "ls", ls, "evaln", evaln, "every", every, "articles", w.n, "slots", w.R, "clusters", w.K,
          "train/val/test starts", len(w.starts[0]), len(w.starts[1]), len(w.starts[2]), "d", D, "layers", L, "np", NP, "nt", NT,
          "gemm", "fp32-exact" if exact else "bf16-wmma", "data", dir)
    var ctx = DeviceContext()
    var rng = Rng(seed)
    var init = List[Float32](length=NT, fill=0)
    for i in range(NT):
        init[i] = Float32(0.02 * rng.normal())
    for l in range(L):
        var o = V * D + l * LAYER
        for j in range(D):
            init[o + j] = 1
            init[o + D + 4 * DD + j] = 1
        for j in range(DD):
            init[o + D + 3 * DD + j] *= Float32(1 / sqrt(Float64(2 * L)))
        for j in range(DF):
            init[o + 2 * D + 4 * DD + 2 * DF + j] *= Float32(1 / sqrt(Float64(2 * L)))
    for j in range(D):
        init[GF + j] = 1
    var nedge = w.n * w.R
    var M = ceildiv(nedge, 64) * 64
    var keys = List[Float32](length=M * EB, fill=0)
    var vals = List[Float32](length=M * EB, fill=0)
    var code = List[Float32](length=w.R * EB, fill=0)
    var crng = Rng(7)
    for r in range(w.R):
        var ss: Float64 = 0
        for d in range(EB):
            var x = crng.normal()
            code[r * EB + d] = Float32(x)
            ss += x * x
        for d in range(EB):
            code[r * EB + d] = Float32(Float64(code[r * EB + d]) / sqrt(ss))
    var perm = List[Int](length=nedge, fill=0)
    for e in range(nedge):
        perm[e] = e
    if arm == "shuffled":
        var prng = Rng(seed * 1000 + 3)
        for e in range(nedge - 1, 0, -1):
            var j = prng.below(e + 1)
            var t = perm[e]
            perm[e] = perm[j]
            perm[j] = t
    for e in range(nedge):
        var i = e // w.R
        var r = e % w.R
        var ss: Float64 = 0
        for d in range(EB):
            var x = Float64(w.ent[i * EB + d]) + Float64(code[r * EB + d])
            keys[e * EB + d] = Float32(x)
            ss += x * x
        for d in range(EB):
            keys[e * EB + d] = Float32(Float64(keys[e * EB + d]) / sqrt(ss))
        var src = perm[e]
        var tgt_art = Int(w.nbr[src])
        for d in range(EB):
            vals[e * EB + d] = w.ent[tgt_art * EB + d]
    var pb = ctx.enqueue_create_buffer[DType.float32](NT)
    var gb = ctx.enqueue_create_buffer[DType.float32](NT)
    var mb = ctx.enqueue_create_buffer[DType.float32](NT)
    var vb = ctx.enqueue_create_buffer[DType.float32](NT)
    var npart = ceildiv(NT, CH)
    var partb = ctx.enqueue_create_buffer[DType.float32](npart)
    var scb = ctx.enqueue_create_buffer[DType.float32](2)
    upload(ctx, pb, init)
    ctx.enqueue_memset(mb, 0)
    ctx.enqueue_memset(vb, 0)
    var p = F(unsafe_from_address=Int(pb.unsafe_ptr()))
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    var mp = F(unsafe_from_address=Int(mb.unsafe_ptr()))
    var vp = F(unsafe_from_address=Int(vb.unsafe_ptr()))
    var part = F(unsafe_from_address=Int(partb.unsafe_ptr()))
    var sc = F(unsafe_from_address=Int(scb.unsafe_ptr()))
    var m = Model[exact](ctx, b)
    m.lk = Look(ctx, b, NH, M, nedge, NP, NP + D * EB)
    m.lk.ls = ls
    m.lk.aw = aux if arm != "none" else 0
    m.lk.load(ctx, keys, vals)
    var bt = Batch(ctx, b)
    var drng = Rng(seed + 5)
    var lrow = List[Float32](length=(1 + 2 * NH) * b, fill=0)
    var b1: Float32 = 0.9
    var b2: Float32 = 0.999
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(steps):
        var H = hops if hops > 0 else (2 + (it % 2))
        var kinds = layout(arm, H)
        var S = len(kinds)
        bt.fill(drng, w, w.starts[0], H, b)
        bt.push(ctx, m, S, b)
        ctx.enqueue_memset(gb, 0)
        m.step(ctx, p, g, I(unsafe_from_address=Int(bt.ib.unsafe_ptr())), I(unsafe_from_address=Int(bt.tb.unsafe_ptr())), F(unsafe_from_address=Int(bt.wb.unsafe_ptr())), kinds, S, True)
        ctx.enqueue_function[sumsq_k](g, part, Int32(NT), grid_dim=grid(npart), block_dim=THREADS)
        ctx.enqueue_function[clip_k](part, Int32(npart), sc, clipn, grid_dim=1, block_dim=256)
        var t = Float64(it + 1)
        var f = Float64(1)
        if warmup > 0:
            f = min(f, t / Float64(warmup))
        if decay != 0:
            f *= max(0.0, 1 - Float64(it) / Float64(steps))
        ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NT), Float32(Float64(lr) * f), b1, b2, Float32(1e-8), wd,
            Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NT), block_dim=THREADS)
        var show = it % max(1, steps // 10) == 0 or it == steps - 1
        if show:
            ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
            ctx.synchronize()
            var loss: Float64 = 0
            for i in range(S * b):
                loss += Float64(lrow[i])
            if loss != loss:
                raise Error("FAIL train: loss is NaN at update " + String(it))
            print("step", it, "H", H, "loss", loss)
        if every > 0 and (it + 1) % every == 0 and it + 1 < steps:
            var r2 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[1], arm, 2, b, 1024, 10000 + seed)
            var r3 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[1], arm, 3, b, 1024, 10000 + seed)
            print("VAL step", it + 1, "acc2", r2[0], "acc3", r3[0], "readout2", r2[1], "readout3", r3[1])
    ctx.synchronize()
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    var v2 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[1], arm, 2, b, evaln, 10000 + seed)
    var v3 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[1], arm, 3, b, evaln, 10000 + seed)
    var r2 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[2], arm, 2, b, evaln, 20000 + seed)
    var r3 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[2], arm, 3, b, evaln, 20000 + seed)
    var tr3 = evaluate[exact](ctx, m, p, g, bt, w, w.starts[0], arm, 3, b, evaln, 30000 + seed)
    print("VAL final acc2", v2[0], "acc3", v3[0])
    print("TRAIN-starts acc3", tr3[0])
    print("RESULT arm", arm, "seed", seed, "steps", steps, "lr", lr, "aux", aux, "acc2", r2[0], "acc3", r3[0], "mean", (r2[0] + r3[0]) / 2,
          "readout2", r2[1], "readout3", r3[1], "wall_s", wall)
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mb.unsafe_ptr(), vb.unsafe_ptr())
