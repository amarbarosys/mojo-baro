"""Arithmetic expressions with a program checker: arith ARM STEPS [B=256] [SEED=1] [OUT=-] [k=v ...].

ARM: cot (EXPR = S1 ; S2 ; ... ; N END, loss on every token after =) or nocot (EXPR = N END). Generator and checker: arithgen.mojo.
k=v: fams=0,1 (training families), test=2 (families scored, comma list), n=5 (operands), S=40 (rows), lr=1e-3, warmup=50, decay=1,
     every=0 (val eval interval), evaln=512, neval=2048 (untouched test stream per family), wd=0.01, clip=1, init=PATH, freeze=1 (embedding and layer 0 get no gradient; needs wd=0).
Scoring: the model decodes greedily from the prompt (EXPR =) with its own tokens fed back; the checker `value` of the prompt expression
must equal the number the model wrote before END. Val stream seed 10000+SEED, test stream 20000+SEED, train stream SEED.
"""
from std.sys import argv
from std.math import sqrt, log, cos, ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload
from lmodel import *
from lops import CH, sumsq_k, clip_k, adamw_k, zero_k
import arithgen as ag


def normal(mut r: ag.Rng) -> Float64:
    var u = (Float64(r.below(1 << 30)) + 1) / 1073741825.0
    var v = Float64(r.below(1 << 30)) / 1073741824.0
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v)


def parse(s: String) raises -> List[Int]:
    var o = List[Int]()
    for x in s.split(","):
        if x.byte_length() > 0:
            o.append(Int(String(x)))
    return o^


def argmax_row(z: List[Float32], o: Int) -> Int:
    var best = 0
    for v in range(1, V):
        if z[o + v] > z[o + best]:
            best = v
    return best


struct Bufs(Movable):
    var ib: DeviceBuffer[DType.int32]
    var tb: DeviceBuffer[DType.int32]
    var wb: DeviceBuffer[DType.float32]
    var ids: List[Int32]
    var tgt: List[Int32]
    var wts: List[Float32]
    var z: List[Float32]

    def __init__(out self, ctx: DeviceContext, S: Int, b: Int) raises:
        self.ib = ctx.enqueue_create_buffer[DType.int32](S * b)
        self.tb = ctx.enqueue_create_buffer[DType.int32](S * b)
        self.wb = ctx.enqueue_create_buffer[DType.float32](S * b)
        self.ids = List[Int32](length=S * b, fill=-1)
        self.tgt = List[Int32](length=S * b, fill=-1)
        self.wts = List[Float32](length=S * b, fill=0)
        self.z = List[Float32](length=S * b * V, fill=0)

    def idp(self) -> I:
        return I(unsafe_from_address=Int(self.ib.unsafe_ptr()))

    def tgp(self) -> I:
        return I(unsafe_from_address=Int(self.tb.unsafe_ptr()))

    def wp(self) -> F:
        return F(unsafe_from_address=Int(self.wb.unsafe_ptr()))


def fill_train(mut rng: ag.Rng, fams: List[Int], n: Int, cot: Bool, S: Int, b: Int, mut bf: Bufs) raises:
    """Sequence i, row r at r * b + i. Row r holds token r and predicts token r + 1 from the = row on; weight 1 / targeted rows."""
    var cnt = 0
    for k in range(S * b):
        bf.ids[k] = -1
        bf.tgt[k] = -1
        bf.wts[k] = 0
    for i in range(b):
        var s = ag.make(rng, fams[rng.below(len(fams))], n, cot)
        if len(s.toks) > S:
            raise Error("FAIL length: " + String(len(s.toks)) + " > S " + String(S))
        for r in range(len(s.toks)):
            bf.ids[r * b + i] = Int32(s.toks[r])
            if r >= s.plen - 1 and r + 1 < len(s.toks):
                bf.tgt[r * b + i] = Int32(s.toks[r + 1])
                cnt += 1
    for k in range(S * b):
        if bf.tgt[k] >= 0:
            bf.wts[k] = Float32(1 / Float64(cnt))


def score[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, fam: Int, n: Int, cot: Bool, S: Int, b: Int, count: Int, seed: Int) raises -> List[Float64]:
    """Greedy decode, then the checker. Returns [answer accuracy, exact-chain accuracy, no-END fraction]."""
    var rng = ag.Rng(seed)
    var kinds = List[Int](length=S, fill=TOK)
    var total = 0
    var ok = 0
    var chain = 0
    var noend = 0
    while total < count:
        var smp = List[ag.Sample]()
        var minp = S
        for k in range(S * b):
            bf.ids[k] = -1
            bf.tgt[k] = -1
            bf.wts[k] = 0
        for i in range(b):
            var s = ag.make(rng, fam, n, cot)
            for r in range(s.plen):
                bf.ids[r * b + i] = Int32(s.toks[r])
            minp = min(minp, s.plen)
            smp.append(s^)
        upload(ctx, bf.tb, bf.tgt)
        upload(ctx, bf.wb, bf.wts)
        for t in range(minp, S):
            upload(ctx, bf.ib, bf.ids)
            m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
            ctx.enqueue_copy(dst_ptr=bf.z.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.z, S * b * V, owning=False))
            ctx.synchronize()
            for i in range(b):
                if t >= smp[i].plen:
                    bf.ids[t * b + i] = Int32(argmax_row(bf.z, ((t - 1) * b + i) * V))
        for i in range(b):
            var out = List[Int]()
            for r in range(S):
                out.append(Int(bf.ids[r * b + i]))
            var a = ag.answer(out, smp[i].plen)
            ok += Int(a == ag.value(smp[i].expr))
            noend += Int(a < 0)
            var same = True
            for r in range(len(smp[i].toks)):
                same = same and out[r] == smp[i].toks[r]
            chain += Int(same)
        total += b
    return [Float64(ok) / Float64(total), Float64(chain) / Float64(total), Float64(noend) / Float64(total)]


def main() raises:
    comptime assert V <= 32, "no sort_ids path here"
    var a = argv()
    var arm = String(a[1])
    var steps = Int(String(a[2]))
    var b = Int(String(a[3])) if len(a) > 3 else 256
    var seed = Int(String(a[4])) if len(a) > 4 else 1
    var out = String(a[5]) if len(a) > 5 else "-"
    if arm != "cot" and arm != "nocot":
        raise Error("FAIL arm: " + arm)
    var cot = arm == "cot"
    var fams: List[Int] = [0, 1, 2]
    var tests: List[Int] = [0, 1, 2]
    var n = 5
    var S = 40
    var lr: Float32 = 1e-3
    var wd: Float32 = 0.01
    var clipn: Float32 = 1
    var warmup = 50
    var decay = 1
    var every = 0
    var evaln = 512
    var neval = 2048
    var initp = String("")
    var freeze = 0
    for i in range(6, len(a)):
        var kv = String(a[i]).split("=")
        var key = String(kv[0])
        var val = String(kv[1])
        if key == "fams":
            fams = parse(val)
        elif key == "test":
            tests = parse(val)
        elif key == "n":
            n = Int(val)
        elif key == "S":
            S = Int(val)
        elif key == "lr":
            lr = Float32(Float64(val))
        elif key == "warmup":
            warmup = Int(val)
        elif key == "decay":
            decay = Int(val)
        elif key == "every":
            every = Int(val)
        elif key == "evaln":
            evaln = Int(val)
        elif key == "neval":
            neval = Int(val)
        elif key == "wd":
            wd = Float32(Float64(val))
        elif key == "clip":
            clipn = Float32(Float64(val))
        elif key == "init":
            initp = val
        elif key == "freeze":
            freeze = (V * D + LAYER) * Int(val)
        else:
            raise Error("FAIL arg: " + String(a[i]))
    if S > SMAX:
        raise Error("FAIL S " + String(S) + " > SMAX " + String(SMAX) + " (build with -D LAT_S)")
    print("params arm", arm, "steps", steps, "b", b, "seed", seed, "fams", fams, "test", tests, "n", n, "S", S, "layers", L, "d", D, "heads", H,
          "ff", FF, "vocab", V, "np", NP, "lr", lr, "warmup", warmup, "decay", decay, "wd", wd, "clip", clipn, "every", every, "evaln", evaln,
          "neval", neval, "init", initp, "adam", "fresh (moments 0, step counter 1)", "gemm", "bf16-wmma", "tokens_per_update", S * b)
    var tw = perf_counter_ns()
    var ctx = DeviceContext()
    var rng = ag.Rng(seed)
    var init = List[Float32](length=NP, fill=0)
    for i in range(NP):
        init[i] = Float32(0.02 * normal(rng))
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
    if initp != "":
        var raw = open(initp, "r").read_bytes()
        if len(raw) != NP * 4:
            raise Error("FAIL init: " + initp + " has " + String(len(raw)) + " bytes, want " + String(NP * 4))
        var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(NP):
            init[i] = src[i]
    var pb = ctx.enqueue_create_buffer[DType.float32](NP)
    var gb = ctx.enqueue_create_buffer[DType.float32](NP)
    var mb = ctx.enqueue_create_buffer[DType.float32](NP)
    var vb = ctx.enqueue_create_buffer[DType.float32](NP)
    var npart = ceildiv(NP, CH)
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
    var m = Model[False](ctx, b)
    var bf = Bufs(ctx, S, b)
    var kinds = List[Int](length=S, fill=TOK)
    var lrow = List[Float32](length=S * b, fill=0)
    var b1: Float32 = 0.9
    var b2: Float32 = 0.999
    var trng = ag.Rng(seed)
    var eval_ns = 0
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(steps):
        fill_train(trng, fams, n, cot, S, b, bf)
        upload(ctx, bf.ib, bf.ids)
        upload(ctx, bf.tb, bf.tgt)
        upload(ctx, bf.wb, bf.wts)
        ctx.enqueue_memset(gb, 0)
        m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, True)
        if freeze > 0:
            ctx.enqueue_function[zero_k](g, Int32(freeze), grid_dim=grid(freeze), block_dim=THREADS)
        ctx.enqueue_function[sumsq_k](g, part, Int32(NP), grid_dim=grid(npart), block_dim=THREADS)
        ctx.enqueue_function[clip_k](part, Int32(npart), sc, clipn, grid_dim=1, block_dim=256)
        var t = Float64(it + 1)
        var f = Float64(1)
        if warmup > 0:
            f = min(f, t / Float64(warmup))
        if decay != 0:
            f *= max(0.0, 1 - Float64(it) / Float64(steps))
        ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NP), Float32(Float64(lr) * f), b1, b2, Float32(1e-8), wd,
            Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
            ctx.synchronize()
            var loss: Float64 = 0
            for i in range(S * b):
                loss += Float64(lrow[i])
            print(arm, "it", it, "loss", loss)
        if every > 0 and ((it + 1) % every == 0 or it == steps - 1):
            ctx.synchronize()
            var e0 = perf_counter_ns()
            var line = String("EVAL it ") + String(it + 1)
            for f in tests:
                var r = score(ctx, m, p, g, bf, f, n, cot, S, b, evaln, 10000 + seed)
                line += " | fam " + String(f) + " acc " + String(r[0]) + " chain " + String(r[1])
            eval_ns += Int(perf_counter_ns() - e0)
            print(line, "train_sec", Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9)
    ctx.synchronize()
    var sec = Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9
    for f in tests:
        var r = score(ctx, m, p, g, bf, f, n, cot, S, b, neval, 20000 + seed)
        print("RESULT", arm, "fam", f, "trained", "1" if f in fams else "0", "acc", r[0], "chain", r[1], "noend", r[2], "n", neval, "steps", steps,
              "train_sec", sec, "ms_per_step", sec * 1000 / Float64(max(steps, 1)), "wall_sec", Float64(perf_counter_ns() - tw) / 1e9)
    if out != "-":
        var fin = List[Float32](length=NP, fill=0)
        ctx.enqueue_copy(dst_ptr=fin.unsafe_ptr(), src_buf=pb)
        ctx.synchronize()
        with open(out + "/params.f32", "w") as fo:
            fo.write_bytes(Span[UInt8](unsafe_ptr=fin.unsafe_ptr().unsafe_bitcast[UInt8](), length=NP * 4))
        print("written to", out)
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mb.unsafe_ptr(), vb.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), bf.ib.unsafe_ptr())
