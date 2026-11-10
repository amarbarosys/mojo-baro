"""Latent-thought pointer chasing: run ARM STEPS [B=256] [SEED=1] [OUT=-] [EVAL=4096] [k=v ...].

ARM: nocot | text | latent | pause (torch pilot exchange/receipts/latent_pointer.py, run 2: 8 hops, 2 layers).
Task: random f over 16 symbols as a 32-token (key, f(key)) table, start x, answer f^8(x).
latent: Coconut curriculum, stage s = it // (STEPS // 8) puts thoughts 0..s-1 in latent rows; eval at s = 7.
OUT: writes OUT/params.f32 and OUT/result.json when not '-'.
k=v: lr=3e-4, warmup=0, decay=0 (1 = linear to 0), stages=a,...,h (latent steps per stage, sum replaces STEPS),
     fix=s (latent: every step at stage s), every=0 (eval interval), evaln=1024, target=0.99, stop=0 (1 = stop at HIT),
     aux=0 (readout loss weight on rows feeding latent rows, hop label c[j]; answer/main and aux each averaged),
     init=PATH (start from params.f32), feed=raw|soft (latent rows take the final-norm hidden or softmax(z) E of the
     previous row, in training and eval; soft is differentiable into the previous row's logits and E), exact=1 (fp32 GEMMs),
     rw=1 (text, short layout: row j of the 8 supervised rows weighs 1 + j/7, normalised), temp=1 (soft feed is softmax(z / temp) E), ft=0 ftlr=3e-4 (after training: ft updates of stop-gradient
     soft-feed fine-tune at stage 7, CE on every thought row and the answer, Adam carried; PRE line before it),
     ema=0 (Polyak decay; >0 evaluates, stops on and saves the averaged weights, raw also printed),
     emafrom=300 (ema starts as a copy at this update), nobot=1 (with short: no BOT, the x row predicts c[0], S = 41), b2=0.999, clip=1.0, short=1 (drop EOT/ANS: row P0+7 predicts the answer as an eighth hop), wd=0.01,
     evalfrom=0 (no validation eval before this update), w8=1 (text: row P0+7 predicts c8, hop/row41/ANS CE thirds), hit=soft (text arm: HIT and RESULT measure latent stage 7 with soft feedback instead of the text chain),
     rule=0 (latent: advance stage when the readout of the row feeding the next latent row >= rule on validation,
     at least 50 updates per stage; replaces the stage table), horizon=STEPS (linear decay reaches 0 here),
     freeze=1 (zero the gradient of the embedding and layer 0; wd 0 needed to hold them), tail=0 (lr after the horizon; Adam moments and step counter continue),
     star=1 (text: before each update the model generates its own chain; sequences whose generated answer is right
     train on that chain, the rest get no loss; ground-truth hops are never trained, only the answer filters).
Eval: latent at stage 7, text both teacher-forced (tf) and generated chain (gen, its own argmax fed back); the
HIT metric is latent stage 7 / text gen / nocot / pause answer accuracy on a validation stream (seed 10000+SEED);
the final RESULT uses an untouched test stream (seed 20000+SEED). hops = argmax readout at rows P0..P0+6 vs c[j].
"""
from stagecheck import check_stage_args
from std.sys import argv
from std.os import getenv
from std.math import log, sqrt, cos, ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload
from lmodel import *
from lops import CH, sumsq_k, clip_k, adamw_k, ema_k, zero_k

comptime NSYM = 16
comptime HOPS = NTH + 1
comptime Q = 16
comptime BOT = 17
comptime EOT = 18
comptime ANS = 19
comptime P0 = 2 * NSYM + 2


struct Rng(Movable):
    var s: UInt64
    var task: Int
    var rules: List[Int]

    def __init__(out self, seed: Int) raises:
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1
        self.task = Int(getenv("LAT_TASK", "0"))
        self.rules = List[Int]()
        for x in getenv("LAT_RULES", "").split(","):
            if x.byte_length() > 0:
                self.rules.append(Int(String(x)))

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


def base(S: Int) -> Int:
    """Row predicting c[0]: BOT (P0), or the x row (P0 - 1) in the no-BOT layout (S = P0 + NTH)."""
    return P0 - 1 if S == P0 + NTH else P0


def layout(arm: String, stage: Int, gen: Bool = False, feed: Int = LAT, short: Bool = False, nobot: Bool = False) -> List[Int]:
    """short: no EOT/ANS rows, the last thought row predicts the answer as an eighth hop (S = 42).
    nobot (with short): no BOT row, the x row predicts c[0] (S = 41)."""
    var k = List[Int]()
    var S = P0 + 1 if arm == "nocot" else P0 + 1 + NTH + (0 if short else 2) - (1 if nobot and short else 0)
    var q0 = base(S) if arm != "nocot" else P0
    for r in range(S):
        k.append(TOK)
    if arm == "pause":
        for j in range(NTH):
            k[q0 + 1 + j] = PSE
    if arm == "latent":
        for j in range(stage):
            k[q0 + 1 + j] = feed
    if arm == "text" and gen:
        for j in range(NTH):
            k[q0 + 1 + j] = GEN
    return k^


def hop_rule(v: Int, task: Int) -> Int:
    """LAT_TASK: 0 plain f; 1-4 add 1,3,5,7; 5-10 xor 1,2,4,8,3,6 after each lookup (mod 16)."""
    if task == 0:
        return v
    if task <= 4:
        return (v + 2 * task - 1) % NSYM
    var k = (1, 2, 4, 8, 3, 6)
    return v ^ [k[0], k[1], k[2], k[3], k[4], k[5]][task - 5]


def fill(mut rng: Rng, arm: String, kinds: List[Int], b: Int, mut ids: List[Int32], mut tgt: List[Int32], mut chain: List[Int]):
    """Position-major: row r * b + i is position r of sequence i."""
    var S = len(kinds)
    var f = List[Int](length=NSYM, fill=0)
    for i in range(b):
        var rule = rng.task
        if len(rng.rules) > 0:
            rule = rng.rules[rng.below(len(rng.rules))]
        for s in range(NSYM):
            f[s] = rng.below(NSYM)
        var x = rng.below(NSYM)
        var c = List[Int](length=HOPS, fill=0)
        var cur = x
        for h in range(HOPS):
            cur = hop_rule(f[cur], rule)
            c[h] = cur
        for h in range(HOPS):
            chain[i * HOPS + h] = c[h]
        for r in range(S):
            ids[r * b + i] = -1
            tgt[r * b + i] = -1
        for s in range(NSYM):
            ids[(2 * s) * b + i] = Int32(s)
            ids[(2 * s + 1) * b + i] = Int32(f[s])
        ids[(2 * NSYM) * b + i] = Int32(20 + rule) if len(rng.rules) > 0 else Q
        ids[(2 * NSYM + 1) * b + i] = Int32(x)
        if arm == "nocot":
            ids[P0 * b + i] = ANS
            tgt[P0 * b + i] = Int32(c[HOPS - 1])
            continue
        var q0 = base(S)
        if q0 == P0:
            ids[P0 * b + i] = BOT
        var lat = 0
        for j in range(NTH):
            var r = q0 + 1 + j
            if kinds[r] == TOK:
                ids[r * b + i] = Int32(c[j])
            else:
                lat = j + 1
        if arm != "pause":
            for j in range(lat, NTH):
                tgt[(q0 + j) * b + i] = Int32(c[j])
        if S == q0 + 1 + NTH:
            tgt[(S - 1) * b + i] = Int32(c[HOPS - 1])
            continue
        ids[(S - 2) * b + i] = EOT
        ids[(S - 1) * b + i] = ANS
        tgt[(S - 1) * b + i] = Int32(c[HOPS - 1])



struct Bufs(Movable):
    var ib: DeviceBuffer[DType.int32]
    var tb: DeviceBuffer[DType.int32]
    var wb: DeviceBuffer[DType.float32]
    var wts: List[Float32]
    var ids: List[Int32]
    var tgt: List[Int32]
    var chain: List[Int]
    var z: List[Float32]

    def __init__(out self, ctx: DeviceContext, b: Int) raises:
        var SM = P0 + 1 + NTH + 2
        self.ib = ctx.enqueue_create_buffer[DType.int32](SM * b)
        self.tb = ctx.enqueue_create_buffer[DType.int32](SM * b)
        self.wb = ctx.enqueue_create_buffer[DType.float32](SM * b)
        self.wts = List[Float32](length=SM * b, fill=0)
        ctx.enqueue_memset(self.wb, 0)
        self.ids = List[Int32](length=SM * b, fill=0)
        self.tgt = List[Int32](length=SM * b, fill=0)
        self.chain = List[Int](length=b * HOPS, fill=0)
        self.z = List[Float32](length=SM * b * V, fill=0)

    def idp(self) -> I:
        return I(unsafe_from_address=Int(self.ib.unsafe_ptr()))

    def tgp(self) -> I:
        return I(unsafe_from_address=Int(self.tb.unsafe_ptr()))

    def wp(self) -> F:
        return F(unsafe_from_address=Int(self.wb.unsafe_ptr()))

    def weigh(mut self, ctx: DeviceContext, kinds: List[Int], b: Int, aux: Float64, w8: Bool = False, rw: Bool = False) raises:
        """Main rows share weight 1; rows feeding a latent row get readout targets and weight aux; each group averaged.
        w8 (text): row P0+7 also predicts c8 and the loss is (mean hop CE + row P0+7 CE + ANS CE) / 3."""
        var S = len(kinds)
        if w8 and S <= P0 + NTH:
            raise Error("w8 needs row P0+7, absent with short=1 nobot=1")
        if w8:
            for r in range(S):
                for i in range(b):
                    var w: Float64 = 0
                    if r == P0 + NTH:
                        self.tgt[r * b + i] = Int32(self.chain[i * HOPS + HOPS - 1])
                        w = 1 / Float64(3 * b)
                    elif r == S - 1:
                        w = 1 / Float64(3 * b)
                    elif r >= P0 and r < P0 + NTH and self.tgt[r * b + i] >= 0:
                        w = 1 / Float64(3 * NTH * b)
                    self.wts[r * b + i] = Float32(w)
            upload(ctx, self.wb, self.wts)
            return
        if aux > 0:
            for r in range(S - 1):
                if kinds[r + 1] == LAT or kinds[r + 1] == SOFT:
                    for i in range(b):
                        self.tgt[r * b + i] = Int32(self.chain[i * HOPS + r - base(S)])
        var w = row_weights(kinds, self.tgt, b, aux, rw, base(S))
        for k in range(len(w)):
            self.wts[k] = w[k]
        upload(ctx, self.wb, self.wts)


def argmax_row(z: List[Float32], o: Int) -> Int:
    var best = 0
    for v in range(1, V):
        if z[o + v] > z[o + best]:
            best = v
    return best


def evaluate[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, arm: String, kinds: List[Int], b: Int, n: Int, seed: Int, report: Bool = False) raises -> List[Float64]:
    """Answer accuracy at index HOPS - 1, hop-j readout accuracy at index j. report: print miss buckets."""
    var rng = Rng(seed)
    var S = len(kinds)
    var hits = List[Int](length=HOPS, fill=0)
    var bk = List[Int](length=6, fill=0)
    var total = 0
    while total < n:
        fill(rng, arm, kinds, b, bf.ids, bf.tgt, bf.chain)
        upload(ctx, bf.ib, bf.ids)
        upload(ctx, bf.tb, bf.tgt)
        m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
        ctx.enqueue_copy(dst_ptr=bf.z.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.z, S * b * V, owning=False))
        ctx.synchronize()
        for i in range(b):
            var ok = argmax_row(bf.z, ((S - 1) * b + i) * V) == bf.chain[i * HOPS + HOPS - 1]
            hits[HOPS - 1] += Int(ok)
            if report:
                var seen = List[Int](length=NSYM, fill=0)
                seen[Int(bf.ids[(2 * NSYM + 1) * b + i])] = 1
                var rev = False
                for h in range(HOPS):
                    var c = bf.chain[i * HOPS + h]
                    rev = rev or seen[c] != 0
                    seen[c] = 1
                var fixp = bf.chain[i * HOPS + HOPS - 1] == bf.chain[i * HOPS + HOPS - 2]
                var k = 0 if rev else 2
                bk[k] += 1
                bk[k + 1] += Int(not ok)
                if fixp:
                    bk[4] += 1
                    bk[5] += Int(not ok)
            if arm != "nocot":
                for j in range(NTH):
                    hits[j] += Int(argmax_row(bf.z, ((base(S) + j) * b + i) * V) == bf.chain[i * HOPS + j])
        total += b
    if report:
        print("BUCKETS revisit n", bk[0], "miss", bk[1], "| no-revisit n", bk[2], "miss", bk[3], "| fixed-point c8==c7 n", bk[4], "miss", bk[5])
    var r = List[Float64]()
    for h in range(HOPS):
        r.append(Float64(hits[h]) / Float64(total))
    return r^


def self_label[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, gk: List[Int], b: Int) raises -> Int:
    """star: generate the chain for the filled batch, replace the hop inputs and targets with it, drop wrong-answer sequences."""
    var S = len(gk)
    var q0 = base(S)
    upload(ctx, bf.ib, bf.ids)
    upload(ctx, bf.tb, bf.tgt)
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), gk, S, False)
    ctx.enqueue_copy(dst_ptr=bf.z.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.z, S * b * V, owning=False))
    ctx.synchronize()
    var kept = 0
    for i in range(b):
        var ok = argmax_row(bf.z, ((S - 1) * b + i) * V) == bf.chain[i * HOPS + HOPS - 1]
        for j in range(NTH):
            var gj = argmax_row(bf.z, ((q0 + j) * b + i) * V)
            bf.ids[(q0 + 1 + j) * b + i] = Int32(gj)
            bf.tgt[(q0 + j) * b + i] = Int32(gj) if ok else -1
        if not ok:
            bf.tgt[(S - 1) * b + i] = -1
        kept += Int(ok)
    return kept


def show(r: List[Float64]) -> String:
    var s = String("acc ") + String(r[HOPS - 1]) + " hops"
    for j in range(NTH):
        s += " " + String(Float64(Int(r[j] * 1000 + 0.5)) / 10)
    return s


def main() raises:
    var a = argv()
    for i in range(7, len(a)):
        if String(a[i]) == "exact=1":
            go[True]()
            return
    go[False]()


def go[exact: Bool]() raises:
    var a = argv()
    var arm = String(a[1])
    var steps = Int(String(a[2]))
    var b = Int(String(a[3])) if len(a) > 3 else 256
    var seed = Int(String(a[4])) if len(a) > 4 else 1
    var out = String(a[5]) if len(a) > 5 else "-"
    var neval = Int(String(a[6])) if len(a) > 6 else 4096
    if arm != "nocot" and arm != "text" and arm != "latent" and arm != "pause":
        raise Error("FAIL arm: " + arm)
    var lr: Float32 = 3e-4
    var wd: Float32 = 0.01
    var b2: Float32 = 0.999
    var clipn: Float32 = 1
    var warmup = 0
    var decay = 0
    var plan = List[Int]()
    var fix = -1
    var every = 0
    var evaln = 1024
    var target = 0.99
    var stop = 0
    var aux = 0.0
    var initp = String("")
    var star = False
    var feed = LAT
    var rule = 0.0
    var horizon = 0
    var hitsoft = False
    var evalfrom = 0
    var w8 = False
    var rw = False
    var short = False
    var nobot = False
    var ema = 0.0
    var temp = 1.0
    var ft = 0
    var ftlr = 3e-4
    var freeze = 0
    var emafrom = 300
    var floor = 0.0
    var tail = 0.0
    for i in range(7, len(a)):
        var kv = String(a[i]).split("=")
        var key = String(kv[0])
        var val = String(kv[1])
        if key == "lr":
            lr = Float32(Float64(val))
        elif key == "warmup":
            warmup = Int(val)
        elif key == "decay":
            decay = Int(val)
        elif key == "stages":
            for x in val.split(","):
                plan.append(Int(String(x)))
        elif key == "fix":
            fix = Int(val)
        elif key == "every":
            every = Int(val)
        elif key == "evaln":
            evaln = Int(val)
        elif key == "target":
            target = Float64(val)
        elif key == "stop":
            stop = Int(val)
        elif key == "aux":
            aux = Float64(val)
        elif key == "star":
            star = val == "1"
        elif key == "init":
            initp = val
        elif key == "feed":
            feed = SOFT if val == "soft" else LAT
        elif key == "rule":
            rule = Float64(val)
        elif key == "floor":
            floor = Float64(val)
        elif key == "temp":
            temp = Float64(val)
        elif key == "ft":
            ft = Int(val)
        elif key == "freeze":
            freeze = (V * D + LAYER) * Int(val)
        elif key == "ftlr":
            ftlr = Float64(val)
        elif key == "ema":
            ema = Float64(val)
        elif key == "emafrom":
            emafrom = Int(val)
        elif key == "nobot":
            nobot = val == "1"
        elif key == "short":
            short = val == "1"
        elif key == "b2":
            b2 = Float32(Float64(val))
        elif key == "clip":
            clipn = Float32(Float64(val))
        elif key == "wd":
            wd = Float32(Float64(val))
        elif key == "rw":
            rw = val == "1"
        elif key == "w8":
            w8 = val == "1"
        elif key == "evalfrom":
            evalfrom = Int(val)
        elif key == "hit":
            hitsoft = val == "soft"
        elif key == "exact":
            pass
        elif key == "horizon":
            horizon = Int(val)
        elif key == "tail":
            tail = Float64(val)
        else:
            raise Error("FAIL arg: " + String(a[i]))
    # The legal combinations of the stage selector, checked once before the loop (stagecheck.mojo).
    check_stage_args(arm, steps, rule, every, fix, plan)
    if len(plan) > 0:
        steps = 0
        for x in plan:
            steps += x
    if horizon == 0:
        horizon = steps
    var b1: Float32 = 0.9
    print("params arm", arm, "steps", steps, "b", b, "seed", seed, "hops", HOPS, "layers", L, "d", D, "heads", H, "ff", FF,
          "vocab", V, "lr", lr, "warmup", warmup, "decay", decay, "wd", wd, "betas", b1, b2, "clip", clipn, "np", NP,
          "stages", plan, "fix", fix, "every", every, "evaln", evaln, "target", target, "stop", stop, "aux", aux, "init", initp, "feed", feed, "rule", rule, "horizon", horizon, "floor", floor, "tail", tail,
          "adam", "fresh (moments 0, step counter 1)", "gemm", "fp32-exact" if exact else "bf16-wmma", "hit", "soft" if hitsoft else "arm", "evalfrom", evalfrom, "w8", w8, "short", short, "nobot", nobot, "ema", ema, "emafrom", emafrom, "temp", temp, "ft", ft, "ftlr", ftlr, "rw", rw, "star", star, "heads", H)
    var tw = perf_counter_ns()
    var ctx = DeviceContext()
    var rng = Rng(seed)
    var init = List[Float32](length=NP, fill=0)
    for i in range(NP):
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
    var eb = ctx.enqueue_create_buffer[DType.float32](NP if ema > 0 else 1)
    var pe = F(unsafe_from_address=Int(eb.unsafe_ptr()))
    var pv = pe if ema > 0 else p
    var g = F(unsafe_from_address=Int(gb.unsafe_ptr()))
    var mp = F(unsafe_from_address=Int(mb.unsafe_ptr()))
    var vp = F(unsafe_from_address=Int(vb.unsafe_ptr()))
    var part = F(unsafe_from_address=Int(partb.unsafe_ptr()))
    var sc = F(unsafe_from_address=Int(scb.unsafe_ptr()))
    var m = Model[exact](ctx, b)
    m.itemp = Float32(1 / temp)
    var bf = Bufs(ctx, b)
    var lrow = List[Float32](length=(P0 + 1 + NTH + 2) * b, fill=0)
    var stages = NTH + 1 if arm == "latent" else 1
    var per = max(1, steps // stages)
    var ekinds = layout(arm, NTH if arm == "latent" else 0, arm == "text", feed, short, nobot)
    var earm = arm
    if hitsoft:
        earm = String("latent")
        ekinds = layout(earm, NTH, False, SOFT, short, nobot)
    if star and arm != "text":
        raise Error("FAIL star: text arm only")
    var gk = layout(arm, 0, True, feed, short, nobot)
    var kept = 0
    var seen = 0
    var rstage = 0
    var rsince = 0
    var hit = -1
    var hit_train = 0.0
    var eval_ns = 0
    ctx.synchronize()
    var t0 = perf_counter_ns()
    var done = 0
    var prev_stage = 0
    for it in range(steps):
        var stage = min(it // per, stages - 1) if arm == "latent" else 0
        if len(plan) > 0 and arm == "latent":
            var acc = 0
            stage = 0
            while stage < len(plan) - 1 and it >= acc + plan[stage]:
                acc += plan[stage]
                stage += 1
        if fix >= 0:
            stage = fix
        if rule > 0:
            stage = rstage
        rsince += 1
        var kinds = layout(arm, stage, False, feed, short, nobot)
        var S = len(kinds)
        fill(rng, arm, kinds, b, bf.ids, bf.tgt, bf.chain)
        if star:
            kept += self_label(ctx, m, p, g, bf, gk, b)
            seen += b
        bf.weigh(ctx, kinds, b, aux, w8, rw)
        upload(ctx, bf.ib, bf.ids)
        upload(ctx, bf.tb, bf.tgt)
        comptime if V > 32:
            m.sort_ids(ctx, bf.ids)
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
            f *= max(floor, 1 - Float64(it) / Float64(horizon))
        var lr_now = Float64(lr) * f
        if it >= horizon and tail > 0:
            lr_now = tail
        ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NP), Float32(lr_now), b1, b2, Float32(1e-8), wd,
            Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
        done = it + 1
        if ema > 0 and it + 1 >= emafrom:
            ctx.enqueue_function[ema_k](p, pe, Int32(NP), Float32(ema), Int32(it + 1 == emafrom), grid_dim=grid(NP), block_dim=THREADS)
        var boundary = it > 0 and stage != prev_stage
        prev_stage = stage
        if it % max(1, steps // 10) == 0 or it == steps - 1 or boundary or it == horizon:
            ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
            var nrm = List[Float32](length=2, fill=0)
            ctx.enqueue_copy(dst_ptr=nrm.unsafe_ptr(), src_buf=scb)
            ctx.synchronize()
            var loss: Float64 = 0
            for i in range(S * b):
                loss += Float64(lrow[i])
            if star:
                print(arm, "it", it, "stage", stage, "lr", lr_now, "loss", loss, "gnorm", nrm[0], "clip", nrm[1], "boundary" if boundary else "", "kept", Float64(kept) / Float64(max(seen, 1)))
            else:
                print(arm, "it", it, "stage", stage, "lr", lr_now, "loss", loss, "gnorm", nrm[0], "clip", nrm[1], "boundary" if boundary else "")
            kept = 0
            seen = 0
        if every > 0 and it + 1 >= evalfrom and ((it + 1) % every == 0 or it == steps - 1):
            ctx.synchronize()
            var e0 = perf_counter_ns()
            var r = evaluate(ctx, m, pv if it + 1 >= emafrom else p, g, bf, earm, ekinds, b, evaln, 10000 + seed)
            var line = String("EVAL it ") + String(it + 1) + " stage " + String(stage) + " " + show(r)
            if ema > 0:
                line += " | raw " + show(evaluate(ctx, m, p, g, bf, earm, ekinds, b, evaln, 10000 + seed))
            if arm == "text":
                line += " | tf " + show(evaluate(ctx, m, p, g, bf, arm, layout(arm, 0, False, LAT, short, nobot), b, evaln, 10000 + seed))
            if rule > 0 and stage < NTH and rsince >= 50:
                var pr = evaluate(ctx, m, p, g, bf, arm, layout(arm, stage, False, LAT, short, nobot), b, evaln, 10000 + seed)
                line += " | probe hop " + String(stage) + " " + String(pr[stage])
                if pr[stage] >= rule:
                    rstage += 1
                    rsince = 0
            eval_ns += Int(perf_counter_ns() - e0)
            var train_s = Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9
            print(line, "train_sec", train_s)
            var final_cfg = arm != "latent" or stage == NTH
            if hit < 0 and r[HOPS - 1] >= target and final_cfg:
                hit = it + 1
                hit_train = train_s
                print("HIT", arm, "step", hit, "train_sec", hit_train, "wall_sec", Float64(perf_counter_ns() - tw) / 1e9)
                if stop != 0:
                    break
    if ft > 0:
        ctx.synchronize()
        var pre = evaluate(ctx, m, p, g, bf, earm, ekinds, b, neval, 20000 + seed)
        print("PRE", show(pre), "tf", show(evaluate(ctx, m, p, g, bf, "latent", layout("latent", 0, False, LAT, short, nobot), b, neval, 20000 + seed)))
        var fk = layout("latent", NTH, False, SOFT, short, nobot)
        m.sg = True
        var e1 = perf_counter_ns()
        for k in range(ft):
            fill(rng, "latent", fk, b, bf.ids, bf.tgt, bf.chain)
            bf.weigh(ctx, fk, b, Float64(NTH))
            upload(ctx, bf.ib, bf.ids)
            upload(ctx, bf.tb, bf.tgt)
            ctx.enqueue_memset(gb, 0)
            m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), fk, len(fk), True)
            ctx.enqueue_function[sumsq_k](g, part, Int32(NP), grid_dim=grid(npart), block_dim=THREADS)
            ctx.enqueue_function[clip_k](part, Int32(npart), sc, clipn, grid_dim=1, block_dim=256)
            var t = Float64(done + k + 1)
            ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NP), Float32(ftlr), b1, b2, Float32(1e-8), wd,
                Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
        ctx.synchronize()
        print("FT updates", ft, "lr", ftlr, "sec", Float64(perf_counter_ns() - e1) / 1e9, "eval_excluded_before", Float64(eval_ns) / 1e9)
        m.sg = False
    ctx.synchronize()
    var sec = Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9
    var pf = pv if done >= emafrom else p
    var res = evaluate(ctx, m, pf, g, bf, earm, ekinds, b, neval, 20000 + seed, True)
    var wall = Float64(perf_counter_ns() - tw) / 1e9
    var extra = String("")
    if arm == "text":
        extra = " | tf " + show(evaluate(ctx, m, p, g, bf, arm, layout(arm, 0, False, LAT, short, nobot), b, neval, 20000 + seed))
    print("RESULT", arm, show(res), "n", neval, "steps", done, "hit_step", hit, "hit_train_sec", hit_train, "train_sec", sec,
          "ms_per_step", sec * 1000 / Float64(max(done, 1)), "wall_sec", wall, extra)
    if out != "-":
        var fin = List[Float32](length=NP, fill=0)
        ctx.enqueue_copy(dst_ptr=fin.unsafe_ptr(), src_buf=eb if ema > 0 and done >= emafrom else pb)
        ctx.synchronize()
        with open(out + "/params.f32", "w") as fo:
            fo.write_bytes(Span[UInt8](unsafe_ptr=fin.unsafe_ptr().unsafe_bitcast[UInt8](), length=NP * 4))
        with open(out + "/result.json", "w") as fo:
            fo.write("{\"arm\": \"" + arm + "\", \"acc\": " + String(res[HOPS - 1]) + ", \"n\": " + String(neval) + ", \"steps\": " + String(done) +
                    ", \"hit_step\": " + String(hit) + ", \"b\": " + String(b) + ", \"seed\": " + String(seed) + ", \"lr\": " + String(lr) +
                    ", \"train_sec\": " + String(sec) + ", \"wall_sec\": " + String(wall) + ", \"np\": " + String(NP) + "}\n")
        print("written to", out)
    _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mb.unsafe_ptr(), vb.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), eb.unsafe_ptr())
