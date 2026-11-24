"""Latent-thought pointer chasing: run ARM STEPS [B=256] [SEED=1] [OUT=-] [EVAL=4096] [k=v ...].

ARM: nocot | text | latent | pause | bytes (byte-level text: corpus=PATH seq=S, LAT_V=256; val/test bpb on the last 1%) (torch pilot exchange/receipts/latent_pointer.py, run 2: 8 hops, 2 layers).
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
     muon=1 (Muon on the layer matrices, AdamW on the rest; mlr=LR mbeta=0.95 ns=5; muon=2 NS self-test and exit), cool=0 (WSD: linear decay to 0 over the last cool * STEPS), zinit=1 (wo and w2 init 0), lpd=K (Lazy Polar Drift: Muon NS every K updates, cached polar factor between; lpdv=2 caches the NS polynomial P and applies P X_t every update), bhl=0.3 (block-head loss weight, build -D LAT_BH=K), bhd=1 (detached drafter: heads learn only W_k), mat=1 (Matryoshka FFN: widths FF,768,FF,1408,FF,2048 by update), ffm=M (fixed FFN width, also for score / gen / eval), score=REQ sout=OUT (baro-bench: score benchprep candidates with init= params, then exit), gen=REQ gout=OUT gmax=320 gtools=1 (baro-bench part 2: greedy generation with the calc tool loop),
     freeze=1 (zero the gradient of the embedding and layer 0; wd 0 needed to hold them), tail=0 (lr after the horizon; Adam moments and step counter continue),
     shuf=0 (1 table rows in random order everywhere; 2 sorted train/val, sealed HELDOUT test on shuffled tables), dseed=30000+SEED (data stream; init uses SEED),
     ckpt=PATH (text arms: self-describing GGUF at the end: sources, build line from env LAT_BUILD, vocab, weights,
     Adam moments, step, rng, a closure set; resume=DIR / closure=DIR take a `ckpt extract` directory),
     save=N (text arms: rewrite ckpt every N updates, atomically; with ckpt=PATH), stopat=K (stop after update K and
     write ckpt: an interruption). Resume = the SAME command plus resume=DIR: the loop continues at the saved update and
     the result is byte-identical to an uninterrupted run (miniagi/latent/resume-gate.sh; miniagi/latent/train-resume.sh),

     warm=W (prints STEADY ms_per_step over the updates after the first W, one process: no two-run subtraction),
     gdump=PATH (raw first-update gradient, pre-clip, f32; gmut=x2|nan|zero:OFF:LEN plants a fault for the gate test),
     star=1 (text: before each update the model generates its own chain; sequences whose generated answer is right
     train on that chain, the rest get no loss; ground-truth hops are never trained, only the answer filters),
     ecool=F (toks/bytes with cool>0: error-annealed cooldown, in the cooldown the round(F b) sequences of an update with the highest loss
     are replayed as the first sequences of the next batch; ECOOL log lines), ecorp=PATH emix=G (cooldown: fraction G of the fresh sequences from a second u16 corpus),
     emixall=1 (with emix, ecool=0: the second corpus in every update, not only the cooldown; update it takes floor(G b (it + 1)) - floor(G b it) of its b sequences from it),
     mdist=L (needs mat=1, toks/bytes: every small-width update of the mat schedule trains on the batch of the full-width update before it, loss (1 - L) CE + L KL(p_full || p_small) against that update's saved bf16 logits; MDIST log lines; mdist=2: kernel self-test and exit),
     grow=STEP:FROM:TO gseed=E (toks/bytes, bf16 build: FFN width FROM until update STEP, then TO; w2 rows [FROM, TO) forced 0 at the switch so the loss is unchanged; gseed=E > 0 seeds the new neurons from the switch batch's highest-CE rows: w1/w3 column = the layer's FFN input of that row, w2 row = E * tied-embedding row of its target; GROW log lines; gmut=grow:ROW plants a nonzero w2 row for the gate),
     dgrow=STEP:L0 (toks/bytes: only the first L0 layers run (no kernels for the rest) until update STEP, then all L; wo and w2 of the inactive layers are forced 0 at the switch so the loss is unchanged; STEP = steps never switches (timing); DGROW log lines; gmut=dgrow:LAYER plants a nonzero wo for the gate).
Eval: latent at stage 7, text both teacher-forced (tf) and generated chain (gen, its own argmax fed back); the
HIT metric is latent stage 7 / text gen / nocot / pause answer accuracy on a validation stream (seed 10000+SEED);
the final RESULT uses an untouched test stream (seed 20000+SEED). hops = argmax readout at rows P0..P0+6 vs c[j].
"""
from stagecheck import check_stage_args
from std.sys import argv
from std.os import getenv
from std.os.path import exists
from std.os import remove
from std.ffi import external_call
from std.collections import Dict
from std.math import log, sqrt, cos, ceildiv, exp
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload
from ckpt import Blob, write_ckpt, GG_F32, GG_I32, extract
from lora import Experts, EP, XRng, tname
from lmodel import *
from muon import Muon, mu_selftest
from calc import calc, fmt
from lops import copy_k, cast16_k, argmax_k, ce_kd_k, BHK, BHS, CAP, CH, OPT, FLASH, sumsq_k, sumsqb_k, clip_k, adamw_k, adamw16_k, ema_k, zero_k, H16

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
    var shuf: Bool

    def __init__(out self, seed: Int, shuf: Bool = False) raises:
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1
        self.shuf = shuf
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
        var order = List[Int](length=NSYM, fill=0)
        for s in range(NSYM):
            order[s] = s
        if rng.shuf:  # table rows in random order: the key is no longer its position
            for s in range(NSYM - 1, 0, -1):
                var t = rng.below(s + 1)
                var u = order[s]
                order[s] = order[t]
                order[t] = u
        for s in range(NSYM):
            ids[(2 * s) * b + i] = Int32(order[s])
            ids[(2 * s + 1) * b + i] = Int32(f[order[s]])
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

    def __init__(out self, ctx: DeviceContext, b: Int, smax: Int = 0) raises:
        var SM = max(P0 + 1 + NTH + 2, smax)
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

    def weigh(mut self, ctx: DeviceContext, kinds: List[Int], b: Int, aux: Float64, w8: Bool = False, rw: Bool = False, up: Bool = True) raises:
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
            if up:
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
        if up:
            upload(ctx, self.wb, self.wts)


def argmax_row(z: List[Float32], o: Int) -> Int:
    var best = 0
    for v in range(1, V):
        if z[o + v] > z[o + best]:
            best = v
    return best


def check_answer(ids: List[Int32], b: Int, i: Int, rule: Int, want: Int) raises:
    """Independent checker: walk the table tokens of sequence i (whatever their order) from x; must reach want."""
    var f = List[Int](length=NSYM, fill=-1)
    for s in range(NSYM):
        f[Int(ids[(2 * s) * b + i])] = Int(ids[(2 * s + 1) * b + i])
    var cur = Int(ids[(2 * NSYM + 1) * b + i])
    for _ in range(HOPS):
        cur = hop_rule(f[cur], rule)
    if cur != want:
        raise Error("FAIL checker: sequence " + String(i) + " table walk gives " + String(cur) + ", generator " + String(want))


def evaluate[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, arm: String, kinds: List[Int], b: Int, n: Int, seed: Int, report: Bool = False, shuf: Bool = False) raises -> List[Float64]:
    """Answer accuracy at index HOPS - 1, hop-j readout accuracy at index j, over exactly n cases (the last batch is
    scored in part). report: print miss buckets. shuf: tables in random row order."""
    var rng = Rng(seed, shuf)
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
        var nb = min(b, n - total)
        for i in range(nb):
            if len(rng.rules) == 0:
                check_answer(bf.ids, b, i, rng.task, bf.chain[i * HOPS + HOPS - 1])
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
        total += nb
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


struct Corpus(Movable):
    """Training text as ids: bytes (w 1) or bpe8k uint16 ids (w 2, little endian); blen[id] = bytes the id spells."""
    var raw: List[UInt8]
    var w: Int
    var n: Int
    var blen: List[Int]

    def __init__(out self, path: String, vocab: String) raises:
        self.raw = open(path, "r").read_bytes() if path != "" else List[UInt8]()
        self.w = 2 if vocab != "" else 1
        self.n = len(self.raw) // self.w
        self.blen = List[Int](length=256, fill=1)
        if vocab != "":
            for line in open(vocab, "r").read().split("\n"):
                var f = String(line).split(" ")
                if len(f) >= 2:
                    self.blen.append(String(f[1]).byte_length() // 2)

    @always_inline
    def at(self, i: Int) -> Int:
        if self.w == 1:
            return Int(self.raw[i])
        return Int(self.raw[2 * i]) + 256 * Int(self.raw[2 * i + 1])


def fill_bytes(mut rng: Rng, c: Corpus, lo: Int, hi: Int, S: Int, b: Int, nb: Int, mut bf: Bufs) -> Int:
    """b windows of ids [lo, hi): ids = c[o, o + S), tgt = c[o + 1, o + S]; every row of the first nb sequences
    weighs 1 / (S nb) (mean CE), the rest 0. Returns the bytes the scored targets spell."""
    var w = Float32(1 / Float64(S * nb))
    var nbytes = 0
    for i in range(b):
        var o = lo + rng.below(hi - lo - S - 1)
        for r in range(S):
            var t = c.at(o + r + 1)
            bf.ids[r * b + i] = Int32(c.at(o + r))
            bf.tgt[r * b + i] = Int32(t)
            bf.wts[r * b + i] = w if i < nb else 0
            if i < nb:
                nbytes += c.blen[t]
    return nbytes


def fill_win(mut rng: Rng, c: Corpus, c2: Corpus, lo: Int, hi: Int, S: Int, b: Int, mut bf: Bufs, mut offs: List[Int], top: List[Int], nmix: Int):
    """Like fill_bytes (every sequence scored, weight 1 / (S b)), and records each window start in offs (a window of c2 as -(o + 1)).
    Sequence i < len(top) reuses window top[i] (error-annealed cooldown); the last nmix sequences are fresh windows of c2;
    the rest are fresh windows of [lo, hi) with the same rng draws as fill_bytes."""
    var w = Float32(1 / Float64(S * b))
    for i in range(b):
        var e = 0
        if i < len(top):
            e = top[i]
        elif i >= b - nmix:
            e = -(rng.below(c2.n - S - 1) + 1)
        else:
            e = lo + rng.below(hi - lo - S - 1)
        offs[i] = e
        var o = e if e >= 0 else -e - 1
        for r in range(S):
            var x = c.at(o + r) if e >= 0 else c2.at(o + r)
            var t = c.at(o + r + 1) if e >= 0 else c2.at(o + r + 1)
            bf.ids[r * b + i] = Int32(x)
            bf.tgt[r * b + i] = Int32(t)
            bf.wts[r * b + i] = w


def hardest(lrow: List[Float32], offs: List[Int], S: Int, b: Int, k: Int) -> List[Int]:
    """Window starts of the k sequences with the highest summed row loss (lrow[r * b + i]); ties keep the lower index."""
    var tot = List[Float64](length=b, fill=0)
    for r in range(S):
        for i in range(b):
            tot[i] += Float64(lrow[r * b + i])
    var top = List[Int]()
    for _ in range(k):
        var j = 0
        for i in range(b):
            if tot[i] > tot[j]:
                j = i
        top.append(offs[j])
        tot[j] = -1.0
    return top^


def offs_str(offs: List[Int], n: Int) -> String:
    var s = String("")
    for i in range(n):
        s += String(offs[i]) if i == 0 else "," + String(offs[i])
    return s


def grow_switch[exact: Bool](ctx: DeviceContext, mut m: Model[exact], p: F, g: F, mp: F, vp: F, bf: Bufs, kinds: List[Int], S: Int, f0: Int, f1: Int, e: Float32, plant: Int, it: Int) raises:
    """grow=STEP:FROM:TO at update `it`, on the batch already uploaded: forward at width f0 (its highest-CE rows seed the new neurons
    when e > 0), widen to f1 (Model.grow_ffn), forward again. Prints the summed row loss of both: equal when w2 rows [f0, f1) are 0."""
    var nr = S * m.b
    var lrow = List[Float32](length=nr, fill=0)
    m.ffm = f0
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
    ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, nr, owning=False))
    ctx.synchronize()
    var la = Float64(0)
    for x in lrow:
        la += Float64(x)
    var rows = List[Int]()
    if e > 0:  # the f1 - f0 rows the model gets most wrong (ties keep the lower row)
        if f1 - f0 > nr:
            raise Error("FAIL grow: " + String(f1 - f0) + " new neurons but " + String(nr) + " rows")
        for _ in range(f1 - f0):
            var j = 0
            for i in range(nr):
                if lrow[i] > lrow[j]:
                    j = i
            rows.append(j)
            lrow[j] = -1.0
    m.grow_ffn(ctx, p, mp, vp, f0, f1, e, rows, bf.tgt, nr, plant)
    if len(rows) > 0:
        var s = String("")
        for k in range(len(rows)):
            s += String(bf.tgt[rows[k]]) if k == 0 else "," + String(bf.tgt[rows[k]])
        print("GROW tgt", s)
    m.ffm = f1
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
    var lb = List[Float32](length=nr, fill=0)
    ctx.enqueue_copy(dst_ptr=lb.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, nr, owning=False))
    ctx.synchronize()
    var lt = Float64(0)
    for x in lb:
        lt += Float64(x)
    print("GROW it", it, "from", f0, "to", f1, "gseed", e, "loss_from", la, "loss_to", lt)


def dgrow_switch[exact: Bool](ctx: DeviceContext, mut m: Model[exact], p: F, g: F, mp: F, vp: F, bf: Bufs, kinds: List[Int], S: Int, l0: Int, plant: Int, it: Int) raises:
    """dgrow=STEP:L0 at update `it`, on the batch already uploaded: forward with l0 layers, join the rest (Model.dgrow_zero: wo and w2
    of the layers [l0, L) = 0), forward again with all L. Prints the summed row loss of both: equal when those layers are the identity."""
    var nr = S * m.b
    var lrow = List[Float32](length=nr, fill=0)
    m.nl = l0
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
    ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, nr, owning=False))
    ctx.synchronize()
    var la = Float64(0)
    for x in lrow:
        la += Float64(x)
    m.dgrow_zero(ctx, p, mp, vp, l0, plant)
    m.nl = L
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
    ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, nr, owning=False))
    ctx.synchronize()
    var lt = Float64(0)
    for x in lrow:
        lt += Float64(x)
    print("DGROW it", it, "layers", l0, "to", L, "loss_from", la, "loss_to", lt)


def ids_hash(ids: List[Int32], n: Int) -> Int:
    """Order-sensitive checksum of the first n token ids (the MDIST log lines prove a small update saw the previous batch)."""
    var h = 7
    for k in range(n):
        h = (h * 1000003 + Int(ids[k]) + 1) % 2147483629
    return h


def kd_ref(zh: List[Float32], qh: List[Float32], tg: List[Int32], wr: List[Float32], n: Int, lam: Float64, mut loss: List[Float64], mut dz: List[Float64]):
    """float64 host reference of ce_kd_k: row loss w ((1 - lam) CE + lam KL(q || p)), dz = w (p - (1 - lam) onehot - lam q)."""
    for t in range(n):
        var o = t * V
        var y = Int(tg[t])
        var w = Float64(wr[t])
        if y < 0 or w == 0:
            loss[t] = 0
            for j in range(V):
                dz[o + j] = 0
            continue
        var m = Float64(zh[o])
        var mq = Float64(qh[o])
        for j in range(V):
            m = max(m, Float64(zh[o + j]))
            mq = max(mq, Float64(qh[o + j]))
        var s = Float64(0)
        var sq = Float64(0)
        for j in range(V):
            s += exp(Float64(zh[o + j]) - m)
            sq += exp(Float64(qh[o + j]) - mq)
        var ls = log(s) + m
        var lq = log(sq) + mq
        var kl = Float64(0)
        for j in range(V):
            var lqj = Float64(qh[o + j]) - lq
            kl += exp(lqj) * (lqj - (Float64(zh[o + j]) - ls))
            var pj = exp(Float64(zh[o + j]) - ls)
            dz[o + j] = w * (pj - (1 - lam) * Float64(Int(j == y)) - lam * exp(lqj))
        loss[t] = w * ((1 - lam) * (ls - Float64(zh[o + y])) + lam * kl)


def kd_run(ctx: DeviceContext, zh: List[Float32], zb: DeviceBuffer[DType.float32], qd: DeviceBuffer[DType.bfloat16], tb: DeviceBuffer[DType.int32],
           wb: DeviceBuffer[DType.float32], lb: DeviceBuffer[DType.float32], db: DeviceBuffer[DType.float32], d16: DeviceBuffer[DType.bfloat16],
           n: Int, lam: Float64, mut loss: List[Float32], mut dz: List[Float32]) raises:
    """Upload the student logits zh, run ce_kd_k (fp32 dz) on n rows, bring loss [n] and dz [n, V] back."""
    upload(ctx, zb, zh)
    ctx.enqueue_function[ce_kd_k[True]](F(unsafe_from_address=Int(zb.unsafe_ptr())), H16(unsafe_from_address=Int(qd.unsafe_ptr())),
        I(unsafe_from_address=Int(tb.unsafe_ptr())), F(unsafe_from_address=Int(wb.unsafe_ptr())), F(unsafe_from_address=Int(lb.unsafe_ptr())),
        F(unsafe_from_address=Int(db.unsafe_ptr())), Int32(n), Int32(V), H16(unsafe_from_address=Int(d16.unsafe_ptr())), Float32(lam),
        grid_dim=grid(n * 32), block_dim=THREADS)
    ctx.enqueue_copy(dst_ptr=loss.unsafe_ptr(), src_buf=lb)
    ctx.enqueue_copy(dst_ptr=dz.unsafe_ptr(), src_buf=db)
    ctx.synchronize()


def mdist_selftest(ctx: DeviceContext) raises:
    """mdist=2: ce_kd_k (fp32 dz) on 16 rows (the last one masked), then exit. Checks (FAIL mdist ... on any miss):
    (1) loss and dz against the float64 host reference at L = 0, 0.5, 1 (rel 1e-4); (2) the directional derivative of the summed loss
    along dz (central difference of the kernel's own loss, step 0.05 on the unit direction dz / |dz|) equals |dz| (rel 0.05) at L = 0.5, 1;
    (3) teacher = student (bf16-exact logits) at L = 1: dz is 0 (max |dz| < 1e-2 of the L = 0 maximum); (4) planted: an unrelated teacher
    at L = 1 gives dz well above that bound, so check (3) can fail."""
    comptime if CAP > 0:
        raise Error("FAIL mdist: the self-test needs LAT_CAP=0")
    var n = 16
    var nv = n * V
    var rng = Rng(7)
    var zh = List[Float32](length=nv, fill=0)
    var qh = List[Float32](length=nv, fill=0)
    var qb = List[Scalar[DType.bfloat16]](length=nv, fill=Scalar[DType.bfloat16](0))
    var tg = List[Int32](length=n, fill=0)
    var wr = List[Float32](length=n, fill=Float32(1 / Float64(n)))
    for i in range(nv):
        zh[i] = Float32(2 * rng.normal())
        var qv = Float32(2 * rng.normal()).cast[DType.bfloat16]()
        qb[i] = qv
        qh[i] = qv.cast[DType.float32]()
    for t in range(n):
        tg[t] = Int32(rng.below(V))
    tg[n - 1] = -1
    var zb = ctx.enqueue_create_buffer[DType.float32](nv)
    var qd = ctx.enqueue_create_buffer[DType.bfloat16](nv)
    var tb = ctx.enqueue_create_buffer[DType.int32](n)
    var wb = ctx.enqueue_create_buffer[DType.float32](n)
    var lb = ctx.enqueue_create_buffer[DType.float32](n)
    var db = ctx.enqueue_create_buffer[DType.float32](nv)
    var d16 = ctx.enqueue_create_buffer[DType.bfloat16](1)
    upload(ctx, qd, qb)
    upload(ctx, tb, tg)
    upload(ctx, wb, wr)
    var loss = List[Float32](length=n, fill=0)
    var dz = List[Float32](length=nv, fill=0)
    var rl = List[Float64](length=n, fill=0)
    var rz = List[Float64](length=nv, fill=0)
    var lams = [0.0, 0.5, 1.0]
    var dz0max = Float64(0)
    for li in range(3):
        var lam = lams[li]
        kd_run(ctx, zh, zb, qd, tb, wb, lb, db, d16, n, lam, loss, dz)
        kd_ref(zh, qh, tg, wr, n, lam, rl, rz)
        var zmax = Float64(0)
        var zerr = Float64(0)
        for i in range(nv):
            zmax = max(zmax, abs(rz[i]))
            zerr = max(zerr, abs(Float64(dz[i]) - rz[i]))
        var lmax = Float64(0)
        var lerr = Float64(0)
        for t in range(n):
            lmax = max(lmax, abs(rl[t]))
            lerr = max(lerr, abs(Float64(loss[t]) - rl[t]))
        print("MDIST selftest ref L", lam, "dz rel", zerr / zmax, "loss rel", lerr / lmax)
        if not (zerr <= 1e-4 * zmax and lerr <= 1e-4 * lmax):
            raise Error("FAIL mdist ref: kernel differs from the float64 reference at L " + String(lam))
        if li > 0:  # the L = 0 case is the plain CE gradient: its max bounds check (3)
            var nrm = Float64(0)
            for i in range(nv):
                nrm += Float64(dz[i]) * Float64(dz[i])
            nrm = sqrt(nrm)
            var up = zh.copy()
            var dn = zh.copy()
            var ls_ = List[Float32](length=n, fill=0)
            var eps = 0.05
            for i in range(nv):
                up[i] = zh[i] + Float32(eps * Float64(dz[i]) / nrm)
                dn[i] = zh[i] - Float32(eps * Float64(dz[i]) / nrm)
            kd_run(ctx, up, zb, qd, tb, wb, lb, db, d16, n, lam, ls_, dz)
            var lu = Float64(0)
            for t in range(n):
                lu += Float64(ls_[t])
            kd_run(ctx, dn, zb, qd, tb, wb, lb, db, d16, n, lam, ls_, dz)
            var ld = Float64(0)
            for t in range(n):
                ld += Float64(ls_[t])
            var fd = (lu - ld) / (2 * eps)
            print("MDIST selftest fd L", lam, "directional", fd, "|dz|", nrm, "rel", abs(fd - nrm) / nrm)
            if abs(fd - nrm) > 0.05 * nrm:
                raise Error("FAIL mdist fd: finite difference " + String(fd) + " vs |dz| " + String(nrm) + " at L " + String(lam))
        else:
            dz0max = zmax
    # (3) teacher = student: the student logits are the bf16-exact teacher values
    var same = qh.copy()
    kd_run(ctx, same, zb, qd, tb, wb, lb, db, d16, n, 1.0, loss, dz)
    var smax = Float64(0)
    for i in range(nv):
        smax = max(smax, abs(Float64(dz[i])))
    print("MDIST selftest same teacher L 1 max |dz|", smax, "vs plain CE max", dz0max)
    if not (smax < 1e-2 * dz0max):
        raise Error("FAIL mdist same: dz " + String(smax) + " with teacher = student")
    # (4) planted: the unrelated teacher gives a clearly nonzero dz, so (3) can fail
    kd_run(ctx, zh, zb, qd, tb, wb, lb, db, d16, n, 1.0, loss, dz)
    var pmax = Float64(0)
    for i in range(nv):
        pmax = max(pmax, abs(Float64(dz[i])))
    print("MDIST selftest planted unrelated teacher L 1 max |dz|", pmax)
    if not (pmax > 1e-2 * dz0max):
        raise Error("FAIL mdist planted: an unrelated teacher left dz near 0 (" + String(pmax) + ")")
    print("MDIST selftest PASS")


def eval_bytes[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, c: Corpus, lo: Int, hi: Int, S: Int, b: Int, n: Int, seed: Int) raises -> Float64:
    """Bits per byte over exactly n fixed windows of ids [lo, hi) (stream seed): summed CE in bits / bytes spelled."""
    var rng = Rng(seed)
    var kinds = List[Int](length=S, fill=TOK)
    var lrow = List[Float32](length=S * b, fill=0)
    var tot: Float64 = 0
    var nbytes = 0
    var done = 0
    while done < n:
        var nb = min(b, n - done)
        nbytes += fill_bytes(rng, c, lo, hi, S, b, nb, bf)
        upload(ctx, bf.ib, bf.ids)
        upload(ctx, bf.tb, bf.tgt)
        upload(ctx, bf.wb, bf.wts)
        m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
        ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
        ctx.synchronize()
        var sb: Float64 = 0
        for x in lrow:
            sb += Float64(x)
        tot += sb * Float64(S * nb)  # sb is the mean CE per row of the nb scored sequences
        done += nb
    return tot / log(2.0) / Float64(nbytes)


def run_score[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, req: String, dst: String, S: Int, b: Int) raises:
    """baro-bench scorer: each candidate (benchprep request: task item choice gold nctx ncont nbytes, then the ids) is one
    sequence; rows whose target is a continuation token weigh 1, so lrow is -log p per scored token. Writes one line per
    candidate: task item choice gold ncont nbytes sum_logprob greedy (1 when every continuation token is the argmax)."""
    var raw = open(req, "r").read_bytes()
    var q = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var nq = Int(q[0])
    var at = 1
    var heads = List[Int]()  # offset of each record
    for _ in range(nq):
        heads.append(at)
        at += 7 + Int(q[at + 4]) + Int(q[at + 5])
    var kinds = List[Int](length=S, fill=TOK)
    var lrow = List[Float32](length=S * b, fill=0)
    var am = List[Int32](length=S * b, fill=0)
    var amb = ctx.enqueue_create_buffer[DType.int32](S * b)
    var out = String()
    var i0 = 0
    while i0 < nq:
        var nb = min(b, nq - i0)
        for r in range(S * b):
            bf.ids[r] = 0
            bf.tgt[r] = 0
            bf.wts[r] = 0
        for j in range(nb):
            var h = heads[i0 + j]
            var nc = Int(q[h + 4])
            var nn = nc + Int(q[h + 5])
            if nn > S + 1 or nc < 1:
                raise Error("FAIL score: candidate " + String(i0 + j) + " length " + String(nn) + " > S + 1 or empty context")
            for r in range(nn - 1):
                bf.ids[r * b + j] = q[h + 7 + r]
                bf.tgt[r * b + j] = q[h + 8 + r]
                bf.wts[r * b + j] = 1 if r >= nc - 1 else 0
        upload(ctx, bf.ib, bf.ids)
        upload(ctx, bf.tb, bf.tgt)
        upload(ctx, bf.wb, bf.wts)
        m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
        ctx.enqueue_function[argmax_k](m.z, I(unsafe_from_address=Int(amb.unsafe_ptr())), Int32(S * b), Int32(V), grid_dim=grid(S * b), block_dim=THREADS)
        ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
        ctx.enqueue_copy(dst_ptr=am.unsafe_ptr(), src_buf=amb)
        ctx.synchronize()
        for j in range(nb):
            var h = heads[i0 + j]
            var nc = Int(q[h + 4])
            var nn = nc + Int(q[h + 5])
            var lp: Float64 = 0
            var greedy = 1
            for r in range(nc - 1, nn - 1):
                lp -= Float64(lrow[r * b + j])
                if am[r * b + j] != q[h + 8 + r]:
                    greedy = 0
            out += String(q[h]) + " " + String(q[h + 1]) + " " + String(q[h + 2]) + " " + String(q[h + 3]) + " " + String(q[h + 5]) + " " + String(q[h + 6]) + " " + String(lp) + " " + String(greedy) + "\n"
        i0 += nb
    var f = open(dst, "w")
    f.write(out)
    f.close()
    print("SCORE", nq, "candidates ->", dst)


def hexbytes(h: String) -> List[UInt8]:
    var o = List[UInt8]()
    var b = h.as_bytes()
    for i in range(0, len(b) - 1, 2):
        var x = 0
        for k in range(2):
            var c = Int(b[i + k])
            x = x * 16 + (c - 48 if c <= 57 else c - 87)
        o.append(UInt8(x))
    return o^


def run_gen[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, req: String, dst: String, vpath: String, S: Int, b: Int, gmax: Int, tools: Bool) raises:
    """baro-bench part 2: greedy generation (full forward per token, b prompts at once) with the calc tool loop. Request
    (benchprep gbuild): count, im_end, tool_call, /tool_call ids, the tool-response prefix and suffix ids, then per item
    its index and prompt ids. When an assistant turn holding a tool call ends, the expr of {"expr": "..."} goes through
    calc and prefix + bytes(value) + suffix are appended. Output line: item calls valid_calls HEX(generated bytes)."""
    var tab = List[List[UInt8]]()
    for i in range(256):
        tab.append([UInt8(i)])
    for l in open(vpath, "r").read().split("\n"):
        if l.byte_length() > 0:
            var sp = l.find(" ")
            tab.append(hexbytes(String(l[byte = sp + 1 :])))
    var raw = open(req, "r").read_bytes()
    var q = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var nq = Int(q[0])
    var IE = q[1]
    var TO = q[2]
    var TC = q[3]
    var at = 4
    var pre = List[Int32]()
    for _ in range(Int(q[at])):
        at += 1
        pre.append(q[at])
    at += 1
    var suf = List[Int32]()
    for _ in range(Int(q[at])):
        at += 1
        suf.append(q[at])
    at += 1
    var items = List[Int]()
    var prompts = List[List[Int32]]()
    for _ in range(nq):
        items.append(Int(q[at]))
        var n = Int(q[at + 1])
        var pr = List[Int32]()
        for k in range(n):
            pr.append(q[at + 2 + k])
        prompts.append(pr^)
        at += 2 + n
    var kinds = List[Int](length=S, fill=TOK)
    var am = List[Int32](length=S * b, fill=0)
    var amb = ctx.enqueue_create_buffer[DType.int32](S * b)
    var out = String()
    comptime H = "0123456789abcdef"
    var i0 = 0
    while i0 < nq:
        var nb = min(b, nq - i0)
        var seqs = List[List[Int32]]()
        var start = List[Int]()
        var turn = List[Int]()
        var live = List[Bool]()
        var calls = List[Int](length=nb, fill=0)
        var good = List[Int](length=nb, fill=0)
        for j in range(nb):
            seqs.append(prompts[i0 + j].copy())
            start.append(len(prompts[i0 + j]))
            turn.append(len(prompts[i0 + j]))
            live.append(len(prompts[i0 + j]) < S)
        while True:
            var any = False
            for j in range(nb):
                any = any or live[j]
            if not any:
                break
            for r in range(S * b):
                bf.ids[r] = 0
                bf.tgt[r] = 0
                bf.wts[r] = 0
            for j in range(nb):
                for r in range(len(seqs[j])):
                    bf.ids[r * b + j] = seqs[j][r]
            upload(ctx, bf.ib, bf.ids)
            upload(ctx, bf.tb, bf.tgt)
            upload(ctx, bf.wb, bf.wts)
            m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
            ctx.enqueue_function[argmax_k](m.z, I(unsafe_from_address=Int(amb.unsafe_ptr())), Int32(S * b), Int32(V), grid_dim=grid(S * b), block_dim=THREADS)
            ctx.enqueue_copy(dst_ptr=am.unsafe_ptr(), src_buf=amb)
            ctx.synchronize()
            for j in range(nb):
                if not live[j]:
                    continue
                var t = am[(len(seqs[j]) - 1) * b + j]
                seqs[j].append(t)
                if len(seqs[j]) >= S or len(seqs[j]) - start[j] >= gmax:
                    live[j] = False
                    continue
                if t != IE:
                    continue
                var o0 = -1
                var o1 = -1
                for k in range(turn[j], len(seqs[j])):
                    if seqs[j][k] == TO:
                        o0 = k
                    if seqs[j][k] == TC:
                        o1 = k
                if not tools or o0 < 0 or o1 < o0 or calls[j] >= 8:
                    live[j] = False
                    continue
                calls[j] += 1
                var body = String()
                for k in range(o0 + 1, o1):
                    body += String(unsafe_from_utf8=Span(tab[Int(seqs[j][k])]))
                var val = String("error")
                var e0 = body.find('"expr": "')
                if e0 >= 0:
                    var e1 = body.find('"', e0 + 9)
                    if e1 > e0:
                        var c = calc(String(body[byte = e0 + 9 : e1]))
                        if c[0]:
                            val = fmt(c[1])
                            good[j] += 1
                for x in pre:
                    seqs[j].append(x)
                for x in val.as_bytes():
                    seqs[j].append(Int32(x))
                for x in suf:
                    seqs[j].append(x)
                turn[j] = len(seqs[j])
                if len(seqs[j]) >= S:
                    live[j] = False
        for j in range(nb):
            var hx = String()
            for k in range(start[j], len(seqs[j])):
                for x in tab[Int(seqs[j][k])]:
                    hx += H[byte = Int(x >> 4)] + H[byte = Int(x & 15)]
            out += String(items[i0 + j]) + " " + String(calls[j]) + " " + String(good[j]) + " " + hx + "\n"
        i0 += nb
        print("GEN", i0, "of", nq)
    var f = open(dst, "w")
    f.write(out)
    f.close()
    print("GEN done ->", dst)


def upload_raw(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], path: String) raises:
    var raw = open(path, "r").read_bytes()
    if len(raw) != len(dst) * 4:
        raise Error("FAIL resume: " + path + " has " + String(len(raw)) + " bytes, want " + String(len(dst) * 4))
    ctx.enqueue_copy(dst_buf=dst, src_ptr=raw.unsafe_ptr().unsafe_bitcast[Float32]())
    ctx.synchronize()


def eval_fixed[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, path: String, S: Int, b: Int, c: Corpus) raises -> Float64:
    """Bits per byte over the windows stored in path (i32, S + 1 ids each, at most b): the closure check."""
    var raw = open(path, "r").read_bytes()
    var ids = raw.unsafe_ptr().unsafe_bitcast[Int32]()
    var nw = len(raw) // 4 // (S + 1)
    var w = Float32(1 / Float64(S * nw))
    var nbytes = 0
    for i in range(b):
        for r in range(S):
            var ok = i < nw
            bf.ids[r * b + i] = ids[i * (S + 1) + r] if ok else 0
            bf.tgt[r * b + i] = ids[i * (S + 1) + r + 1] if ok else 0
            bf.wts[r * b + i] = w if ok else 0
            if ok:
                nbytes += c.blen[Int(ids[i * (S + 1) + r + 1])]
    upload(ctx, bf.ib, bf.ids)
    upload(ctx, bf.tb, bf.tgt)
    upload(ctx, bf.wb, bf.wts)
    var kinds = List[Int](length=S, fill=TOK)
    m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
    var lrow = List[Float32](length=S * b, fill=0)
    ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
    ctx.synchronize()
    var sb: Float64 = 0
    for x in lrow:
        sb += Float64(x)
    return sb * Float64(S * nw) / log(2.0) / Float64(nbytes)


def _f32_bytes(ctx: DeviceContext, src: DeviceBuffer[DType.float32]) raises -> List[UInt8]:
    var h = List[UInt8](length=len(src) * 4, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr().unsafe_bitcast[Float32](), src_buf=src)
    ctx.synchronize()
    return h^


def write_self[exact: Bool](ctx: DeviceContext, m: Model[exact], p: F, g: F, mut bf: Bufs, c: Corpus, pb: DeviceBuffer[DType.float32],
                            mb: DeviceBuffer[DType.float32], vb: DeviceBuffer[DType.float32], path: String, corpus: String, vocab: String,
                            S: Int, b: Int, seed: Int, dseed: Int, step: Int, rngs: UInt64, lr: Float32, nc: Int, total: Int) raises:
    """Self-describing checkpoint: sources, build line, vocab, provenance, resume state, weights, Adam moments and a
    closure set (min(b, 64) test windows, their bpb computed now by this binary). Written to PATH.tmp, then renamed over
    PATH, so an interrupted save never leaves a torn checkpoint; `total` is the run's step count (resume checks it)."""
    var tmp = path + ".tmp"
    var av = String()
    for i in range(len(argv())):
        av += (" " if i > 0 else "") + String(argv()[i])
    var nw = min(b, 64)
    var crng = Rng(30000 + seed)
    var cl = List[Int32]()
    var lo = nc - nc // 200
    for _ in range(nw):
        var o = lo + crng.below(nc - lo - S - 1)
        for r in range(S + 1):
            cl.append(Int32(c.at(o + r)))
    var cpath = path + ".closure_ids.bin"
    var fc = open(cpath, "w")
    fc.write_bytes(Span[UInt8](unsafe_ptr=cl.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(cl) * 4))
    fc.close()
    var cbpb = eval_fixed(ctx, m, p, g, bf, cpath, S, b, c)
    var keys = List[String]()
    var vals = List[String]()
    var here = String(argv()[0])
    for f in ["miniagi/latent/run.mojo", "miniagi/latent/lmodel.mojo", "miniagi/latent/lops.mojo", "miniagi/latent/ckpt.mojo",
              "miniagi/latent/stagecheck.mojo", "miniagi/latent/nlook.mojo", "miniagi/latent/bpe8k.mojo", "miniagi/train/ops.mojo"]:
        if exists(f):
            keys.append(String("baro.kernel.src.") + f)
            vals.append(open(f, "r").read())
    keys.append("baro.build")
    vals.append(getenv("LAT_BUILD", "(set LAT_BUILD to the mojo build line)"))
    if vocab != "":
        keys.append("baro.vocab")
        vals.append(open(vocab, "r").read())
    var meta = [("np", String(NP)), ("step", String(step)), ("rng", String(Int(rngs))), ("seed", String(seed)), ("dseed", String(dseed)),
                ("lr", String(lr)), ("seq", String(S)), ("b", String(b)), ("corpus", corpus), ("corpus_ids", String(nc)),
                ("split", "train [0, 99%) val [99%, 99.5%) test [99.5%, 100%) of ids"), ("closure_bpb", String(cbpb)), ("opt", String(OPT)),
                ("steps", String(total)), ("argv", av)]
    for kv in meta:
        keys.append(String("baro.meta.") + kv[0])
        vals.append(kv[1])
    if m.xe.NX > 0:
        keys.append("baro.x.cfg")
        vals.append(m.xe.cfg())
    var blobs = List[Blob]()
    blobs.append(Blob("params", _f32_bytes(ctx, pb), NP, GG_F32))
    blobs.append(Blob("adam_m", _f32_bytes(ctx, mb), len(mb), GG_F32))  # 1 element under frz (no base moments)
    blobs.append(Blob("adam_v", _f32_bytes(ctx, vb), len(vb), GG_F32))
    if m.xe.NX > 0:
        blobs.append(Blob("experts", _f32_bytes(ctx, m.xe.xb), m.xe.NX, GG_F32))
        blobs.append(Blob("xadam_m", _f32_bytes(ctx, m.xe.xm), m.xe.NX, GG_F32))
        blobs.append(Blob("xadam_v", _f32_bytes(ctx, m.xe.xv), m.xe.NX, GG_F32))
    var cb8 = List[UInt8](Span[UInt8](unsafe_ptr=cl.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(cl) * 4))
    blobs.append(Blob("closure_ids", cb8^, len(cl), GG_I32))
    write_ckpt(tmp, keys, vals, blobs)
    remove(cpath)
    var z = path
    if external_call["rename", Int32](tmp.as_c_string_slice().unsafe_ptr(), z.as_c_string_slice().unsafe_ptr()) != 0:
        raise Error("FAIL ckpt: rename " + tmp + " -> " + path)
    print("CKPT", path, "step", step, "of", total, "closure_bpb", cbpb, "sources", len(keys) - len(meta) - 2 - Int(m.xe.NX > 0))


def read_f32(path: String, n: Int) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != n * 4:
        raise Error("FAIL read: " + path + " has " + String(len(raw)) + " bytes, want " + String(n * 4))
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var h = List[Float32](length=n, fill=0)
    for i in range(n):
        h[i] = src[i]
    return h^


def write_experts(ctx: DeviceContext, xe: Experts, path: String) raises:
    """The experts alone (router included when routed), self-describing: the shape line, the layout, the base it adds to."""
    var keys = List[String]()
    var vals = List[String]()
    var av = String()
    for i in range(len(argv())):
        av += (" " if i > 0 else "") + String(argv()[i])
    var kv = [("baro.x.cfg", xe.cfg()), ("baro.x.r", String(xe.r)), ("baro.x.E", String(xe.E)), ("baro.x.null", String(xe.null)),
              ("baro.x.topk", String(xe.topk)), ("baro.x.tm", String(xe.tm)), ("baro.x.alpha", String(xe.alpha)), ("baro.x.base_np", String(NP)),
              ("baro.x.layout", "per layer: router R [D, 8] (routed only, column c = option c, option 0 = base path when null = 1), then per target in w1,w3,w2,wo order: AT [E r, in] then B [E r, out]; y = x W + (alpha / r) sum_e g_e (x AT_e^T) B_e"),
              ("baro.x.argv", av)]
    for p in kv:
        keys.append(p[0])
        vals.append(p[1])
    var blobs = List[Blob]()
    blobs.append(Blob("experts", _f32_bytes(ctx, xe.xb), xe.NX, GG_F32))
    write_ckpt(path, keys, vals, blobs)
    print("XEXPORT", path, "nx", xe.NX, "routed", xe.routed())


def load_experts(ctx: DeviceContext, mut xe: Experts, path: String, tm: Int) raises:
    """xinit=FILE (write_experts output): same r, null, alpha and targets; its E0 <= E experts fill slots [0, E0) and the router
    columns [0, null + E0); the rest keep this run's init (a new random expert with B = 0, the paper's added expert)."""
    var dir = path + ".d"
    _ = extract(path, dir)
    var meta = Dict[String, String]()
    for line in open(dir + "/meta.txt", "r").read().split("\n"):
        var kv = String(line).split("\t")
        if len(kv) == 2:
            meta[String(kv[0])] = String(kv[1])
    var e0 = Int(meta["baro.x.E"])
    if Int(meta["baro.x.r"]) != xe.r or Int(meta["baro.x.null"]) != xe.null or Int(meta["baro.x.tm"]) != tm or Float64(meta["baro.x.alpha"]) != xe.alpha or e0 > xe.E:
        raise Error("FAIL xinit: " + path + " is '" + meta["baro.x.cfg"] + "', this run '" + xe.cfg() + "' (same r, null, alpha, targets; E0 <= E)")
    var src = Experts(ctx)
    src.setup(ctx, xe.r, e0, xe.null, 0, tm, xe.alpha, xe.L, xe.FF, 64, 1, 0)
    var hs = read_f32(dir + "/experts.bin", src.NX)
    var h = xe.download(ctx, xe.xb)
    var r = xe.r
    for l in range(xe.L):
        if xe.routed() and src.routed():
            for d in range(DM):
                for c in range(src.nopt):
                    h[xe.rofs(l) + d * EP + c] = hs[src.rofs(l) + d * EP + c]
        for t in range(4):
            if not xe.has(t):
                continue
            for j in range(e0 * r * xe.tin(t)):
                h[xe.aofs(l, t) + j] = hs[src.aofs(l, t) + j]
            for j in range(e0 * r * xe.tout(t)):
                h[xe.bofs(l, t) + j] = hs[src.bofs(l, t) + j]
    xe.upload(ctx, h)
    print("XINIT", path, "experts", e0, "of", xe.E, "loaded")


def show(r: List[Float64]) -> String:
    var s = String("acc ") + String(r[HOPS - 1]) + " hops"
    for j in range(NTH):
        s += " " + String(Float64(Int(r[j] * 1000 + 0.5)) / 10)
    return s


def finite(x: Float64) -> Bool:
    return x == x and abs(x) < 1e30


def dump_grad(ctx: DeviceContext, gb: DeviceBuffer[DType.float32], path: String, mut_: String) raises:
    """Raw first-update gradient (pre-clip) to PATH as f32. mut_ plants a known fault for the gate's own test:
    x2 doubles it, nan sets element 0 to NaN, zero:OFF:LEN zeroes one block."""
    var n = len(gb)
    var h = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=gb)
    ctx.synchronize()
    if mut_ == "x2":
        for i in range(n):
            h[i] *= 2
    elif mut_ == "nan":
        h[0] = Float32(0) / Float32(0)
    elif mut_.startswith("zero:"):
        var f = mut_.split(":")
        for i in range(Int(String(f[1])), Int(String(f[1])) + Int(String(f[2]))):
            h[i] = 0
    elif mut_ != "":
        raise Error("FAIL gmut: " + mut_)
    with open(path, "w") as fo:
        fo.write_bytes(Span[UInt8](unsafe_ptr=h.unsafe_ptr().unsafe_bitcast[UInt8](), length=n * 4))


def dump_rows(ctx: DeviceContext, src: F, n: Int, path: String) raises:
    """n floats from the device (zdump: the logits of the first update's forward) to PATH."""
    var h = List[Float32](length=n, fill=0)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, src, n, owning=False))
    ctx.synchronize()
    with open(path, "w") as fo:
        fo.write_bytes(Span[UInt8](unsafe_ptr=h.unsafe_ptr().unsafe_bitcast[UInt8](), length=n * 4))


def expert_fd[exact: Bool](ctx: DeviceContext, mut m: Model[exact], p: F, g: F, mut bf: Bufs, kinds: List[Int], S: Int, b: Int, spec: String, mut_: String) raises:
    """xfd=K:EPS, the gradient oracle of the expert path. Per block (each layer's router R, and per target its AT and B) the
    summed row loss is re-evaluated at x +- EPS d with d = g_block / |g_block| (forward only, same batch, same teacher logits):
    the central difference must equal |g_block| from the backward that just ran; a zero block gradient counts as a failure
    (with B random and the router on, every block has one). Then K random unit directions over all floats (reported).
    mut_ x2 doubles the analytic gradient (the gate's planted fault: it must fail)."""
    var sp = spec.split(":")
    var k = Int(String(sp[0]))
    var eps = Float64(String(sp[1]))
    var x0 = m.xe.download(ctx, m.xe.xb)
    var gx = m.xe.download(ctx, m.xe.xg)
    if mut_ == "x2":
        for i in range(len(gx)):
            gx[i] *= 2
    var n = m.xe.NX
    var lrow = List[Float32](length=S * b, fill=0)
    var bo = List[Int]()
    var bn = List[Int]()
    var bname = List[String]()
    var nbl = m.xe.nblocks() // m.xe.L
    for j in range(m.xe.nblocks()):  # the first and the last layer: every block kind, half the forwards of all layers
        if j // nbl != 0 and j // nbl != m.xe.L - 1:
            continue
        var bk = m.xe.block(j)
        bo.append(bk[0]); bn.append(bk[1]); bname.append(bk[2])
    var rng = XRng(777)
    var worst = Float64(0)
    var zero = 0
    for j in range(len(bo) + k):
        var d = List[Float64](length=n, fill=0)
        var nn = Float64(0)
        if j < len(bo):
            for i in range(bo[j], bo[j] + bn[j]):
                d[i] = Float64(gx[i])
                nn += d[i] * d[i]
        else:
            for i in range(n):
                d[i] = rng.normal()
                nn += d[i] * d[i]
        nn = sqrt(nn)
        if nn == 0:
            print("XFD block", bname[j], "gradient is zero: FAIL")
            zero += 1
            continue
        var an = Float64(0)
        for i in range(n):
            d[i] /= nn
            an += Float64(gx[i]) * d[i]
        var ls = List[Float64]()
        for sg in [1.0, -1.0]:
            var xs = List[Float32](length=n, fill=0)
            for i in range(n):
                xs[i] = Float32(Float64(x0[i]) + sg * eps * d[i])
            m.xe.upload(ctx, xs)
            m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
            ctx.enqueue_copy(dst_ptr=lrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
            ctx.synchronize()
            var s = Float64(0)
            for v in lrow:
                s += Float64(v)
            ls.append(s)
        var fd = (ls[0] - ls[1]) / (2 * eps)
        var rel = abs(fd - an) / max(abs(an), 1e-12)
        if j < len(bo):
            worst = max(worst, rel)
        print("XFD", bname[j] if j < len(bo) else "random " + String(j - len(bo)), "fd", fd, "analytic", an, "rel", rel)
    m.xe.upload(ctx, x0)
    print("XFD worst block rel", worst, "blocks", len(bo), "zero", zero, "eps", eps)


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
    if arm != "nocot" and arm != "text" and arm != "latent" and arm != "pause" and arm != "bytes" and arm != "toks":
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
    var muon = 0
    var mlr = -1.0
    var mbeta = 0.95
    var nsit = 5
    var cool = 0.0
    var zinit = 0
    var lpd = 1
    var lpdv = 1
    var mhyb = 0
    var leps = 7e-3
    var mat = 0
    var ffm = FF
    var score = String("")
    var genr = String("")
    var gout = String("")
    var gmax = 320
    var gtools = 0
    var sout = String("")
    var bhl = 0.3
    var bhd = 0
    var shuf = 0
    var corpus = String("")
    var vocab = String("")
    var warm = 0
    var ckpt = String("")
    var resume = String("")
    var save = 0  # save=N: checkpoint (ckpt=PATH) every N updates; resume with the same command plus resume=DIR
    var stopat = 0  # stopat=K: stop after update K and write the checkpoint (an interruption, for the resume gate)
    var closure = String("")
    var seq = 0
    var dseed = -1
    var gdump = String("")
    var gmut = String("")
    var ecool = 0.0
    var ecorp = String("")
    var emix = 0.0
    var emixall = 0
    var mdist = 0.0
    var grow = String("")
    var gseed = 0.0
    var dgrow = String("")
    var xr = 0  # experts (lora.mojo): rank; 0 = off
    var xe = 1
    var xnull = 0
    var xtopk = 0
    var xrt = -1
    var xt = String("w1,w3,w2")
    var xalpha = 0.0
    var xlr = -1.0
    var xseed = -1
    var xbstd = 0.0
    var xinit = String("")
    var xtrain = 0
    var xmerge = String("")
    var xgdump = String("")
    var xgref = String("")
    var xfd = String("")
    var frz = 0
    var xkd = 0.0
    var zdump = String("")
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
        elif key == "muon":
            muon = Int(val)
        elif key == "mlr":
            mlr = Float64(val)
        elif key == "mbeta":
            mbeta = Float64(val)
        elif key == "ns":
            nsit = Int(val)
        elif key == "cool":
            cool = Float64(val)
        elif key == "bhd":
            bhd = Int(val)
        elif key == "bhl":
            bhl = Float64(val)
        elif key == "gen":
            genr = val
        elif key == "gout":
            gout = val
        elif key == "gmax":
            gmax = Int(val)
        elif key == "gtools":
            gtools = Int(val)
        elif key == "score":
            score = val
        elif key == "sout":
            sout = val
        elif key == "mat":
            mat = Int(val)
        elif key == "ffm":
            ffm = Int(val)
        elif key == "leps":
            leps = Float64(val)
        elif key == "mhyb":
            mhyb = Int(val)
        elif key == "lpdv":
            lpdv = Int(val)
        elif key == "lpd":
            lpd = Int(val)
        elif key == "zinit":
            zinit = Int(val)
        elif key == "corpus":
            corpus = val
        elif key == "vocab":
            vocab = val
        elif key == "warm":
            warm = Int(val)
        elif key == "ckpt":
            ckpt = val
        elif key == "resume":
            resume = val
        elif key == "save":
            save = Int(val)
        elif key == "stopat":
            stopat = Int(val)
        elif key == "closure":
            closure = val
        elif key == "seq":
            seq = Int(val)
        elif key == "shuf":
            shuf = Int(val)
        elif key == "dseed":
            dseed = Int(val)
        elif key == "gdump":
            gdump = val
        elif key == "gmut":
            gmut = val
        elif key == "ecool":
            ecool = Float64(val)
        elif key == "ecorp":
            ecorp = val
        elif key == "emix":
            emix = Float64(val)
        elif key == "emixall":
            emixall = Int(val)
        elif key == "mdist":
            mdist = Float64(val)
        elif key == "grow":
            grow = val
        elif key == "gseed":
            gseed = Float64(val)
        elif key == "dgrow":
            dgrow = val
        elif key == "xr":
            xr = Int(val)
        elif key == "xe":
            xe = Int(val)
        elif key == "xnull":
            xnull = Int(val)
        elif key == "xtopk":
            xtopk = Int(val)
        elif key == "xrt":
            xrt = Int(val)
        elif key == "xt":
            xt = val
        elif key == "xalpha":
            xalpha = Float64(val)
        elif key == "xlr":
            xlr = Float64(val)
        elif key == "xseed":
            xseed = Int(val)
        elif key == "xbstd":
            xbstd = Float64(val)
        elif key == "xinit":
            xinit = val
        elif key == "xtrain":
            xtrain = Int(val)
        elif key == "xmerge":
            xmerge = val
        elif key == "xgdump":
            xgdump = val
        elif key == "xfd":
            xfd = val
        elif key == "xgref":
            xgref = val
        elif key == "frz":
            frz = Int(val)
        elif key == "xkd":
            xkd = Float64(val)
        elif key == "zdump":
            zdump = val
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
    var lm = arm == "bytes" or arm == "toks"
    var cb = Corpus(corpus if lm else "", vocab if arm == "toks" else "")
    var nc = cb.n
    if lm:  # text: train on ids [0, 99%), validation [99%, 99.5%), test [99.5%, 100%)
        var need = 256 if arm == "bytes" else len(cb.blen)
        if corpus == "" or seq <= 0 or seq > SMAX or V < need or (arm == "toks" and vocab == ""):
            raise Error("FAIL " + arm + ": needs corpus=PATH (toks: vocab=PATH), 0 < seq <= LAT_S (" + String(SMAX) + "), LAT_V >= " + String(need) + " (" + String(V) + ")")
        if nc // 200 < 4 * seq:
            raise Error("FAIL bytes: corpus " + String(nc) + " bytes too small for seq " + String(seq))
    var cb2 = Corpus(ecorp, vocab if arm == "toks" else "")
    var ecp = ecool > 0 or emix > 0  # error-annealed cooldown or cooldown corpus mix: the recording batch fill
    if ecool < 0 or ecool > 1 or emix < 0 or emix > 1:
        raise Error("FAIL ecool: ecool and emix are fractions in [0, 1] (" + String(ecool) + ", " + String(emix) + ")")
    if ecp and not (lm and cool > 0):
        raise Error("FAIL ecool: needs arm bytes or toks and cool > 0")
    if (emix > 0) != (ecorp != ""):
        raise Error("FAIL emix: ecorp=PATH and emix=G (> 0) go together")
    if emixall != 0 and (emixall != 1 or emix <= 0 or ecool > 0):
        raise Error("FAIL emixall: emixall=1 needs emix > 0 and ecool=0")
    if ecorp != "" and cb2.n < 4 * seq:
        raise Error("FAIL ecorp: " + String(cb2.n) + " ids too small for seq " + String(seq))
    var mdl = mdist > 0 and mdist <= 1  # in-place distillation of the small widths of the mat schedule
    if mdist < 0 or (mdist > 1 and mdist != 2):
        raise Error("FAIL mdist: a weight in (0, 1], or 2 for the kernel self-test (" + String(mdist) + ")")
    if mdl and not (mat != 0 and lm and V > 256 and not exact):
        raise Error("FAIL mdist: needs mat=1, arm bytes or toks, V > 256 and the bf16 path")
    if mdl and every % 2 == 1:  # a validation eval after an even update refills bf.ids and would replace the batch the next small update reuses
        raise Error("FAIL mdist: every=" + String(every) + " must be even (the eval would overwrite the reused batch)")
    var gstep = -1  # grow: update at which the FFN widens
    var gfrom = 0
    var gto = 0
    var gplant = -1
    if gseed < 0 or (gseed > 0 and grow == ""):
        raise Error("FAIL gseed: a scale >= 0, and only with grow=STEP:FROM:TO (" + String(gseed) + ")")
    if grow != "":
        var gp = grow.split(":")
        if len(gp) != 3:
            raise Error("FAIL grow: STEP:FROM:TO (" + grow + ")")
        gstep = Int(String(gp[0]))
        gfrom = Int(String(gp[1]))
        gto = Int(String(gp[2]))
        if gstep < 1 or gstep >= steps or gfrom < 8 or gfrom % 8 != 0 or gto <= gfrom or gto > FF or gto % 8 != 0:
            raise Error("FAIL grow: 1 <= STEP < steps, FROM and TO multiples of 8, 8 <= FROM < TO <= " + String(FF) + " (" + grow + ")")
        if not (lm and not exact and mat == 0 and not mdl and ffm == FF and muon == 0 and ema == 0 and resume == "" and closure == "" and score == "" and genr == ""):
            raise Error("FAIL grow: needs arm bytes or toks, the bf16 path, a fresh run, and no mat, mdist, ffm, muon, ema, resume, closure, score or gen")
        ffm = gfrom
    if gmut.startswith("grow:"):
        var gm = gmut.split(":")
        gplant = Int(String(gm[1]))
        if grow == "" or gplant < gfrom or gplant >= gto:
            raise Error("FAIL gmut: grow:ROW needs grow=STEP:FROM:TO and FROM <= ROW < TO")
    var dgstep = -1  # dgrow: update at which the inactive layers join
    var dgl0 = L
    var dgplant = -1
    if dgrow != "":
        var dp = dgrow.split(":")
        if len(dp) != 2:
            raise Error("FAIL dgrow: STEP:L0 (" + dgrow + ")")
        dgstep = Int(String(dp[0]))
        dgl0 = Int(String(dp[1]))
        if dgstep < 1 or dgstep > steps or dgl0 < 1 or dgl0 > L:
            raise Error("FAIL dgrow: 1 <= STEP <= steps, 1 <= L0 <= " + String(L) + " (" + dgrow + ")")
        if not (lm and muon == 0 and ema == 0 and resume == "" and closure == "" and score == "" and genr == ""):
            raise Error("FAIL dgrow: needs arm bytes or toks, a fresh run, and no muon, ema, resume, closure, score or gen")
    if gmut.startswith("dgrow:"):
        dgplant = Int(String(gmut.split(":")[1]))
        if dgrow == "" or dgplant < dgl0 or dgplant >= L:
            raise Error("FAIL gmut: dgrow:LAYER needs dgrow=STEP:L0 and L0 <= LAYER < " + String(L))
    var xtm = 0  # expert target mask over NAMES (w1, w3, w2, wo)
    for tn in xt.split(","):
        var hit_t = False
        for q in range(4):
            if String(tn) == tname(q):
                xtm |= 1 << q
                hit_t = True
        if not hit_t and xr > 0:
            raise Error("FAIL xt: targets from w1,w3,w2,wo (" + xt + ")")
    if xr > 0:
        comptime if not exact and not (AB16 and BT and DWG and BF2 and RT and FQKV and F16):
            raise Error("FAIL experts: need exact=1 or the bf16 AB16/BT/DWG/BF2/RT/FQKV build (LAT_OPT 7864319)")
        if not (lm and mat == 0 and ffm == FF and grow == "" and dgrow == "" and muon == 0 and ema == 0 and score == "" and genr == "" and ft == 0):
            raise Error("FAIL experts: need arm bytes or toks, and no mat, ffm, grow, dgrow, muon, ema, score, gen or ft")
    if frz != 0 and (frz != 1 or xr == 0 or freeze != 0 or NBH > 0):
        raise Error("FAIL frz: frz=1 (frozen base) needs experts (xr > 0), no freeze= and no block heads")
    if xkd < 0 or xkd > 1 or (xkd > 0 and not (lm and V > 256 and mdist == 0 and mat == 0 and grow == "" and dgrow == "")):
        raise Error("FAIL xkd: a weight in (0, 1], arm bytes or toks with V > 256, no mdist, mat, grow or dgrow (" + String(xkd) + ")")
    if (xinit != "" or xtrain > 0 or xmerge != "" or xgdump != "" or xfd != "" or xbstd > 0) and xr == 0:
        raise Error("FAIL experts: xinit, xtrain, xmerge, xgdump, xfd and xbstd need xr > 0")
    if xtrain < 0 or xtrain >= xe:
        raise Error("FAIL xtrain: experts [0, xtrain) are held fixed; 0 <= xtrain < xe (" + String(xtrain) + ")")
    if xseed < 0:
        xseed = 50000 + seed
    if dseed < 0:
        dseed = 30000 + seed
    if shuf < 0 or shuf > 2:
        raise Error("FAIL shuf: 0 sorted tables, 1 shuffled everywhere, 2 sorted train/val + shuffled held-out test")
    # shape guards: attention softmax covers SMAX <= 64 columns; dwg_k needs K = rows per segment % 64 == 0
    if SMAX > 64 and not FLASH:
        raise Error("FAIL shape: LAT_S " + String(SMAX) + " > 64 needs flash attention (LAT_OPT bit 4194304)")
    comptime if DWG:
        var nrow = seq * b if lm else b
        if nrow % 64 != 0:
            raise Error("FAIL shape: " + String(nrow) + " rows per segment not a multiple of 64 (dwg_k K tiles)")
    var b1: Float32 = 0.9
    print("params arm", arm, "steps", steps, "b", b, "seed", seed, "hops", HOPS, "layers", L, "d", D, "heads", H, "ff", FF,
          "vocab", V, "lr", lr, "warmup", warmup, "decay", decay, "wd", wd, "betas", b1, b2, "clip", clipn, "np", NP,
          "stages", plan, "fix", fix, "every", every, "evaln", evaln, "target", target, "stop", stop, "aux", aux, "init", initp, "feed", feed, "rule", rule, "horizon", horizon, "floor", floor, "tail", tail, "muon", muon, "mlr", mlr, "mbeta", mbeta, "ns", nsit, "cool", cool, "zinit", zinit, "lpd", lpd, "lpdv", lpdv, "mhyb", mhyb, "leps", leps, "mat", mat, "ffm", ffm, "ecool", ecool, "ecorp", ecorp, "emix", emix, "emixall", emixall, "mdist", mdist, "grow", grow, "gseed", gseed, "dgrow", dgrow, "bh", BHK, "bhl", bhl, "bhd", bhd, "cap", CAP,
          "adam", "fresh (moments 0, step counter 1)", "gemm", "fp32-exact" if exact else "bf16-wmma", "hit", "soft" if hitsoft else "arm", "evalfrom", evalfrom, "w8", w8, "short", short, "nobot", nobot, "ema", ema, "emafrom", emafrom, "temp", temp, "ft", ft, "ftlr", ftlr, "rw", rw, "star", star, "heads", H, "shuf", shuf, "dseed", dseed, "opt", OPT)
    var tw = perf_counter_ns()
    var ctx = DeviceContext()
    if mdist == 2:
        mdist_selftest(ctx)
        return
    var irng = Rng(seed)  # init only: the data stream (dseed) does not move when the model shape changes
    var rng = Rng(dseed, shuf == 1)
    var init = List[Float32](length=NP, fill=0)
    for i in range(NP):
        init[i] = Float32(0.02 * irng.normal())
    for l in range(L):
        var o = V * D + l * LAYER
        for j in range(D):
            init[o + j] = 1
            init[o + D + 4 * DD + j] = 1
        var ps = Float32(0) if zinit != 0 else Float32(1 / sqrt(Float64(2 * L)))  # zinit: wo, w2 start at 0 (blocks start as identity)
        for j in range(DD):
            init[o + D + 3 * DD + j] *= ps
        for j in range(DF):
            init[o + 2 * D + 4 * DD + 2 * DF + j] *= ps
    for j in range(D):
        init[GF + j] = 1
    for j in range(BH0, NP):  # block heads start at W_k = 0: h_k = x, each head a copy of the main one
        init[j] = 0
    if initp != "":
        var raw = open(initp, "r").read_bytes()
        if len(raw) != NP * 4:
            raise Error("FAIL init: " + initp + " has " + String(len(raw)) + " bytes, want " + String(NP * 4))
        var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(NP):
            init[i] = src[i]
    var pb = ctx.enqueue_create_buffer[DType.float32](NP)
    var gb = ctx.enqueue_create_buffer[DType.float32](NP)
    var mb = ctx.enqueue_create_buffer[DType.float32](NP if frz == 0 else 1)  # frozen base: no Adam moments for it
    var vb = ctx.enqueue_create_buffer[DType.float32](NP if frz == 0 else 1)
    var npart = ceildiv(NP, CH)
    var ng = BH0 if bhd != 0 else NP  # bhd: the clip norm skips the detached heads (they must not scale the trunk update)
    var npg = ceildiv(ng, CH)
    var partb = ctx.enqueue_create_buffer[DType.float32](npart + (ceildiv(1 << 28, CH) if xr > 0 else 0))  # + room for the expert partials
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
    var mu = Muon(ctx, nsit if muon != 0 else 0, lpd if muon != 0 else 1, lpdv)
    mu.hyb = mhyb != 0
    mu.leps = Float32(leps)
    mu.vb = vp
    mu.b2 = b2
    if muon == 2:
        mu_selftest(ctx, m, mu)
        return
    m.itemp = Float32(1 / temp)
    m.bhl = Float32(bhl)
    if ffm % 8 != 0 or ffm < 8 or ffm > FF:
        raise Error("FAIL ffm: " + String(ffm) + " must be a multiple of 8 in [8, FF]")
    m.ffm = ffm
    m.nl = dgl0  # dgrow: the first dgl0 layers until the switch (L when off)
    var mats = [FF, 768, FF, 1408, FF, 2048]  # Matryoshka schedule: full width every other update, nested widths between
    if mdl:
        m.teacher_alloc(ctx)
    m.bhd = bhd != 0
    m.frz = frz != 0
    var npx = 0  # expert gradient partials (sumsq_k: one per thread; sumsqb_k: one per block)
    var p0b = ctx.enqueue_create_buffer[DType.float32](NP if xkd > 0 and frz == 0 else 1)  # xkd teacher = the start params
    if xr > 0:
        m.xe.setup(ctx, xr, xe, xnull, xtopk, xtm, xalpha, L, FF, m.SR, xseed, xbstd, xrt)
        if m.xe.NX >= (1 << 28):
            raise Error("FAIL experts: " + String(m.xe.NX) + " floats > the partial buffer's 2^28")
        npx = ceildiv(m.xe.NX, CH)
        if xinit != "":
            load_experts(ctx, m.xe, xinit, xtm)
        print("XPERTS", m.xe.cfg(), "s", m.xe.s, "frz", frz, "xkd", xkd, "xlr", xlr, "xtrain", xtrain, "xseed", xseed, "xinit", xinit, "routed", m.xe.routed())
    if frz != 0:
        ctx.enqueue_memset(gb, 0)  # zeroed once: the end-of-run check proves no base gradient kernel wrote into it
    if xkd > 0:
        m.teacher_alloc(ctx)
        if frz == 0:
            ctx.enqueue_copy(dst_buf=p0b, src_buf=pb)
    var p0 = F(unsafe_from_address=Int(p0b.unsafe_ptr()))
    var bf = Bufs(ctx, b, seq)
    if score != "":
        run_score(ctx, m, p, g, bf, score, sout, seq, b)
        return
    if genr != "":
        run_gen(ctx, m, p, g, bf, genr, gout, vocab, seq, b, gmax, gtools != 0)
        return
    var lrow = List[Float32](length=max(P0 + 1 + NTH + 2, seq) * b, fill=0)
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
    var S0 = len(layout(arm, 0, False, feed, short, nobot))
    var kept = 0
    var seen = 0
    var rstage = 0
    var rsince = 0
    var hit = -1
    var hit_train = 0.0
    var eval_ns = 0
    var nb = len(bf.ids)
    var hid = ctx.enqueue_create_host_buffer[DType.int32](2 * nb)
    var htg = ctx.enqueue_create_host_buffer[DType.int32](2 * nb)
    var hwt = ctx.enqueue_create_host_buffer[DType.float32](2 * nb)
    var evs = [ctx.create_event(), ctx.create_event()]
    var eoffs = List[Int](length=b, fill=0)  # ecool: window start of every sequence of the current batch
    var etop = List[Int]()  # ecool: windows replayed at the start of the next batch (hardest of the last update)
    var elrow = List[Float32](length=seq * b, fill=0)
    ctx.synchronize()
    var t_off = 0  # resume: Adam step counter and lr schedule continue from the checkpoint
    if resume != "" or closure != "":
        var rd = resume if resume != "" else closure
        var meta = Dict[String, String]()
        for line in open(rd + "/meta.txt", "r").read().split("\n"):
            var kv = String(line).split("\t")
            if len(kv) == 2:
                meta[String(kv[0])] = String(kv[1])
        if Int(meta["baro.meta.np"]) != NP:
            raise Error("FAIL resume: checkpoint np " + meta["baro.meta.np"] + " != " + String(NP))
        upload_raw(ctx, pb, rd + "/params.bin")
        var xc = meta.get("baro.x.cfg", String(""))
        if xc != (m.xe.cfg() if xr > 0 else String("")):
            raise Error("FAIL resume: checkpoint experts '" + xc + "' != this run's '" + (m.xe.cfg() if xr > 0 else String("")) + "'")
        if xr > 0:  # the experts the checkpoint was saved with (closure= scores them too); moments only to resume
            m.xe.upload(ctx, read_f32(rd + "/experts.bin", m.xe.NX))
            if resume != "":
                upload_raw(ctx, m.xe.xm, rd + "/xadam_m.bin")
                upload_raw(ctx, m.xe.xv, rd + "/xadam_v.bin")
        if resume != "":
            upload_raw(ctx, mb, rd + "/adam_m.bin")
            upload_raw(ctx, vb, rd + "/adam_v.bin")
            t_off = Int(meta["baro.meta.step"])
            rng.s = UInt64(Int(meta["baro.meta.rng"]))
            # resume = the same command plus resume=DIR: the loop continues at the saved update, so warmup, decay, cool
            # and the Adam step counter see the global step, as in an uninterrupted run
            if "baro.meta.steps" in meta and Int(meta["baro.meta.steps"]) != steps:
                raise Error("FAIL resume: checkpoint was saved by a " + meta["baro.meta.steps"] + "-step run; pass the same steps (" + String(steps) + " given)")
            if t_off >= steps:
                raise Error("FAIL resume: checkpoint is at step " + String(t_off) + ", nothing left of " + String(steps))
            print("RESUME from", rd, "step", t_off, "of", steps)
        if closure != "":
            var want = meta["baro.meta.closure_bpb"]
            var got = eval_fixed(ctx, m, p, g, bf, rd + "/closure_ids.bin", seq, b, cb)
            print("CLOSURE bpb", got, "want", want, "PASS" if String(got) == want else "FAIL")
            if String(got) != want:
                raise Error("FAIL closure: bpb " + String(got) + " != stored " + want)
            return
    var xhb = ctx.enqueue_create_buffer[DType.float32](m.xe.NX if xtrain > 0 else 1)  # xtrain: the held experts as loaded
    if xtrain > 0:
        ctx.enqueue_copy(dst_buf=xhb, src_buf=m.xe.xb)
    var xh = F(unsafe_from_address=Int(xhb.unsafe_ptr()))
    var t0 = perf_counter_ns()
    var done = 0
    var prev_stage = 0
    var tw0 = t0
    var wait_ns = 0  # host time blocked on the GPU (slot event) after warm: ~0 means the host, not the GPU, sets the pace
    var ev0 = 0
    if (save > 0 or stopat > 0) and ckpt == "":
        raise Error("FAIL save: save= and stopat= need ckpt=PATH")
    if (save > 0 or stopat > 0 or resume != "") and (ema > 0 or (muon != 0 and lpd > 1)):
        raise Error("FAIL resume: ema and Muon's cached polar factor (lpd > 1) are not in the checkpoint, so the run cannot continue exactly")
    for it in range(t_off, steps):
        if it - t_off == warm and warm > 0:  # steady-state clock: restart after the warm-up updates (JIT, first touch)
            ctx.synchronize()
            tw0 = perf_counter_ns()
            ev0 = eval_ns
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
        var kinds = layout(arm, stage, False, feed, short, nobot) if not lm else List[Int](length=seq, fill=TOK)
        var S = len(kinds)
        var dstep = mdl and it % 2 == 1  # odd updates are the small widths of mats (full width at it % 6 = 0, 2, 4)
        if lm:
            if dstep:
                pass  # mdist: the small width trains on the batch the full-width update before it just saw (the data stream stays)
            elif ecp:
                var incool = cool > 0 and Float64(it) >= Float64(steps) * (1 - cool)
                var nmix = Int(emix * Float64(b - len(etop)) + 0.5) if incool else 0
                if emixall == 1:  # every update: floor(G b (it + 1)) - floor(G b it) sequences, G of all sequences over any span
                    nmix = Int(emix * Float64(b) * Float64(it + 1)) - Int(emix * Float64(b) * Float64(it))
                fill_win(rng, cb, cb2, 0, nc - nc // 100, S, b, bf, eoffs, etop, nmix)
                if len(etop) > 0:
                    print("ECOOL it", it, "replay", len(etop))
                    print("ECOOL use it", it, "offs", offs_str(eoffs, len(etop)))
            else:
                _ = fill_bytes(rng, cb, 0, nc - nc // 100, S, b, b, bf)
            comptime if V > 256 and not exact:
                m.set_sort(ctx, bf.ids, S * b)
        else:
            fill(rng, arm, kinds, b, bf.ids, bf.tgt, bf.chain)
        if star:
            kept += self_label(ctx, m, p, g, bf, gk, b)
            seen += b
        comptime if (OPT & 8192) != 0:
            # async batch upload from pinned slot it % 2: wait only for that slot's copies (two steps back),
            # so the GPU never drains at the step boundary (upload() synchronized three times per step)
            if not lm:
                bf.weigh(ctx, kinds, b, aux, w8, rw, False)
            var sl = it % 2
            var wq0 = perf_counter_ns()
            evs[sl].synchronize()
            if it >= warm:
                wait_ns += Int(perf_counter_ns() - wq0)
            var o = sl * nb
            for k in range(nb):
                hid[o + k] = bf.ids[k]
                htg[o + k] = bf.tgt[k]
                hwt[o + k] = bf.wts[k]
            ctx.enqueue_copy(dst_buf=bf.ib, src_ptr=hid.unsafe_ptr().unsafe_offset(o))
            ctx.enqueue_copy(dst_buf=bf.tb, src_ptr=htg.unsafe_ptr().unsafe_offset(o))
            ctx.enqueue_copy(dst_buf=bf.wb, src_ptr=hwt.unsafe_ptr().unsafe_offset(o))
            ctx.stream().record_event(evs[sl])
        else:
            if not lm:
                bf.weigh(ctx, kinds, b, aux, w8, rw)
            else:
                upload(ctx, bf.wb, bf.wts)
            upload(ctx, bf.ib, bf.ids)
            upload(ctx, bf.tb, bf.tgt)
        if frz == 0:
            ctx.enqueue_memset(gb, 0)
        if xr > 0:
            ctx.enqueue_memset(m.xe.xg, 0)
        if it == gstep:
            grow_switch(ctx, m, p, g, mp, vp, bf, kinds, S, gfrom, gto, Float32(gseed), gplant, it)
            ffm = gto
        if it == dgstep:
            dgrow_switch(ctx, m, p, g, mp, vp, bf, kinds, S, dgl0, dgplant, it)
        if mat != 0:
            m.ffm = min(mats[it % len(mats)], FF)
        if mdl:
            var hh = ids_hash(bf.ids, S * b)
            if dstep:
                m.dist = Float32(mdist)
                print("MDIST use it", it, "ffm", m.ffm, "lam", mdist, "ids", hh)
            else:
                print("MDIST fresh it", it, "ffm", m.ffm, "ids", hh)
        if xkd > 0:  # teacher = the frozen start model on this batch (experts off; frz: the same p): its logits -> zt
            var xon = m.xe.on
            m.xe.on = False
            m.step(ctx, p if frz != 0 else p0, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, False)
            m.save_teacher(ctx, S * b)
            m.xe.on = xon
            if frz == 0:
                m.w16src = 0  # the teacher pass cast the start params into w16
            m.dist = Float32(xkd)
        m.step(ctx, p, g, bf.idp(), bf.tgp(), bf.wp(), kinds, S, True)
        m.dist = 0
        if frz != 0 and it == t_off:  # the base never changes: its bf16 copy and transposes stay as this step left them
            m.w16src = Int(p)
            m.t16p = Int(p)
        if zdump != "" and it == t_off:
            dump_rows(ctx, m.z, S * b * V, zdump)
        if mdl and not dstep and it + 1 < steps:  # the next update distils from this full-width update's logits
            m.save_teacher(ctx, S * b)
        m.ffm = ffm  # evals and the next update's default: the fixed width
        if ecool > 0 and (not mdl or it % 2 == 0) and it + 1 < steps and Float64(it + 1) >= Float64(steps) * (1 - cool):  # the next fresh batch is in the cooldown (with mdist: only full-width updates pick the replay; small ones reuse the batch and carry the distilled loss)
            ctx.enqueue_copy(dst_ptr=elrow.unsafe_ptr(), src_buf=DeviceBuffer[DType.float32](ctx, m.lrow, S * b, owning=False))
            ctx.synchronize()
            etop = hardest(elrow, eoffs, S, b, Int(ecool * Float64(b) + 0.5))
            print("ECOOL top it", it, "offs", offs_str(etop, len(etop)))
        if freeze > 0:
            ctx.enqueue_function[zero_k](g, Int32(freeze), grid_dim=grid(freeze), block_dim=THREADS)
        if xtrain > 0:  # experts [0, xtrain) held fixed (the router still trains)
            for l in range(L):
                for q in range(4):
                    if m.xe.has(q):
                        var na = xtrain * m.xe.r * m.xe.tin(q)
                        var nb = xtrain * m.xe.r * m.xe.tout(q)
                        ctx.enqueue_function[zero_k](m.xe.gp().unsafe_offset(m.xe.aofs(l, q)), Int32(na), grid_dim=grid(na), block_dim=THREADS)
                        ctx.enqueue_function[zero_k](m.xe.gp().unsafe_offset(m.xe.bofs(l, q)), Int32(nb), grid_dim=grid(nb), block_dim=THREADS)
        if it == (1 if mdl else 0) and gdump != "":  # mdist: the first small-width update, distilled
            dump_grad(ctx, gb, gdump, gmut)
        if it == 0 and xgdump != "":
            dump_grad(ctx, m.xe.xg, xgdump, gmut)
        if it == 0 and xgref != "":  # this build's raw expert gradient vs a reference dump (exact=1), per block
            var gr = read_f32(xgref, m.xe.NX)
            var gh = m.xe.download(ctx, m.xe.xg)
            if gmut == "x2":
                for i in range(len(gh)):
                    gh[i] *= 2
            var worst = Float64(0)
            var bad = 0
            for j in range(m.xe.nblocks()):
                var bk = m.xe.block(j)
                var e = Float64(0)
                var rr = Float64(0)
                for i in range(bk[0], bk[0] + bk[1]):
                    var df = Float64(gh[i]) - Float64(gr[i])
                    if df != df:
                        bad += 1
                    e += df * df
                    rr += Float64(gr[i]) * Float64(gr[i])
                var q = sqrt(e / rr) if rr > 0 else sqrt(e)
                if q > worst or q != q:
                    worst = q if q == q else 1e30
                print("XGREF", bk[2], "rel", q)
            print("XGREF worst", worst, "nonfinite", bad, "blocks", m.xe.nblocks())
        if it == 0 and xfd != "":
            m.dist = Float32(xkd)  # the same loss the backward differentiated (KL against the same teacher rows)
            expert_fd(ctx, m, p, g, bf, kinds, S, b, xfd, gmut)
            return
        comptime if (OPT & 1024) != 0:
            var nb0 = 0
            if frz == 0:
                ctx.enqueue_function[sumsqb_k](g, part, Int32(ng), grid_dim=grid(npg), block_dim=256)
                nb0 = ceildiv(npg, 256)
            if xr > 0:
                ctx.enqueue_function[sumsqb_k](m.xe.gp(), part.unsafe_offset(nb0), Int32(m.xe.NX), grid_dim=grid(npx), block_dim=256)
            ctx.enqueue_function[clip_k](part, Int32(nb0 + (ceildiv(npx, 256) if xr > 0 else 0)), sc, clipn, grid_dim=1, block_dim=256)
        else:
            var nb0 = 0
            if frz == 0:
                ctx.enqueue_function[sumsq_k](g, part, Int32(ng), grid_dim=grid(npg), block_dim=THREADS)
                nb0 = npg
            if xr > 0:
                ctx.enqueue_function[sumsq_k](m.xe.gp(), part.unsafe_offset(nb0), Int32(m.xe.NX), grid_dim=grid(npx), block_dim=THREADS)
            ctx.enqueue_function[clip_k](part, Int32(nb0 + npx), sc, clipn, grid_dim=1, block_dim=256)
        var t = Float64(it + 1)
        var f = Float64(1)
        if warmup > 0:
            f = min(f, t / Float64(warmup))
        if decay != 0:
            f *= max(floor, 1 - Float64(it) / Float64(horizon))
        var lr_now = Float64(lr) * f
        if it >= horizon and tail > 0:
            lr_now = tail
        if cool > 0 and Float64(it) >= Float64(steps) * (1 - cool):
            lr_now *= max(0.0, Float64(steps - it) / (Float64(steps) * cool))
        comptime if (OPT & 512) != 0:
            var w16 = H16(unsafe_from_address=Int(m.w16.unsafe_ptr()))
            var c1 = Float32(1 / (1 - Float64(b1) ** t))
            var c2 = Float32(1 / (1 - Float64(b2) ** t))
            if muon != 0:  # AdamW on embedding, gains and the tail; Muon on the matrices
                var rs = List[Int]()
                var rn = List[Int]()
                rs.append(0); rn.append(V * D)
                for l in range(L):
                    var o = V * D + l * LAYER
                    rs.append(o); rn.append(D)
                    rs.append(o + D + 4 * DD); rn.append(D)
                    if mhyb != 0:  # wo and w2 on AdamW
                        rs.append(o + D + 3 * DD); rn.append(DD)
                        rs.append(o + 2 * D + 4 * DD + 2 * DF); rn.append(DF)
                rs.append(V * D + L * LAYER); rn.append(NP - V * D - L * LAYER)
                for j in range(len(rs)):
                    var o = rs[j]
                    ctx.enqueue_function[adamw16_k](p.unsafe_offset(o), g.unsafe_offset(o), mp.unsafe_offset(o), vp.unsafe_offset(o), sc, w16.unsafe_offset(o),
                        Int32(rn[j]), Float32(lr_now), b1, b2, Float32(1e-8), wd, c1, c2, grid_dim=grid(rn[j]), block_dim=THREADS)
                var mr = (mlr if mlr > 0 else Float64(lr)) * lr_now / Float64(lr)
                mu.c2 = c2
                mu.step(ctx, m, p, g, mp, sc, w16, Float32(mr), wd, Float32(mbeta), it)
            elif frz == 0:
                ctx.enqueue_function[adamw16_k](p, g, mp, vp, sc, w16, Int32(NP), Float32(lr_now), b1, b2,
                    Float32(1e-8), wd, c1, c2, grid_dim=grid(NP), block_dim=THREADS)
            m.w16src = Int(p)
        else:
            if frz == 0:
                ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NP), Float32(lr_now), b1, b2, Float32(1e-8), wd,
                    Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
        if xr > 0:  # the experts: AdamW (+ their bf16 copy), lr xlr on the same schedule
            var xl = lr_now * (xlr / Float64(lr)) if xlr > 0 and lr > 0 else lr_now
            ctx.enqueue_function[adamw16_k](m.xe.p(), m.xe.gp(), F(unsafe_from_address=Int(m.xe.xm.unsafe_ptr())), F(unsafe_from_address=Int(m.xe.xv.unsafe_ptr())), sc,
                H16(unsafe_from_address=Int(m.xe.x16.unsafe_ptr())), Int32(m.xe.NX), Float32(xl), b1, b2, Float32(1e-8), wd,
                Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(m.xe.NX), block_dim=THREADS)
        if xtrain > 0:  # zero gradient still lets weight decay move them: put the held experts (and their bf16 copy) back
            var x16 = H16(unsafe_from_address=Int(m.xe.x16.unsafe_ptr()))
            for l in range(L):
                for q in range(4):
                    if m.xe.has(q):
                        for ab in range(2):
                            var o = m.xe.aofs(l, q) if ab == 0 else m.xe.bofs(l, q)
                            var nh = xtrain * m.xe.r * (m.xe.tin(q) if ab == 0 else m.xe.tout(q))
                            ctx.enqueue_function[copy_k](xh.unsafe_offset(o), m.xe.p().unsafe_offset(o), Int32(nh), Int32(0), grid_dim=grid(nh), block_dim=THREADS)
                            ctx.enqueue_function[cast16_k](m.xe.p().unsafe_offset(o), x16.unsafe_offset(o), Int32(nh), grid_dim=grid(nh), block_dim=THREADS)
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
            if not finite(loss) or not finite(Float64(nrm[0])):
                raise Error("FAIL nonfinite: it " + String(it) + " loss " + String(loss) + " gnorm " + String(nrm[0]))
            if star:
                print(arm, "it", it, "stage", stage, "lr", lr_now, "loss", loss, "gnorm", nrm[0], "clip", nrm[1], "boundary" if boundary else "", "kept", Float64(kept) / Float64(max(seen, 1)))
            else:
                print(arm, "it", it, "stage", stage, "lr", lr_now, "loss", loss, "gnorm", nrm[0], "clip", nrm[1], "boundary" if boundary else "")
                if xr > 0 and m.xe.routed():
                    print("ROUTE it", it, "mean/top1", m.xe.route_stats(ctx, S * b))
                if NBH > 0:  # block-head losses on the main loss scale (every BHS-th row, weight bhl)
                    var bl = m.bh_losses(ctx, S)
                    var t = String("bh")
                    for k in range(len(bl)):
                        t += " t+" + String(k + 2) + " " + String(bl[k] / Float64(bhl) * Float64(BHS))
                    print(t)
            kept = 0
            seen = 0
        if every > 0 and it + 1 >= evalfrom and ((it + 1) % every == 0 or it == steps - 1):
            ctx.synchronize()
            var e0 = perf_counter_ns()
            if lm:
                var bpb = eval_bytes(ctx, m, pv if it + 1 >= emafrom else p, g, bf, cb, nc - nc // 100, nc - nc // 200, seq, b, evaln, 10000 + seed)
                eval_ns += Int(perf_counter_ns() - e0)
                print("EVAL it", it + 1, "val_bpb", bpb, "train_sec", Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9)
                continue
            var r = evaluate(ctx, m, pv if it + 1 >= emafrom else p, g, bf, earm, ekinds, b, evaln, 10000 + seed, False, shuf == 1)
            if it + 1 >= emafrom:
                m.w16src = 0  # the eval cast pv into w16
            var line = String("EVAL it ") + String(it + 1) + " stage " + String(stage) + " " + show(r)
            if ema > 0:
                line += " | raw " + show(evaluate(ctx, m, p, g, bf, earm, ekinds, b, evaln, 10000 + seed, False, shuf == 1))
            if arm == "text":
                line += " | tf " + show(evaluate(ctx, m, p, g, bf, arm, layout(arm, 0, False, LAT, short, nobot), b, evaln, 10000 + seed, False, shuf == 1))
            if rule > 0 and stage < NTH and rsince >= 50:
                var pr = evaluate(ctx, m, p, g, bf, arm, layout(arm, stage, False, LAT, short, nobot), b, evaln, 10000 + seed, False, shuf == 1)
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
        if lm and save > 0 and (it + 1) % save == 0 and it + 1 < steps and it + 1 != stopat:
            write_self(ctx, m, p, g, bf, cb, pb, mb, vb, ckpt, corpus, vocab, seq, b, seed, dseed, it + 1, rng.s, lr, nc, steps)
        if stopat > 0 and it + 1 == stopat:
            print("STOPAT", stopat, "of", steps, ": checkpoint", ckpt)
            break
    if ft > 0:
        ctx.synchronize()
        var pre = evaluate(ctx, m, p, g, bf, earm, ekinds, b, neval, 20000 + seed, False, shuf == 1)
        print("PRE", show(pre), "tf", show(evaluate(ctx, m, p, g, bf, "latent", layout("latent", 0, False, LAT, short, nobot), b, neval, 20000 + seed, False, shuf == 1)))
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
            comptime if (OPT & 1024) != 0:
                ctx.enqueue_function[sumsqb_k](g, part, Int32(NP), grid_dim=grid(npart), block_dim=256)
                ctx.enqueue_function[clip_k](part, Int32(ceildiv(npart, 256)), sc, clipn, grid_dim=1, block_dim=256)
            else:
                ctx.enqueue_function[sumsq_k](g, part, Int32(NP), grid_dim=grid(npart), block_dim=THREADS)
                ctx.enqueue_function[clip_k](part, Int32(npart), sc, clipn, grid_dim=1, block_dim=256)
            var t = Float64(done + k + 1)
            comptime if (OPT & 512) != 0:
                ctx.enqueue_function[adamw16_k](p, g, mp, vp, sc, H16(unsafe_from_address=Int(m.w16.unsafe_ptr())), Int32(NP), Float32(ftlr), b1, b2,
                    Float32(1e-8), wd, Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
                m.w16src = Int(p)
            else:
                ctx.enqueue_function[adamw_k](p, g, mp, vp, sc, Int32(NP), Float32(ftlr), b1, b2, Float32(1e-8), wd,
                    Float32(1 / (1 - Float64(b1) ** t)), Float32(1 / (1 - Float64(b2) ** t)), grid_dim=grid(NP), block_dim=THREADS)
        ctx.synchronize()
        print("FT updates", ft, "lr", ftlr, "sec", Float64(perf_counter_ns() - e1) / 1e9, "eval_excluded_before", Float64(eval_ns) / 1e9)
        m.sg = False
    ctx.synchronize()
    var sec = Float64(Int(perf_counter_ns() - t0) - eval_ns) / 1e9
    var ran = done - t_off  # updates this process ran (a resumed run starts at t_off)
    if warm > 0 and ran > warm:
        print("STEADY ms_per_step", Float64(Int(perf_counter_ns() - tw0) - (eval_ns - ev0)) / 1e6 / Float64(ran - warm), "over", ran - warm,
              "updates after", warm, "(eval excluded)")
        print("HOSTWAIT ms_per_step", Float64(wait_ns) / 1e6 / Float64(ran - warm))
    var pf = pv if done >= emafrom else p
    if lm:
        var tb = eval_bytes(ctx, m, pf, g, bf, cb, nc - nc // 200, nc, seq, b, neval, 20000 + seed)
        print("RESULT", arm, "test_bpb", tb, "n", neval, "S", seq, "rows_per_step", seq * b, "supervised_rows", seq * b, "steps", done,
              "train_sec", sec, "ms_per_step", sec * 1000 / Float64(max(ran, 1)), "corpus_bytes", nc)
        if out != "-":
            var fin = List[Float32](length=NP, fill=0)
            ctx.enqueue_copy(dst_ptr=fin.unsafe_ptr(), src_buf=pb)
            ctx.synchronize()
            with open(out + "/params.f32", "w") as fo:
                fo.write_bytes(Span[UInt8](unsafe_ptr=fin.unsafe_ptr().unsafe_bitcast[UInt8](), length=NP * 4))
        if ckpt != "":
            write_self(ctx, m, p, g, bf, cb, pb, mb, vb, ckpt, corpus, vocab, seq, b, seed, dseed, done, rng.s, lr, nc, steps)
        if xr > 0:
            if frz != 0:  # no base gradient kernel ran: g was zeroed once before the loop and never written
                var gh = List[Float32](length=NP, fill=0)
                ctx.enqueue_copy(dst_ptr=gh.unsafe_ptr(), src_buf=gb)
                ctx.synchronize()
                var nz = 0
                for i in range(NP):
                    nz += Int(gh[i] != 0)
                print("FRZ base gradient nonzero", nz, "of", NP, "PASS" if nz == 0 else "FAIL")
                if nz != 0:
                    raise Error("FAIL frz: " + String(nz) + " base gradient entries were written")
            if xtrain > 0:
                var h0 = m.xe.download(ctx, xhb)
                var h1 = m.xe.download(ctx, m.xe.xb)
                var mx = Float64(0)
                var ch = Float64(0)
                for j in range(m.xe.nblocks()):
                    var bk = m.xe.block(j)
                    var held = bk[2].endswith(".AT") or bk[2].endswith(".B")
                    var cut = xtrain * m.xe.r * (bk[1] // m.xe.er())
                    for i in range(bk[0], bk[0] + bk[1]):
                        var d = abs(Float64(h1[i]) - Float64(h0[i]))
                        if held and i - bk[0] < cut:
                            mx = max(mx, d)
                        else:
                            ch = max(ch, d)
                print("XTRAIN held experts [0,", xtrain, ") max change", mx, "trained max change", ch, "PASS" if mx == 0 and ch > 0 else "FAIL")
            if out != "-":
                write_experts(ctx, m.xe, out + "/experts.gguf")
            if xmerge != "":
                var mg = List[Float32](length=NP, fill=0)
                ctx.enqueue_copy(dst_ptr=mg.unsafe_ptr(), src_buf=pb)
                ctx.synchronize()
                m.xe.fold(ctx, mg, V * D, LAYER, [2 * D + 4 * DD, 2 * D + 4 * DD + DF, 2 * D + 4 * DD + 2 * DF, D + 3 * DD])
                with open(xmerge, "w") as fo:
                    fo.write_bytes(Span[UInt8](unsafe_ptr=mg.unsafe_ptr().unsafe_bitcast[UInt8](), length=NP * 4))
                print("XMERGE", xmerge, "experts folded into W (router-free)")
        _ = (pb.unsafe_ptr(), gb.unsafe_ptr(), mb.unsafe_ptr(), vb.unsafe_ptr(), partb.unsafe_ptr(), scb.unsafe_ptr(), eb.unsafe_ptr(), p0b.unsafe_ptr())
        return
    var res = evaluate(ctx, m, pf, g, bf, earm, ekinds, b, neval, 20000 + seed, True, shuf == 1)
    if shuf == 2:  # sealed structural test: table row order never seen in training or validation
        print("HELDOUT shuffled-tables", show(evaluate(ctx, m, pf, g, bf, earm, ekinds, b, neval, 40000 + seed, False, True)), "n", neval)
    var wall = Float64(perf_counter_ns() - tw) / 1e9
    var extra = String("")
    if arm == "text":
        extra = " | tf " + show(evaluate(ctx, m, p, g, bf, arm, layout(arm, 0, False, LAT, short, nobot), b, neval, 20000 + seed, False, shuf == 1))
    var sup = 0
    for k in range(S0 * b):
        sup += Int(bf.wts[k] > 0)
    print("RESULT", arm, show(res), "n", neval, "S", S0, "rows_per_step", S0 * b, "supervised_rows", sup, "steps", done, "hit_step", hit, "hit_train_sec", hit_train, "train_sec", sec,
          "ms_per_step", sec * 1000 / Float64(max(ran, 1)), "wall_sec", wall, extra)
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
