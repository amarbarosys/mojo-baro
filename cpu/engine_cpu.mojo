"""CPU decode engine for the dense qwen35 pack: the m=1 launch path of serve/window.mojo, op for op, in cpu/ops.mojo.

usage: engine_cpu [PACKDIR]  (else BARO_PACK) requests on stdin, one JSON line each (serve/serve_proto.mojo), BARO_TMAX (default 16384),
       BARO_THREADS (default online CPUs / 2: on the 7800X3D 8 threads gave 7.3 tok/s, 16 gave 3.7)
Prefill replays the prompt through the decode step (v1). Buffers, KV and state are allocated once at boot; the token
loop allocates nothing (token lines go out through a stack buffer). VmRSS/VmHWM are printed at ready and at done.
"""
from std.ffi import external_call, c_ssize_t
from std.os import getenv
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from model import H, FFN, VOCAB, QF, KV, HD, NQH, NKVH, N_LAYERS, N_SSM, N_ATT, NROT, YARN_LOW, YARN_HIGH, FREQ_BASE, FREQ_SCALE, MSCALE
from bexp import fexp32 as exp
from ops import q4_row_dot, rmsnorm_cast, head_rmsnorm, rope_yarn, attn_head, swiglu, ssm_gates, ssm_conv, ssm_delta, ssm_gated
from pack import Pack
from pool import Pool
from serve_proto import read_line, parse_request, default_sample_params, is_cancel_line, idle_cancel_note, json_key, json_int
from wire import err_line, window_exceeded, window_error, finish_word, FIN_LENGTH, FIN_STOP, DoneWords, ready_words
from sample_ref import sample_row_ref

comptime NH_K = get_defined_int["BARO_SSM_NH_K", 16]()
comptime NH_V = get_defined_int["BARO_SSM_NH_V", 32]()
comptime SS = get_defined_int["BARO_SSM_STATE", 128]()
comptime CONV = 2 * NH_K * SS + NH_V * SS
comptime INNER = NH_V * SS
comptime ATT = NQH * HD
comptime EPS = Float32(1e-6)
comptime NW = 14
comptime F = MutPointer[Float32, MutUntrackedOrigin]
comptime B = MutPointer[BFloat16, MutUntrackedOrigin]
# f32 arena: x, qf, q, gate, k, v, ao, g, u, qkv, z, co, so, gb, logits
comptime OX = 0
comptime OQF = OX + H
comptime OQ = OQF + QF
comptime OGATE = OQ + ATT
comptime OK = OGATE + ATT
comptime OV = OK + KV
comptime OAO = OV + KV
comptime OG = OAO + ATT
comptime OU = OG + FFN
comptime OQKV = OU + FFN
comptime OZ = OQKV + CONV
comptime OCO = OZ + INNER
comptime OSO = OCO + CONV
comptime OGB = OSO + INNER
comptime OLOG = OGB + 2 * NH_V
comptime NF32 = OLOG + VOCAB


def fp(a: Int) -> F:
    return F(unsafe_from_address=a)


def bp(a: Int) -> B:
    return B(unsafe_from_address=a)


def is_attn(i: Int) -> Bool:
    return (i + 1) % 4 == 0


def gemv_rows(ctx: Int, lo: Int, hi: Int):
    """Rows [lo, hi) of the job at ctx (a, w, n, k, dst, add as 6 Ints): dst[r] = 0 + dot, or dst[r] += 0 + dot
    (the q4rowb partial and its NSPLIT=1 reduce)."""
    var j = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=ctx)
    var a = j[unsafe_offset=0]
    var w = j[unsafe_offset=1]
    var n = j[unsafe_offset=2]
    var k = j[unsafe_offset=3]
    var o = fp(j[unsafe_offset=4])
    for r in range(lo, hi):
        var d = Float32(0) + q4_row_dot(bp(a), MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=w + r * (k // 2)),
                                         MutPointer[Float16, MutUntrackedOrigin](unsafe_from_address=w + n * (k // 2) + r * (k // 16)), k // 32)
        o[unsafe_offset=r] = o[unsafe_offset=r] + d if j[unsafe_offset=5] == 1 else d


def say(s: String):
    _ = external_call["__write", c_ssize_t](1, s.unsafe_ptr(), s.byte_length())


def emit_tok(id: Int, tok: Int):
    var b = Array[UInt8, 64](fill=0)
    var n = 0
    for part in range(2):
        var lit = StaticString("{\"id\":") if part == 0 else StaticString(",\"tok\":")
        for c in lit.as_bytes():
            b[n] = c
            n += 1
        var v = id if part == 0 else tok
        var d = Array[UInt8, 20](fill=0)
        var nd = 0
        while True:
            d[nd] = UInt8(48 + v % 10)
            nd += 1
            v //= 10
            if v == 0:
                break
        for j in range(nd):
            b[n + j] = d[nd - 1 - j]
        n += nd
    b[n] = 125
    b[n + 1] = 10
    _ = external_call["__write", c_ssize_t](1, b.unsafe_ptr(), n + 2)


def int_array(line: String, key: String) -> List[Int]:
    """The integers of `"key":[...]`, empty when the key is absent (the GPU engine's "force" field)."""
    var out = List[Int]()
    var i = json_key(line, key)
    if i < 0 or line.as_bytes()[i] != 91:
        return out^
    i += 1
    var v = 0
    while True:
        while i < line.byte_length() and (line.as_bytes()[i] == 32 or line.as_bytes()[i] == 44):
            i += 1
        if not json_int(line, i, v):
            return out^
        out.append(v)


def rss() -> String:
    try:
        var out = String("")
        with open("/proc/self/status", "r") as f:
            for l in f.read().split("\n"):
                if l.startswith("VmRSS") or l.startswith("VmHWM") or l.startswith("RssAnon"):
                    out += ",\"" + String(l.split(":")[0]) + "_kb\":" + String(String(l.split(":")[1]).strip().split(" ")[0])
        return out
    except:
        return ""


struct Engine(Movable):
    var pack: Pack
    var w: List[Int]
    var top: List[Int]
    var f: List[Float32]
    var bf: List[BFloat16]
    var kv: List[Float32]
    var cst: List[Float32]
    var sst: List[Float32]
    var tmax: Int
    var pool: Pool[gemv_rows]
    var d4: Int
    var d4l: Int
    var job: List[Int]

    def __init__(out self, dir: String, tmax: Int, threads: Int) raises:
        self.pack = Pack(dir)
        self.pool = Pool[gemv_rows](threads)
        self.job = List[Int](length=8, fill=0)
        self.d4 = 0
        self.d4l = Int(getenv("BARO_DUMP_LAYER", "-1"))
        self.tmax = tmax
        self.w = List[Int]()
        for i in range(N_LAYERS):
            var p = "blk." + String(i) + "."
            var names: List[String] = ["attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output"] if is_attn(i) else ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_conv1d", "ssm_a", "ssm_dt.bias", "ssm_norm", "ssm_out"]
            names += ["attn_norm", "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down"]
            for nm in names:
                self.w.append(Int(self.pack.ptr[DType.uint8](p + nm + ("" if nm == "ssm_a" or nm == "ssm_dt.bias" else ".weight"))))
            for _ in range(NW - len(names)):
                self.w.append(0)
        self.top = List[Int]()
        for nm in ["token_embd.weight", "output_norm.weight", "output.weight"]:
            self.top.append(Int(self.pack.ptr[DType.uint8](nm)))
        self.f = List[Float32](length=NF32, fill=0)
        self.bf = List[BFloat16](length=FFN, fill=0)
        self.kv = List[Float32](unsafe_uninit_length=2 * N_ATT * NKVH * tmax * HD)
        self.cst = List[Float32](length=N_SSM * 3 * CONV, fill=0)
        self.sst = List[Float32](length=N_SSM * NH_V * SS * SS, fill=0)

    def mv(mut self, a: Int, w: Int, n: Int, k: Int, dst: Int, add: Bool):
        self.job[0] = a
        self.job[1] = w
        self.job[2] = n
        self.job[3] = k
        self.job[4] = dst
        self.job[5] = Int(add)
        self.pool.run(Int(self.job.unsafe_ptr()), n, max(16, n // (8 * self.pool.n)))

    def lw(self, i: Int, j: Int) -> Int:
        return self.w[i * NW + j]

    def step(mut self, tok: Int, pos: Int, head: Bool, dump: Int = 0):
        """One token; dump != 0: x after each sub-block and each FFN goes to rows 2i, 2i + 1 at that address (BARO_DUMP)."""
        var fa = Int(self.f.unsafe_ptr())
        var cur = Int(self.bf.unsafe_ptr())
        var x = fp(fa + OX * 4)
        var emb = MutPointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=self.top[0] + tok * H * 2)
        for i in range(H):
            x[unsafe_offset=i] = emb[unsafe_offset=i].cast[DType.float32]()
        var a = 0
        var s = 0
        for i in range(N_LAYERS):
            var o = 6 if is_attn(i) else 9
            rmsnorm_cast(x, fp(self.lw(i, o)), bp(cur), H, EPS)
            if is_attn(i):
                self.attn(i, a, pos, fa, cur)
                a += 1
            else:
                self.ssm(i, s, fa, cur)
                s += 1
            if dump != 0 and self.d4 == 0:
                for j in range(H):
                    fp(dump + 2 * i * H * 4)[unsafe_offset=j] = x[unsafe_offset=j]
            rmsnorm_cast(x, fp(self.lw(i, o + 1)), bp(cur), H, EPS)
            self.mv(cur, self.lw(i, o + 2), FFN, H, fa + OG * 4, False)
            self.mv(cur, self.lw(i, o + 3), FFN, H, fa + OU * 4, False)
            for j in range(FFN):
                bp(cur)[unsafe_offset=j] = swiglu(self.f[OG + j], self.f[OU + j])
            self.mv(cur, self.lw(i, o + 4), H, FFN, fa + OX * 4, True)
            if dump != 0 and self.d4 == 0:
                for j in range(H):
                    fp(dump + (2 * i + 1) * H * 4)[unsafe_offset=j] = x[unsafe_offset=j]
        if head:
            rmsnorm_cast(x, fp(self.top[1]), bp(cur), H, EPS)
            self.mv(cur, self.top[2], VOCAB, H, fa + OLOG * 4, False)

    def attn(mut self, i: Int, a: Int, pos: Int, fa: Int, cur: Int):
        self.mv(cur, self.lw(i, 0), QF, H, fa + OQF * 4, False)
        self.mv(cur, self.lw(i, 1), KV, H, fa + OK * 4, False)
        self.mv(cur, self.lw(i, 2), KV, H, fa + OV * 4, False)
        for h in range(NQH):
            for d in range(HD):
                self.f[OQ + h * HD + d] = self.f[OQF + h * 2 * HD + d]
                self.f[OGATE + h * HD + d] = self.f[OQF + h * 2 * HD + HD + d]
        for h in range(NQH):
            head_rmsnorm(fp(fa + (OQ + h * HD) * 4), HD, fp(self.lw(i, 3)), EPS)
            rope_yarn(fp(fa + (OQ + h * HD) * 4), pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
        var kc = Int(self.kv.unsafe_ptr()) + a * NKVH * self.tmax * HD * 4
        var vc = kc + N_ATT * NKVH * self.tmax * HD * 4
        for h in range(NKVH):
            head_rmsnorm(fp(fa + (OK + h * HD) * 4), HD, fp(self.lw(i, 4)), EPS)
            rope_yarn(fp(fa + (OK + h * HD) * 4), pos, NROT, FREQ_BASE, FREQ_SCALE, MSCALE, YARN_LOW, YARN_HIGH)
            var row = (h * self.tmax + pos) * HD * 4
            for d in range(HD):
                fp(kc + row)[unsafe_offset=d] = self.f[OK + h * HD + d]
                fp(vc + row)[unsafe_offset=d] = self.f[OV + h * HD + d]
        for h in range(NQH):
            var hb = (h // (NQH // NKVH)) * self.tmax * HD * 4
            attn_head(fp(fa + (OQ + h * HD) * 4), fp(kc + hb), fp(vc + hb), pos + 1, HD, self.tmax, Float32(0.0625), fp(fa + (OAO + h * HD) * 4))
        for j in range(ATT):
            bp(cur)[unsafe_offset=j] = (self.f[OAO + j] * (Float32(1) / (1 + exp(-self.f[OGATE + j])))).cast[DType.bfloat16]()
        self.mv(cur, self.lw(i, 5), H, ATT, fa + OX * 4, True)

    def ssm(mut self, i: Int, s: Int, fa: Int, cur: Int):
        self.mv(cur, self.lw(i, 0), CONV, H, fa + OQKV * 4, False)
        self.mv(cur, self.lw(i, 1), INNER, H, fa + OZ * 4, False)
        self.mv(cur, self.lw(i, 2), NH_V, H, fa + OGB * 4, False)
        self.mv(cur, self.lw(i, 3), NH_V, H, fa + (OGB + NH_V) * 4, False)
        for h in range(NH_V):
            var g = ssm_gates(self.f[OGB + h], self.f[OGB + NH_V + h], fp(self.lw(i, 5))[unsafe_offset=h], fp(self.lw(i, 6))[unsafe_offset=h])
            self.f[OGB + h] = g[0]
            self.f[OGB + NH_V + h] = g[1]
        ssm_conv(fp(fa + OQKV * 4), fp(Int(self.cst.unsafe_ptr()) + s * 3 * CONV * 4), fp(self.lw(i, 4)), fp(fa + OCO * 4), CONV, NH_K, SS, EPS)
        ssm_delta(fp(Int(self.sst.unsafe_ptr()) + s * NH_V * SS * SS * 4), fp(fa + OCO * 4), fp(fa + OGB * 4), fp(fa + OSO * 4), NH_V, NH_K, SS)
        if self.d4 != 0 and i == self.d4l:
            for j in range(CONV):
                fp(self.d4 + (8 * H + j) * 4)[unsafe_offset=0] = self.f[OCO + j]
            for j in range(2 * NH_V):
                fp(self.d4 + (5 * H + j) * 4)[unsafe_offset=0] = self.f[OGB + j]
            for j in range(INNER):
                fp(self.d4 + (6 * H + j) * 4)[unsafe_offset=0] = self.f[OSO + j]
        ssm_gated(fp(fa + OSO * 4), fp(fa + OZ * 4), fp(self.lw(i, 7)), bp(cur), INNER, SS, EPS)
        self.mv(cur, self.lw(i, 8), H, INNER, fa + OX * 4, True)

    def margin(self, fi: Int, got: Int, want: Int) -> String:
        """BARO_MARGIN=1 debug line, the GPU engine's MARGIN format: top-2 logits, gap, the forced id's logit."""
        var i1 = 0
        var i2 = -1
        for i in range(1, VOCAB):
            if self.f[OLOG + i] > self.f[OLOG + i1]:
                i2 = i1
                i1 = i
            elif i2 < 0 or self.f[OLOG + i] > self.f[OLOG + i2]:
                i2 = i
        return ("MARGIN pos " + String(fi) + "  engine_argmax " + String(got) + "  ref " + String(want) + "  top1 " + String(i1) + " " + String(self.f[OLOG + i1])
                + "  top2 " + String(i2) + " " + String(self.f[OLOG + i2]) + "  gap " + String(self.f[OLOG + i1] - self.f[OLOG + i2]) + "  ref_logit " + String(self.f[OLOG + want]) + "\n")

    def pick(self, temperature: Float64, top_k: Int, top_p: Float64, min_p: Float64, seed: UInt64, counter: Int) -> Int:
        if temperature <= 0:
            var best = 0
            var bv = Float32(-3.4e38)
            for i in range(VOCAB):
                if self.f[OLOG + i] > bv:
                    bv = self.f[OLOG + i]
                    best = i
            return best
        var row = List[Float32](length=VOCAB, fill=0)
        for i in range(VOCAB):
            row[i] = self.f[OLOG + i]
        return sample_row_ref(row, Float32(temperature), top_k, Float32(top_p), Float32(min_p), seed, UInt64(counter), 0)[0]


def main() raises:
    var a = argv()
    var dir = String(a[1]) if len(a) == 2 else getenv("BARO_PACK", "")
    if len(a) > 2 or dir == "":
        raise Error("usage: engine_cpu PACKDIR (or BARO_PACK=PACKDIR, as baro-front starts it)")
    var tmax = Int(getenv("BARO_TMAX", "16384"))
    var threads = Int(getenv("BARO_THREADS", String(max(1, external_call["sysconf", Int](Int32(84)) // 2))))
    var e = Engine(dir, tmax, threads)
    say("{\"ready\":true,\"tmax\":" + String(tmax) + ",\"mrows\":1,\"kmax\":0,\"spec_k\":0,\"kv\":\"float32\",\"device\":\"cpu\",\"threads\":" + String(e.pool.n) + ",\"pack\":\"" + dir + "\"" + ready_words("off", 1) + rss() + "}\n")
    while True:
        var got = read_line(0)
        if not got:
            break
        var line = got.value()
        var c = is_cancel_line(line)
        if c[0]:
            say(idle_cancel_note(c[1]) + "\n")
            continue
        var id = 0
        var prompt = List[Int]()
        var n = 0
        var spec = False
        var has_spec = False
        var stop = List[List[Int]]()
        var ckpt = List[Int]()
        var sp = default_sample_params()
        var s1 = String("")
        var s2 = String("")
        var s3 = False
        var s4 = String("")
        var s5 = String("")
        var s6 = 0
        var s7 = Int64(0)
        var err = parse_request(line, id, prompt, n, spec, has_spec, stop, ckpt, sp, s1, s2, s3, s4, s5, s6, s7)
        if err == "" and len(prompt) == 0:
            err = "empty prompt"
        if err == "" and window_exceeded(len(prompt), n, tmax):
            err = window_error(len(prompt), n, tmax)
        if err != "":
            say(err_line(id, err) + "\n")
            continue
        for i in range(len(e.cst)):
            e.cst[i] = 0
        for i in range(len(e.sst)):
            e.sst[i] = 0
        var force = int_array(line, "force")
        var margin = getenv("BARO_MARGIN", "0") == "1"
        var t0 = perf_counter_ns()
        var dpath = getenv("BARO_DUMP", "")
        var dbuf = List[Float32](length=(32 * 2 * N_LAYERS * H) if dpath != "" else 0, fill=0)
        e.d4 = Int(dbuf.unsafe_ptr()) if getenv("BARO_DUMP4", "0") == "1" else 0
        for p in range(len(prompt)):
            e.step(prompt[p], p, p == len(prompt) - 1, Int(dbuf.unsafe_ptr()) if dpath != "" and p == len(prompt) - 1 else 0)
        var t1 = perf_counter_ns()
        e.d4 = 0
        var gen = List[Int](capacity=n)
        var fin = FIN_LENGTH
        var pos = len(prompt)
        while len(gen) < n:
            var tok = e.pick(sp.temperature, sp.top_k, sp.top_p, sp.min_p, sp.seed, len(gen))
            gen.append(tok)
            emit_tok(id, tok)
            if margin and len(gen) <= len(force):
                say(e.margin(len(gen) - 1, tok, force[len(gen) - 1]))
            for ref st in stop:
                var hit = len(st) > 0 and len(st) <= len(gen)
                for j in range(len(st) if hit else 0):
                    hit = hit and gen[len(gen) - len(st) + j] == st[j]
                if hit:
                    fin = FIN_STOP
            if fin == FIN_STOP or len(gen) == n:
                break
            var dk = len(gen)
            e.step(force[len(gen) - 1] if len(gen) <= len(force) else tok, pos, True, Int(dbuf.unsafe_ptr()) + dk * 2 * N_LAYERS * H * 4 if dpath != "" and dk < 32 else 0)
            pos += 1
        if len(force) > 0:
            var ps = String("")
            var agree = 0
            for i in range(len(gen)):
                ps += String(gen[i]) + " "
                agree += Int(i < len(force) and gen[i] == force[i])
            say("predicted: " + ps + "\nforced agreement: " + String(agree) + " / " + String(min(len(gen), len(force))) + "\n")
        if dpath != "":
            with open(dpath, "w") as df:
                df.write_bytes(Span[UInt8](unsafe_ptr=dbuf.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(dbuf) * 4))
        var t2 = perf_counter_ns()
        var dec_s = Float64(t2 - t1) / 1e9
        say("{\"id\":" + String(id) + ",\"done\":true,\"n\":" + String(len(gen)) + ",\"prefill_s\":" + String(Float64(t1 - t0) / 1e9)
            + ",\"decode_s\":" + String(dec_s) + ",\"tok_s\":" + String(Float64(len(gen)) / max(dec_s, 1e-9)) + ",\"cached\":0,\"prefill_rows\":0,\"finish\":\""
            + finish_word(fin) + "\"" + DoneWords().json() + rss() + "}\n")
