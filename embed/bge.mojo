# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
"""bge-small-en-v1.5 forward on the CPU: BERT 12 x 384, CLS pooling, L2 norm. Weights are read from safetensors (F32);
sequences run in parallel, GEMMs are 4x4 register tiles of SIMD dot products. Tokenizer: tools/wordpiece.mojo.

    bge <model_dir> <norm.tsv> <leads.tsv> <rows> <out.f32>
"""
from std.math import erf, exp, rsqrt
from std.sys import argv, exit, stderr
from std.time import perf_counter_ns
from std.os.path import getsize
from max.algorithm import parallelize
from wordpiece import Tok

comptime D = 384
comptime F = 1536
comptime HD = 32
comptime NH = 12
comptime NL = 12
comptime W = 16
comptime EPS = Float32(1e-12)
comptime V = SIMD[DType.float32, W]
comptime Ptr = Pointer[Float32, MutAnyOrigin]
comptime PARTS = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                  "attention.output.LayerNorm", "intermediate.dense", "output.dense", "output.LayerNorm"]


def tensor_offset(hdr: List[UInt8], name: String) raises -> Int:
    """Float offset of a tensor in the data section: the first number of its data_offsets."""
    var key = ('"' + name + '"').as_bytes()
    var i = 0
    while i + len(key) <= len(hdr):
        var hit = True
        for j in range(len(key)):
            if hdr[i + j] != key[j]:
                hit = False
                break
        if hit:
            break
        i += 1
    if i + len(key) > len(hdr):
        raise Error("tensor not found: " + name)
    while hdr[i] != 91 or hdr[i - 3] != 115:  # the '[' after "data_offsets" ("shape" ends in e)
        i += 1
    var n = 0
    i += 1
    while hdr[i] != 44:
        n = n * 10 + Int(hdr[i]) - 48
        i += 1
    return n // 4


def linear(x: Ptr, w: Ptr, b: Ptr, y: Ptr, T: Int, K: Int, O: Int):
    """y[T, O] = x[T, K] w[O, K]^T + b; T, O multiples of 4, K of W."""
    for t in range(0, T, 4):
        for o in range(0, O, 4):
            var acc = Array[V, 16](fill=V(0))
            for k in range(0, K, W):
                comptime for c in range(4):
                    var wv = w.unsafe_load[width=W]((o + c) * K + k)
                    comptime for r in range(4):
                        acc[r * 4 + c] = x.unsafe_load[width=W]((t + r) * K + k).fma(wv, acc[r * 4 + c])
            comptime for r in range(4):
                comptime for c in range(4):
                    y.unsafe_store((t + r) * O + o + c, acc[r * 4 + c].reduce_add() + b[unsafe_offset=o + c])


def layer_norm(a: Ptr, r: Ptr, g: Ptr, b: Ptr, y: Ptr, T: Int):
    """y = LayerNorm(a + r) per row of D."""
    for t in range(T):
        var s = V(0)
        for i in range(t * D, (t + 1) * D, W):
            var v = a.unsafe_load[width=W](i) + r.unsafe_load[width=W](i)
            y.unsafe_store(i, v)
            s += v
        var mean = s.reduce_add() / Float32(D)
        var q = V(0)
        for i in range(t * D, (t + 1) * D, W):
            var d = y.unsafe_load[width=W](i) - mean
            q += d * d
        var inv = rsqrt(q.reduce_add() / Float32(D) + EPS)
        for i in range(t * D, (t + 1) * D, W):
            var v = (y.unsafe_load[width=W](i) - mean) * inv
            y.unsafe_store(i, v * g.unsafe_load[width=W](i - t * D) + b.unsafe_load[width=W](i - t * D))


def attention(q: Ptr, k: Ptr, v: Ptr, ctx: Ptr, s: Ptr, T: Int):
    """ctx = softmax(q k^T / sqrt(HD)) v per head; every token is real, so there is no mask."""
    for h in range(NH):
        for i in range(T):
            var top = Float32(-1e30)
            for j in range(T):
                var d = q.unsafe_load[width=W](i * D + h * HD) * k.unsafe_load[width=W](j * D + h * HD)
                d += q.unsafe_load[width=W](i * D + h * HD + W) * k.unsafe_load[width=W](j * D + h * HD + W)
                var e = d.reduce_add() * 0.17677669529663687
                s[unsafe_offset=j] = e
                top = max(top, e)
            var sum = Float32(0)
            for j in range(T):
                s[unsafe_offset=j] = exp(s[unsafe_offset=j] - top)
                sum += s[unsafe_offset=j]
            var c0 = V(0)
            var c1 = V(0)
            for j in range(T):
                c0 += v.unsafe_load[width=W](j * D + h * HD) * s[unsafe_offset=j]
                c1 += v.unsafe_load[width=W](j * D + h * HD + W) * s[unsafe_offset=j]
            ctx.unsafe_store(i * D + h * HD, c0 / sum)
            ctx.unsafe_store(i * D + h * HD + W, c1 / sum)


def embed(w: List[Ptr], ids: List[Int], dst: Ptr, nl: Int = NL, types: List[Int] = List[Int](), unit: Bool = True):
    """CLS vector of one token sequence (unit length unless unit=False); w holds the tensors (embeddings, then 16 per
    layer), nl layers; types are the token-type ids (all 0 when empty)."""
    var T = len(ids)
    var Tp = (T + 3) // 4 * 4
    var buf = List[Float32](length=Tp * (6 * D + F) + T, fill=0)
    var bp = Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))
    var xp = bp
    var q = bp.unsafe_offset(Tp * D)
    var k = bp.unsafe_offset(2 * Tp * D)
    var v = bp.unsafe_offset(3 * Tp * D)
    var ctx = bp.unsafe_offset(4 * Tp * D)
    var tmp = bp.unsafe_offset(5 * Tp * D)
    var h = bp.unsafe_offset(6 * Tp * D)
    var s = bp.unsafe_offset(Tp * (6 * D + F))
    for t in range(T):
        for i in range(0, D, W):
            var ty = types[t] if len(types) > 0 else 0
            xp.unsafe_store(t * D + i, w[0].unsafe_load[width=W](ids[t] * D + i) + w[2].unsafe_load[width=W](ty * D + i))
    layer_norm(xp, w[1], w[3], w[4], xp, T)
    for l in range(nl):
        var p = 5 + l * 16
        linear(xp, w[p], w[p + 1], q, Tp, D, D)
        linear(xp, w[p + 2], w[p + 3], k, Tp, D, D)
        linear(xp, w[p + 4], w[p + 5], v, Tp, D, D)
        attention(q, k, v, ctx, s, T)
        linear(ctx, w[p + 6], w[p + 7], tmp, Tp, D, D)
        layer_norm(tmp, xp, w[p + 8], w[p + 9], xp, T)
        linear(xp, w[p + 10], w[p + 11], h, Tp, D, F)
        for i in range(0, T * F, W):
            var u = h.unsafe_load[width=W](i)
            var g = u * 0.5 * (1 + erf(u * 0.7071067811865476))  # GELU
            h.unsafe_store(i, g)
        linear(h, w[p + 12], w[p + 13], tmp, Tp, F, D)
        layer_norm(tmp, xp, w[p + 14], w[p + 15], xp, T)
    var n = V(0)
    for i in range(0, D, W):
        n += xp.unsafe_load[width=W](i) * xp.unsafe_load[width=W](i)
    var inv = rsqrt(n.reduce_add()) if unit else Float32(1)
    for i in range(0, D, W):
        dst.unsafe_store(i, xp.unsafe_load[width=W](i) * inv)
    _ = len(buf)


struct Bge(Movable):
    """bge-small loaded once: safetensors bytes (kept alive), tensor pointers into them, tokenizer."""
    var raw: List[UInt8]
    var w: List[Int]  # tensor addresses into raw (a struct field cannot hold an AnyOrigin pointer)
    var tok: Tok
    var hdr: List[UInt8]
    var nl: Int

    def __init__(out self, dir: String, norm: String, prefix: String = "", nl: Int = NL) raises:
        """prefix: tensor name prefix ("bert." for a BertForSequenceClassification checkpoint); nl: encoder layers."""
        var path = dir + "/model.safetensors"
        self.raw = open(path, "r").read_bytes(getsize(path))  # sized read: 25 ms for 90 MB, unsized 135 ms
        var hlen = Int(self.raw.unsafe_ptr().unsafe_bitcast[UInt64]()[unsafe_offset=0])
        self.hdr = List[UInt8](capacity=hlen)
        for i in range(hlen):
            self.hdr.append(self.raw[8 + i])
        self.nl = nl
        # keep only the data section, in its own (aligned) buffer: a header length that is not a multiple of 4
        # leaves the floats misaligned in the file (ms-marco MiniLM: 12,090-byte header)
        self.raw = List[UInt8](Span(self.raw)[8 + hlen :])
        var base = Ptr(unsafe_from_address=Int(self.raw.unsafe_ptr())).unsafe_bitcast[Float32]()
        self.w = List[Int]()
        for n in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
            self.w.append(Int(base.unsafe_offset(tensor_offset(self.hdr, prefix + "embeddings." + String(n) + ".weight"))))
        for n in ["weight", "bias"]:
            self.w.append(Int(base.unsafe_offset(tensor_offset(self.hdr, prefix + "embeddings.LayerNorm." + String(n)))))
        for l in range(nl):
            for part in materialize[PARTS]():
                for n in ["weight", "bias"]:
                    self.w.append(Int(base.unsafe_offset(tensor_offset(self.hdr, prefix + "encoder.layer." + String(l) + "." + String(part) + "." + String(n)))))
        self.tok = Tok(dir + "/vocab.txt", norm)

    def embed(self, texts: List[String]) raises -> List[Float32]:
        """Unit CLS vectors, len(texts) x D, rows in parallel."""
        var ids = List[List[Int]]()
        for t in texts:
            ids.append(self.tok.encode(t))
        var out = List[Float32](length=len(ids) * D, fill=0)
        var op = Ptr(unsafe_from_address=Int(out.unsafe_ptr()))
        var w = List[Ptr]()
        for a in self.w:
            w.append(Ptr(unsafe_from_address=a))

        var nl = self.nl

        def work(i: Int) {imm w, imm ids, imm op, imm nl}:
            embed(w, ids[i], op.unsafe_offset(i * D), nl)

        parallelize(work, len(ids))
        return out^

    def tensor(self, name: String) raises -> Int:
        """Address of a named F32 tensor in the loaded checkpoint."""
        return Int(self.raw.unsafe_ptr()) + 4 * tensor_offset(self.hdr, name)


def main() raises:
    var args = argv()
    if len(args) != 6:
        print("usage: bge <model_dir> <norm.tsv> <leads.tsv> <rows> <out.f32>", file=stderr)
        exit(2)
    var t0 = perf_counter_ns()
    var m = Bge(String(args[1]), String(args[2]))
    var rows = Int(String(args[4]))
    var texts = List[String]()
    for line in String(open(String(args[3]), "r").read()).split("\n"):
        if len(texts) == rows:
            break
        var cols = line.split("\t")
        if len(cols) >= 3:
            texts.append(String(cols[1]) + "\n" + String(cols[2]))
    var t1 = perf_counter_ns()
    var out = m.embed(texts)
    var secs = Float64(perf_counter_ns() - t1) / 1e9
    with open(String(args[5]), "w") as f:
        f.write_bytes(Span(unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(out) * 4))
    print("rows", len(texts), "load+tokenize s", Float64(t1 - t0) / 1e9, "forward s", secs, "rows/s", Float64(len(texts)) / secs, file=stderr)
