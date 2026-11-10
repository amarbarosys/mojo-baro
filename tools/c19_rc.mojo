# C19 stage 3 (AMDHQ 06 "## C19:", stage 3 amendment): exact bytes and CPU decode
# time for CacheGen's shipped codec on our KV, with our own range coder.
#
# usage: c19_rc DIR PAYLOAD_OUT [REPS]
# DIR holds tools/c19_export.py's meta.txt, q.bin, max.bin, cdf.bin, fixed.bin.
# One stream per (layer, channel, 256 tokens), LZMA-style carryless range coder,
# 16-bit CDFs from stage 1. Encodes every stream, writes the payload file
#   fixed | fp16 maxes [nl][T] | int16 cdf [nl][C][lp_l] | int32 lengths [chunk][nl][C] | streams
# then REPS times: read the payload file, decode every stream, dequantize into
# an f32 KV buffer (all cores), timed; checks every decoded symbol against q.bin.
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns

from max.algorithm import parallelize

comptime CHUNK = 256
comptime SLOT = 600
comptime TOP = UInt64(1) << 24
comptime MASK32 = UInt64(0xFFFFFFFF)
comptime WORKERS = 64


def read_file(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def cdf_at(cdfv: UnsafePointer[UInt16, _], cb: Int, s: Int, lp: Int) -> UInt64:
    # int16 CDF entries as CacheGen stores them: 65536 wraps to 0, and only
    # entry 0 is a true 0 (the table is strictly increasing); s == lp is the total.
    if s >= lp:
        return UInt64(65536)
    var v = UInt64(Int(cdfv[cb + s]))
    if s > 0 and v == 0:
        return UInt64(65536)
    return v


def put_le(mut out: List[UInt8], v: Int, n: Int):
    for k in range(n):
        out.append(UInt8((v >> (8 * k)) & 0xFF))


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: c19_rc DIR PAYLOAD_OUT [REPS]")
    var dir = String(a[1])
    var payload_path = String(a[2])
    var reps = atol(String(a[3])) if len(a) > 3 else 3

    var meta = String(unsafe_from_utf8=Span(read_file(dir + "/meta.txt"))).split("\n")
    var head = String(meta[0]).split(" ")
    var nl = atol(String(head[0]))
    var T = atol(String(head[1]))
    var C = atol(String(head[2]))
    var lpmax = atol(String(head[3]))
    var bins = List[Int]()
    var lps = List[Int]()
    for l in range(nl):
        var f = String(meta[1 + l]).split(" ")
        bins.append(atol(String(f[0])))
        lps.append(atol(String(f[1])))
    var nch = (T + CHUNK - 1) // CHUNK
    var nstreams = nch * nl * C
    print("c19_rc: layers", nl, "tokens", T, "channels", C, "chunks", nch, "streams", nstreams)

    var q = read_file(dir + "/q.bin")
    var mx = read_file(dir + "/max.bin")
    var cdfb = read_file(dir + "/cdf.bin")
    var fixed = read_file(dir + "/fixed.bin")
    if len(q) != nl * T * C or len(mx) != nl * T * 2 or len(cdfb) != nl * C * (lpmax + 1) * 4:
        raise Error("FAIL inputs: sizes do not match meta.txt")
    var qp = q.unsafe_ptr()
    var cdfp = cdfb.unsafe_ptr().unsafe_bitcast[Int32]()
    var lpp = lps.unsafe_ptr()

    # ---- encode: every stream into its own SLOT-byte slot, lengths recorded
    var slots = List[UInt8](length=nstreams * SLOT, fill=0)
    var lens = List[Int32](length=nstreams, fill=0)
    var over = List[Int32](length=WORKERS, fill=0)
    var sp = slots.unsafe_ptr()
    var lnp = lens.unsafe_ptr()
    var ovp = over.unsafe_ptr()
    var t_enc = perf_counter_ns()

    def enc_worker(w: Int) {imm qp, imm cdfp, imm lpp, imm sp, imm lnp, imm ovp, imm nl, imm T, imm C, imm lpmax, imm nch}:
        var per = (nl * C + WORKERS - 1) // WORKERS
        var lo_i = w * per
        var hi_i = min(nl * C, lo_i + per)
        if lo_i >= hi_i:
            return
        var n = hi_i - lo_i
        var low = List[UInt64](length=n, fill=0)
        var rng = List[UInt64](length=n, fill=0)
        var cache = List[UInt64](length=n, fill=0)
        var csize = List[Int](length=n, fill=0)
        var outn = List[Int](length=n, fill=0)
        for k in range(nch):
            for j in range(n):
                low[j] = 0
                rng[j] = MASK32
                cache[j] = 0
                csize[j] = 1
                outn[j] = 0
            var t0 = k * CHUNK
            var t1 = min(T, t0 + CHUNK)
            for t in range(t0, t1 + 5):
                for j in range(n):
                    var i = lo_i + j
                    var l = i // C
                    var c = i % C
                    var sid = (k * nl + l) * C + c
                    var nsh = 0
                    if t < t1:
                        var s = Int(qp[(l * T + t) * C + c])
                        var base = (l * C + c) * (lpmax + 1)
                        var start = UInt64(Int(cdfp[base + s]))
                        var size = UInt64(Int(cdfp[base + s + 1]) - Int(cdfp[base + s]))
                        var r = rng[j] >> 16
                        low[j] += start * r
                        rng[j] = r * size
                        while rng[j] < TOP:
                            rng[j] = (rng[j] << 8) & MASK32
                            nsh += 1
                    else:
                        nsh = 1
                    for _ in range(nsh):
                        var lw = low[j]
                        if (lw & MASK32) < UInt64(0xFF000000) or (lw >> 32) != 0:
                            var carry = lw >> 32
                            var temp = cache[j]
                            while True:
                                if outn[j] < SLOT:
                                    sp[sid * SLOT + outn[j]] = UInt8((temp + carry) & 0xFF)
                                else:
                                    ovp[w] = 1
                                outn[j] += 1
                                temp = 0xFF
                                csize[j] -= 1
                                if csize[j] == 0:
                                    break
                            cache[j] = (lw >> 24) & 0xFF
                        csize[j] += 1
                        low[j] = (lw & UInt64(0x00FFFFFF)) << 8
            for j in range(n):
                var i = lo_i + j
                lnp[(k * nl + i // C) * C + i % C] = Int32(outn[j])

    parallelize(enc_worker, WORKERS)
    var enc_s = Float64(perf_counter_ns() - t_enc) / 1e9
    for w in range(WORKERS):
        if over[w] != 0:
            raise Error("FAIL encode: a stream exceeded its " + String(SLOT) + "-byte slot")

    # ---- payload file
    var stream_bytes = 0
    for s in range(nstreams):
        stream_bytes += Int(lens[s])
    var pay = List[UInt8](capacity=len(fixed) + len(mx) + nl * C * lpmax * 2 + nstreams * 4 + stream_bytes)
    for b in fixed:
        pay.append(b)
    for b in mx:
        pay.append(b)
    for l in range(nl):
        for c in range(C):
            for s in range(lps[l]):
                put_le(pay, Int(cdfp[(l * C + c) * (lpmax + 1) + s]) & 0xFFFF, 2)
    for s in range(nstreams):
        put_le(pay, Int(lens[s]), 4)
    for s in range(nstreams):
        for b in range(Int(lens[s])):
            pay.append(slots[s * SLOT + b])
    with open(payload_path, "w") as f:
        f.write_bytes(Span(pay))
    var cdf_bytes = 0
    for l in range(nl):
        cdf_bytes += C * lps[l] * 2
    print("payload_bytes", len(pay), "fixed", len(fixed), "maxes", len(mx), "cdf", cdf_bytes, "lengths", nstreams * 4, "streams", stream_bytes, "encode_s", enc_s)

    # ---- decode, timed, REPS times
    var fixed_n = len(fixed)
    var times = List[Float64]()
    var sym = List[UInt8](length=nl * T * C, fill=0)
    var kv = List[Float32](length=nl * T * C, fill=0)
    var symp = sym.unsafe_ptr()
    var kvp = kv.unsafe_ptr()
    var binp = bins.unsafe_ptr()
    for rep in range(reps):
        var t0 = perf_counter_ns()
        var d = read_file(payload_path)
        var dp = d.unsafe_ptr()
        var o_max = fixed_n
        var o_cdf = o_max + nl * T * 2
        var o_len = o_cdf + cdf_bytes
        var o_str = o_len + nstreams * 4
        var offs = List[Int](length=nstreams + 1, fill=0)
        var cdfo = List[Int](length=nl + 1, fill=0)
        for l in range(nl):
            cdfo[l + 1] = cdfo[l] + C * lps[l]
        var lenp = dp.unsafe_offset(o_len).unsafe_bitcast[Int32]()
        for s in range(nstreams):
            offs[s + 1] = offs[s] + Int(lenp[s])
        if o_str + offs[nstreams] != len(d):
            raise Error("FAIL decode: payload size does not match its lengths")
        var offp = offs.unsafe_ptr()
        var cdfop = cdfo.unsafe_ptr()

        def dec_worker(w: Int) {imm dp, imm offp, imm cdfop, imm lpp, imm binp, imm symp, imm kvp, imm nl, imm T, imm C, imm nch, imm o_max, imm o_cdf, imm o_str}:
            var per = (nl * C + WORKERS - 1) // WORKERS
            var lo_i = w * per
            var hi_i = min(nl * C, lo_i + per)
            if lo_i >= hi_i:
                return
            var n = hi_i - lo_i
            var code = List[UInt64](length=n, fill=0)
            var rng = List[UInt64](length=n, fill=0)
            var pos = List[Int](length=n, fill=0)
            var end = List[Int](length=n, fill=0)
            var cdfv = dp.unsafe_offset(o_cdf).unsafe_bitcast[UInt16]()
            var maxv = dp.unsafe_offset(o_max).unsafe_bitcast[UInt16]()
            for k in range(nch):
                for j in range(n):
                    var i = lo_i + j
                    var sid = (k * nl + i // C) * C + i % C
                    pos[j] = o_str + offp[sid]
                    end[j] = o_str + offp[sid + 1]
                    rng[j] = MASK32
                    var cd = UInt64(0)
                    for _ in range(5):
                        var b = UInt64(0)
                        if pos[j] < end[j]:
                            b = UInt64(Int(dp[pos[j]]))
                        pos[j] += 1
                        cd = ((cd << 8) | b) & MASK32
                    code[j] = cd
                var t0 = k * CHUNK
                var t1 = min(T, t0 + CHUNK)
                for t in range(t0, t1):
                    for j in range(n):
                        var i = lo_i + j
                        var l = i // C
                        var c = i % C
                        var lp = lpp[l]
                        var cb = cdfop[l] + c * lp
                        var r = rng[j] >> 16
                        var v = code[j] // r
                        if v > 65535:
                            v = 65535
                        var s = lp - 1
                        while s > 0 and cdf_at(cdfv, cb, s, lp) > v:
                            s -= 1
                        var st = cdf_at(cdfv, cb, s, lp)
                        var hi = cdf_at(cdfv, cb, s + 1, lp)
                        code[j] = (code[j] - st * r) & MASK32
                        rng[j] = r * (hi - st)
                        while rng[j] < TOP:
                            var b = UInt64(0)
                            if pos[j] < end[j]:
                                b = UInt64(Int(dp[pos[j]]))
                            pos[j] += 1
                            code[j] = ((code[j] << 8) | b) & MASK32
                            rng[j] = (rng[j] << 8) & MASK32
                        var e = (l * T + t) * C + c
                        symp[e] = UInt8(s)
                        var mxl = Float32(binp[l] // 2 - 1)
                        var m16 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](maxv[l * T + t])).cast[DType.float32]()
                        kvp[e] = (Float32(s) - mxl) / mxl * m16

        parallelize(dec_worker, WORKERS)
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        times.append(dt)
        print("decode rep", rep, "decode_s", dt)

    var bad = 0
    for e in range(nl * T * C):
        if sym[e] != q[e]:
            bad += 1
    if bad != 0:
        raise Error("FAIL round trip: " + String(bad) + " decoded symbols differ from q.bin")
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[j] < times[i]:
                var tmp = times[i]
                times[i] = times[j]
                times[j] = tmp
    print("round_trip PASS symbols", nl * T * C, "streams", nstreams)
    print("decode_s_median", times[len(times) // 2], "reps", len(times))
