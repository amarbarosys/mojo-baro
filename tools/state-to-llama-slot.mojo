# Convert a mojo-baro engine state into a llama.cpp server slot file.
#
# usage: state-to-llama-slot STATE OUT_SLOT [--kv f16|f32] [--att N] [--nkvh N] [--hd N] [--ext 0|1]
#
# LatentOS P1 item 4, the forward direction of tools/llama-slot-to-state.mojo:
# our engine prefills (state_save, or POST /v1/state/export), this tool rewrites
# that state into the file llama-server restores with /slots/0?action=restore,
# and llama.cpp decodes from there. STATE is a BAROST01 (f32) or BAROST02 (int8
# pages) file, bare or behind the 256-byte LAT1 header export streams.
#   K/V   f32 page-major pool, kv_off(t, att_i, kvh) -> f16 or f32 cell rows
#   conv  ours [tap*CONV + channel]            -> llama [tap + channel*3]
#   ssm   ours [h][k*128 + v] (value fastest)  -> llama [h][k + v*128] (key fastest)
# The receiver must run one stream with a non-transposed V cache (-np 1 -fa on)
# and the K/V type named by --kv (llama.cpp's default is f16). Defaults are the
# qwen35 hybrid (8 full-attention + 24 gated-delta-net layers, M-RoPE cell
# ext); an attention-only state (conv_n = ssm_n = 0) writes no recurrent
# section, with --att/--nkvh/--hd/--ext 0 naming its geometry.
from std.math import ceildiv
from std.memory import bitcast
from std.sys import argv

from registry import N_ATT, NKVH, N_SSM, KVPAGE, HD

comptime GGSQ = 0x67677371
comptime CONVC = 8192
comptime NHV = 32
comptime ST = 128


def u32(d: List[UInt8], o: Int) -> Int:
    return Int(d[o]) | (Int(d[o + 1]) << 8) | (Int(d[o + 2]) << 16) | (Int(d[o + 3]) << 24)


def i64(d: List[UInt8], o: Int) -> Int:
    return u32(d, o) | (u32(d, o + 4) << 32)


def f32_at(d: List[UInt8], o: Int) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](UInt32(u32(d, o))))


struct Out(Movable):
    var b: List[UInt8]
    var o: Int

    def __init__(out self, n: Int):
        self.b = List[UInt8](length=n, fill=0)
        self.o = 0

    def put32(mut self, v: Int):
        for k in range(4):
            self.b[self.o + k] = UInt8((v >> (8 * k)) & 0xFF)
        self.o += 4

    def put64(mut self, v: Int):
        self.put32(v & 0xFFFFFFFF)
        self.put32(v >> 32)

    def put_elem(mut self, v: Float32, es: Int):
        if es == 4:
            self.put32(Int(bitcast[DType.uint32, 1](SIMD[DType.float32, 1](v))))
        else:
            var h = Int(bitcast[DType.uint16, 1](SIMD[DType.float16, 1](v.cast[DType.float16]())))
            self.b[self.o] = UInt8(h & 0xFF)
            self.b[self.o + 1] = UInt8(h >> 8)
            self.o += 2


def main() raises:
    var a = argv()
    if len(a) < 3:
        print("usage: state-to-llama-slot STATE OUT_SLOT [--kv f16|f32] [--att N] [--nkvh N] [--hd N] [--ext 0|1]")
        raise Error("missing arguments")
    var es = 2
    var n_att = N_ATT
    var nkvh = NKVH
    var hd = HD
    var ext = 1
    var i = 3
    while i + 1 < len(a):
        var k = String(a[i])
        var v = String(a[i + 1])
        if k == "--kv":
            if v != "f16" and v != "f32":
                raise Error("--kv takes f16 or f32, got " + v)
            es = 4 if v == "f32" else 2
        elif k == "--att":
            n_att = atol(v)
        elif k == "--nkvh":
            nkvh = atol(v)
        elif k == "--hd":
            hd = atol(v)
        elif k == "--ext":
            ext = atol(v)
        else:
            raise Error("unknown option " + k)
        i += 2
    if i != len(a):
        raise Error("option " + String(a[i]) + " needs a value")

    var d: List[UInt8]
    with open(String(a[1]), "r") as f:
        d = f.read_bytes()
    var base = 0
    if len(d) >= 256 and d[0] == 0x4C and d[1] == 0x41 and d[2] == 0x54 and d[3] == 0x31:
        base = 256
    if len(d) < base + 72:
        raise Error("state file too short")
    var magic = String("")
    for k in range(8):
        magic += chr(Int(d[base + k]))
    if magic != "BAROST01" and magic != "BAROST02":
        raise Error("not a BAROST01/BAROST02 state (bare or LAT1-wrapped)")
    var int8 = magic == "BAROST02"
    var pos = i64(d, base + 8)
    var conv_n = i64(d, base + 16)
    var ssm_n = i64(d, base + 24)
    var kvn = i64(d, base + 32)
    var kvhstr = KVPAGE * hd
    var pages = ceildiv(pos, KVPAGE)
    if pos < 1 or kvn != pages * n_att * nkvh * kvhstr:
        raise Error("kv_n " + String(kvn) + " does not match pos " + String(pos) + " with att " + String(n_att) + " nkvh " + String(nkvh) + " hd " + String(hd))
    var hybrid = conv_n != 0 or ssm_n != 0
    if hybrid and (conv_n != N_SSM * 3 * CONVC or ssm_n != N_SSM * NHV * ST * ST):
        raise Error("recurrent slot sizes are not the qwen35 hybrid's: conv " + String(conv_n) + " ssm " + String(ssm_n))
    var o = base + 72
    var tokens = List[Int](capacity=pos)
    for t in range(pos):
        tokens.append(u32(d, o + 4 * t))
    o += 4 * pos
    var conv_o = o
    var ssm_o = conv_o + conv_n * 4
    var k_o = ssm_o + ssm_n * 4
    var ngroups = kvn // kvhstr
    var kv_bytes = 2 * (ngroups * 4 + kvn) if int8 else 2 * kvn * 4
    if len(d) != k_o + kv_bytes:
        raise Error("payload length mismatch: file " + String(len(d)) + " expected " + String(k_o + kv_bytes))
    var v_o = k_o + (ngroups * 4 + kvn if int8 else kvn * 4)

    var row = nkvh * hd
    var cell_meta = 12 + (12 if ext == 1 else 0)
    var total = 12 + 4 * (pos + 4) + 8 + pos * cell_meta + 8 + 2 * n_att * (12 + pos * row * es)
    if hybrid:
        total += 4 + 8 + 8 + N_SSM * (12 + 3 * CONVC * 4) + N_SSM * (12 + NHV * ST * ST * 4)
    var w = Out(total)

    # --- header and server_tokens framing: [-1, 1, n, ids..., 0] --------------
    w.put32(GGSQ)
    w.put32(3)
    w.put32(pos + 4)
    w.put32(0xFFFFFFFF)
    w.put32(1)
    w.put32(pos)
    for t in range(pos):
        w.put32(tokens[t])
    w.put32(0)

    # --- KV cache: one stream, cells are positions 0..pos-1 --------------------
    w.put32(1)
    w.put32(pos)
    for t in range(pos):
        w.put32(t)
        w.put32(1)
        if ext == 1:
            w.put32(t)
            w.put32(t)
            w.put32(tokens[t])
        w.put32(0)
    w.put32(0)
    w.put32(n_att)
    for side in range(2):
        var src = k_o if side == 0 else v_o
        for ai in range(n_att):
            w.put32(0 if es == 4 else 1)
            w.put64(row * es)
            for t in range(pos):
                for e in range(row):
                    var idx = (((t // KVPAGE) * n_att + ai) * nkvh + e // hd) * kvhstr + (t % KVPAGE) * hd + e % hd
                    var x: Float32
                    if int8:
                        var q = Int(d[src + ngroups * 4 + idx])
                        if q >= 128:
                            q -= 256
                        x = Float32(q) * f32_at(d, src + 4 * (idx // kvhstr))
                    else:
                        x = f32_at(d, src + 4 * idx)
                    w.put_elem(x, es)

    # --- recurrent state: one cell at the last position ------------------------
    if hybrid:
        w.put32(1)
        w.put32(pos - 1)
        w.put32(0)
        w.put32(0)
        w.put32(n_att + N_SSM)
        for si in range(N_SSM):
            w.put32(0)
            w.put64(3 * CONVC * 4)
            for ch in range(CONVC):
                for tap in range(3):
                    w.put32(u32(d, conv_o + (si * 3 * CONVC + tap * CONVC + ch) * 4))
        for si in range(N_SSM):
            w.put32(0)
            w.put64(NHV * ST * ST * 4)
            var sb = ssm_o + si * NHV * ST * ST * 4
            for h in range(NHV):
                for v in range(ST):
                    for k in range(ST):
                        w.put32(u32(d, sb + (h * ST * ST + k * ST + v) * 4))
    if w.o != total:
        raise Error("internal: wrote " + String(w.o) + " of " + String(total) + " bytes")
    with open(String(a[2]), "w") as f:
        f.write_bytes(Span(w.b))
    print("state-to-llama-slot:", magic, "pos", pos, "kv", "f32" if es == 4 else "f16", "hybrid", hybrid, "->", String(a[2]), total, "bytes")
