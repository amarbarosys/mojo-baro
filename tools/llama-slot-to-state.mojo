# Convert a llama.cpp server slot file into a mojo-baro engine state file.
#
# usage: llama-slot-to-state SLOT_FILE PACKDIR OUT_STATE
#
# LatentOS use 1: llama.cpp prefills the prompt (/slots/0?action=save), this
# tool rewrites that state into the BAROST01 file serve/engine.mojo loads with
# BARO_STATE_LOAD, and mojo-baro decodes from there. qwen35 hybrid only
# (8 full-attention + 24 gated-delta-net layers). Byte layout and every
# transform: .work/latent-llama/spec.md (from llama.cpp llama-context.cpp,
# llama-kv-cache.cpp, llama-memory-recurrent.cpp, models/qwen35.cpp).
#   K/V   f16 or f32 cell rows -> f32 page-major pool, kv_off(t, att_i, kvh)
#   conv  llama [tap + channel*3]            -> ours [tap*CONV + channel]
#   ssm   llama [h][k + v*128] (key fastest) -> ours [h][k*128 + v] (value fastest)
# PACKDIR must be the same string the engine gets as BARO_PACK: its sha256 is
# the salt the engine checks before it trusts the file.
from std.math import ceildiv
from std.memory import bitcast
from std.sys import argv

from registry import CONV_SLOT, SSM_SLOT, N_ATT, NKVH, N_SSM, KVPAGE, KVHSTR, HD
from prefix import sha256_bytes, string_bytes

comptime GGSQ = 0x67677371
comptime CONVC = 8192
comptime NHV = 32
comptime ST = 128


def u32(d: List[UInt8], o: Int) -> Int:
    return Int(d[o]) | (Int(d[o + 1]) << 8) | (Int(d[o + 2]) << 16) | (Int(d[o + 3]) << 24)


def i32(d: List[UInt8], o: Int) -> Int:
    var v = u32(d, o)
    return v - (1 << 32) if v >= (1 << 31) else v


def u64(d: List[UInt8], o: Int) -> Int:
    return u32(d, o) | (u32(d, o + 4) << 32)


def f32_at(d: List[UInt8], o: Int) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](UInt32(u32(d, o))))


def f16_at(d: List[UInt8], o: Int) -> Float32:
    var h = UInt16(Int(d[o]) | (Int(d[o + 1]) << 8))
    return bitcast[DType.float16, 1](SIMD[DType.uint16, 1](h)).cast[DType.float32]()


def elem(d: List[UInt8], o: Int, ty: Int) raises -> Float32:
    if ty == 0:
        return f32_at(d, o)
    if ty == 1:
        return f16_at(d, o)
    raise Error("unsupported KV dtype " + String(ty) + " (need f32 or f16)")


def put_i64(mut out: List[UInt8], v: Int):
    for b in range(8):
        out.append(UInt8((v >> (8 * b)) & 0xFF))


def main() raises:
    var a = argv()
    if len(a) != 4:
        print("usage: llama-slot-to-state SLOT_FILE PACKDIR OUT_STATE")
        return
    var d: List[UInt8]
    with open(String(a[1]), "r") as f:
        d = f.read_bytes()
    if u32(d, 0) != GGSQ or u32(d, 4) != 3:
        raise Error("not a llama.cpp seq state file v3 (magic/version)")
    var ntok = u32(d, 8)
    var o = 12
    var raw = List[Int](capacity=ntok)
    for t in range(ntok):
        raw.append(i32(d, o + 4 * t))
    o += 4 * ntok
    # The server writes server_tokens::serialize(), not a flat id list: a text
    # run is framed as [-1, 1, n, ids..., 0] (measured on a real slot, 8004
    # entries for an 8000-token prompt). Take the ids; refuse other framings.
    var tokens = List[Int]()
    if ntok >= 4 and raw[0] == -1:
        var n = raw[2]
        if raw[1] != 1 or 3 + n + 1 != ntok or raw[3 + n] != 0:
            raise Error("unrecognised server_tokens framing: [" + String(raw[0]) + ", " + String(raw[1]) + ", " + String(raw[2]) + ", ...]")
        for t in range(n):
            tokens.append(raw[3 + t])
    else:
        tokens = raw^

    # --- KV cache (8 full-attention layers) ---------------------------------
    if u32(d, o) != 1:
        raise Error("expected one KV stream, got " + String(u32(d, o)))
    o += 4
    var cells = u32(d, o)
    o += 4
    var cpos = List[Int](capacity=cells)
    for _ in range(cells):
        cpos.append(i32(d, o))
        var nseq = u32(d, o + 4)
        o += 8 + 12 + 4 * nseq
    var pos = cells
    var seen = List[Bool](length=pos, fill=False)
    for c in range(cells):
        if cpos[c] < 0 or cpos[c] >= pos or seen[cpos[c]]:
            raise Error("KV cells are not positions 0..n-1")
        seen[cpos[c]] = True
    var vtrans = u32(d, o)
    var nl = u32(d, o + 4)
    o += 8
    if nl != N_ATT:
        raise Error("expected " + String(N_ATT) + " attention layers, got " + String(nl))
    var kvn = ceildiv(pos, KVPAGE) * N_ATT * NKVH * KVHSTR
    var K = List[Float32](length=kvn, fill=0)
    var V = List[Float32](length=kvn, fill=0)
    comptime ROW = NKVH * HD
    for ai in range(N_ATT):
        var ty = i32(d, o)
        var rsz = u64(d, o + 4)
        o += 12
        var es = rsz // ROW
        for c in range(cells):
            var t = cpos[c]
            for e in range(ROW):
                var kvh = e // HD
                var dd = e % HD
                var dst = (((t // KVPAGE) * N_ATT + ai) * NKVH + kvh) * KVHSTR + (t % KVPAGE) * HD + dd
                K[dst] = elem(d, o + c * rsz + e * es, ty)
        o += cells * rsz
    for ai in range(N_ATT):
        var ty = i32(d, o)
        if vtrans == 0:
            var rsz = u64(d, o + 4)
            o += 12
            var es = rsz // ROW
            for c in range(cells):
                var t = cpos[c]
                for e in range(ROW):
                    var dst = (((t // KVPAGE) * N_ATT + ai) * NKVH + e // HD) * KVHSTR + (t % KVPAGE) * HD + e % HD
                    V[dst] = elem(d, o + c * rsz + e * es, ty)
            o += cells * rsz
        else:
            var es = u32(d, o + 4)
            var nemb = u32(d, o + 8)
            o += 12
            if nemb != ROW:
                raise Error("V row width " + String(nemb))
            for e in range(ROW):
                for c in range(cells):
                    var t = cpos[c]
                    var dst = (((t // KVPAGE) * N_ATT + ai) * NKVH + e // HD) * KVHSTR + (t % KVPAGE) * HD + e % HD
                    V[dst] = elem(d, o + (e * cells + c) * es, ty)
            o += ROW * cells * es

    # --- recurrent state (24 gated-delta-net layers) -------------------------
    if u32(d, o) != 1:
        raise Error("expected one recurrent cell, got " + String(u32(d, o)))
    o += 4
    o += 8
    if u32(d, o) != 0:
        raise Error("s_trans != 0 is not handled")
    o += 8
    var conv = List[Float32](length=CONV_SLOT, fill=0)
    var ssm = List[Float32](length=SSM_SLOT, fill=0)
    for si in range(N_SSM):
        if i32(d, o) != 0 or u64(d, o + 4) != 3 * CONVC * 4:
            raise Error("conv row type/size at layer " + String(si))
        o += 12
        for ch in range(CONVC):
            for tap in range(3):
                conv[si * 3 * CONVC + tap * CONVC + ch] = f32_at(d, o + (tap + ch * 3) * 4)
        o += 3 * CONVC * 4
    for si in range(N_SSM):
        if i32(d, o) != 0 or u64(d, o + 4) != NHV * ST * ST * 4:
            raise Error("ssm row type/size at layer " + String(si))
        o += 12
        var base = si * NHV * ST * ST
        for h in range(NHV):
            for k in range(ST):
                for v in range(ST):
                    ssm[base + h * ST * ST + k * ST + v] = f32_at(d, o + (h * ST * ST + v * ST + k) * 4)
        o += NHV * ST * ST * 4
    if o != len(d):
        raise Error("trailing bytes: parsed " + String(o) + " of " + String(len(d)))

    # --- BAROST01 (serve/engine.mojo save_state / load_state) ----------------
    var head = List[UInt8]()
    var magic = String("BAROST01")
    for i in range(8):
        head.append(magic.as_bytes()[i])
    put_i64(head, pos)
    put_i64(head, CONV_SLOT)
    put_i64(head, SSM_SLOT)
    put_i64(head, kvn)
    for b in sha256_bytes(string_bytes(String(a[2]))):
        head.append(b)
    for t in range(pos):
        for b in range(4):
            head.append(UInt8((tokens[t] >> (8 * b)) & 0xFF))
    with open(String(a[3]), "w") as f:
        f.write_bytes(Span(head))
        f.write_bytes(Span[UInt8](unsafe_ptr=conv.unsafe_ptr().bitcast[UInt8](), length=CONV_SLOT * 4))
        f.write_bytes(Span[UInt8](unsafe_ptr=ssm.unsafe_ptr().bitcast[UInt8](), length=SSM_SLOT * 4))
        f.write_bytes(Span[UInt8](unsafe_ptr=K.unsafe_ptr().bitcast[UInt8](), length=kvn * 4))
        f.write_bytes(Span[UInt8](unsafe_ptr=V.unsafe_ptr().bitcast[UInt8](), length=kvn * 4))
    print("llama-slot-to-state:", ntok, "tokens,", cells, "KV cells, v_trans", vtrans, "->", String(a[3]))
