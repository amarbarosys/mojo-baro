# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
"""bge-small-en-v1.5 embed process for baro-serve (serve/src/embed.rs): loads the model once, answers a line protocol on
stdin/stdout. CPU only. Build with embed/build.sh.

    serve_embed MODEL_DIR NORM_TSV [MAX_TOKENS=512]

  startup line   ready
  request line   one batch: texts as hex of their UTF-8 bytes, separated by one space, "-" for an empty text
  reply line     ok NTOK:HEX NTOK:HEX ...   per text its token count (CLS and SEP included) and the L2-normalised
                 CLS vector as 384 little-endian float32 in hex
                 err MESSAGE                the request was refused; the process keeps running
The caller applies any query instruction prefix; this process embeds the text it is given.
"""
from std.ffi import external_call, c_ssize_t
from std.sys import argv, exit, stderr
from max.algorithm import parallelize
from bge import Bge, embed, D, Ptr

def read_line(fd: Int) raises -> Optional[String]:
    var buf = List[UInt8]()
    var b = List[UInt8](unsafe_uninit_length=1)
    while True:
        var n = external_call["__read", c_ssize_t](fd, b.unsafe_ptr(), 1)
        if n <= 0:
            if len(buf) == 0:
                return None
            break
        if b[0] == 10:
            break
        buf.append(b[0])
    return String(from_utf8=Span[UInt8](buf))


def nibble(b: UInt8) raises -> Int:
    var c = Int(b)
    if 48 <= c <= 57:
        return c - 48
    if 97 <= c <= 102:
        return c - 87
    if 65 <= c <= 70:
        return c - 55
    raise Error("bad hex digit")


def unhex(h: StringSlice) raises -> String:
    var b = h.as_bytes()
    if len(b) % 2 != 0:
        raise Error("odd hex length")
    var out = List[UInt8](capacity=len(b) // 2)
    for i in range(0, len(b), 2):
        out.append(UInt8((nibble(b[i]) << 4) | nibble(b[i + 1])))
    return String(from_utf8=Span[UInt8](out))


def main() raises:
    var a = argv()
    if len(a) < 3:
        print("usage: serve_embed MODEL_DIR NORM_TSV [MAX_TOKENS=512]", file=stderr)
        exit(2)
    var max_len = Int(String(a[3])) if len(a) > 3 else 512
    if max_len < 3 or max_len > 512:
        print("FAIL serve_embed: MAX_TOKENS must be 3..512", file=stderr)
        exit(2)
    var m = Bge(String(a[1]), String(a[2]))
    var w = List[Ptr]()
    for p in m.w:
        w.append(Ptr(unsafe_from_address=p))
    var nl = m.nl
    print("ready", flush=True)
    while True:
        var line = read_line(0)
        if not line:
            return
        try:
            var ids = List[List[Int]]()
            for f in line.value().split(" "):
                if f.byte_length() == 0:
                    continue
                ids.append(m.tok.encode("" if f == "-" else unhex(f), max_len))
            if len(ids) == 0:
                raise Error("empty batch")
            var out = List[Float32](length=len(ids) * D, fill=0)
            var op = Ptr(unsafe_from_address=Int(out.unsafe_ptr()))

            def work(i: Int) {imm w, imm ids, imm op, imm nl}:
                embed(w, ids[i], op.unsafe_offset(i * D), nl)

            parallelize(work, len(ids))
            var hex = "0123456789abcdef"
            var raw = out.unsafe_ptr().unsafe_bitcast[UInt8]()
            var reply = String("ok")
            for i in range(len(ids)):
                reply += " " + String(len(ids[i])) + ":"
                for j in range(i * D * 4, (i + 1) * D * 4):
                    reply += hex[byte = Int(raw[j] >> 4)]
                    reply += hex[byte = Int(raw[j] & 15)]
            print(reply, flush=True)
        except e:
            print("err " + String(e).replace("\n", " "), flush=True)
