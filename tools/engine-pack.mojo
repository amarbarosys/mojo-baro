"""Build the engine weight pack from a GGUF: one flat binary + text index + identity.json. Mojo port of
tools/engine-pack.py, which stays as the oracle (byte-identical pack.bin, index.txt and identity.json apart from
pack_mtime_ns, checked by tools/engine-pack-check.sh) and as the library its Python neighbours import.

Modes, as in the oracle (its docstring has the formats):
  MODEL OUTDIR [--q8|--q4|--q2b3|--tq1|--tq2] [--q4-draft]   qwen35 trunk in the fixed engine order (block_count minus the NextN layers, then the NextN
                                          head); without a flag 2D bf16 weights are stored transposed
  MODEL OUTDIR --kq                       as --q8, but 2D Q4_K / Q6_K weights keep the file's raw bytes (dtype q4_k / q6_k);
                                          Mojo only, the engine reads them on the launch path
  MODEL OUTDIR --dense                    llama/qwen2/granite dense transformers for serve/spark.mojo
  MODEL OUTDIR --arch qwen35moe           every tensor in lexical order, experts as the GGUF holds them
  --identity OUTDIR                       refresh OUTDIR/identity.json (cached by pack.bin size and mtime)

Sources: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K rows dequantised as gguf-py does (tools/ggml_quant.mojo),
rounded to bf16 where the oracle rounds, then requantised with ggml's q8_0 / q4_0 rules in f32. Rows stream from
an mmap of the GGUF; nothing larger than one tensor's scales is held.

Build: ./.venv/bin/mojo build --fp-mode contract=off tools/engine-pack.mojo -I serve -I tools -o .work/engine-pack
"""
from std.sys import argv, exit, stderr
from std.os import makedirs, stat
from std.os.path import exists
from std.subprocess import run
from std.memory import bitcast
from std.math import floor, trunc
from max.algorithm import parallelize
from std.sys.info import num_physical_cores
from shard_index import ShardIndex, TensorRef
from ggml_quant import T_F32, T_F16, T_BF16, T_Q8_0, T_Q4_K, T_Q6_K, row_f32, to_bf16, from_bf16, type_name
from ggml_quant import release as ggml_release

comptime FLUSH = 1 << 24
comptime RAW_CHUNK = 1 << 28
comptime CHUNK_ROWS = 256


struct Out(Movable):
    var f: FileHandle
    var buf: List[UInt8]
    var n: Int

    def __init__(out self, path: String) raises:
        self.f = open(path, "w")
        self.buf = List[UInt8](capacity=FLUSH + 64)
        self.n = 0

    def flush(mut self) raises:
        if len(self.buf) > 0:
            self.f.write_bytes(Span(self.buf))
            self.buf.clear()

    def b(mut self, v: UInt8) raises:
        self.buf.append(v)
        self.n += 1
        if len(self.buf) >= FLUSH:
            self.flush()

    def u16(mut self, v: UInt16) raises:
        self.b(UInt8(v & 0xFF))
        self.b(UInt8(v >> 8))

    def f32(mut self, x: Float32) raises:
        var u = bitcast[DType.uint32](x)
        for k in range(4):
            self.b(UInt8((u >> UInt32(8 * k)) & 0xFF))

    def bytes(mut self, src: List[UInt8]) raises:
        self.flush()
        self.f.write_bytes(Span(src))
        self.n += len(src)

    def raw(mut self, a: Int, n: Int) raises:
        self.flush()
        var done = 0
        while done < n:
            var k = min(RAW_CHUNK, n - done)
            self.f.write_bytes(Span[UInt8](unsafe_ptr=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a + done), length=k))
            done += k
        self.n += n

    def close(mut self) raises:
        self.flush()


def release(t: TensorRef):
    ggml_release(t.addr, t.nbytes)


def n_elem(t: TensorRef) -> Int:
    var n = 1
    for d in t.dims:
        n *= d
    return n


def n_rows(t: TensorRef) -> Int:
    return n_elem(t) // t.dims[0]


def row_addr(t: TensorRef, r: Int) -> Int:
    return t.addr + r * (t.nbytes // n_rows(t))


def bf16_row(t: TensorRef, r: Int, mut row: List[Float32]) raises:
    """Row r rounded to bf16 (the oracle's _read_bf16 / dequantize_kquant), as f32 values."""
    row_f32(t.gtype, row_addr(t, r), t.dims[0], row)
    if t.gtype != T_BF16:
        for k in range(t.dims[0]):
            row[k] = from_bf16(to_bf16(row[k]))


def f16_bytes(d: Float32, s: Int):
    var h = bitcast[DType.uint16](Float16(d))
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=s)
    p[unsafe_offset=0] = UInt8(h & 0xFF)
    p[unsafe_offset=1] = UInt8(h >> 8)


def q8_row(row: List[Float32], ne0: Int, o: Int, scales: Int):
    """Ggml quantize_q8_0: d = amax / 127, q = sign(x / d) * floor(|x / d| + 0.5), all f32."""
    var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=o)
    for g in range(ne0 // 32):
        var amax = Float32(0)
        for l in range(32):
            amax = max(amax, abs(row[g * 32 + l]))
        var d = amax / Float32(127)
        var id = Float32(1) / d if d != 0 else Float32(0)
        for l in range(32):
            var x0 = row[g * 32 + l] * id
            var a = abs(x0)
            var m = a + Float32(0.5)
            var q = floor(m)
            if x0 < 0:
                q = -q
            elif x0 == 0:
                q = 0
            q = min(max(q, Float32(-128)), Float32(127))
            dst[unsafe_offset=g * 32 + l] = UInt8(Int(q) & 0xFF)
        f16_bytes(d, scales + g * 2)


def q4_row(row: List[Float32], ne0: Int, o: Int, scales: Int):
    """Ggml quantize_row_q4_0_ref: d = (signed value at the first largest |x|) / -8, nibble = clip(trunc(x / d + 8.5))."""
    var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=o)
    for g in range(ne0 // 32):
        var amax = Float32(-1)
        var at = 0
        for l in range(32):
            var v = abs(row[g * 32 + l])
            if v > amax:
                amax = v
                at = l
        var d = row[g * 32 + at] / Float32(-8)
        var id = Float32(1) / d if d != 0 else Float32(0)
        for j in range(16):
            var x0 = row[g * 32 + j] * id
            var x1 = row[g * 32 + 16 + j] * id
            var s0 = x0 + Float32(8.5)
            var s1 = x1 + Float32(8.5)
            var i0 = Int(min(max(trunc(s0), Float32(0)), Float32(15)))
            var i1 = Int(min(max(trunc(s1), Float32(0)), Float32(15)))
            dst[unsafe_offset=g * 16 + j] = UInt8(i0 | (i1 << 4))
        f16_bytes(d, scales + g * 2)


def ternary_code(x: Float32, inv: Float32) -> Int:
    var v = x * inv
    var magnitude = abs(v)
    var q = floor(magnitude)
    if magnitude - q >= Float32(0.5):
        q += Float32(1)
    if v < 0:
        q = -q
    return min(max(Int(q) + 1, 0), 2)


def ternary_row(row: List[Float32], ne0: Int, mode: String, mut o: Out, mut scales: List[UInt8]) raises:
    var block = 128 if mode == "q2b3" else 256
    if ne0 % block != 0:
        raise Error(mode + ": row width " + String(ne0) + " is not divisible by " + String(block))
    for start in range(0, ne0, block):
        var amax = Float32(0)
        for j in range(block):
            amax = max(amax, abs(row[start + j]))
        var inv = Float32(1) / amax if amax > 0 else Float32(0)
        if mode == "q2b3":
            var payload = List[UInt8](length=26, fill=0)
            for j in range(128):
                var c = j // 32
                var t = j % 32
                var byte = 6 * c + t // 5 if t < 30 else 24 + c // 2
                var digit = t % 5 if t < 30 else 2 * (c % 2) + (t - 30)
                var pow3 = 1
                for k in range(digit):
                    pow3 *= 3
                payload[byte] += UInt8(ternary_code(row[start + j], inv) * pow3)
            for b in range(26):
                o.b(payload[b])
        elif mode == "tq1":
            for m in range(32):
                var packed = 0
                for n in range(5):
                    packed = packed * 3 + ternary_code(row[start + m + n * 32], inv)
                o.b(UInt8((packed * 256 + 242) // 243))
            for m in range(16):
                var packed = 0
                for n in range(5):
                    packed = packed * 3 + ternary_code(row[start + 160 + m + n * 16], inv)
                o.b(UInt8((packed * 256 + 242) // 243))
            for j in range(4):
                var packed = 0
                for m in range(4):
                    packed = packed * 3 + ternary_code(row[start + 240 + j + m * 4], inv)
                packed *= 3
                o.b(UInt8((packed * 256 + 242) // 243))
        else:
            for j in range(0, 64, 32):
                for m in range(32):
                    var packed = 0
                    for n in range(4):
                        packed += ternary_code(row[start + j * 4 + m + n * 32], inv) << (2 * n)
                    o.b(UInt8(packed))
        var h = bitcast[DType.uint16](Float16(amax))
        scales.append(UInt8(h & 0xFF))
        scales.append(UInt8(h >> 8))


def emit_quant(mut o: Out, parts: List[TensorRef], q4: Bool) raises -> Int:
    """Rows of every part quantised in order on all cores, then all their block scales; returns the weight count."""
    var scales = List[UInt8]()
    var ne = 0
    for t in parts:
        var ne0 = t.dims[0]
        var rows = n_rows(t)
        var dbytes = ne0 // 32 * (16 if q4 else 32)
        var sbytes = ne0 // 32 * 2
        var data = List[UInt8](length=CHUNK_ROWS * dbytes, fill=0)
        var sc = List[UInt8](length=CHUNK_ROWS * sbytes, fill=0)
        var da = Int(data.unsafe_ptr())
        var sa = Int(sc.unsafe_ptr())
        var failed = List[Int](length=1, fill=0)
        var fa = Int(failed.unsafe_ptr())
        var tr = t.copy()
        var r0 = 0
        while r0 < rows:
            var nr = min(CHUNK_ROWS, rows - r0)

            def work(i: Int) {imm}:
                var row = List[Float32](length=ne0, fill=0)
                try:
                    bf16_row(tr, r0 + i, row)
                except:
                    Pointer[Int, MutAnyOrigin](unsafe_from_address=fa)[] = 1
                    return
                if q4:
                    q4_row(row, ne0, da + i * dbytes, sa + i * sbytes)
                else:
                    q8_row(row, ne0, da + i * dbytes, sa + i * sbytes)

            parallelize(work, nr, num_physical_cores())
            if failed[0] != 0:
                raise Error("row dequantise failed")
            o.raw(da, nr * dbytes)
            for k in range(nr * sbytes):
                scales.append(sc[k])
            r0 += nr
        _ = (data.unsafe_ptr(), sc.unsafe_ptr(), failed.unsafe_ptr())
        ne += n_elem(t)
        release(t)
    o.bytes(scales)
    return ne


def kq_name(gtype: Int) -> String:
    return "q4k" if gtype == T_Q4_K else "q6k"


def kq_dt(parts: List[TensorRef]) -> String:
    """The oracle's _read_kq verdict: "q4k" / "q6k" for one type, "A/B/<A elements>" for one A-rows-then-B-rows
    split, "" when a part is not Q4_K / Q6_K or the types change more than once."""
    var changes = 0
    var ea = 0
    for i in range(len(parts)):
        if parts[i].gtype != T_Q4_K and parts[i].gtype != T_Q6_K:
            return ""
        if i > 0 and parts[i].gtype != parts[i - 1].gtype:
            changes += 1
        if parts[i].gtype == parts[0].gtype:
            ea += n_elem(parts[i])
    if changes > 1:
        return ""
    if changes == 0:
        return kq_name(parts[0].gtype)
    return kq_name(parts[0].gtype) + "/" + kq_name(parts[len(parts) - 1].gtype) + "/" + String(ea)


def emit_mat(mut o: Out, name: String, parts: List[TensorRef], mut lines: String) raises:
    """A dense matrix: Q4_K / Q6_K parts raw (the engine's native K-quant GEMV), anything else bf16-rounded q8."""
    var off = o.n
    var dt = kq_dt(parts)
    var ne = 0
    if dt == "":
        ne = emit_quant(o, parts, False)
        dt = "q8"
    else:
        for t in parts:
            o.raw(t.addr, t.nbytes)
            ne += n_elem(t)
            release(t)
    lines += line(name, dt, off, ne)


def emit_ternary(mut o: Out, parts: List[TensorRef], mode: String) raises -> Int:
    var scales = List[UInt8]()
    var ne = 0
    for t in parts:
        var ne0 = t.dims[0]
        var row = List[Float32](length=ne0, fill=0)
        for r in range(n_rows(t)):
            bf16_row(t, r, row)
            ternary_row(row, ne0, mode, o, scales)
        ne += n_elem(t)
        release(t)
    o.bytes(scales)
    return ne


def emit_bf16(mut o: Out, t: TensorRef) raises:
    if t.gtype == T_BF16:
        o.raw(t.addr, t.nbytes)
        return
    var row = List[Float32](length=t.dims[0], fill=0)
    for r in range(n_rows(t)):
        bf16_row(t, r, row)
        for k in range(t.dims[0]):
            o.u16(to_bf16(row[k]))


def emit_bf16_t(mut o: Out, t: TensorRef) raises:
    """A 2D matrix [out, in] as bf16 transposed to [in, out]."""
    var ne0 = t.dims[0]
    var rows = n_rows(t)
    var m = List[UInt16](length=rows * ne0, fill=0)
    var row = List[Float32](length=ne0, fill=0)
    for r in range(rows):
        bf16_row(t, r, row)
        for k in range(ne0):
            m[r * ne0 + k] = to_bf16(row[k])
    for k in range(ne0):
        for r in range(rows):
            o.u16(m[r * ne0 + k])


def emit_f32(mut o: Out, parts: List[TensorRef]) raises -> Int:
    """The oracle's _read_f32: full-precision f32, no bf16 round trip."""
    var ne = 0
    for t in parts:
        ne += n_elem(t)
        if t.gtype == T_F32:
            o.raw(t.addr, t.nbytes)
            release(t)
            continue
        var row = List[Float32](length=t.dims[0], fill=0)
        for r in range(n_rows(t)):
            row_f32(t.gtype, row_addr(t, r), t.dims[0], row)
            for k in range(t.dims[0]):
                o.f32(row[k])
        release(t)
    return ne


def gib(off: Int) -> String:
    var c = Int(Float64(off) / Float64(1 << 30) * 100 + 0.5)
    return String(c // 100) + "." + ("0" if c % 100 < 10 else "") + String(c % 100)


def py_bool(b: Bool) -> String:
    return "True" if b else "False"


def layer_names(i: Int) -> List[String]:
    var core: List[String]
    if (i + 1) % 4 == 0:
        core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight",
                "attn_k_norm.weight", "attn_output.weight"]
    else:
        core = ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
                "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"]
    for n in ["post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
        core.append(n)
    var out = List[String]()
    for n in core:
        out.append("blk." + String(i) + "." + n)
    return out^


def mtp_names(n_layers: Int) -> List[String]:
    var out = List[String]()
    for n in ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight",
              "attn_k_norm.weight", "attn_output.weight", "post_attention_norm.weight", "ffn_gate.weight",
              "ffn_up.weight", "ffn_down.weight", "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
              "nextn.shared_head_norm.weight"]:
        out.append("blk." + String(n_layers) + "." + n)
    return out^


def line(name: String, dt: String, off: Int, ne: Int) -> String:
    return name + " " + dt + " " + String(off) + " " + String(ne) + "\n"


def pack_trunk(idx: ShardIndex, mut o: Out, q8: Bool, q4: Bool, q4_draft: Bool, kq: Bool = False, ternary: String = "") raises -> Tuple[String, Int]:
    # Trunk depth from the GGUF: block_count minus the MTP draft block(s).
    var n_layers = idx.gguf.akv_int("block_count", -1) - idx.gguf.akv_int("nextn_predict_layers", 0)
    if n_layers < 0:
        raise Error("no qwen35.block_count")
    var order: List[String] = ["token_embd.weight"]
    for i in range(n_layers):
        for n in layer_names(i):
            order.append(n)
    order.append("output_norm.weight")
    order.append("output.weight")
    for n in mtp_names(n_layers):
        order.append(n)
    var lines = String()
    for name in order:
        # Tied embeddings (Qwen3.5-4B and smaller): no output.weight, the head is token_embd.
        var tied = name == "output.weight" and "output.weight" not in idx.t
        var t = idx["token_embd.weight" if tied else name]
        var off = o.n
        var two_d = len(t.dims) == 2
        var embd = name == "token_embd.weight"
        var native = t.gtype == T_F32 or t.gtype == T_F16 or t.gtype == T_BF16
        var dt: String
        if kq and two_d and not embd and (t.gtype == T_Q4_K or t.gtype == T_Q6_K):
            o.raw(t.addr, t.nbytes)
            dt = "q4_k" if t.gtype == T_Q4_K else "q6_k"
        elif two_d and (not native or t.gtype == T_BF16) and (q8 or q4 or ternary != "" or embd):
            if embd:
                emit_bf16(o, t)
                dt = "bf16"
            elif ternary != "":
                _ = emit_ternary(o, [t.copy()], ternary)
                dt = ternary
            else:
                _ = emit_quant(o, [t.copy()], q4)
                dt = "q4" if q4 else "q8"
        elif native:
            dt = type_name(t.gtype)
            if t.gtype == T_BF16 and two_d and not embd:
                emit_bf16_t(o, t)
            else:
                o.raw(t.addr, t.nbytes)
        else:
            dt = "bf16"
            if two_d and not embd:
                emit_bf16_t(o, t)
            else:
                emit_bf16(o, t)
        lines += line(name, dt, off, n_elem(t))
        release(t)
    if q4_draft:
        var t = idx["output.weight"]
        if (t.gtype == T_F32 or t.gtype == T_F16):
            raise Error("--q4-draft: output.weight is " + type_name(t.gtype) + ", the oracle asserts bf16")
        var off = o.n
        var ne = emit_quant(o, [t.copy()], True)
        lines += line("output.weight.q4draft", "q4", off, ne)
    return (lines^, len(order))


def pack_dense(idx: ShardIndex, mut o: Out) raises -> String:
    var arch = idx.gguf.arch
    var n_layers = idx.gguf.akv_int("block_count", -1)
    if n_layers < 0:
        raise Error("--dense: no " + arch + ".block_count")
    var has_bias = "blk.0.attn_q.bias" in idx.t
    var has_gate = "blk.0.attn_gate.weight" in idx.t
    var tied = "output.weight" not in idx.t
    var lines = String()
    var off = o.n
    var ne = emit_f32(o, [idx["token_embd.weight"]])
    lines += line("token_embd.weight", "f32", off, ne)
    for i in range(n_layers):
        var b = "blk." + String(i) + "."
        off = o.n
        ne = emit_f32(o, [idx[b + "attn_norm.weight"]])
        lines += line(b + "attn_norm.weight", "f32", off, ne)
        if b + "attn_qkv.weight" in idx.t:
            emit_mat(o, b + "attn_qkv.weight", [idx[b + "attn_qkv.weight"]], lines)
        else:
            emit_mat(o, b + "attn_qkv.weight", [idx[b + "attn_q.weight"], idx[b + "attn_k.weight"], idx[b + "attn_v.weight"]], lines)
        if has_bias:
            off = o.n
            ne = emit_f32(o, [idx[b + "attn_q.bias"], idx[b + "attn_k.bias"], idx[b + "attn_v.bias"]])
            lines += line(b + "attn_qkv.bias", "f32", off, ne)
        var parts = List[String]()
        if has_gate:
            parts.append("attn_gate.weight")
        parts.append("attn_output.weight")
        for p in parts:
            emit_mat(o, b + p, [idx[b + p]], lines)
        off = o.n
        ne = emit_f32(o, [idx[b + "ffn_norm.weight"]])
        lines += line(b + "ffn_norm.weight", "f32", off, ne)
        for p in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            emit_mat(o, b + p, [idx[b + p]], lines)
    off = o.n
    ne = emit_f32(o, [idx["output_norm.weight"]])
    lines += line("output_norm.weight", "f32", off, ne)
    emit_mat(o, "output.weight", [idx["token_embd.weight" if tied else "output.weight"]], lines)
    if "rope_freqs.weight" in idx.t: # Llama 3 rope scaling factors, last so every earlier index stays put
        off = o.n
        ne = emit_f32(o, [idx["rope_freqs.weight"]])
        lines += line("rope_freqs.weight", "f32", off, ne)
    print("packed " + arch + ", " + String(n_layers) + " layers, bias=" + py_bool(has_bias) + ", gate="
          + py_bool(has_gate) + ", tied=" + py_bool(tied) + ", " + gib(o.n) + " GiB")
    return lines^


def moe_dense_q8(name: String) -> Bool:
    var parts = name.split(".")
    if len(parts) < 2:
        return False
    var p = String(parts[len(parts) - 2])
    return (p == "attn_q" or p == "attn_k" or p == "attn_v" or p == "attn_output" or p == "attn_qkv"
            or p == "attn_gate" or p == "ssm_out")


def pack_moe(idx: ShardIndex, mut o: Out) raises -> String:
    if idx.gguf.arch != "qwen35moe":
        raise Error("--arch qwen35moe requires qwen35moe GGUF, got " + idx.gguf.arch)
    var names = idx.gguf.tensor_names.copy()
    sort(names)
    var lines = String()
    var seen = List[String]()
    var counts = List[Int]()
    for name in names:
        var t = idx[name]
        var off = o.n
        var dt = type_name(t.gtype)
        if name == "output.weight":
            _ = emit_quant(o, [t.copy()], False)
            dt = "q8"
        elif t.gtype == T_Q8_0 and moe_dense_q8(name):
            var nb = t.nbytes // 34
            var qs = List[UInt8](capacity=nb * 32)
            var ds = List[UInt8](capacity=nb * 2)
            for k in range(nb):
                var a = t.addr + k * 34
                for l in range(32):
                    qs.append(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a + 2 + l)[])
                ds.append(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a)[])
                ds.append(Pointer[UInt8, MutAnyOrigin](unsafe_from_address=a + 1)[])
            o.bytes(qs)
            o.bytes(ds)
            dt = "q8"
        else:
            o.raw(t.addr, t.nbytes)
        lines += line(name, dt, off, n_elem(t))
        release(t)
        var hit = False
        for j in range(len(seen)):
            if seen[j] == dt:
                counts[j] += 1
                hit = True
        if not hit:
            seen.append(dt)
            counts.append(1)
    var cs = String()
    for j in range(len(seen)):
        cs += ("" if j == 0 else ", ") + "'" + seen[j] + "': " + String(counts[j])
    print("packed " + String(len(names)) + " tensors, " + gib(o.n) + " GiB, dtypes={" + cs + "}")
    return lines^


def shq(s: String) -> String:
    return "'" + s.replace("'", "'\\''") + "'"


def sha256(path: String) raises -> String:
    var r = run("sha256sum " + shq(path))
    var h = String(r.split(" ")[0])
    if h.byte_length() != 64:
        raise Error("sha256sum " + path + ": " + r)
    return h


def json_str(s: String) -> String:
    """Python json.dumps of a str (ensure_ascii)."""
    var out = String("\"")
    for cp in s.codepoints():
        var c = Int(cp.to_u32())
        if c == 0x22:
            out += "\\\""
        elif c == 0x5C:
            out += "\\\\"
        elif c == 0x0A:
            out += "\\n"
        elif c == 0x0D:
            out += "\\r"
        elif c == 0x09:
            out += "\\t"
        elif c == 0x08:
            out += "\\b"
        elif c == 0x0C:
            out += "\\f"
        elif c < 0x20 or c > 0x7E:
            if c > 0xFFFF:
                var v = c - 0x10000
                out += u_esc(0xD800 + (v >> 10)) + u_esc(0xDC00 + (v & 0x3FF))
            else:
                out += u_esc(c)
        else:
            out += String(cp)
    return out + "\""


def u_esc(c: Int) -> String:
    var hexd = "0123456789abcdef"
    var s = String("\\u")
    for k in range(4):
        s += String(hexd[byte=(c >> (12 - 4 * k)) & 0xF])
    return s


def py_path(p: String) -> String:
    """Python str(pathlib.Path(p)): empty and '.' components dropped, one slash between, a leading '/' kept."""
    var abs_ = p.startswith("/")
    var out = String()
    for c in p.split("/"):
        var s = String(c)
        if s == "" or s == ".":
            continue
        out += ("/" if out.byte_length() > 0 else "") + s
    if abs_:
        return "/" + out
    return out if out.byte_length() > 0 else "."


def json_raw(text: String, key: String) -> String:
    """The raw JSON token after `"key": ` in an indent=2 flat object, or "" when the key is absent."""
    var needle = "\"" + key + "\": "
    var i = text.find(needle)
    if i < 0:
        return ""
    var rest = String(text[byte=i + needle.byte_length() :])
    var e = rest.find("\n")
    var v = String(rest[byte=: e]) if e >= 0 else rest
    if v.endswith(","):
        var n = v.byte_length() - 1
        v = String(String(v)[byte=:n])
    return v


def write_identity(outdir: String, source: String) raises:
    """The oracle's write_identity: source is the JSON token for "source" ("" keeps the prior one)."""
    var pack_bin = outdir + "/pack.bin"
    var st = stat(pack_bin)
    var mtime_ns = Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)
    var tok_path = outdir + "/tokenizer.json"
    var id_path = outdir + "/identity.json"
    var prior = String()
    if exists(id_path):
        with open(id_path, "r") as f:
            prior = f.read()
    var tok = "\"" + sha256(tok_path) + "\"" if exists(tok_path) else json_raw(prior, "tokenizer_sha256")
    var src = source if source != "" else json_raw(prior, "source")
    var uuid = json_raw(prior, "general_uuid")
    var ph = sha256(pack_bin)
    var nul = String("null")
    var body = ("{\n  \"pack_sha256\": \"" + ph + "\",\n  \"pack_bytes\": " + String(Int(st.st_size))
                + ",\n  \"pack_mtime_ns\": " + String(mtime_ns) + ",\n  \"tokenizer_sha256\": "
                + (tok if tok != "" else nul) + ",\n  \"source\": " + (src if src != "" else nul)
                + ",\n  \"general_uuid\": " + (uuid if uuid != "" else nul) + "\n}\n")
    with open(id_path, "w") as f:
        f.write(body)
    var tok_head = String(tok[byte=1:17]) if tok != "" and tok != "null" else "None"
    print("identity: pack_sha256=" + String(ph[byte=:16]) + "... tokenizer_sha256=" + tok_head)


def ensure_identity(outdir: String) raises:
    var pack_bin = outdir + "/pack.bin"
    if not exists(pack_bin):
        raise Error("engine-pack --identity: no pack.bin in " + py_path(outdir))
    var id_path = outdir + "/identity.json"
    if exists(id_path):
        var st = stat(pack_bin)
        var mtime_ns = Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)
        var text: String
        with open(id_path, "r") as f:
            text = f.read()
        var tk = json_raw(text, "tokenizer_sha256")
        var tok_covered = (tk != "" and tk != "null") or not exists(outdir + "/tokenizer.json")
        if (json_raw(text, "pack_bytes") == String(Int(st.st_size)) and json_raw(text, "pack_mtime_ns") == String(mtime_ns)
                and tok_covered):
            var ph = json_raw(text, "pack_sha256")
            print("identity: cached (pack unchanged), pack_sha256=" + String(ph[byte=1:17]) + "...")
            return
    write_identity(outdir, "")


def main() raises:
    var args = List[String]()
    var av = argv()
    for i in range(1, len(av)):
        args.append(String(av[i]))
    var q8 = False
    var q4 = False
    var q4_draft = False
    var ternary = String()
    var kq = False
    var dense = False
    var arch = String()
    var identity = String()
    var pos = List[String]()
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--identity" and i + 1 < len(args):
            identity = args[i + 1]
            i += 1
        elif a == "--arch" and i + 1 < len(args):
            arch = args[i + 1]
            i += 1
        elif a == "--dense":
            dense = True
        elif a == "--q8":
            q8 = True
        elif a == "--q4":
            q4 = True
        elif a == "--kq":
            kq = True
            q8 = True
        elif a == "--q4-draft":
            q4_draft = True
        elif a == "--q2b3":
            ternary = "q2b3"
        elif a == "--tq1":
            ternary = "tq1"
        elif a == "--tq2":
            ternary = "tq2"
        else:
            pos.append(a)
        i += 1
    if identity != "":
        ensure_identity(identity)
        return
    if len(pos) != 2:
        print("usage: engine-pack MODEL.gguf OUTDIR [--q8|--q4|--kq|--q2b3|--tq1|--tq2] [--q4-draft] | --dense | --arch qwen35moe;"
              + " engine-pack --identity OUTDIR", file=stderr)
        exit(2)
    if arch != "" and arch != "qwen35moe":
        raise Error("unsupported architecture: " + arch)
    if (q8 and q4) or ((q8 or q4) and ternary != ""):
        raise Error("--q8, --q4 and ternary quantization flags are exclusive")
    var model = pos[0]
    var outdir = pos[1]
    makedirs(outdir, exist_ok=True)
    var idx = ShardIndex(model)
    var o = Out(outdir + "/pack.bin")
    var lines: String
    if arch != "":
        lines = pack_moe(idx, o)
    elif dense:
        lines = pack_dense(idx, o)
    else:
        var r = pack_trunk(idx, o, q8, q4, q4_draft, kq, ternary)
        lines = r[0]
        print("packed " + String(r[1]) + " tensors, " + gib(o.n) + " GiB")
    o.close()
    with open(outdir + "/index.txt", "w") as f:
        f.write(lines)
    write_identity(outdir, json_str(py_path(model)))
