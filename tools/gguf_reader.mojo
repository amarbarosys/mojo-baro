# VENDORED COPY. Upstream is ~/iTools/lib/mojo/gguf-reader.mojo; this repo keeps
# a real file rather than a symlink, because a clone must build without anything
# outside it. Sync by hand if the upstream changes; tools/ci-checks.sh compares
# the two when the upstream is present and says so when they drift.
#
# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
# Generic GGUF v3 header reader for Mojo tools that need a model's own metadata
# and tensor shapes (dims, dtype, offset) -- not just the tokenizer.* keys.
# Self-contained: no dependency on any other project's tokenizer/parser code.
# Reads the whole header + tensor-info section into memory (HEADER_MAX cap,
# same 96 MiB budget as serve/tokenizer.mojo's Reader in mojo-baro) and records
# each tensor's type and data offset (`tensor_type`, `tensor_off`, `data_start`);
# never touches tensor data bytes itself.
#
# Usage:
#   from gguf_reader import GGUFModel
#   var m = GGUFModel(path)                    # raises on bad magic/version
#   var h = m.akv_int("embedding_length", 0)   # looks up "<arch>.embedding_length"
#   var eps = m.akv_float("attention.layer_norm_rms_epsilon", 1e-6)
#   var tied = not m.tensor_present("output.weight")
#   var dims = m.tensor_dims["token_embd.weight"]   # innermost-first, as GGUF stores it
#
# Origin: extracted from tools/gen-profile.mojo (mojo-baro, lane PROFILE,
# 2026-09-11) after it duplicated serve/tokenizer.mojo's Reader struct with
# float-scalar support added. See ~/Brain/mojo/mojo-baro/2026-09-11-llama-arch-recipe-facts.md
# for the kind of per-architecture fact this is meant to help look up quickly
# (rope type, activation, gate, tied-embed are NOT in the GGUF -- only the
# dims and per-checkpoint tensor presence are).
from std.collections import Dict
from std.memory import bitcast

comptime HEADER_MAX = 96 << 20


struct Reader:
    var buf: List[UInt8]
    var pos: Int

    def __init__(out self, var buf: List[UInt8]):
        self.buf = buf^
        self.pos = 0

    def need(self, n: Int) raises:
        if self.pos + n > len(self.buf):
            raise Error("gguf: header longer than " + String(HEADER_MAX) + " bytes")

    def u32(mut self) raises -> Int:
        self.need(4)
        var v = 0
        for i in range(4):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 4
        return v

    def u64(mut self) raises -> Int:
        self.need(8)
        var v = 0
        for i in range(8):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 8
        return v

    def string(mut self) raises -> String:
        var n = self.u64()
        self.need(n)
        var s = String(unsafe_from_utf8=Span(self.buf)[self.pos : self.pos + n])
        self.pos += n
        return s^

    def skip_scalar(mut self, vtype: Int) raises:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5 or vtype == 6:
            n = 4
        elif vtype == 10 or vtype == 11 or vtype == 12:
            n = 8
        else:
            raise Error("gguf: unknown scalar type " + String(vtype))
        self.need(n)
        self.pos += n

    def scalar_int(mut self, vtype: Int) raises -> Int:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5:
            n = 4
        elif vtype == 10 or vtype == 11:
            n = 8
        else:
            self.skip_scalar(vtype)
            return -1
        self.need(n)
        var v = 0
        for i in range(n):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += n
        return v


def scalar_f32(mut r: Reader) raises -> Float64:
    r.need(4)
    var bits: UInt32 = 0
    for i in range(4):
        bits |= UInt32(r.buf[r.pos + i]) << UInt32(8 * i)
    r.pos += 4
    return Float64(bitcast[DType.float32, 1](bits)[0])


def scalar_f64(mut r: Reader) raises -> Float64:
    r.need(8)
    var bits: UInt64 = 0
    for i in range(8):
        bits |= UInt64(r.buf[r.pos + i]) << UInt64(8 * i)
    r.pos += 8
    return bitcast[DType.float64, 1](bits)[0]


struct GGUFModel:
    """A GGUF v3 file's metadata + tensor shapes, generically parsed (no
    architecture-specific key names baked in -- callers ask for `<suffix>`
    and it's looked up under `<arch>.<suffix>` after `self.arch` is known)."""
    var arch: String
    var ints: Dict[String, Int]
    var floats: Dict[String, Float64]
    var int_arrays: Dict[String, List[Int]]
    var tensor_dims: Dict[String, List[Int]]
    var strings: Dict[String, String]
    var tensor_type: Dict[String, Int]
    var tensor_off: Dict[String, Int]
    var tensor_names: List[String]
    var data_start: Int

    def __init__(out self, path: String) raises:
        try:
            self = GGUFModel(path, 16 << 20)
        except:
            self = GGUFModel(path, HEADER_MAX)

    def __init__(out self, path: String, head: Int) raises:
        self.arch = String("")
        self.ints = Dict[String, Int]()
        self.floats = Dict[String, Float64]()
        self.int_arrays = Dict[String, List[Int]]()
        self.tensor_dims = Dict[String, List[Int]]()
        self.strings = Dict[String, String]()
        self.tensor_type = Dict[String, Int]()
        self.tensor_off = Dict[String, Int]()
        self.tensor_names = List[String]()
        self.data_start = 0
        var buf: List[UInt8]
        with open(path, "r") as f:
            buf = f.read_bytes(head)
        var r = Reader(buf^)
        var magic = r.u32()
        if magic != 0x46554747:
            raise Error("gguf: bad magic")
        var version = r.u32()
        if version != 3:
            raise Error("gguf: unsupported version " + String(version))
        var n_tensors = r.u64()
        var n_kv = r.u64()
        for _ in range(n_kv):
            var key = r.string()
            var vtype = r.u32()
            if vtype == 8:
                var s = r.string()
                self.strings[key] = s
                if key == "general.architecture":
                    self.arch = s
            elif vtype == 9:
                self._array(r, key)
            elif vtype == 6:
                self.floats[key] = scalar_f32(r)
            elif vtype == 12:
                self.floats[key] = scalar_f64(r)
            else:
                self.ints[key] = r.scalar_int(vtype)
        if self.arch == "" and self.ints.get("split.no", 0) == 0:
            raise Error("gguf: no general.architecture key")
        for _ in range(n_tensors):
            var name = r.string()
            var n_dims = r.u32()
            var dims = List[Int]()
            for _ in range(n_dims):
                dims.append(r.u64())
            self.tensor_type[name] = r.u32()
            self.tensor_off[name] = r.u64()
            self.tensor_names.append(name)
            self.tensor_dims[name] = dims^
        var align = self.ints.get("general.alignment", 32)
        self.data_start = ((r.pos + align - 1) // align) * align

    def _array(mut self, mut r: Reader, key: String) raises:
        var etype = r.u32()
        var n = r.u64()
        if etype == 8:
            for _ in range(n):
                _ = r.string()
            return
        if etype == 9:
            raise Error("gguf: nested arrays unsupported")
        var small_int = etype == 0 or etype == 1 or etype == 2 or etype == 3 or etype == 4 or etype == 5 or etype == 7 or etype == 10 or etype == 11
        if small_int and n <= 512:
            var vals = List[Int]()
            for _ in range(n):
                vals.append(r.scalar_int(etype))
            self.int_arrays[key] = vals^
            return
        for _ in range(n):
            if etype == 6:
                _ = scalar_f32(r)
            elif etype == 12:
                _ = scalar_f64(r)
            else:
                r.skip_scalar(etype)

    def akv_int(self, suffix: String, default: Int) -> Int:
        return self.ints.get(self.arch + "." + suffix, default)

    def akv_float(self, suffix: String, default: Float64) -> Float64:
        return self.floats.get(self.arch + "." + suffix, default)

    def tensor_present(self, name: String) -> Bool:
        return len(self.tensor_dims.get(name, List[Int]())) > 0
