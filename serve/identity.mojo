# identity.mojo — U3: computes the four handoff-identity fields once per
# model load, cached, never per handoff (docs/design/latent-os/09-roadmap.md
# "model universal" discussion; AMDHQ 04 SS1/SS6; plan:
# AMDHQ/runs/latent-os/U3-plan-2026-09-11.md).
#
# weights_uuid: GGUF's own general.uuid if present, else a UUIDv5-shaped id
#   (RFC4122 version/variant bits stamped over a SHA-256 digest, since only
#   SHA-256 is available here -- see the plan for why this never needs to be
#   a byte-exact RFC4122 UUIDv5) over the raw tensor-data bytes only (04 SS1's
#   proven "tensor data only" property).
# runtime: sha256 of "mojo-baro <git sha> <mojo --version output> rocm
#   <rocm version>", the shape of 04 SS3's manifest example.
# tokenizer_sha: sha256 over vocab, then merges (original order), then the
#   chat template, all as read by serve/tokenizer.mojo's Tokenizer.
# sigma_id: always zero -- Phase 1 is model-native space, no adapter to hash.
from std.subprocess import run

from sha256 import sha256
from tokenizer import Reader, Tokenizer

comptime GGUF_MAGIC = 0x46554747
comptime HEADER_SCAN_MAX = 96 << 20


@fieldwise_init
struct EngineIdentity(Copyable, Movable):
    var weights_uuid: Array[UInt8, 16]
    var runtime: Array[UInt8, 32]
    var sigma_id: Array[UInt8, 32]
    var tokenizer_sha: Array[UInt8, 32]


def strip_ws(s: String) -> String:
    var b = s.as_bytes()
    var lo = 0
    var hi = len(b)
    while lo < hi and (b[lo] == 32 or b[lo] == 9 or b[lo] == 10 or b[lo] == 13):
        lo += 1
    while hi > lo and (b[hi - 1] == 32 or b[hi - 1] == 9 or b[hi - 1] == 10 or b[hi - 1] == 13):
        hi -= 1
    return String(unsafe_from_utf8=Span(b)[lo:hi])


def sha256_to_inline32(buf: List[UInt8]) -> Array[UInt8, 32]:
    var digest = sha256(Span(buf))
    var out = Array[UInt8, 32](fill=0)
    for i in range(32):
        out[i] = digest[i]
    return out^


def hex_nibble(c: UInt8) raises -> UInt8:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 97 + 10
    if c >= 65 and c <= 70:
        return c - 65 + 10
    raise Error("identity: bad hex digit in general.uuid")


def parse_uuid_string(s: String) raises -> Array[UInt8, 16]:
    """Parses "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" into 16 bytes."""
    var hexonly = List[UInt8]()
    for b in s.as_bytes():
        if b != 45:  # '-'
            hexonly.append(b)
    if len(hexonly) != 32:
        raise Error("identity: general.uuid is not a 36-char UUID string: " + s)
    var out = Array[UInt8, 16](fill=0)
    for i in range(16):
        out[i] = (hex_nibble(hexonly[i * 2]) << 4) | hex_nibble(hexonly[i * 2 + 1])
    return out^


def truncate_to_uuid_v5_shaped(digest: Array[UInt8, 32]) -> Array[UInt8, 16]:
    var out = Array[UInt8, 16](fill=0)
    for i in range(16):
        out[i] = digest[i]
    out[6] = (out[6] & 0x0F) | 0x50
    out[8] = (out[8] & 0x3F) | 0x80
    return out^


def gguf_general_uuid_and_data_offset(gguf_path: String) raises -> Tuple[String, Int]:
    """One pass over the GGUF header (metadata + tensor-info list): returns
    (general.uuid string, "" if absent; aligned tensor-data start offset)."""
    var buf: List[UInt8]
    with open(gguf_path, "r") as f:
        buf = f.read_bytes(HEADER_SCAN_MAX)
    var r = Reader(buf^)
    var magic = r.u32()
    if magic != GGUF_MAGIC:
        raise Error("identity: bad GGUF magic in " + gguf_path)
    var version = r.u32()
    if version != 3:
        raise Error("identity: unsupported GGUF version " + String(version))
    var n_tensors = r.u64()
    var n_kv = r.u64()
    var general_uuid = String("")
    var alignment = 32
    for _ in range(n_kv):
        var key = r.string()
        var vtype = r.u32()
        if vtype == 8:
            var s = r.string()
            if key == "general.uuid":
                general_uuid = s
        elif vtype == 9:
            var etype = r.u32()
            var n = r.u64()
            if etype == 8:
                for _ in range(n):
                    _ = r.string()
            elif etype == 9:
                raise Error("identity: nested arrays unsupported")
            else:
                for _ in range(n):
                    r.skip_scalar(etype)
        else:
            var v = r.scalar_int(vtype)
            if key == "general.alignment" and v > 0:
                alignment = v
    for _ in range(n_tensors):
        _ = r.string()  # name
        var n_dims = r.u32()
        for _ in range(n_dims):
            _ = r.u64()  # dim
        _ = r.u32()  # ggml_type
        _ = r.u64()  # offset, relative to data section (unused: we want the section start)
    var data_offset = r.pos
    if (data_offset % alignment) != 0:
        data_offset += alignment - (data_offset % alignment)
    return (general_uuid, data_offset)


def compute_weights_uuid(gguf_path: String) raises -> Array[UInt8, 16]:
    var probe = gguf_general_uuid_and_data_offset(gguf_path)
    var general_uuid = probe[0]
    if general_uuid != "":
        return parse_uuid_string(general_uuid)
    var data_offset = probe[1]
    var tensor_bytes: List[UInt8]
    with open(gguf_path, "r") as f:
        _ = f.seek(data_offset)
        tensor_bytes = f.read_bytes()
    var digest = sha256_to_inline32(tensor_bytes^)
    return truncate_to_uuid_v5_shaped(digest)


def compute_tokenizer_sha(tok: Tokenizer) -> Array[UInt8, 32]:
    var buf = List[UInt8]()
    for t in tok.tokens:
        for b in t.as_bytes():
            buf.append(b)
        buf.append(0)
    for ab in tok.merges:
        for b in tok.tokens[ab >> 32].as_bytes():
            buf.append(b)
        buf.append(32)
        for b in tok.tokens[ab & 0xFFFFFFFF].as_bytes():
            buf.append(b)
        buf.append(0)
    for b in tok.chat_template.as_bytes():
        buf.append(b)
    return sha256_to_inline32(buf^)


def read_git_head_sha(mojo_baro_dir: String) raises -> String:
    var head_bytes: List[UInt8]
    with open(mojo_baro_dir + "/.git/HEAD", "r") as f:
        head_bytes = f.read_bytes()
    var head = strip_ws(String(unsafe_from_utf8=Span(head_bytes)))
    var sha: String
    if head.startswith("ref: "):
        var git_ref = String(head.removeprefix("ref: "))
        var ref_bytes: List[UInt8]
        with open(mojo_baro_dir + "/.git/" + git_ref, "r") as f2:
            ref_bytes = f2.read_bytes()
        sha = strip_ws(String(unsafe_from_utf8=Span(ref_bytes)))
    else:
        sha = head
    var out = String("")
    var n = 7 if sha.byte_length() >= 7 else sha.byte_length()
    for i in range(n):
        out += String(sha[byte=i])
    return out^


def runtime_build_string(mojo_baro_dir: String, mojo_bin: String) raises -> String:
    var git_sha = read_git_head_sha(mojo_baro_dir)
    var mojo_ver = strip_ws(run(mojo_bin + " --version"))
    var rocm_ver = String("unknown")
    try:
        var rv: List[UInt8]
        with open("/opt/rocm/.info/version", "r") as f:
            rv = f.read_bytes()
        rocm_ver = strip_ws(String(unsafe_from_utf8=Span(rv)))
    except:
        pass
    return "mojo-baro " + git_sha + " " + mojo_ver + " rocm " + rocm_ver


def compute_runtime_sha(mojo_baro_dir: String, mojo_bin: String) raises -> Array[UInt8, 32]:
    var s = runtime_build_string(mojo_baro_dir, mojo_bin)
    var buf = List[UInt8]()
    for b in s.as_bytes():
        buf.append(b)
    return sha256_to_inline32(buf^)


def compute_engine_identity(
    gguf_path: String,
    mojo_baro_dir: String,
    mojo_bin: String,
    tok: Tokenizer,
) raises -> EngineIdentity:
    """Computed once per model load by the caller and cached; never call this
    per handoff (mint/ingest are hot paths, GGUF header parsing and sha256
    over the tensor-data blob are not free)."""
    var weights_uuid = compute_weights_uuid(gguf_path)
    var runtime = compute_runtime_sha(mojo_baro_dir, mojo_bin)
    var sigma_id = Array[UInt8, 32](fill=0)
    var tokenizer_sha = compute_tokenizer_sha(tok)
    return EngineIdentity(weights_uuid^, runtime^, sigma_id^, tokenizer_sha^)
