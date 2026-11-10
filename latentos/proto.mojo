# CANONICAL. LatentOS lives here (mojo-baro latentos/); AMDHQ src/latentos is a
# pointer to this directory (plan 12 M1, 2026-09-24). Tests and the check run
# from AMDHQ tools/latent-os/latentos-check against this copy.
#
# proto.mojo — LatentOS IPC contract & header protocol (03 §2, 04 §1).
# Defines the exact 256-byte LatentHeader binary layout, wire serialization, and validation.

from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

comptime LATENT_MAGIC = 0x3154414C # 'LAT1'
comptime LATENT_VERSION = 1
comptime LATENT_HEADER_SIZE = 256

# IPC Kinds (03 §2)
comptime KIND_KV_PAGES = 1
comptime KIND_SSM_CKPT = 2
comptime KIND_HIDDEN = 3
comptime KIND_LOGITS_TOPK = 4
comptime KIND_TEXT = 5
comptime KIND_SEMANTIC_LATENT = 6

# Data types
comptime DTYPE_F32 = 1
comptime DTYPE_BF16 = 2
comptime DTYPE_INT8_BLOCK = 3

# Batch classes (03 §6)
comptime BATCH_M1 = 0
comptime BATCH_FIXED_M = 1
comptime BATCH_DYN = 2

# Standard sizes for Qwythos 27B / 4B packs
comptime QWYTHOS_KV_BYTES_PER_TOK = 65536
comptime QWYTHOS_SSM_CKPT_BYTES = 52690944
comptime QWYTHOS_HIDDEN_BYTES_PER_STEP = 16384

def byte_to_hex(b: UInt8) -> String:
    var hex_chars = "0123456789abcdef"
    var hi = Int(b >> 4)
    var lo = Int(b & 0x0F)
    return String(hex_chars[byte=hi]) + String(hex_chars[byte=lo])

def u64_to_hex(val: UInt64) -> String:
    var hex_chars = "0123456789abcdef"
    var s = String(capacity_bytes=18)
    s += "0x"
    for i in range(16):
        var shift = UInt64((15 - i) * 4)
        var nibble = Int((val >> shift) & 0x0F)
        s += String(hex_chars[byte=nibble])
    return s^

struct LatentHeader(Copyable, Movable):
    """The 256-byte LatentOS handle contract header (03 §2).

    Binary layout (all multi-byte fields are naturally aligned):
      0..3:   magic ('LAT1' = 0x3154414C) [u32]
      4..5:   version (1) [u16]
      6:      kind (KV_PAGES, SSM_CKPT, HIDDEN, LOGITS_TOPK, TEXT) [u8]
      7:      dtype (f32=1, bf16=2) [u8]
      8..23:  weights_uuid (16 bytes GGUF UUIDv5)
      24..55: role_sha (32 bytes plain artifact sha256)
      56..87: runtime (32 bytes engine build string sha256)
      88..119: sigma_id (32 bytes adapter set sha256; 0 = native)
      120..151: tokenizer_sha (32 bytes tokenizer/template sha256)
      152..153: layer_lo [u16]
      154..155: layer_hi [u16]
      156..159: pos_lo [u32]
      160..163: pos_hi [u32]
      164..167: rope_id [u32]
      168:    batch_class (M1=0, FIXED_M=1, DYN=2) [u8]
      169:    batch_m [u8]
      170..171: reserved [u16]
      172..175: ttl_s (lease duration in seconds) [u32]
      176..183: prefix_hash (prefix cache key) [u64]
      184..191: payload_len (bytes in payload memfd) [u64]
      192..223: payload_sha (32 bytes sha256 of payload)
      224..255: hmac (32 bytes optional fleet authentication)
    """
    var magic: UInt32
    var version: UInt16
    var kind: UInt8
    var dtype: UInt8
    var weights_uuid: Array[UInt8, 16]
    var role_sha: Array[UInt8, 32]
    var runtime: Array[UInt8, 32]
    var sigma_id: Array[UInt8, 32]
    var tokenizer_sha: Array[UInt8, 32]
    var layer_lo: UInt16
    var layer_hi: UInt16
    var pos_lo: UInt32
    var pos_hi: UInt32
    var rope_id: UInt32
    var batch_class: UInt8
    var batch_m: UInt8
    var ttl_s: UInt32
    var prefix_hash: UInt64
    var payload_len: UInt64
    var payload_sha: Array[UInt8, 32]
    var hmac: Array[UInt8, 32]

    def __init__(out self):
        self.magic = LATENT_MAGIC
        self.version = UInt16(LATENT_VERSION)
        self.kind = UInt8(KIND_SSM_CKPT)
        self.dtype = UInt8(DTYPE_F32)
        self.weights_uuid = Array[UInt8, 16](fill=0)
        self.role_sha = Array[UInt8, 32](fill=0)
        self.runtime = Array[UInt8, 32](fill=0)
        self.sigma_id = Array[UInt8, 32](fill=0)
        self.tokenizer_sha = Array[UInt8, 32](fill=0)
        self.layer_lo = 0
        self.layer_hi = 0
        self.pos_lo = 0
        self.pos_hi = 0
        self.rope_id = 0
        self.batch_class = UInt8(BATCH_M1)
        self.batch_m = 1
        self.ttl_s = 600
        self.prefix_hash = 0
        self.payload_len = 0
        self.payload_sha = Array[UInt8, 32](fill=0)
        self.hmac = Array[UInt8, 32](fill=0)

    def serialize(self, dst: BytePtr):
        """Serializes this header into exactly 256 bytes in dst."""
        unsafe_memset(dst, 0, LATENT_HEADER_SIZE)
        dst.unsafe_offset(0).unsafe_bitcast[UInt32]()[] = self.magic
        dst.unsafe_offset(4).unsafe_bitcast[UInt16]()[] = self.version
        dst[unsafe_offset=6] = self.kind
        dst[unsafe_offset=7] = self.dtype
        for i in range(16):
            dst[unsafe_offset=8 + i] = self.weights_uuid[i]
        for i in range(32):
            dst[unsafe_offset=24 + i] = self.role_sha[i]
            dst[unsafe_offset=56 + i] = self.runtime[i]
            dst[unsafe_offset=88 + i] = self.sigma_id[i]
            dst[unsafe_offset=120 + i] = self.tokenizer_sha[i]
        dst.unsafe_offset(152).unsafe_bitcast[UInt16]()[] = self.layer_lo
        dst.unsafe_offset(154).unsafe_bitcast[UInt16]()[] = self.layer_hi
        dst.unsafe_offset(156).unsafe_bitcast[UInt32]()[] = self.pos_lo
        dst.unsafe_offset(160).unsafe_bitcast[UInt32]()[] = self.pos_hi
        dst.unsafe_offset(164).unsafe_bitcast[UInt32]()[] = self.rope_id
        dst[unsafe_offset=168] = self.batch_class
        dst[unsafe_offset=169] = self.batch_m
        dst.unsafe_offset(172).unsafe_bitcast[UInt32]()[] = self.ttl_s
        dst.unsafe_offset(176).unsafe_bitcast[UInt64]()[] = self.prefix_hash
        dst.unsafe_offset(184).unsafe_bitcast[UInt64]()[] = self.payload_len
        for i in range(32):
            dst[unsafe_offset=192 + i] = self.payload_sha[i]
            dst[unsafe_offset=224 + i] = self.hmac[i]

    @staticmethod
    def deserialize(src: BytePtr) -> LatentHeader:
        """Deserializes a 256-byte buffer into a LatentHeader."""
        var h = LatentHeader()
        h.magic = src.unsafe_offset(0).unsafe_bitcast[UInt32]()[]
        h.version = src.unsafe_offset(4).unsafe_bitcast[UInt16]()[]
        h.kind = src[unsafe_offset=6]
        h.dtype = src[unsafe_offset=7]
        for i in range(16):
            h.weights_uuid[i] = src[unsafe_offset=8 + i]
        for i in range(32):
            h.role_sha[i] = src[unsafe_offset=24 + i]
            h.runtime[i] = src[unsafe_offset=56 + i]
            h.sigma_id[i] = src[unsafe_offset=88 + i]
            h.tokenizer_sha[i] = src[unsafe_offset=120 + i]
        h.layer_lo = src.unsafe_offset(152).unsafe_bitcast[UInt16]()[]
        h.layer_hi = src.unsafe_offset(154).unsafe_bitcast[UInt16]()[]
        h.pos_lo = src.unsafe_offset(156).unsafe_bitcast[UInt32]()[]
        h.pos_hi = src.unsafe_offset(160).unsafe_bitcast[UInt32]()[]
        h.rope_id = src.unsafe_offset(164).unsafe_bitcast[UInt32]()[]
        h.batch_class = src[unsafe_offset=168]
        h.batch_m = src[unsafe_offset=169]
        h.ttl_s = src.unsafe_offset(172).unsafe_bitcast[UInt32]()[]
        h.prefix_hash = src.unsafe_offset(176).unsafe_bitcast[UInt64]()[]
        h.payload_len = src.unsafe_offset(184).unsafe_bitcast[UInt64]()[]
        for i in range(32):
            h.payload_sha[i] = src[unsafe_offset=192 + i]
            h.hmac[i] = src[unsafe_offset=224 + i]
        return h^

    def is_valid(self) -> Bool:
        if self.magic != LATENT_MAGIC:
            return False
        if self.version != UInt16(LATENT_VERSION):
            return False
        if self.kind < 1 or self.kind > 5:
            return False
        if self.dtype < DTYPE_F32 or self.dtype > DTYPE_INT8_BLOCK:
            return False
        if self.dtype == DTYPE_INT8_BLOCK and self.kind != KIND_KV_PAGES:
            return False
        if self.payload_len == 0:
            return False
        return True

    def kind_name(self) -> String:
        if self.kind == KIND_KV_PAGES:
            return "KV_PAGES"
        if self.kind == KIND_SSM_CKPT:
            return "SSM_CKPT"
        if self.kind == KIND_HIDDEN:
            return "HIDDEN"
        if self.kind == KIND_LOGITS_TOPK:
            return "LOGITS_TOPK"
        if self.kind == KIND_TEXT:
            return "TEXT"
        return "UNKNOWN"

    def content_address(self) -> String:
        """Returns the human-readable content address prefix."""
        return "lat:" + self.kind_name() + "@pos" + String(self.pos_hi) + ":" + u64_to_hex(self.prefix_hash)


def bytes_equal_16(a: Array[UInt8, 16], b: Array[UInt8, 16]) -> Bool:
    for i in range(16):
        if a[i] != b[i]:
            return False
    return True


def bytes_equal_32(a: Array[UInt8, 32], b: Array[UInt8, 32]) -> Bool:
    for i in range(32):
        if a[i] != b[i]:
            return False
    return True


def headers_equal(a: LatentHeader, b: LatentHeader) -> Bool:
    if a.kind != b.kind or a.dtype != b.dtype or a.layer_lo != b.layer_lo or a.layer_hi != b.layer_hi:
        return False
    if a.pos_lo != b.pos_lo or a.pos_hi != b.pos_hi or a.rope_id != b.rope_id:
        return False
    if a.batch_class != b.batch_class or a.batch_m != b.batch_m:
        return False
    if a.prefix_hash != b.prefix_hash or a.payload_len != b.payload_len:
        return False
    if not bytes_equal_16(a.weights_uuid, b.weights_uuid) or not bytes_equal_32(a.role_sha, b.role_sha):
        return False
    if not bytes_equal_32(a.runtime, b.runtime) or not bytes_equal_32(a.sigma_id, b.sigma_id):
        return False
    if not bytes_equal_32(a.tokenizer_sha, b.tokenizer_sha) or not bytes_equal_32(a.payload_sha, b.payload_sha):
        return False
    return bytes_equal_32(a.hmac, b.hmac)


def check_identity(
    header: LatentHeader,
    weights_uuid: Array[UInt8, 16],
    runtime: Array[UInt8, 32],
    sigma_id: Array[UInt8, 32],
    tokenizer_sha: Array[UInt8, 32],
) raises:
    """U3 (09-roadmap.md "model universal" discussion): refuse a handle minted
    by a different model, tokenizer, engine build or Sigma space. Checks the
    four identity fields against the receiver's own; raises a typed error
    naming the first mismatching field. role_sha is deliberately not part of
    this check (04 SS1: it names the prompt/role artifact, not the model).
    Callers must run this BEFORE any restore so a mismatch leaves nothing
    ingested.
    """
    if not bytes_equal_16(header.weights_uuid, weights_uuid):
        raise Error("IDENTITY_MISMATCH:weights_uuid")
    if not bytes_equal_32(header.runtime, runtime):
        raise Error("IDENTITY_MISMATCH:runtime")
    if not bytes_equal_32(header.sigma_id, sigma_id):
        raise Error("IDENTITY_MISMATCH:sigma_id")
    if not bytes_equal_32(header.tokenizer_sha, tokenizer_sha):
        raise Error("IDENTITY_MISMATCH:tokenizer_sha")
