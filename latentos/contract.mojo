from std.memory import Pointer, unsafe_memset

import latentos.proto as proto
import latentos.hash as sha

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime CONTRACT_VERSION: UInt16 = 1
comptime SEMANTIC_MAGIC: UInt32 = 0x3142534C
comptime SEMANTIC_VERSION: UInt16 = 1
comptime SEMANTIC_HEADER_SIZE: Int = 512

comptime OP_DESCRIBE: UInt32 = 1
comptime OP_CREATE: UInt32 = 2
comptime OP_FORK: UInt32 = 4
comptime OP_RESUME: UInt32 = 8
comptime OP_RELEASE: UInt32 = 16
comptime OP_EXPORT: UInt32 = 32
comptime OP_IMPORT: UInt32 = 64
comptime OP_CONSUME_SEMANTIC: UInt32 = 128
comptime ALL_OPERATIONS: UInt32 = 255

comptime REPRESENTATION_EXECUTION_CHECKPOINT: UInt8 = 1
comptime REPRESENTATION_SEMANTIC_LATENT: UInt8 = 2

comptime TRANSFER_LOCAL_HANDLE: UInt8 = 1
comptime TRANSFER_PAYLOAD: UInt8 = 2
comptime TRANSFER_RECIPE: UInt8 = 3
comptime TRANSFER_TEXT: UInt8 = 4

comptime TIER_GPU: UInt8 = 1
comptime TIER_PINNED: UInt8 = 2
comptime TIER_DISK: UInt8 = 3

comptime BINDING_FLAG_REQUIRES_AUTH: UInt32 = 4


def zero16() -> Array[UInt8, 16]:
    return Array[UInt8, 16](fill=0)


def zero32() -> Array[UInt8, 32]:
    return Array[UInt8, 32](fill=0)


def bytes_equal(a: Array[UInt8, 32], b: Array[UInt8, 32]) -> Bool:
    for i in range(32):
        if a[i] != b[i]:
            return False
    return True


def bytes_equal_16(a: Array[UInt8, 16], b: Array[UInt8, 16]) -> Bool:
    for i in range(16):
        if a[i] != b[i]:
            return False
    return True


def byte_hex(value: UInt8) -> String:
    var digits = "0123456789abcdef"
    var v = Int(value)
    return String(digits[byte=v >> 4]) + String(digits[byte=v & 15])



def hex32(value: Array[UInt8, 32]) -> String:
    var out = String(capacity_bytes=64)
    for i in range(32):
        out += byte_hex(value[i])
    return out^


def nonzero32(value: Array[UInt8, 32]) -> Bool:
    for i in range(32):
        if value[i] != 0:
            return True
    return False

def payload_sha256(payload: List[UInt8]) -> Array[UInt8, 32]:
    var digest = sha.sha256(Span(payload))
    var out = Array[UInt8, 32](fill=0)
    for i in range(32):
        out[i] = digest[i]
    return out^
struct EngineIdentity(Copyable, Movable):
    var weights_uuid: Array[UInt8, 16]
    var runtime: Array[UInt8, 32]
    var sigma_id: Array[UInt8, 32]
    var tokenizer_sha: Array[UInt8, 32]

    def __init__(out self):
        self.weights_uuid = zero16()
        self.runtime = zero32()
        self.sigma_id = zero32()
        self.tokenizer_sha = zero32()

    def check_exact(self, header: proto.LatentHeader) raises:
        proto.check_identity(
            header,
            self.weights_uuid,
            self.runtime,
            self.sigma_id,
            self.tokenizer_sha,
        )


def identity_weights_digest(identity: EngineIdentity) -> Array[UInt8, 32]:
    var out = zero32()
    for i in range(16):
        out[i] = identity.weights_uuid[i]
    return out^


def identity_runtime_digest(identity: EngineIdentity) -> Array[UInt8, 32]:
    return identity.runtime.copy()


def identity_tokenizer_digest(identity: EngineIdentity) -> Array[UInt8, 32]:
    return identity.tokenizer_sha.copy()


struct CapabilityDescriptor(Copyable, Movable):
    var contract_version: UInt16
    var operations: UInt32
    var transfer_formats: UInt32
    var representation: UInt8
    var exact_resume: Bool
    var approximation_permitted: Bool
    var max_payload_bytes: UInt64
    var architecture_family: UInt32
    var state_abi: UInt32
    var weights_digest: Array[UInt8, 32]
    var tokenizer_digest: Array[UInt8, 32]
    var runtime_digest: Array[UInt8, 32]
    var adapter_digest: Array[UInt8, 32]

    def __init__(out self):
        self.contract_version = CONTRACT_VERSION
        self.operations = ALL_OPERATIONS
        self.transfer_formats = 15
        self.representation = REPRESENTATION_EXECUTION_CHECKPOINT
        self.exact_resume = True
        self.approximation_permitted = False
        self.max_payload_bytes = UInt64(1) << 40
        self.architecture_family = 0
        self.state_abi = proto.LATENT_VERSION
        self.weights_digest = zero32()
        self.tokenizer_digest = zero32()
        self.runtime_digest = zero32()
        self.adapter_digest = zero32()

    def supports(self, operation: UInt32) -> Bool:
        return (self.operations & operation) == operation

    def validate(self) raises:
        if self.contract_version != CONTRACT_VERSION:
            raise Error("CAPABILITY:unsupported_contract_version")
        if self.operations == 0:
            raise Error("CAPABILITY:no_operations")
        if self.max_payload_bytes == 0:
            raise Error("CAPABILITY:zero_max_payload")
        if self.representation != REPRESENTATION_EXECUTION_CHECKPOINT and self.representation != REPRESENTATION_SEMANTIC_LATENT:
            raise Error("CAPABILITY:unknown_representation")
        if self.exact_resume and self.representation == REPRESENTATION_SEMANTIC_LATENT:
            raise Error("CAPABILITY:semantic_exact_resume_conflict")

    def to_json(self) -> String:
        var out = String("{\"contract_version\":")
        out += String(self.contract_version)
        out += ",\"operations\":" + String(self.operations)
        out += ",\"transfer_formats\":" + String(self.transfer_formats)
        out += ",\"representation\":" + String(self.representation)
        out += ",\"exact_resume\":" + ("true" if self.exact_resume else "false")
        out += ",\"approximation_permitted\":" + ("true" if self.approximation_permitted else "false")
        out += ",\"max_payload_bytes\":" + String(self.max_payload_bytes)
        out += ",\"architecture_family\":" + String(self.architecture_family)
        out += ",\"state_abi\":" + String(self.state_abi)
        out += ",\"weights_digest\":\"" + hex32(self.weights_digest) + "\""
        out += ",\"tokenizer_digest\":\"" + hex32(self.tokenizer_digest) + "\""
        out += ",\"runtime_digest\":\"" + hex32(self.runtime_digest) + "\""
        out += ",\"adapter_digest\":\"" + hex32(self.adapter_digest) + "\"}"
        return out^


struct Authority(Copyable, Movable):
    var tenant: String
    var room: String
    var session: String
    var sender: String
    var recipient: String
    var revision: UInt64
    var expires_at_s: Int64
    var token: Array[UInt8, 32]
    var revoked: Bool

    def __init__(out self):
        self.tenant = ""
        self.room = ""
        self.session = ""
        self.sender = ""
        self.recipient = ""
        self.revision = 0
        self.expires_at_s = 0
        self.token = zero32()
        self.revoked = False

    def validate(
        self,
        expected_tenant: String,
        expected_room: String,
        expected_session: String,
        expected_recipient: String,
        now_s: Int64,
    ) raises:
        if self.revoked:
            raise Error("AUTHORITY:revoked")
        if self.tenant != expected_tenant or self.tenant == "":
            raise Error("AUTHORITY:tenant_scope")
        if self.room != expected_room or self.room == "":
            raise Error("AUTHORITY:room_scope")
        if self.session != expected_session or self.session == "":
            raise Error("AUTHORITY:session_scope")
        if self.recipient != expected_recipient or self.recipient == "":
            raise Error("AUTHORITY:recipient_scope")
        if self.sender == "":
            raise Error("AUTHORITY:sender_missing")
        if self.expires_at_s <= now_s:
            raise Error("AUTHORITY:expired")
        if not nonzero32(self.token):
            raise Error("AUTHORITY:token_missing")


struct BindingManifest(Copyable, Movable):
    var version: UInt16
    var representation: UInt8
    var dtype: UInt8
    var source_layer: UInt16
    var injection_layer: UInt16
    var source_dim: UInt32
    var receiver_dim: UInt32
    var positions: UInt32
    var max_payload_bytes: UInt64
    var flags: UInt32
    var adapter_version: UInt32
    var normalization_id: UInt32
    var injection_site_id: UInt32
    var task_domain: UInt32
    var sender_digest: Array[UInt8, 32]
    var receiver_digest: Array[UInt8, 32]
    var sender_runtime: Array[UInt8, 32]
    var receiver_runtime: Array[UInt8, 32]
    var sender_tokenizer: Array[UInt8, 32]
    var receiver_tokenizer: Array[UInt8, 32]
    var adapter_digest: Array[UInt8, 32]
    var provenance_digest: Array[UInt8, 32]
    var expires_at_s: Int64
    var authority_epoch: UInt64
    var revision: UInt64
    var payload_len: UInt64
    var payload_digest: Array[UInt8, 32]
    var scope_digest: Array[UInt8, 32]

    def __init__(out self):
        self.version = SEMANTIC_VERSION
        self.representation = REPRESENTATION_SEMANTIC_LATENT
        self.dtype = proto.DTYPE_F32
        self.source_layer = 0
        self.injection_layer = 0
        self.source_dim = 0
        self.receiver_dim = 0
        self.positions = 0
        self.max_payload_bytes = 0
        self.flags = BINDING_FLAG_REQUIRES_AUTH
        self.adapter_version = 0
        self.normalization_id = 0
        self.injection_site_id = 0
        self.task_domain = 0
        self.sender_digest = zero32()
        self.receiver_digest = zero32()
        self.sender_runtime = zero32()
        self.receiver_runtime = zero32()
        self.sender_tokenizer = zero32()
        self.receiver_tokenizer = zero32()
        self.adapter_digest = zero32()
        self.provenance_digest = zero32()
        self.expires_at_s = 0
        self.authority_epoch = 0
        self.revision = 0
        self.payload_len = 0
        self.payload_digest = zero32()
        self.scope_digest = zero32()

    def validate(self, now_s: Int64) raises:
        if self.version != SEMANTIC_VERSION:
            raise Error("BINDING:unsupported_version")
        if self.representation != REPRESENTATION_SEMANTIC_LATENT:
            raise Error("BINDING:not_semantic")
        if self.source_dim == 0 or self.receiver_dim == 0 or self.positions == 0:
            raise Error("BINDING:zero_geometry")
        if self.dtype != proto.DTYPE_F32:
            raise Error("BINDING:unsupported_dtype")
        if self.max_payload_bytes == 0:
            raise Error("BINDING:zero_max_payload")
        if self.payload_len > self.max_payload_bytes:
            raise Error("BINDING:payload_over_quota")
        if self.positions > UInt32(1 << 20):
            raise Error("BINDING:position_overflow")
        var elements = UInt64(self.positions) * UInt64(self.receiver_dim)
        if elements == 0 or elements > (UInt64(1) << 40) or elements * 4 > self.max_payload_bytes or self.payload_len != elements * 4:
            raise Error("BINDING:shape_overflow")
        if self.expires_at_s <= now_s:
            raise Error("BINDING:expired")
        if not nonzero32(self.sender_digest) or not nonzero32(self.receiver_digest):
            raise Error("BINDING:identity_missing")
        if not nonzero32(self.adapter_digest) or not nonzero32(self.provenance_digest):
            raise Error("BINDING:provenance_missing")
    def matches_receiver(self, identity: EngineIdentity) -> Bool:
        var weights = identity_weights_digest(identity)
        var runtime = identity_runtime_digest(identity)
        var tokenizer = identity_tokenizer_digest(identity)
        return bytes_equal(self.receiver_digest, weights) and bytes_equal(self.receiver_runtime, runtime) and bytes_equal(
            self.receiver_tokenizer, tokenizer
        )

    def serialize(self, dst: BytePtr):
        unsafe_memset(dst, 0, SEMANTIC_HEADER_SIZE)
        dst.unsafe_offset(0).unsafe_bitcast[UInt32]()[] = SEMANTIC_MAGIC
        dst.unsafe_offset(4).unsafe_bitcast[UInt16]()[] = self.version
        dst[unsafe_offset=6] = self.representation
        dst[unsafe_offset=7] = self.dtype
        dst.unsafe_offset(8).unsafe_bitcast[UInt16]()[] = self.source_layer
        dst.unsafe_offset(10).unsafe_bitcast[UInt16]()[] = self.injection_layer
        dst.unsafe_offset(12).unsafe_bitcast[UInt32]()[] = self.source_dim
        dst.unsafe_offset(16).unsafe_bitcast[UInt32]()[] = self.receiver_dim
        dst.unsafe_offset(20).unsafe_bitcast[UInt32]()[] = self.positions
        dst.unsafe_offset(24).unsafe_bitcast[UInt64]()[] = self.max_payload_bytes
        dst.unsafe_offset(32).unsafe_bitcast[UInt32]()[] = self.flags
        dst.unsafe_offset(36).unsafe_bitcast[UInt32]()[] = self.adapter_version
        dst.unsafe_offset(40).unsafe_bitcast[UInt32]()[] = self.normalization_id
        dst.unsafe_offset(44).unsafe_bitcast[UInt32]()[] = self.injection_site_id
        dst.unsafe_offset(48).unsafe_bitcast[UInt32]()[] = self.task_domain
        for i in range(32):
            dst[unsafe_offset=56 + i] = self.sender_digest[i]
            dst[unsafe_offset=88 + i] = self.receiver_digest[i]
            dst[unsafe_offset=120 + i] = self.sender_runtime[i]
            dst[unsafe_offset=152 + i] = self.receiver_runtime[i]
            dst[unsafe_offset=184 + i] = self.sender_tokenizer[i]
            dst[unsafe_offset=216 + i] = self.receiver_tokenizer[i]
            dst[unsafe_offset=248 + i] = self.adapter_digest[i]
            dst[unsafe_offset=280 + i] = self.provenance_digest[i]
        dst.unsafe_offset(312).unsafe_bitcast[Int64]()[] = self.expires_at_s
        dst.unsafe_offset(320).unsafe_bitcast[UInt64]()[] = self.authority_epoch
        dst.unsafe_offset(328).unsafe_bitcast[UInt64]()[] = self.revision
        dst.unsafe_offset(336).unsafe_bitcast[UInt64]()[] = self.payload_len
        for i in range(32):
            dst[unsafe_offset=344 + i] = self.payload_digest[i]
            dst[unsafe_offset=376 + i] = self.scope_digest[i]

    @staticmethod
    def deserialize(src: BytePtr) raises -> BindingManifest:
        if src.unsafe_offset(0).unsafe_bitcast[UInt32]()[] != SEMANTIC_MAGIC:
            raise Error("BINDING:magic")
        var out = BindingManifest()
        out.version = src.unsafe_offset(4).unsafe_bitcast[UInt16]()[]
        out.representation = src[unsafe_offset=6]
        out.dtype = src[unsafe_offset=7]
        out.source_layer = src.unsafe_offset(8).unsafe_bitcast[UInt16]()[]
        out.injection_layer = src.unsafe_offset(10).unsafe_bitcast[UInt16]()[]
        out.source_dim = src.unsafe_offset(12).unsafe_bitcast[UInt32]()[]
        out.receiver_dim = src.unsafe_offset(16).unsafe_bitcast[UInt32]()[]
        out.positions = src.unsafe_offset(20).unsafe_bitcast[UInt32]()[]
        out.max_payload_bytes = src.unsafe_offset(24).unsafe_bitcast[UInt64]()[]
        out.flags = src.unsafe_offset(32).unsafe_bitcast[UInt32]()[]
        out.adapter_version = src.unsafe_offset(36).unsafe_bitcast[UInt32]()[]
        out.normalization_id = src.unsafe_offset(40).unsafe_bitcast[UInt32]()[]
        out.injection_site_id = src.unsafe_offset(44).unsafe_bitcast[UInt32]()[]
        out.task_domain = src.unsafe_offset(48).unsafe_bitcast[UInt32]()[]
        for i in range(32):
            out.sender_digest[i] = src[unsafe_offset=56 + i]
            out.receiver_digest[i] = src[unsafe_offset=88 + i]
            out.sender_runtime[i] = src[unsafe_offset=120 + i]
            out.receiver_runtime[i] = src[unsafe_offset=152 + i]
            out.sender_tokenizer[i] = src[unsafe_offset=184 + i]
            out.receiver_tokenizer[i] = src[unsafe_offset=216 + i]
            out.adapter_digest[i] = src[unsafe_offset=248 + i]
            out.provenance_digest[i] = src[unsafe_offset=280 + i]
        out.expires_at_s = src.unsafe_offset(312).unsafe_bitcast[Int64]()[]
        out.authority_epoch = src.unsafe_offset(320).unsafe_bitcast[UInt64]()[]
        out.revision = src.unsafe_offset(328).unsafe_bitcast[UInt64]()[]
        out.payload_len = src.unsafe_offset(336).unsafe_bitcast[UInt64]()[]
        for i in range(32):
            out.payload_digest[i] = src[unsafe_offset=344 + i]
            out.scope_digest[i] = src[unsafe_offset=376 + i]
        return out^

    def to_json(self) -> String:
        var out = String("{\"version\":")
        out += String(self.version)
        out += ",\"representation\":\"semantic_latent\""
        out += ",\"dtype\":" + String(self.dtype)
        out += ",\"source_layer\":" + String(self.source_layer)
        out += ",\"injection_layer\":" + String(self.injection_layer)
        out += ",\"source_dim\":" + String(self.source_dim)
        out += ",\"receiver_dim\":" + String(self.receiver_dim)
        out += ",\"positions\":" + String(self.positions)
        out += ",\"max_payload_bytes\":" + String(self.max_payload_bytes)
        out += ",\"adapter_version\":" + String(self.adapter_version)
        out += ",\"sender_digest\":\"" + hex32(self.sender_digest) + "\""
        out += ",\"receiver_digest\":\"" + hex32(self.receiver_digest) + "\""
        out += ",\"adapter_digest\":\"" + hex32(self.adapter_digest) + "\""
        out += ",\"provenance_digest\":\"" + hex32(self.provenance_digest) + "\"}"
        return out^


struct StateEnvelope(Copyable, Movable):
    var representation: UInt8
    var header: proto.LatentHeader
    var binding: BindingManifest
    var authority: Authority
    var payload_len: UInt64
    var payload_tag: UInt64

    def __init__(out self):
        self.representation = REPRESENTATION_EXECUTION_CHECKPOINT
        self.header = proto.LatentHeader()
        self.binding = BindingManifest()
        self.authority = Authority()
        self.payload_len = 0
        self.payload_tag = 0
