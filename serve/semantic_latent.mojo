from std.memory import Pointer, bitcast

import latentos.contract as contract
import latentos.binding as affine_binding


comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]



def read_bytes_exact(path: String, size: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var actual = Int(f.seek(0, 2))
        if actual != size:
            raise Error("SEMANTIC:file_size:" + path)
        _ = f.seek(0)
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("SEMANTIC:file_read:" + path)
        return data^

def read_f32_exact(path: String, count: Int) raises -> List[Float32]:
    var raw = read_bytes_exact(path, count * 4)
    var out = List[Float32](unsafe_uninit_length=count)
    for i in range(count):
        var b0 = UInt32(raw[i * 4])
        var b1 = UInt32(raw[i * 4 + 1])
        var b2 = UInt32(raw[i * 4 + 2])
        var b3 = UInt32(raw[i * 4 + 3])
        var bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        out[i] = bitcast[DType.float32, 1](bits)[0]
    return out^


struct SemanticInput(Movable):
    var active: Bool
    var positions: Int
    var dim: Int
    var values: List[Float32]
    var binding: contract.BindingManifest
    var consumed: Int

    def __init__(out self):
        self.active = False
        self.positions = 0
        self.dim = 0
        self.values = List[Float32]()
        self.binding = contract.BindingManifest()
        self.consumed = 0

    def binding_id(self) -> String:
        return contract.hex32(self.binding.adapter_digest)

    def mark_consumed(mut self):
        self.consumed += 1


struct DirectionalPins(Copyable, Movable):
    var sender_digest: String
    var receiver_digest: String
    var sender_runtime: String
    var sender_tokenizer: String
    var receiver_runtime: String
    var receiver_tokenizer: String
    var scope_digest: String
    var binding_digest: String
    var source_layer: Int
    var injection_layer: Int
    var normalization_id: Int
    var adapter_version: Int
    var injection_site: Int
    var task_domain: Int
    var scope_revision: UInt64
    var authority_epoch: UInt64

    def __init__(out self):
        self.sender_digest = ""
        self.receiver_digest = ""
        self.binding_digest = ""
        self.receiver_runtime = ""
        self.receiver_tokenizer = ""
        self.sender_runtime = ""
        self.sender_tokenizer = ""
        self.scope_digest = ""
        self.source_layer = 0
        self.injection_site = 0
        self.task_domain = 0
        self.injection_layer = 0
        self.normalization_id = 0
        self.adapter_version = 0
        self.scope_revision = 0
        self.authority_epoch = 0

    def validate(self, manifest: contract.BindingManifest) raises:
        if self.sender_digest == "" or self.receiver_digest == "" or self.receiver_runtime == "" or self.receiver_tokenizer == "" or self.scope_digest == "" or self.sender_runtime == "" or self.sender_tokenizer == "":
            raise Error("SEMANTIC:direction_pin_missing")
        if self.binding_digest == "" or self.binding_digest != contract.hex32(manifest.adapter_digest):
            raise Error("SEMANTIC:binding_digest")
        if self.scope_revision == 0 or self.authority_epoch == 0:
            raise Error("SEMANTIC:authority_context")
        if self.sender_digest != contract.hex32(manifest.sender_digest) or self.receiver_digest != contract.hex32(manifest.receiver_digest):
            raise Error("SEMANTIC:direction_identity")
        if self.receiver_runtime != contract.hex32(manifest.receiver_runtime) or self.receiver_tokenizer != contract.hex32(manifest.receiver_tokenizer):
            raise Error("SEMANTIC:receiver_runtime")
        if self.sender_runtime != contract.hex32(manifest.sender_runtime) or self.sender_tokenizer != contract.hex32(manifest.sender_tokenizer):
            raise Error("SEMANTIC:sender_runtime")
        if self.scope_digest != contract.hex32(manifest.scope_digest):
            raise Error("SEMANTIC:scope_digest")
        if self.source_layer != Int(manifest.source_layer) or self.injection_layer != Int(manifest.injection_layer) or self.injection_site != Int(manifest.injection_site_id) or self.task_domain != Int(manifest.task_domain):
            raise Error("SEMANTIC:injection_site")
        if self.normalization_id != Int(manifest.normalization_id) or self.adapter_version == 0 or self.adapter_version != Int(manifest.adapter_version):
            raise Error("SEMANTIC:adapter_contract")
        if self.scope_revision != manifest.revision or self.authority_epoch != manifest.authority_epoch:
            raise Error("SEMANTIC:authority_context")
        if (manifest.flags & contract.BINDING_FLAG_REQUIRES_AUTH) == 0:
            raise Error("SEMANTIC:authority_flag")

def load_semantic_input(
    manifest_path: String,
    input_path: String,
    expected_dim: Int,
    expected_positions: Int,
    now_s: Int64,
    pins: DirectionalPins,
) raises -> SemanticInput:
    var raw_manifest = read_bytes_exact(manifest_path, contract.SEMANTIC_HEADER_SIZE)
    var manifest_array = Array[UInt8, contract.SEMANTIC_HEADER_SIZE](fill=0)
    for i in range(contract.SEMANTIC_HEADER_SIZE):
        manifest_array[i] = raw_manifest[i]
    var manifest = contract.BindingManifest.deserialize(manifest_array.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    manifest.validate(now_s)
    pins.validate(manifest)
    if Int(manifest.receiver_dim) != expected_dim or Int(manifest.positions) != expected_positions:
        raise Error("SEMANTIC:geometry_mismatch")
    var expected_values = expected_dim * expected_positions
    if Int(manifest.payload_len) != expected_values * 4:
        raise Error("SEMANTIC:payload_length")
    var raw_input = read_bytes_exact(input_path, expected_values * 4)
    if not contract.bytes_equal(manifest.payload_digest, contract.payload_sha256(raw_input)):
        raise Error("SEMANTIC:payload_digest")
    var values = affine_binding.bytes_to_f32(raw_input, expected_values)
    var out = SemanticInput()
    out.active = True
    out.positions = expected_positions
    out.dim = expected_dim
    out.values = values^
    out.binding = manifest.copy()
    return out^