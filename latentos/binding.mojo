from std.memory import bitcast

import latentos.contract as contract


comptime FNV_OFFSET: UInt64 = 0xcbf29ce484222325
comptime FNV_PRIME: UInt64 = 0x100000001b3


def bytes_to_f32(data: List[UInt8], count: Int) raises -> List[Float32]:
    if len(data) != count * 4:
        raise Error("BINDING:short_weight_read")
    var out = List[Float32](unsafe_uninit_length=count)
    for i in range(count):
        var b0 = UInt32(data[i * 4])
        var b1 = UInt32(data[i * 4 + 1])
        var b2 = UInt32(data[i * 4 + 2])
        var b3 = UInt32(data[i * 4 + 3])
        var bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        out[i] = bitcast[DType.float32, 1](bits)[0]
    return out^


def f32_bits(value: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](value)[0]


def weights_fingerprint(weights: List[Float32], bias: List[Float32]) -> UInt64:
    var h = FNV_OFFSET
    for i in range(len(weights)):
        var bits = f32_bits(weights[i])
        for j in range(4):
            h = (h ^ UInt64(UInt8((bits >> UInt32(j * 8)) & UInt32(255)))) * FNV_PRIME
    for i in range(len(bias)):
        var bits = f32_bits(bias[i])
        for j in range(4):
            h = (h ^ UInt64(UInt8((bits >> UInt32(j * 8)) & UInt32(255)))) * FNV_PRIME
    return h

def artifact_sha256(weights: List[Float32], bias: List[Float32]) -> Array[UInt8, 32]:
    var raw = List[UInt8](unsafe_uninit_length=(len(weights) + len(bias)) * 4)
    var offset = 0
    for i in range(len(weights)):
        var bits = f32_bits(weights[i])
        for j in range(4):
            raw[offset + j] = UInt8((bits >> UInt32(j * 8)) & UInt32(255))
        offset += 4
    for i in range(len(bias)):
        var bits = f32_bits(bias[i])
        for j in range(4):
            raw[offset + j] = UInt8((bits >> UInt32(j * 8)) & UInt32(255))
        offset += 4
    return contract.payload_sha256(raw)


struct AffineBinding(Movable):
    var sender_dim: Int
    var receiver_dim: Int
    var weights: List[Float32]
    var bias: List[Float32]
    var fingerprint: UInt64

    def __init__(
        out self,
        sender_dim: Int,
        receiver_dim: Int,
        var weights: List[Float32],
        var bias: List[Float32],
    ) raises:
        if sender_dim <= 0 or receiver_dim <= 0:
            raise Error("BINDING:invalid_dimensions")
        if len(weights) != sender_dim * receiver_dim:
            raise Error("BINDING:weight_shape")
        if len(bias) != receiver_dim:
            raise Error("BINDING:bias_shape")
        self.sender_dim = sender_dim
        self.receiver_dim = receiver_dim
        self.weights = weights^
        self.bias = bias^
        self.fingerprint = weights_fingerprint(self.weights, self.bias)

    def validate_manifest(self, manifest: contract.BindingManifest, now_s: Int64) raises:
        manifest.validate(now_s)
        if manifest.representation != contract.REPRESENTATION_SEMANTIC_LATENT:
            raise Error("BINDING:manifest_representation")
        if Int(manifest.source_dim) != self.sender_dim or Int(manifest.receiver_dim) != self.receiver_dim:
            raise Error("BINDING:manifest_dimensions")
        if manifest.adapter_version == 0:
            raise Error("BINDING:manifest_adapter_version")
        if not contract.bytes_equal(manifest.adapter_digest, artifact_sha256(self.weights, self.bias)):
            raise Error("BINDING:adapter_digest")

    def apply(self, inputs: List[Float32], positions: Int) raises -> List[Float32]:
        if positions <= 0 or len(inputs) != positions * self.sender_dim:
            raise Error("BINDING:input_shape")
        var out = List[Float32](unsafe_uninit_length=positions * self.receiver_dim)
        for position in range(positions):
            var in_base = position * self.sender_dim
            var out_base = position * self.receiver_dim
            for o in range(self.receiver_dim):
                var acc = self.bias[o]
                var row = o * self.sender_dim
                for i in range(self.sender_dim):
                    acc += self.weights[row + i] * inputs[in_base + i]
                out[out_base + o] = acc
        return out^


def load_affine(path: String, sender_dim: Int, receiver_dim: Int) raises -> AffineBinding:
    if sender_dim <= 0 or receiver_dim <= 0:
        raise Error("BINDING:invalid_dimensions")
    var expected = (sender_dim * receiver_dim + receiver_dim) * 4
    with open(path, "r") as f:
        var total = Int(f.seek(0, 2))
        if total != expected:
            raise Error("BINDING:file_size")
        _ = f.seek(0)
        var weight_bytes = f.read_bytes(sender_dim * receiver_dim * 4)
        var bias_bytes = f.read_bytes(receiver_dim * 4)
        var weights = bytes_to_f32(weight_bytes, sender_dim * receiver_dim)
        var bias = bytes_to_f32(bias_bytes, receiver_dim)
        return AffineBinding(sender_dim, receiver_dim, weights^, bias^)
