from std.memory import Pointer
from std.sys import argv
from std.os import getenv

import latentos.binding as binding
import latentos.contract as contract


comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]


def read_exact(path: String, size: Int) raises -> List[UInt8]:
    with open(path, "r") as f:
        var actual = Int(f.seek(0, 2))
        if actual != size:
            raise Error("binding:file_size:" + path)
        _ = f.seek(0)
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("binding:short_read:" + path)
        return data^


def main() raises:
    var args = argv()
    if len(args) != 8:
        raise Error("usage: latent_binding_check manifest weights input expected sender_dim receiver_dim positions")
    var manifest_path = args[1]
    var weights_path = args[2]
    var input_path = args[3]
    var expected_path = args[4]
    var sender_dim = Int(args[5])
    var receiver_dim = Int(args[6])
    var positions = Int(args[7])
    var manifest_bytes = read_exact(manifest_path, contract.SEMANTIC_HEADER_SIZE)
    var manifest_array = Array[UInt8, contract.SEMANTIC_HEADER_SIZE](fill=0)
    for i in range(contract.SEMANTIC_HEADER_SIZE):
        manifest_array[i] = manifest_bytes[i]
    var manifest_ptr = manifest_array.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var manifest = contract.BindingManifest.deserialize(manifest_ptr)
    var affine = binding.load_affine(weights_path, sender_dim, receiver_dim)
    var now_s = Int64(Int(getenv("LATENT_BINDING_NOW_S", "0")))
    affine.validate_manifest(manifest, now_s)
    var input_count = positions * sender_dim
    var output_count = positions * receiver_dim
    var inputs = binding.bytes_to_f32(read_exact(input_path, input_count * 4), input_count)
    var expected = binding.bytes_to_f32(read_exact(expected_path, output_count * 4), output_count)
    var actual = affine.apply(inputs, positions)
    if len(actual) != len(expected):
        raise Error("binding:output_length")
    var max_abs = Float32(0.0)
    var worst = 0
    for i in range(len(expected)):
        var diff = actual[i] - expected[i]
        if diff != diff:
            raise Error("binding:nan_output:" + String(i))
        if diff < 0.0:
            diff = -diff
        if diff > max_abs:
            max_abs = diff
            worst = i
    if max_abs > Float32(1e-5):
        raise Error("binding:parity:" + String(max_abs) + ":index=" + String(worst))
    var wrong_dim_rejected = False
    try:
        _ = binding.load_affine(weights_path, sender_dim + 1, receiver_dim)
    except _:
        wrong_dim_rejected = True
    if not wrong_dim_rejected:
        raise Error("binding:wrong_dimensions_accepted")
    print("LATENT_BINDING_PASS max_abs=" + String(max_abs) + " worst=" + String(worst) + " positions=" + String(positions))
