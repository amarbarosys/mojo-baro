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
            raise Error("golden:size:" + path)
        _ = f.seek(0)
        var data = f.read_bytes()
        if len(data) != size:
            raise Error("golden:read:" + path)
        return data^


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("usage: golden_binding manifest weights input expected dimensions")
    var sender_dim = 4
    var receiver_dim = 3
    var positions = 2
    var manifest_bytes = read_exact(args[1], contract.SEMANTIC_HEADER_SIZE)
    var manifest_array = Array[UInt8, contract.SEMANTIC_HEADER_SIZE](fill=0)
    for i in range(contract.SEMANTIC_HEADER_SIZE):
        manifest_array[i] = manifest_bytes[i]
    var manifest = contract.BindingManifest.deserialize(manifest_array.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    var affine = binding.load_affine(args[2], sender_dim, receiver_dim)
    var now_s = Int64(Int(getenv("LATENT_BINDING_NOW_S", "0")))
    affine.validate_manifest(manifest, now_s)
    var inputs = binding.bytes_to_f32(read_exact(args[3], positions * sender_dim * 4), positions * sender_dim)
    var expected = binding.bytes_to_f32(read_exact(args[4], positions * receiver_dim * 4), positions * receiver_dim)
    var actual = affine.apply(inputs, positions)
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            raise Error("golden:parity:" + String(i))
    print("GOLDEN_BINDING_PASS")
