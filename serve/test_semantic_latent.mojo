from std.sys import argv
import semantic_latent


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("usage: test_semantic_latent manifest input dim positions")
    var dim = Int(args[3])
    var positions = Int(args[4])
    var pins = semantic_latent.DirectionalPins()
    pins.sender_digest = "f671511beb72db94ddc0751bf350278871145b91789af0cdb407613d5666c624"
    pins.binding_digest = "e8c021fe510686767475a87311728656945bbab1b8c3ad493dab964e23cef950"
    pins.receiver_digest = "e63fdfc8a4574a4e10321a42e7e1d63e95ad1022040acf7831e91ac0c0db9495"
    pins.receiver_runtime = "1ba45f9db57a86f355714146881b897a2e22a4dc3ba4464c771f36fe0fc20df7"
    pins.receiver_tokenizer = "1c5d165b38ec5cf713d80469b9c3dbe48337cd3d79ba07a8fe309a3513f8310c"
    pins.sender_runtime = "3a92a23d932c1b7a3dc746ff30884814d41b236d1d0529b6bf0a6570cf9420a1"
    pins.sender_tokenizer = "671dc033c1014709e128f10a909e9d8b4215f96a2f21a6873aea805f038093d0"
    pins.scope_digest = "b056401c0c3d34e4b56ad4142db554b5a067cfc2ee026af37734525a3efbf7e6"
    pins.source_layer = 39
    pins.injection_site = 1
    pins.injection_layer = 0
    pins.normalization_id = 1
    pins.adapter_version = 1
    pins.task_domain = 1
    pins.scope_revision = 1
    pins.authority_epoch = 1
    var input = semantic_latent.load_semantic_input(args[1], args[2], dim, positions, 0, pins)
    if not input.active or input.dim != dim or input.positions != positions:
        raise Error("semantic loader shape")
    if len(input.values) != dim * positions:
        raise Error("semantic loader values")
    input.mark_consumed()
    if input.consumed != 1:
        raise Error("semantic consume counter")
    print("SEMANTIC_LOADER_PASS")
