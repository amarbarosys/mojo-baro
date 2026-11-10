#!/usr/bin/env python3
"""Generate a deterministic rectangular CPU binding fixture and Mojo manifest."""
import argparse
import hashlib
import json
import struct
from array import array
from pathlib import Path


def digest(label: str) -> bytes:
    return hashlib.sha256(label.encode()).digest()


def write_f32(path: Path, values: array) -> None:
    with path.open("wb") as out:
        for start in range(0, len(values), 8192):
            part = values[start : start + 8192]
            out.write(struct.pack("<%df" % len(part), *part))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sender-dim", type=int, default=2560)
    parser.add_argument("--receiver-dim", type=int, default=2048)
    parser.add_argument("--positions", type=int, default=8)
    args = parser.parse_args()
    if min(args.sender_dim, args.receiver_dim, args.positions) <= 0:
        raise SystemExit("dimensions must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weight_values = array("f")
    bias_values = array("f")
    for row in range(args.receiver_dim):
        for col in range(args.sender_dim):
            weight_values.append(((row * 17 + col * 31 + 7) % 29 - 14) / 64.0)
    for row in range(args.receiver_dim):
        bias_values.append(((row * 13 + 3) % 17 - 8) / 128.0)
    weights_path = out / "affine.f32"
    write_f32(weights_path, weight_values)
    bias_path = out / "bias.f32"
    write_f32(bias_path, bias_values)

    inputs = array("f")
    for position in range(args.positions):
        for col in range(args.sender_dim):
            inputs.append(((position * 19 + col * 7 + 1) % 23 - 11) / 32.0)
    expected = array("f", [0.0]) * (args.positions * args.receiver_dim)
    combined_values = array("f", weight_values)
    combined_values.extend(bias_values)
    write_f32(weights_path, combined_values)
    for position in range(args.positions):
        in_base = position * args.sender_dim
        out_base = position * args.receiver_dim
        for row in range(args.receiver_dim):
            acc = bias_values[row]
            weight_base = row * args.sender_dim
            for col in range(args.sender_dim):
                acc += weight_values[weight_base + col] * inputs[in_base + col]
            expected[out_base + row] = acc
    input_path = out / "input.f32"
    expected_path = out / "expected.f32"
    write_f32(input_path, inputs)
    write_f32(expected_path, expected)

    sender_digest = digest("sender:granite-4.2-3b")
    receiver_digest = digest("receiver:llama-3.2-1b")
    sender_runtime = digest("sender-runtime:spark-fixture-v1")
    receiver_runtime = digest("receiver-runtime:spark-fixture-v1")
    sender_tokenizer = digest("sender-tokenizer:granite-fixture-v1")
    receiver_tokenizer = digest("receiver-tokenizer:llama-fixture-v1")
    adapter_bytes = weights_path.read_bytes()
    adapter_digest = hashlib.sha256(adapter_bytes).digest()
    provenance_digest = digest("provenance:cpu-training-oracle-fixture-v1")
    payload_len = args.positions * args.receiver_dim * 4
    payload_digest = hashlib.sha256(expected_path.read_bytes()).digest()
    scope_digest = digest("scope:tenant/room/session/recipient-fixture")

    manifest = bytearray(512)
    struct.pack_into("<I", manifest, 0, 0x3142534C)
    struct.pack_into("<H", manifest, 4, 1)
    struct.pack_into("<BB", manifest, 6, 2, 1)
    struct.pack_into("<HH", manifest, 8, 39, 0)
    struct.pack_into("<III", manifest, 12, args.sender_dim, args.receiver_dim, args.positions)
    struct.pack_into("<Q", manifest, 24, payload_len)
    struct.pack_into("<I", manifest, 32, 6)
    struct.pack_into("<I", manifest, 36, 1)
    struct.pack_into("<I", manifest, 40, 1)
    struct.pack_into("<I", manifest, 44, 1)
    struct.pack_into("<I", manifest, 48, 1)
    for offset, value in (
        (56, sender_digest),
        (88, receiver_digest),
        (120, sender_runtime),
        (152, receiver_runtime),
        (184, sender_tokenizer),
        (216, receiver_tokenizer),
        (248, adapter_digest),
        (280, provenance_digest),
    ):
        manifest[offset : offset + 32] = value
    struct.pack_into("<q", manifest, 312, 4102444800)
    struct.pack_into("<QQQ", manifest, 320, 1, 1, payload_len)
    manifest[344:376] = payload_digest
    manifest[376:408] = scope_digest
    (out / "binding.manifest").write_bytes(manifest)

    metadata = {
        "fixture": "cpu-training-oracle-v1",
        "sender_dim": args.sender_dim,
        "receiver_dim": args.receiver_dim,
        "positions": args.positions,
        "weights_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "sender_digest": sender_digest.hex(),
        "receiver_digest": receiver_digest.hex(),
        "adapter_digest": adapter_digest.hex(),
    }
    (out / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
