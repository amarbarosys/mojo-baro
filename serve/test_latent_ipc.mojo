"""T3b header checks on the CPU: B refuses a wrong pack, a wrong prefix and a bad size before any
HIP call (ipc_refusal reaches no HIP code), and accepts the header A builds for the same prefix."""
from std.memory import Pointer

import latentos.proto as proto
from prefix import prefix_hash
from latent import hash64
from latent_ipc import ipc_refusal, ipc_payload_bytes, hex_encode, tokens_from_hex, MSG_BYTES, weights_record, weights_refusal, HANDLE_BYTES


def header_for(salt: List[UInt8], tokens: List[Int]) -> proto.LatentHeader:
    var h = proto.LatentHeader()
    h.kind = UInt8(proto.KIND_KV_PAGES)
    h.dtype = UInt8(proto.DTYPE_F32)
    h.pos_hi = UInt32(len(tokens))
    h.prefix_hash = hash64(prefix_hash(salt, tokens, len(tokens)))
    h.payload_len = UInt64(ipc_payload_bytes(len(tokens)))
    for i in range(32):
        h.role_sha[i] = salt[i]
    return h^


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL " + what)
    print("ok", what)


def main() raises:
    var salt = List[UInt8]()
    for i in range(32):
        salt.append(UInt8(i * 7 + 1))
    var tokens = List[Int]()
    for t in range(300):
        tokens.append(1000 + t * 37)

    var msg = List[UInt8](length=MSG_BYTES, fill=0)
    header_for(salt, tokens).serialize(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr())))
    var h = proto.LatentHeader.deserialize(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr())))
    check(ipc_refusal(h, salt, tokens, 9216) == "", "A's header for this prefix is accepted after a serialize round trip")

    var other = salt.copy()
    other[5] ^= 1
    check(ipc_refusal(h, other, tokens, 9216) == "IPC refused: saved from a different pack", "wrong pack salt refused")

    var shifted = tokens.copy()
    shifted[17] += 1
    check(ipc_refusal(h, salt, shifted, 9216) == "IPC refused: prefix_hash mismatch", "same length, other prefix refused")

    var bad = h.copy()
    bad.prefix_hash ^= 1
    check(ipc_refusal(bad, salt, tokens, 9216) == "IPC refused: prefix_hash mismatch", "tampered header prefix_hash refused")

    bad = h.copy()
    bad.payload_len += 4
    check(ipc_refusal(bad, salt, tokens, 9216).startswith("IPC refused: payload_len"), "wrong payload_len refused")
    check(ipc_refusal(h, salt, tokens, 300).startswith("IPC refused: pos"), "pos at TMAX refused")

    bad = h.copy()
    bad.kind = UInt8(proto.KIND_SSM_CKPT)
    check(ipc_refusal(bad, salt, tokens, 9216) == "IPC refused: not an f32 KV_PAGES LatentHeader", "wrong kind refused")

    check(ipc_payload_bytes(7999) == 581173248, "8k payload = 63 pages x 8 MiB + conv + SSM = 581,173,248 B")

    var raw = List[UInt8]()
    for t in tokens:
        for b in range(4):
            raw.append(UInt8((t >> (8 * b)) & 0xFF))
    var back = tokens_from_hex(hex_encode(Span(raw)))
    var same = len(back) == len(tokens)
    for i in range(min(len(back), len(tokens))):
        same = same and back[i] == tokens[i]
    check(same, "token ids survive the hex u32 LE round trip")
    check(hex_encode(Span(msg)).byte_length() == 640, "the 320-byte message is 640 hex digits")
    var handle = List[UInt8](length=HANDLE_BYTES, fill=0xAB)
    var total = 6639194112
    var rec = weights_record(total, salt, handle)
    check(weights_refusal(rec, salt, total) == "", "weight record for this pack and size is accepted")
    check(weights_refusal(rec, other, total) == "weights import refused: exported from a different pack", "weight record from another pack refused")
    check(weights_refusal(rec, salt, total - 1).startswith("weights import refused: exporter holds 6639194112"), "weight record of another size refused")
    var badrec = rec.copy()
    badrec[0] ^= 1
    check(weights_refusal(badrec, salt, total) == "weights import refused: not a BAROWT01 record", "bad magic refused")
    var short = rec.copy()
    _ = short.pop()
    check(weights_refusal(short, salt, total).startswith("weights import refused: record is 111"), "short record refused")
    check(rec[48] == 0xAB and rec[111] == 0xAB, "handle carried at bytes 48..111")
    print("LATENT_IPC_HEADER_PASS")
