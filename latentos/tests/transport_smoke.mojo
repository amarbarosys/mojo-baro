import latentos.contract as contract
import latentos.crypto as crypto
import latentos.transport as transport
import latentos.ipc as ipc
import latentos.sys as sys


def fill32(v: UInt8) -> Array[UInt8, 32]:
    return Array[UInt8, 32](fill=v)


def make_key() raises -> crypto.CryptoKey:
    var bytes = List[UInt8]()
    for i in range(crypto.CRYPTO_KEY_BYTES):
        bytes.append(UInt8(i + 1))
    return crypto.CryptoKey("fixture-key", bytes)



def copy_range(data: List[UInt8], start: Int, count: Int) -> List[UInt8]:
    var out = List[UInt8](unsafe_uninit_length=count)
    for i in range(count):
        out[i] = data[start + i]
    return out^
def make_manifest(total: Int) -> transport.TransferManifest:
    var manifest = transport.TransferManifest()
    manifest.transfer_id = "transfer-1"
    manifest.operation_id = "turn-1"
    manifest.sender_digest = fill32(1)
    manifest.recipient_digest = fill32(2)
    manifest.issued_at_s = 10
    manifest.expires_at_s = 100
    manifest.total_plaintext_len = UInt64(total)
    manifest.chunk_size = 10
    manifest.chunk_count = UInt64((total + 9) // 10)
    manifest.key_id = "fixture-key"
    manifest.scope_revision = 1
    manifest.authority_epoch = 1
    return manifest^


def main() raises:
    crypto.init_crypto()
    var key = make_key()
    var plaintext = List[UInt8]()
    for i in range(20):
        plaintext.append(UInt8(i))
    var first_plain = copy_range(plaintext, 0, 10)
    var second_plain = copy_range(plaintext, 10, 10)
    var manifest = make_manifest(20)
    var wire_id = transport.wire_transfer_id(manifest)
    var first = crypto.seal_chunk(key, wire_id, 0, 0, first_plain)
    var second = crypto.seal_chunk(key, wire_id, 1, 10, second_plain)
    var session = transport.TransferSession(1024)
    session.accept_manifest(manifest.copy(), 20, fill32(2), "turn-1", 1, 1)
    var wrong_context = transport.TransferSession(1024)
    var context_failed = False
    try:
        wrong_context.accept_manifest(manifest.copy(), 20, fill32(2), "wrong-turn", 1, 1)
    except _:
        context_failed = True
    if not context_failed:
        raise Error("transport authority context accepted")
    if not session.accept_chunk(first.clone(), 21):
        raise Error("first chunk")
    var mutated_manifest = manifest.copy()
    mutated_manifest.expires_at_s += 1
    var frozen_failed = False
    try:
        session.accept_manifest(mutated_manifest, 21, fill32(2), "turn-1", 1, 1)
    except _:
        frozen_failed = True
    if not frozen_failed:
        raise Error("transport manifest mutation accepted")
    if session.resume_offset() != 10:
        raise Error("resume offset")
    if not session.accept_chunk(first.clone(), 21):
        raise Error("idempotent retry")
    if not session.accept_chunk(second.clone(), 21):
        raise Error("second chunk")
    var result = session.finalize(key, 22)
    if len(result) != 20 or result[19] != 19:
        raise Error("secure resume")
    var repeated = session.finalize(key, 23)
    if not session.accept_chunk(first.clone(), 23):
        raise Error("completed chunk retry")
    var conflicting = first.clone()
    conflicting.ciphertext[0] = conflicting.ciphertext[0] ^ 1
    var conflict_failed = False
    try:
        _ = session.accept_chunk(conflicting^, 23)
    except _:
        conflict_failed = True
    if not conflict_failed:
        raise Error("conflicting completed retry accepted")
    if len(repeated) != 20 or repeated[0] != 0:
        raise Error("duplicate operation")
    var relay = transport.RelayStore(1024)
    relay.accept_manifest(manifest.copy(), 24, "turn-1", 1, 1)
    if relay.holds_key():
        raise Error("relay key exposure")
    _ = relay.accept_chunk(first.clone(), 25)
    _ = relay.accept_chunk(second.clone(), 25)
    if not relay.accept_chunk(first.clone(), 25):
        raise Error("relay retry")
    if not relay.complete() or relay.chunk_count() != 2:
        raise Error("relay completion")
    var offline_receiver = transport.TransferSession(1024)
    offline_receiver.accept_manifest(relay.manifest.copy(), 26, fill32(2), "turn-1", 1, 1)
    _ = offline_receiver.accept_chunk(relay.clone_chunk(0), 27)
    _ = offline_receiver.accept_chunk(relay.clone_chunk(1), 27)
    var offline_result = offline_receiver.finalize(key, 28)
    if len(offline_result) != 20 or offline_result[0] != 0:
        raise Error("sender-offline relay resume")
    var wire = transport.encode_chunk(second)
    var decoded = transport.decode_chunk(wire, wire_id)
    var decoded_plain = crypto.open_chunk(key, decoded)
    if len(decoded_plain) != 10 or decoded_plain[0] != 10:
        raise Error("wire round trip")
    var pair = ipc.create_ipc_pair()
    transport.send_secure_chunk(pair[0], second.clone())
    var socket_chunk = transport.recv_secure_chunk(pair[1], wire_id, 1024)
    var socket_plain = crypto.open_chunk(key, socket_chunk)
    if len(socket_plain) != 10 or socket_plain[0] != 10:
        raise Error("secure socket round trip")
    _ = sys.sys_close(pair[0])
    _ = sys.sys_close(pair[1])
    var policy = transport.TransportPolicy()
    policy.check(True, True, True, False, 20)
    var refused = False
    try:
        policy.check(False, False, False, False, 20)
    except _:
        refused = True
    if not refused:
        raise Error("plaintext accepted")
    var wrong_recipient = make_manifest(20)
    wrong_recipient.recipient_digest = fill32(9)
    refused = False
    try:
        var other = transport.TransferSession(1024)
        other.accept_manifest(wrong_recipient, 20, fill32(2), "turn-1", 1, 1)
    except _:
        refused = True
    if not refused:
        raise Error("wrong recipient accepted")
    print("TRANSPORT_SMOKE_PASS")
