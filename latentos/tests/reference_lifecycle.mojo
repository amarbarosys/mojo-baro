from std.memory import Pointer

import latentos.contract as contract
import latentos.proto as proto
import latentos.client as client
import latentos.store as store


def fill16(v: UInt8) -> Array[UInt8, 16]:
    return Array[UInt8, 16](fill=v)


def fill32(v: UInt8) -> Array[UInt8, 32]:
    return Array[UInt8, 32](fill=v)


def identity() -> contract.EngineIdentity:
    var out = contract.EngineIdentity()
    out.weights_uuid = fill16(1)
    out.runtime = fill32(2)
    out.sigma_id = fill32(0)
    out.tokenizer_sha = fill32(3)
    return out^


def header(length: Int, runtime_value: UInt8 = 2) -> proto.LatentHeader:
    var out = proto.LatentHeader()
    out.kind = proto.KIND_KV_PAGES
    out.dtype = proto.DTYPE_F32
    out.payload_len = UInt64(length)
    out.weights_uuid = fill16(1)
    out.runtime = fill32(runtime_value)
    out.tokenizer_sha = fill32(3)
    return out^


def authority(now_expiry: Int64 = 1000) -> contract.Authority:
    var out = contract.Authority()
    out.tenant = "tenant"
    out.room = "room"
    out.session = "session"
    out.sender = "sender"
    out.recipient = "recipient"
    out.expires_at_s = now_expiry
    out.token[0] = 1
    return out^


def binding_for(receiver: contract.EngineIdentity, payload: List[UInt8]) -> contract.BindingManifest:
    var out = contract.BindingManifest()
    out.source_dim = 4
    out.receiver_dim = 3
    out.positions = 2
    out.max_payload_bytes = 64
    out.payload_len = UInt64(len(payload))
    out.payload_digest = contract.payload_sha256(payload)
    out.sender_digest = fill32(8)
    out.receiver_digest = contract.identity_weights_digest(receiver)
    out.sender_runtime = fill32(4)
    out.receiver_runtime = receiver.runtime.copy()
    out.sender_tokenizer = fill32(5)
    out.receiver_tokenizer = receiver.tokenizer_sha.copy()
    out.adapter_digest = fill32(6)
    out.provenance_digest = fill32(7)
    out.expires_at_s = 1000
    out.adapter_version = 1
    return out^


def bytes(values: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    for value in values:
        out.append(UInt8(value))
    return out^




def main() raises:
    var id = identity()
    var limits = store.StoreLimits()
    limits.max_handles = 16
    limits.max_bytes = 4096
    limits.default_ttl_s = 50
    var api = client.LatentOSClient(id, limits)
    var capability = api.describe()
    if capability.byte_length() == 0:
        raise Error("capability descriptor")
    var grant = authority()
    var parent_payload = bytes([1, 2, 3, 4])
    var parent = api.create(header(4), parent_payload, grant, "role-a", contract.TIER_DISK, 10)
    var branch = api.fork(parent, "role-b", 11)
    var suffix = bytes([9])
    api.append_branch(branch, suffix, 12)
    var a_bytes = api.resume(parent, 13)
    var b_bytes = api.resume(branch, 13)
    if len(a_bytes) != 4 or len(b_bytes) != 5 or b_bytes[4] != 9:
        raise Error("branch isolation")
    if api.loaded_weights() != 1:
        raise Error("weight owner")
    var exported = api.export(branch, 14)
    var imported = api.import_exact(exported, 15)
    if api.count() != 3:
        raise Error("import lifecycle")
    _ = api.release(imported, 16)
    _ = api.release(branch, 16)
    _ = api.release(parent, 16)
    if api.count() != 0 or api.retained_bytes() != 0:
        raise Error("release baseline")
    var chain_client = client.LatentOSClient(id, limits)
    var grand = chain_client.create(header(1), bytes([1]), grant, "grand", contract.TIER_DISK, 1)
    var middle = chain_client.fork(grand, "middle", 2)
    var leaf = chain_client.fork(middle, "leaf", 3)
    _ = chain_client.release(grand, 4)
    _ = chain_client.release(middle, 4)
    _ = chain_client.release(leaf, 4)
    if chain_client.count() != 0 or chain_client.retained_bytes() != 0:
        raise Error("recursive release cleanup")

    var semantic_payload = bytes([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ])
    var semantic = binding_for(id, semantic_payload)
    var semantic_handle = api.import_semantic(semantic, semantic_payload, grant, "semantic", contract.TIER_PINNED, 20)
    var consumed = api.consume_semantic(semantic_handle, 21)
    if len(consumed) != len(semantic_payload):
        raise Error("semantic consume")
    var exact_semantic_failed = False
    try:
        _ = api.resume(semantic_handle, 22)
    except _:
        exact_semantic_failed = True
    if not exact_semantic_failed:
        raise Error("semantic exact resume accepted")
    var legacy = header(1)
    legacy.kind = proto.KIND_SEMANTIC_LATENT
    if legacy.is_valid():
        raise Error("semantic kind accepted by v1")
    _ = api.release(semantic_handle, 23)

    var wrong_runtime = header(1, 9)
    var wrong_identity_failed = False
    try:
        _ = api.create(wrong_runtime, bytes([1]), grant, "bad", contract.TIER_DISK, 30)
    except _:
        wrong_identity_failed = True
    if not wrong_identity_failed:
        raise Error("wrong identity accepted")
    var expired = store.StoreLimits()
    expired.max_handles = 4
    expired.max_bytes = 32
    expired.default_ttl_s = 2
    var expiring = client.LatentOSClient(id, expired)
    var expiring_header = header(1)
    expiring_header.ttl_s = 2
    var expiring_handle = expiring.create(expiring_header, bytes([1]), grant, "expired", contract.TIER_DISK, 1)
    var expiry_failed = False
    try:
        _ = expiring.resume(expiring_handle, 4)
    except _:
        expiry_failed = True
    if not expiry_failed:
        raise Error("expired state accepted")
    if expiring.sweep(4) != 1 or expiring.count() != 0 or expiring.retained_bytes() != 0:
        raise Error("expired state cleanup")
    var quota = store.StoreLimits()
    quota.max_handles = 4
    quota.max_bytes = 2
    var limited = client.LatentOSClient(id, quota)
    var quota_failed = False
    try:
        _ = limited.create(header(3), bytes([1, 2, 3]), grant, "quota", contract.TIER_DISK, 1)
    except _:
        quota_failed = True
    if not quota_failed:
        raise Error("quota accepted")
    if contract.hex32(contract.payload_sha256(bytes([97, 98, 99]))) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
        raise Error("SHA256 known answer")

    var manifest_bytes = Array[UInt8, contract.SEMANTIC_HEADER_SIZE](fill=0)
    var ptr = manifest_bytes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    semantic.serialize(ptr)
    var roundtrip = contract.BindingManifest.deserialize(ptr)
    if not contract.bytes_equal(semantic.adapter_digest, roundtrip.adapter_digest) or roundtrip.receiver_dim != 3:
        raise Error("semantic manifest roundtrip")
    print("REFERENCE_LIFECYCLE_PASS")
