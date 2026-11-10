from std.memory import Pointer

import latentos.contract as contract
import latentos.crypto as crypto
import latentos.sys as sys


comptime TRANSFER_MAGIC: UInt32 = 0x3152534C
comptime TRANSFER_VERSION: UInt16 = 1
comptime OP_RESUME_EXACT: UInt8 = 1
comptime OP_CONSUME_SEMANTIC: UInt8 = 2
comptime WIRE_PREFIX_BYTES: Int = 78
comptime DEFAULT_CHUNK_BYTES: Int = 64 * 1024


def _u16(value: UInt16) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8(value & UInt16(255)))
    out.append(UInt8((value >> 8) & UInt16(255)))
    return out^


def _u32(value: UInt32) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(4):
        out.append(UInt8((value >> UInt32(i * 8)) & UInt32(255)))
    return out^


def _u64(value: UInt64) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(8):
        out.append(UInt8((value >> UInt64(i * 8)) & UInt64(255)))
    return out^


def _read_u16(data: List[UInt8], offset: Int) -> UInt16:
    return UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)


def _read_u32(data: List[UInt8], offset: Int) -> UInt32:
    var out = UInt32(0)
    for i in range(4):
        out |= UInt32(data[offset + i]) << UInt32(i * 8)
    return out


def _read_u64(data: List[UInt8], offset: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(data[offset + i]) << UInt64(i * 8)
    return out


struct TransferManifest(Copyable, Movable):
    var transfer_id: String
    var operation_id: String
    var representation: UInt8
    var operation: UInt8
    var sender_digest: Array[UInt8, 32]
    var recipient_digest: Array[UInt8, 32]
    var issued_at_s: Int64
    var expires_at_s: Int64
    var total_plaintext_len: UInt64
    var chunk_size: Int
    var chunk_count: UInt64
    var key_id: String
    var binding_digest: Array[UInt8, 32]
    var scope_revision: UInt64
    var authority_epoch: UInt64

    def __init__(out self):
        self.transfer_id = ""
        self.operation_id = ""
        self.representation = contract.REPRESENTATION_EXECUTION_CHECKPOINT
        self.operation = OP_RESUME_EXACT
        self.sender_digest = contract.zero32()
        self.recipient_digest = contract.zero32()
        self.issued_at_s = 0
        self.expires_at_s = 0
        self.total_plaintext_len = 0
        self.chunk_size = DEFAULT_CHUNK_BYTES
        self.chunk_count = 0
        self.key_id = ""
        self.binding_digest = contract.zero32()
        self.scope_revision = 0
        self.authority_epoch = 0

    def validate(self, now_s: Int64, expected_recipient: Array[UInt8, 32], max_bytes: Int) raises:
        if self.transfer_id == "" or self.operation_id == "" or self.key_id == "":
            raise Error("TRANSFER:identity")
        if self.representation != contract.REPRESENTATION_EXECUTION_CHECKPOINT and self.representation != contract.REPRESENTATION_SEMANTIC_LATENT:
            raise Error("TRANSFER:representation")
        if self.representation == contract.REPRESENTATION_EXECUTION_CHECKPOINT and self.operation != OP_RESUME_EXACT:
            raise Error("TRANSFER:exact_operation")
        if self.representation == contract.REPRESENTATION_SEMANTIC_LATENT and self.operation != OP_CONSUME_SEMANTIC:
            raise Error("TRANSFER:semantic_operation")
        if not contract.bytes_equal(self.recipient_digest, expected_recipient):
            raise Error("TRANSFER:recipient")
        if not contract.nonzero32(self.sender_digest):
            raise Error("TRANSFER:sender")
        if self.scope_revision == 0 or self.authority_epoch == 0:
            raise Error("TRANSFER:authority_epoch")
        if self.issued_at_s <= 0 or self.expires_at_s <= self.issued_at_s or now_s >= self.expires_at_s:
            raise Error("TRANSFER:expiry")
        if self.total_plaintext_len == 0 or self.total_plaintext_len > UInt64(max_bytes):
            raise Error("TRANSFER:payload_quota")
        if self.chunk_size <= 0 or self.chunk_size > crypto.MAX_CHUNK_BYTES:
            raise Error("TRANSFER:chunk_size")
        var expected_count = (self.total_plaintext_len + UInt64(self.chunk_size) - 1) // UInt64(self.chunk_size)
        if self.total_plaintext_len > 4096 and self.chunk_size < 4096:
            raise Error("TRANSFER:tiny_chunks")
        if self.chunk_count > UInt64(max_bytes / 256 + 1):
            raise Error("TRANSFER:chunk_overhead")
        if expected_count != self.chunk_count or self.chunk_count == 0 or self.chunk_count > UInt64(1 << 20):
            raise Error("TRANSFER:chunk_count")
        if self.representation == contract.REPRESENTATION_SEMANTIC_LATENT and not contract.nonzero32(self.binding_digest):
            raise Error("TRANSFER:binding_missing")

    def to_json(self) -> String:
        var out = String("{\"transfer_id\":\"" + self.transfer_id + "\"")
        out += ",\"operation_id\":\"" + self.operation_id + "\""
        out += ",\"representation\":" + String(self.representation)
        out += ",\"operation\":" + String(self.operation)
        out += ",\"issued_at_s\":" + String(self.issued_at_s)
        out += ",\"expires_at_s\":" + String(self.expires_at_s)
        out += ",\"total_plaintext_len\":" + String(self.total_plaintext_len)
        out += ",\"chunk_size\":" + String(self.chunk_size)
        out += ",\"chunk_count\":" + String(self.chunk_count)
        out += ",\"key_id\":\"" + self.key_id + "\"}"
        return out^

def wire_part(value: String) -> String:
    return String(value.byte_length()) + ":" + value


def wire_transfer_id(manifest: TransferManifest) -> String:
    return wire_part(manifest.transfer_id) + wire_part(manifest.operation_id) + wire_part(String(manifest.representation)) + wire_part(String(manifest.operation)) + wire_part(contract.hex32(manifest.sender_digest)) + wire_part(contract.hex32(manifest.recipient_digest)) + wire_part(contract.hex32(manifest.binding_digest)) + wire_part(manifest.key_id) + wire_part(String(manifest.scope_revision)) + wire_part(String(manifest.authority_epoch)) + wire_part(String(manifest.issued_at_s)) + wire_part(String(manifest.chunk_size)) + wire_part(String(manifest.chunk_count)) + wire_part(String(manifest.total_plaintext_len)) + wire_part(String(manifest.expires_at_s))

struct TransportPolicy(Copyable, Movable):
    var require_encryption: Bool
    var require_authentication: Bool
    var allow_plaintext_loopback: Bool
    var allow_relay: Bool
    var max_bytes: Int

    def __init__(out self):
        self.require_encryption = True
        self.require_authentication = True
        self.allow_plaintext_loopback = False
        self.allow_relay = True
        self.max_bytes = 1 << 30

    def check(self, encrypted: Bool, authenticated: Bool, relay: Bool, loopback: Bool, bytes: Int) raises:
        if bytes <= 0 or bytes > self.max_bytes:
            raise Error("TRANSPORT:quota")
        if not encrypted and not (loopback and self.allow_plaintext_loopback):
            raise Error("TRANSPORT:encryption_required")
        if self.require_encryption and not encrypted:
            raise Error("TRANSPORT:encryption_required")
        if self.require_authentication and not authenticated:
            raise Error("TRANSPORT:authentication_required")
        if relay and not self.allow_relay:
            raise Error("TRANSPORT:relay_forbidden")

def same_encrypted_chunk(a: crypto.EncryptedChunk, b: crypto.EncryptedChunk) -> Bool:
    if a.index != b.index or a.offset != b.offset or a.plaintext_len != b.plaintext_len:
        return False
    if len(a.ciphertext) != len(b.ciphertext) or len(a.metadata) != len(b.metadata):
        return False
    for i in range(len(a.ciphertext)):
        if a.ciphertext[i] != b.ciphertext[i]:
            return False
    for i in range(len(a.metadata)):
        if a.metadata[i] != b.metadata[i]:
            return False
    for i in range(crypto.CRYPTO_NONCE_BYTES):
        if a.nonce[i] != b.nonce[i]:
            return False
    for i in range(crypto.CRYPTO_TAG_BYTES):
        if a.tag[i] != b.tag[i]:
            return False
    return True


def encrypted_chunk_tag(chunk: crypto.EncryptedChunk) -> UInt64:
    var h = UInt64(1469598103934665603)
    for i in range(len(chunk.metadata)):
        h = (h ^ UInt64(chunk.metadata[i])) * UInt64(1099511628211)
    for i in range(len(chunk.ciphertext)):
        h = (h ^ UInt64(chunk.ciphertext[i])) * UInt64(1099511628211)
    return h


struct TransferSession(Movable):
    var manifest: TransferManifest
    var wire_id: String
    var chunks: List[crypto.EncryptedChunk]
    var next_offset: Int
    var bytes_stored: Int
    var completed_chunk_tags: List[UInt64]
    var max_bytes: Int
    var completed: Bool
    var result: List[UInt8]

    def __init__(out self, max_bytes: Int):
        self.manifest = TransferManifest()
        self.wire_id = ""
        self.chunks = List[crypto.EncryptedChunk]()
        self.next_offset = 0
        self.bytes_stored = 0
        self.completed_chunk_tags = List[UInt64]()
        self.max_bytes = max_bytes
        self.completed = False
        self.result = List[UInt8]()

    def accept_manifest(
        mut self,
        manifest: TransferManifest,
        now_s: Int64,
        recipient: Array[UInt8, 32],
        operation_id: String,
        scope_revision: UInt64,
        authority_epoch: UInt64,
    ) raises:
        manifest.validate(now_s, recipient, self.max_bytes)
        if manifest.operation_id != operation_id or manifest.scope_revision != scope_revision or manifest.authority_epoch != authority_epoch:
            raise Error("TRANSFER:authority_context")
        var new_wire_id = wire_transfer_id(manifest)
        if self.manifest.transfer_id != "":
            if self.wire_id != new_wire_id:
                raise Error("TRANSFER:manifest_frozen")
            return
        self.manifest = manifest.copy()
        self.wire_id = new_wire_id

    def _same_chunk(self, a: crypto.EncryptedChunk, b: crypto.EncryptedChunk) -> Bool:
        if a.index != b.index or a.offset != b.offset or a.plaintext_len != b.plaintext_len:
            return False
        if len(a.ciphertext) != len(b.ciphertext) or len(a.metadata) != len(b.metadata):
            return False
        for i in range(len(a.ciphertext)):
            if a.ciphertext[i] != b.ciphertext[i]:
                return False
        for i in range(len(a.metadata)):
            if a.metadata[i] != b.metadata[i]:
                return False
        for i in range(crypto.CRYPTO_NONCE_BYTES):
            if a.nonce[i] != b.nonce[i]:
                return False
        for i in range(crypto.CRYPTO_TAG_BYTES):
            if a.tag[i] != b.tag[i]:
                return False
        return True

    def accept_chunk(mut self, var chunk: crypto.EncryptedChunk, now_s: Int64) raises -> Bool:
        # No manifest yet: say so (the relay already does), instead of the "identity" error validate() would give.
        if self.wire_id == "":
            raise Error("TRANSFER:manifest_required")
        chunk.validate()
        if self.completed:
            if chunk.transfer_id != self.wire_id or chunk.index >= UInt64(len(self.completed_chunk_tags)) or self.completed_chunk_tags[Int(chunk.index)] != encrypted_chunk_tag(chunk):
                raise Error("TRANSFER:conflicting_retry")
            return True
        self.manifest.validate(now_s, self.manifest.recipient_digest, self.max_bytes)
        if chunk.transfer_id != self.wire_id or chunk.index >= self.manifest.chunk_count:
            raise Error("TRANSFER:chunk_identity")
        if chunk.offset > UInt64(self.max_bytes) or len(chunk.ciphertext) > self.max_bytes:
            raise Error("TRANSFER:chunk_quota")
        var offset = Int(chunk.offset)
        var remaining = Int(self.manifest.total_plaintext_len) - offset
        var expected_len = self.manifest.chunk_size
        if remaining < expected_len:
            expected_len = remaining
        if len(chunk.ciphertext) != expected_len:
            raise Error("TRANSFER:chunk_shape")
        if chunk.index != UInt64(len(self.chunks)) and Int(chunk.offset) >= self.next_offset:
            raise Error("TRANSFER:chunk_index")
        if offset < self.next_offset:
            for i in range(len(self.chunks)):
                if self.chunks[i].index == chunk.index and self._same_chunk(self.chunks[i], chunk):
                    return True
            raise Error("TRANSFER:replay_or_gap")
        if offset != self.next_offset:
            raise Error("TRANSFER:out_of_order")
        if len(chunk.ciphertext) > self.max_bytes or self.bytes_stored > self.max_bytes - len(chunk.ciphertext):
            raise Error("TRANSFER:quota")
        self.bytes_stored += len(chunk.ciphertext)
        self.next_offset += len(chunk.ciphertext)
        self.chunks.append(chunk^)
        return True

    def resume_offset(self) -> Int:
        return self.next_offset

    def finalize(mut self, key: crypto.CryptoKey, now_s: Int64) raises -> List[UInt8]:
        # A completed session returns its result even after the manifest expired: the retry must not lose it.
        if self.completed:
            return self.result.copy()
        if self.wire_id == "":
            raise Error("TRANSFER:manifest_required")
        self.manifest.validate(now_s, self.manifest.recipient_digest, self.max_bytes)
        if self.next_offset != Int(self.manifest.total_plaintext_len):
            raise Error("TRANSFER:incomplete")
        var plaintext = List[UInt8](unsafe_uninit_length=self.next_offset)
        var out_offset = 0
        for i in range(len(self.chunks)):
            var part = crypto.open_chunk(key, self.chunks[i])
            for j in range(len(part)):
                plaintext[out_offset] = part[j]
                out_offset += 1
        for i in range(len(self.chunks)):
            self.completed_chunk_tags.append(encrypted_chunk_tag(self.chunks[i]))
        self.result = plaintext^
        self.chunks = List[crypto.EncryptedChunk]()
        self.bytes_stored = 0
        self.completed = True
        return self.result.copy()


struct RelayStore(Movable):
    var manifest: TransferManifest
    var wire_id: String
    var chunks: List[crypto.EncryptedChunk]
    var next_offset: Int
    var bytes_stored: Int
    var max_bytes: Int
    var active: Bool

    def __init__(out self, max_bytes: Int):
        self.manifest = TransferManifest()
        self.wire_id = ""
        self.chunks = List[crypto.EncryptedChunk]()
        self.next_offset = 0
        self.bytes_stored = 0
        self.max_bytes = max_bytes
        self.active = False

    def holds_key(self) -> Bool:
        return False

    def accept_manifest(
        mut self,
        manifest: TransferManifest,
        now_s: Int64,
        operation_id: String,
        scope_revision: UInt64,
        authority_epoch: UInt64,
    ) raises:
        manifest.validate(now_s, manifest.recipient_digest, self.max_bytes)
        if manifest.operation_id != operation_id or manifest.scope_revision != scope_revision or manifest.authority_epoch != authority_epoch:
            raise Error("RELAY:authority_context")
        var new_wire_id = wire_transfer_id(manifest)
        if self.active and now_s >= self.manifest.expires_at_s:
            self.clear()
        if self.active:
            if self.wire_id != new_wire_id:
                raise Error("RELAY:manifest_frozen")
            return
        self.manifest = manifest.copy()
        self.wire_id = new_wire_id
        self.active = True

    def accept_chunk(mut self, var chunk: crypto.EncryptedChunk, now_s: Int64) raises -> Bool:
        if not self.active:
            raise Error("RELAY:manifest_required")
        self.manifest.validate(now_s, self.manifest.recipient_digest, self.max_bytes)
        chunk.validate()
        if chunk.transfer_id != self.wire_id:
            raise Error("RELAY:transfer_identity")
        if chunk.offset > UInt64(self.max_bytes) or len(chunk.ciphertext) > self.max_bytes:
            raise Error("RELAY:chunk_quota")
        var offset = Int(chunk.offset)
        var remaining = Int(self.manifest.total_plaintext_len) - offset
        var expected_len = self.manifest.chunk_size
        if remaining < expected_len:
            expected_len = remaining
        if len(chunk.ciphertext) != expected_len:
            raise Error("RELAY:chunk_shape")
        if chunk.index != UInt64(len(self.chunks)) and Int(chunk.offset) >= self.next_offset:
            raise Error("RELAY:chunk_index")
        if Int(chunk.offset) < self.next_offset:
            for i in range(len(self.chunks)):
                if same_encrypted_chunk(self.chunks[i], chunk):
                    return True
            raise Error("RELAY:replay")
        if Int(chunk.offset) != self.next_offset:
            raise Error("RELAY:offset")
        if len(chunk.ciphertext) > self.max_bytes or self.bytes_stored > self.max_bytes - len(chunk.ciphertext):
            raise Error("RELAY:quota")
        self.bytes_stored += len(chunk.ciphertext)
        self.next_offset += len(chunk.ciphertext)
        self.chunks.append(chunk^)
        return True

    def chunk_count(self) -> Int:
        return len(self.chunks)

    def complete(self) -> Bool:
        return self.active and self.next_offset == Int(self.manifest.total_plaintext_len)

    def clone_chunk(self, index: Int) raises -> crypto.EncryptedChunk:
        if index < 0 or index >= len(self.chunks):
            raise Error("RELAY:chunk_index")
        return self.chunks[index].clone()

    def clear(mut self):
        self.chunks = List[crypto.EncryptedChunk]()
        self.manifest = TransferManifest()
        self.next_offset = 0
        self.bytes_stored = 0
        self.active = False


def encode_chunk(chunk: crypto.EncryptedChunk) raises -> List[UInt8]:
    chunk.validate()
    var out = List[UInt8]()
    var magic = _u32(TRANSFER_MAGIC)
    var version = _u16(TRANSFER_VERSION)
    var index = _u64(chunk.index)
    var offset = _u64(chunk.offset)
    var length = _u64(UInt64(chunk.plaintext_len))
    var meta_len = _u32(UInt32(len(chunk.metadata)))
    var cipher_len = _u32(UInt32(len(chunk.ciphertext)))
    for i in range(len(magic)):
        out.append(magic[i])
    for i in range(len(version)):
        out.append(version[i])
    for i in range(len(index)):
        out.append(index[i])
    for i in range(len(offset)):
        out.append(offset[i])
    for i in range(len(length)):
        out.append(length[i])
    for i in range(crypto.CRYPTO_NONCE_BYTES):
        out.append(chunk.nonce[i])
    for i in range(crypto.CRYPTO_TAG_BYTES):
        out.append(chunk.tag[i])
    for i in range(len(meta_len)):
        out.append(meta_len[i])
    for i in range(len(cipher_len)):
        out.append(cipher_len[i])
    for i in range(len(chunk.metadata)):
        out.append(chunk.metadata[i])
    for i in range(len(chunk.ciphertext)):
        out.append(chunk.ciphertext[i])
    return out^


def _read_chunk_bytes(data: List[UInt8], offset: Int, count: Int) -> List[UInt8]:
    var out = List[UInt8](unsafe_uninit_length=count)
    for i in range(count):
        out[i] = data[offset + i]
    return out^


def decode_chunk(data: List[UInt8], transfer_id: String) raises -> crypto.EncryptedChunk:
    if len(data) < WIRE_PREFIX_BYTES:
        raise Error("WIRE:short_prefix")
    if _read_u32(data, 0) != TRANSFER_MAGIC or _read_u16(data, 4) != TRANSFER_VERSION:
        raise Error("WIRE:version")
    var index = _read_u64(data, 6)
    var offset = _read_u64(data, 14)
    var length = Int(_read_u64(data, 22))
    var nonce = Array[UInt8, crypto.CRYPTO_NONCE_BYTES](fill=0)
    for i in range(crypto.CRYPTO_NONCE_BYTES):
        nonce[i] = data[30 + i]
    var tag = Array[UInt8, crypto.CRYPTO_TAG_BYTES](fill=0)
    for i in range(crypto.CRYPTO_TAG_BYTES):
        tag[i] = data[54 + i]
    var meta_len = Int(_read_u32(data, 70))
    var cipher_len = Int(_read_u32(data, 74))
    if meta_len < 0 or cipher_len != length or length <= 0 or length > crypto.MAX_CHUNK_BYTES:
        raise Error("WIRE:length")
    if len(data) != WIRE_PREFIX_BYTES + meta_len + cipher_len:
        raise Error("WIRE:trailing_data")
    var metadata = _read_chunk_bytes(data, WIRE_PREFIX_BYTES, meta_len)
    var ciphertext = _read_chunk_bytes(data, WIRE_PREFIX_BYTES + meta_len, cipher_len)
    var out = crypto.EncryptedChunk(transfer_id, index, offset, length, nonce, tag, ciphertext^, metadata^)
    return out^


def _write_all(fd: Int32, data: List[UInt8]) raises:
    var sent = 0
    var mutable_data = data.copy()
    while sent < len(mutable_data):
        var rc = sys.sys_write(fd, mutable_data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_offset(sent), len(mutable_data) - sent)
        if rc <= 0:
            raise Error("WIRE:write")
        sent += Int(rc)

def _read_all(fd: Int32, mut data: List[UInt8], start: Int, count: Int) raises:
    var got = 0
    while got < count:
        var rc = sys.sys_read(fd, data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_offset(start + got), count - got)
        if rc <= 0:
            raise Error("WIRE:read")
        got += Int(rc)


def recv_secure_chunk(fd: Int32, transfer_id: String, max_ciphertext: Int) raises -> crypto.EncryptedChunk:
    var prefix = List[UInt8](unsafe_uninit_length=WIRE_PREFIX_BYTES)
    _read_all(fd, prefix, 0, WIRE_PREFIX_BYTES)
    var meta_len = Int(_read_u32(prefix, 70))
    var cipher_len = Int(_read_u32(prefix, 74))
    var length = Int(_read_u64(prefix, 22))
    if meta_len < 0 or meta_len > 65536 or cipher_len <= 0 or cipher_len != length or cipher_len > max_ciphertext or cipher_len > crypto.MAX_CHUNK_BYTES:
        raise Error("WIRE:quota_or_length")
    var total = WIRE_PREFIX_BYTES + meta_len + cipher_len
    var data = List[UInt8](unsafe_uninit_length=total)
    for i in range(WIRE_PREFIX_BYTES):
        data[i] = prefix[i]
    _read_all(fd, data, WIRE_PREFIX_BYTES, meta_len + cipher_len)
    return decode_chunk(data, transfer_id)


def send_secure_chunk(fd: Int32, chunk: crypto.EncryptedChunk) raises:
    var data = encode_chunk(chunk)
    _write_all(fd, data)
