from std.ffi import external_call
from std.memory import Pointer

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]
comptime CRYPTO_KEY_BYTES = 32
comptime CRYPTO_NONCE_BYTES = 24
comptime CRYPTO_TAG_BYTES = 16
comptime MAX_CHUNK_BYTES = 4 * 1024 * 1024


def init_crypto() raises:
    var rc = external_call["los_crypto_init", Int32]()
    if rc != 0:
        raise Error("CRYPTO:init_failed")


def _array_ptr[N: Int](mut value: Array[UInt8, N]) -> BytePtr:
    return value.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _list_ptr(mut value: List[UInt8]) -> Pointer[UInt8, MutAnyOrigin]:
    return value.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _metadata(transfer_id: String, index: UInt64, offset: UInt64, plaintext_len: Int) -> List[UInt8]:
    var out = List[UInt8]()
    var id = transfer_id.as_bytes()
    for i in range(len(id)):
        out.append(id[i])
    out.append(0)
    for j in range(8):
        out.append(UInt8((index >> UInt64(j * 8)) & UInt64(255)))
    for j in range(8):
        out.append(UInt8((offset >> UInt64(j * 8)) & UInt64(255)))
    for j in range(8):
        out.append(UInt8((UInt64(plaintext_len) >> UInt64(j * 8)) & UInt64(255)))
    return out^


struct CryptoKey(Movable):
    var key_id: String
    var bytes: Array[UInt8, CRYPTO_KEY_BYTES]

    def __init__(out self, key_id: String, key_bytes: List[UInt8]) raises:
        if key_id == "" or len(key_bytes) != CRYPTO_KEY_BYTES:
            raise Error("CRYPTO:invalid_key")
        self.key_id = key_id
        self.bytes = Array[UInt8, CRYPTO_KEY_BYTES](fill=0)
        for i in range(CRYPTO_KEY_BYTES):
            self.bytes[i] = key_bytes[i]

    def validate(self) raises:
        if self.key_id == "":
            raise Error("CRYPTO:key_id")


struct EncryptedChunk(Movable):
    var transfer_id: String
    var index: UInt64
    var offset: UInt64
    var plaintext_len: Int
    var nonce: Array[UInt8, CRYPTO_NONCE_BYTES]
    var tag: Array[UInt8, CRYPTO_TAG_BYTES]
    var ciphertext: List[UInt8]
    var metadata: List[UInt8]

    def __init__(
        out self,
        transfer_id: String,
        index: UInt64,
        offset: UInt64,
        plaintext_len: Int,
        nonce: Array[UInt8, CRYPTO_NONCE_BYTES],
        tag: Array[UInt8, CRYPTO_TAG_BYTES],
        var ciphertext: List[UInt8],
        var metadata: List[UInt8],
    ):
        self.transfer_id = transfer_id
        self.index = index
        self.offset = offset
        self.plaintext_len = plaintext_len
        self.nonce = nonce.copy()
        self.tag = tag.copy()
        self.ciphertext = ciphertext^
        self.metadata = metadata^

    def clone(self) -> EncryptedChunk:
        return EncryptedChunk(
            self.transfer_id,
            self.index,
            self.offset,
            self.plaintext_len,
            self.nonce.copy(),
            self.tag.copy(),
            self.ciphertext.copy(),
            self.metadata.copy(),
        )

    def validate(self) raises:
        if self.transfer_id == "" or self.plaintext_len <= 0 or self.plaintext_len > MAX_CHUNK_BYTES:
            raise Error("CHUNK:shape")
        if len(self.ciphertext) != self.plaintext_len:
            raise Error("CHUNK:ciphertext_length")
        if self.offset > UInt64(1) << 63 or UInt64(self.plaintext_len) > (UInt64(1) << 63) - self.offset:
            raise Error("CHUNK:offset_overflow")
        var expected = _metadata(self.transfer_id, self.index, self.offset, self.plaintext_len)
        if len(expected) != len(self.metadata):
            raise Error("CHUNK:metadata_length")
        for i in range(len(expected)):
            if expected[i] != self.metadata[i]:
                raise Error("CHUNK:metadata_mismatch")
        var nonce_nonzero = False
        for i in range(CRYPTO_NONCE_BYTES):
            if self.nonce[i] != 0:
                nonce_nonzero = True
        if not nonce_nonzero:
            raise Error("CHUNK:zero_nonce")
        var tag_nonzero = False
        for i in range(CRYPTO_TAG_BYTES):
            if self.tag[i] != 0:
                tag_nonzero = True
        if not tag_nonzero:
            raise Error("CHUNK:empty_tag")


def seal_chunk(
    key: CryptoKey,
    transfer_id: String,
    index: UInt64,
    offset: UInt64,
    plaintext: List[UInt8],
) raises -> EncryptedChunk:
    key.validate()
    if len(plaintext) == 0 or len(plaintext) > MAX_CHUNK_BYTES:
        raise Error("CHUNK:plaintext_size")
    if offset > UInt64(1) << 63 or UInt64(len(plaintext)) > (UInt64(1) << 63) - offset:
        raise Error("CHUNK:offset_overflow")
    var nonce = Array[UInt8, CRYPTO_NONCE_BYTES](fill=0)
    var tag = Array[UInt8, CRYPTO_TAG_BYTES](fill=0)
    var metadata = _metadata(transfer_id, index, offset, len(plaintext))
    var message = plaintext.copy()
    var key_bytes = key.bytes.copy()
    var ciphertext = List[UInt8](unsafe_uninit_length=len(plaintext))
    external_call["los_randombytes", NoneType](_array_ptr(nonce), UInt64(CRYPTO_NONCE_BYTES))
    var rc = external_call["los_xchacha_encrypt", Int32](
        _list_ptr(ciphertext),
        _array_ptr(tag),
        _list_ptr(message),
        UInt64(len(message)),
        _array_ptr(nonce),
        _array_ptr(key_bytes),
        _list_ptr(metadata),
        UInt64(len(metadata)),
    )
    if rc != 0:
        raise Error("CRYPTO:seal_failed")
    return EncryptedChunk(transfer_id, index, offset, len(plaintext), nonce.copy(), tag.copy(), ciphertext^, metadata^)


def open_chunk(key: CryptoKey, chunk: EncryptedChunk) raises -> List[UInt8]:
    chunk.validate()
    var plaintext = List[UInt8](unsafe_uninit_length=chunk.plaintext_len)
    var tag_bytes = chunk.tag.copy()
    var nonce_bytes = chunk.nonce.copy()
    var key_bytes = key.bytes.copy()
    var ciphertext = chunk.ciphertext.copy()
    var metadata = chunk.metadata.copy()
    var rc = external_call["los_xchacha_decrypt", Int32](
        _list_ptr(plaintext),
        _array_ptr(tag_bytes),
        _list_ptr(ciphertext),
        UInt64(len(ciphertext)),
        _array_ptr(nonce_bytes),
        _array_ptr(key_bytes),
        _list_ptr(metadata),
        UInt64(len(metadata)),
    )
    if rc != 0:
        raise Error("CRYPTO:authentication_failed")
    return plaintext^
