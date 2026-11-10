import latentos.crypto as crypto


def main() raises:
    crypto.init_crypto()
    var key_bytes = List[UInt8]()
    for i in range(crypto.CRYPTO_KEY_BYTES):
        key_bytes.append(UInt8(i + 1))
    var key = crypto.CryptoKey("fixture-key", key_bytes)
    var plaintext = List[UInt8]()
    for i in range(32):
        plaintext.append(UInt8(i))
    var chunk = crypto.seal_chunk(key, "transfer-1", 0, 0, plaintext)
    if len(chunk.ciphertext) != len(plaintext) or chunk.plaintext_len != 32:
        raise Error("crypto shape")
    var opened = crypto.open_chunk(key, chunk)
    if len(opened) != len(plaintext):
        raise Error("crypto open length")
    for i in range(len(plaintext)):
        if opened[i] != plaintext[i]:
            raise Error("crypto plaintext")
    var bad_key_bytes = List[UInt8]()
    for i in range(crypto.CRYPTO_KEY_BYTES):
        bad_key_bytes.append(UInt8(i + 2))
    var bad_key = crypto.CryptoKey("fixture-key", bad_key_bytes)
    var rejected = False
    try:
        _ = crypto.open_chunk(bad_key, chunk)
    except _:
        rejected = True
    if not rejected:
        raise Error("crypto wrong key accepted")
    chunk.ciphertext[0] = chunk.ciphertext[0] ^ 1
    rejected = False
    try:
        _ = crypto.open_chunk(key, chunk)
    except _:
        rejected = True
    if not rejected:
        raise Error("crypto tamper accepted")
    print("CRYPTO_SMOKE_PASS")
