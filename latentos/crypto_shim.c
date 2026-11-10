#include <sodium.h>
#include <stddef.h>

int los_crypto_init(void) {
    return sodium_init() < 0 ? -1 : 0;
}

void los_randombytes(void *out, unsigned long long n) {
    if (sodium_init() < 0) return;
    randombytes_buf(out, (size_t)n);
}

int los_xchacha_encrypt(
    unsigned char *cipher,
    unsigned char *tag,
    const unsigned char *message,
    unsigned long long message_len,
    const unsigned char *nonce,
    const unsigned char *key,
    const unsigned char *ad,
    unsigned long long ad_len
) {
    unsigned long long maclen = 0;
    if (sodium_init() < 0) return -1; /* idempotent: picks the CPU implementation and makes the library thread-safe */
    return crypto_aead_xchacha20poly1305_ietf_encrypt_detached(
        cipher, tag, &maclen, message, (unsigned long long)message_len,
        ad, (unsigned long long)ad_len, NULL, nonce, key
    );
}

int los_xchacha_decrypt(
    unsigned char *message,
    unsigned char *tag,
    const unsigned char *cipher,
    unsigned long long cipher_len,
    const unsigned char *nonce,
    const unsigned char *key,
    const unsigned char *ad,
    unsigned long long ad_len
) {
    if (sodium_init() < 0) return -1;
    int rc = crypto_aead_xchacha20poly1305_ietf_decrypt_detached(
        message, NULL, cipher, (unsigned long long)cipher_len, tag,
        ad, (unsigned long long)ad_len, nonce, key
    );
    if (rc != 0) sodium_memzero(message, (size_t)cipher_len);
    return rc;
}

int los_equal(const unsigned char *a, const unsigned char *b, unsigned long long n) {
    return sodium_memcmp(a, b, (size_t)n);
}

void los_wipe(void *p, unsigned long long n) {
    sodium_memzero(p, (size_t)n);
}
