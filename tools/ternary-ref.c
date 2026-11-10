// Reference codec for the three ternary block formats, extracted verbatim from
// the staged llama.cpp sources so tools/b3s-check.py can prove the Python
// packer bit-equal to the C that defines each format. Function bodies below
// are byte-for-byte copies (b3s-check.py --selftest re-diffs them against
// .work/b3s-ref/ when that staging dir is present):
//   quantize_row_q2_b3_ref, dequantize_row_q2_b3   <- ggml-quants.c (B3S fork)
//   quantize_row_tq1_0_ref, quantize_row_tq2_0_ref,
//   dequantize_row_tq1_0, dequantize_row_tq2_0      <- mainline ggml-quants.c
// Build: gcc -O2 -shared -fPIC -o .work/ternary-ref.so tools/ternary-ref.c
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define GGML_RESTRICT __restrict
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define QK_K 256
#define QK2_B3 128
typedef uint16_t ggml_half;

static inline ggml_half fp32_to_fp16(float f) {
    _Float16 h = (_Float16) f;
    ggml_half u;
    memcpy(&u, &h, sizeof(u));
    return u;
}
static inline float fp16_to_fp32(ggml_half u) {
    _Float16 h;
    memcpy(&h, &u, sizeof(h));
    return (float) h;
}
#define GGML_FP32_TO_FP16(x) fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) fp16_to_fp32(x)

typedef struct {
    ggml_half d;
    uint8_t qs[26];
} block_q2_b3;
static_assert(sizeof(block_q2_b3) == 28, "wrong q2_b3 block size/padding");

typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5];
    uint8_t qh[QK_K/64];
    ggml_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == 54, "wrong tq1_0 block size/padding");

typedef struct {
    uint8_t qs[QK_K/4];
    ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == 66, "wrong tq2_0 block size/padding");

// ---- verbatim: .work/b3s-ref/ggml-quants.c 439-494 ----
void quantize_row_q2_b3_ref(const float * GGML_RESTRICT x, block_q2_b3 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_B3;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float a = fabsf(x[i*qk + j]);
            if (a > amax) amax = a;
        }
        y[i].d = GGML_FP32_TO_FP16(amax);
        const float id = amax > 0.0f ? 1.0f/amax : 0.0f;

        // base-3 pack, v2 chunk-aligned layout: 32-trit chunk c owns bytes
        // [6c..6c+5] (5 trits each, 30) and 2 trits in straggler byte 24+(c>>1)
        // at digit offset 2*(c&1). Constant per-chunk byte offsets let the GPU
        // mmvq decode with compile-time shifts (no runtime-skip u64 chain).
        static const uint8_t pw3[5] = { 1, 3, 9, 27, 81 };
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < qk; ++j) {
            const float w = x[i*qk + j];
            int q = (int)roundf(w * id) + 1;
            if (q < 0) q = 0;
            if (q > 2) q = 2;
            const int c = j >> 5, t = j & 31;
            const int byte  = t < 30 ? 6*c + t/5      : 24 + (c >> 1);
            const int digit = t < 30 ? t % 5          : 2*(c & 1) + (t - 30);
            y[i].qs[byte] = (uint8_t)(y[i].qs[byte] + q * pw3[digit]);
        }
    }
}

void dequantize_row_q2_b3(const block_q2_b3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_B3;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            static const uint16_t pw3[5] = { 1, 3, 9, 27, 81 };
            // v2 chunk-aligned layout (see quantize_row_q2_b3_ref)
            const int c = j >> 5, t = j & 31;
            const int byte_i = t < 30 ? 6*c + t/5 : 24 + (c >> 1);
            const int digit  = t < 30 ? t % 5     : 2*(c & 1) + (t - 30);
            const int q = (x[i].qs[byte_i] / pw3[digit]) % 3;
            y[i*qk + j] = (q - 1) * d;
        }
    }
}

// ---- verbatim: .work/b3s-ref/mainline-ggml-quants.c 2316-2412 ----
void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}

// ---- verbatim: .work/b3s-ref/mainline-ggml-quants.c 2428-2486 ----
void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }

        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
        }
    }
}
