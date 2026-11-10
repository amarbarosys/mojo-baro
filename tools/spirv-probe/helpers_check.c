/* CPU preflight for helpers.c: the integer bf16/f16 conversions must equal the compiler's
   native __bf16 / _Float16 casts. Strided sweep of all f32 bit patterns plus every rounding
   midpoint and its two neighbours. Prints PASS or FAIL helpers: <first mismatch>. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "helpers.c"

static long bad;
static void one(uint32_t u) {
    float f; memcpy(&f, &u, 4);
    __bf16 b = (__bf16)f; _Float16 h = (_Float16)f; uint16_t wb, wh; memcpy(&wb, &b, 2); memcpy(&wh, &h, 2);
    uint16_t gb = baro_f32_to_bf16(f), gh = baro_f32_to_f16(f);
    if (isnan(f)) { if ((gb & 0x7F80) != 0x7F80 || !(gb & 0x7F) || (gh & 0x7C00) != 0x7C00 || !(gh & 0x3FF)) if (!bad++) printf("FAIL helpers: NaN 0x%08x -> bf16 0x%04x f16 0x%04x\n", u, gb, gh); return; }
    if ((gb != wb || gh != wh) && !bad++) printf("FAIL helpers: 0x%08x -> bf16 0x%04x want 0x%04x, f16 0x%04x want 0x%04x\n", u, gb, wb, gh, wh);
}
int main(void) {
    for (uint64_t u = 0; u < (1ull << 32); u += 97) one((uint32_t)u);
    for (uint32_t h = 0; h < 65536; h++) {
        uint32_t mid = (h << 16) | 0x8000u; one(mid - 1); one(mid); one(mid + 1);
        __bf16 b; uint16_t hb = (uint16_t)h; memcpy(&b, &hb, 2); float want = (float)b, got = baro_bf16_to_f32(hb);
        if (!isnan(want) && memcmp(&want, &got, 4) && !bad++) printf("FAIL helpers: bf16 0x%04x -> f32 mismatch\n", h);
        _Float16 hx; memcpy(&hx, &hb, 2); float hw = (float)hx, hg = baro_f16_to_f32(hb);
        if (!isnan(hw) && memcmp(&hw, &hg, 4) && !bad++) printf("FAIL helpers: f16 0x%04x -> f32 mismatch\n", h);
        if (isnan(hw) && !isnan(hg) && !bad++) printf("FAIL helpers: f16 NaN 0x%04x lost\n", h);
        _Float16 x, y; uint16_t h1 = (uint16_t)(h + 1); memcpy(&x, &hb, 2); memcpy(&y, &h1, 2);
        if (isfinite((float)x) && isfinite((float)y) && (h & 0x7FFF) != 0x7FFF) {
            float m = (float)(((double)x + (double)y) / 2); uint32_t mu; memcpy(&mu, &m, 4); one(mu - 1); one(mu); one(mu + 1);
        }
    }
    if (bad) { printf("FAIL helpers: %ld mismatches\n", bad); return 1; }
    printf("PASS helpers (bf16/f16 integer conversions equal native casts)\n"); return 0;
}
