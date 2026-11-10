/* Narrow-float conversions in integer ops: the R5 M330 has no half support and
   SPIR-V has no bfloat16, so bf16/f16 buffers are i16 on the card. Round to
   nearest even, same as fptrunc. Compiled to LLVM IR and spliced into each
   kernel module by air2spv.py (no llvm-link on the build host). */
static unsigned bits(float f) { unsigned u; __builtin_memcpy(&u, &f, 4); return u; }

float baro_bf16_to_f32(unsigned short b) {
    unsigned u = (unsigned)b << 16; float f; __builtin_memcpy(&f, &u, 4); return f;
}

float baro_f16_to_f32(unsigned short h) {
    unsigned sign = (unsigned)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, u;
    if (e == 0x1Fu) u = sign | 0x7F800000u | (m << 13);
    else if (e) u = sign | ((e + 112u) << 23) | (m << 13);
    else if (!m) u = sign;
    else { unsigned s = (unsigned)__builtin_clz(m) - 21u; u = sign | ((113u - s) << 23) | ((m << (13u + s)) & 0x7FFFFFu); }
    float f; __builtin_memcpy(&f, &u, 4); return f;
}

unsigned short baro_f32_to_bf16(float f) {
    unsigned u = bits(f), hi = u >> 16, lo = u & 0xFFFFu;
    if ((u & 0x7F800000u) == 0x7F800000u) return (unsigned short)(hi | ((u & 0x007FFFFFu) && !(hi & 0x7Fu)));
    return (unsigned short)(hi + (lo > 0x8000u || (lo == 0x8000u && (hi & 1u))));
}

unsigned short baro_f32_to_f16(float f) {
    unsigned u = bits(f), sign = (u >> 16) & 0x8000u, a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return (unsigned short)(sign | 0x7C00u | (a > 0x7F800000u ? 0x200u : 0u));
    if (a >= 0x477FF000u) return (unsigned short)(sign | 0x7C00u);
    if (a <= 0x33000000u) return (unsigned short)sign;
    unsigned e = a >> 23, m = (a & 0x7FFFFFu) | 0x800000u;
    unsigned shift = e >= 113u ? 13u : 126u - e;
    unsigned h = e >= 113u ? (((e - 112u) << 10) | ((m >> 13) & 0x3FFu)) : (m >> shift);
    unsigned rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    h += rem > half || (rem == half && (h & 1u));
    return (unsigned short)(sign | h);
}
