// q4_0_ours_s4.cl with two adjacent blocks per lane per step: 32 B of q (one uchar16x2 load pair) and a half2 of
// scales per row, x for both blocks loaded once and used R times; one sub_group_reduce_add per row.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifndef R
#define R 8
#endif
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
#endif
__kernel void gemv(__global const uchar *q, __global const half *d, __global const float *x, __global float *y, int K,
                   __local float *red) {
  int r0 = get_group_id(0) * R, l = get_local_id(0), nb = K / 32;
  float acc[R];
  for (int i = 0; i < R; i++) acc[i] = 0.0f;
  for (int b = 2 * l; b < nb; b += 128) {
    float16 x0 = vload16(0, x + 32 * b), x1 = vload16(1, x + 32 * b), x2 = vload16(2, x + 32 * b), x3 = vload16(3, x + 32 * b);
    for (int i = 0; i < R; i++) {
      size_t blk = (size_t)(r0 + i) * nb + b;
      uchar16 v = vload16(0, q + 16 * blk), w = vload16(1, q + 16 * blk);
      float2 s = vload_half2(0, d + blk);
      float16 p = (convert_float16(v & (uchar)15) - 8.0f) * x0 + (convert_float16(v >> (uchar)4) - 8.0f) * x1;
      float16 t = (convert_float16(w & (uchar)15) - 8.0f) * x2 + (convert_float16(w >> (uchar)4) - 8.0f) * x3;
      float8 p8 = p.lo + p.hi; float4 p4 = p8.lo + p8.hi; float2 p2 = p4.lo + p4.hi;
      float8 t8 = t.lo + t.hi; float4 t4 = t8.lo + t8.hi; float2 t2 = t4.lo + t4.hi;
      acc[i] = mad(t2.x + t2.y, s.y, mad(p2.x + p2.y, s.x, acc[i]));
    }
  }
  for (int i = 0; i < R; i++) {
    float s = sub_group_reduce_add(acc[i]);
    if (l == i) y[r0 + i] = s;
  }
}
