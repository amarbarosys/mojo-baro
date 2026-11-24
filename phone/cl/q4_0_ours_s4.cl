// Q4_0 GEMV, flat layout (q: 16 B per block, d: one half per block), R rows per 64-lane group (one subgroup):
// each lane takes blocks l, l+64, ... of all R rows, x for a block loaded once and used R times (as q4_0_ours_r4.cl),
// then one sub_group_reduce_add per row instead of the local-memory tree and its 6 barriers.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifndef R
#define R 4
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
  for (int b = l; b < nb; b += 64) {
    float16 xl = vload16(0, x + 32 * b), xh = vload16(1, x + 32 * b);
    for (int i = 0; i < R; i++) {
      size_t blk = (size_t)(r0 + i) * nb + b;
      uchar16 v = vload16(0, q + 16 * blk);
      float16 p = (convert_float16(v & (uchar)15) - 8.0f) * xl + (convert_float16(v >> (uchar)4) - 8.0f) * xh;
      float8 p8 = p.lo + p.hi; float4 p4 = p8.lo + p8.hi; float2 p2 = p4.lo + p4.hi;
      acc[i] = mad(p2.x + p2.y, vload_half(0, d + blk), acc[i]);
    }
  }
  for (int i = 0; i < R; i++) {
    float s = sub_group_reduce_add(acc[i]);
    if (l == i) y[r0 + i] = s;
  }
}
