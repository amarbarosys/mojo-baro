// Read-bandwidth probe, image path: the q bytes as an rgba32ui image1d_buffer (16 B texels), same walk as q4_0_ours_bw.cl.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemv(__read_only image1d_buffer_t q, __global const half *d, __global const float *x, __global float *y,
                   int K, __local float *red) {
  int r0 = get_group_id(0) * 4, l = get_local_id(0), nb = K / 32;
  uint4 acc = 0;
  for (int i = 0; i < 4; i++)
    for (int b = l; b < nb; b += 64) acc ^= read_imageui(q, (r0 + i) * nb + b);
  y[get_global_id(0) & 4095] = (float)(acc.x ^ acc.y ^ acc.z ^ acc.w);
}
