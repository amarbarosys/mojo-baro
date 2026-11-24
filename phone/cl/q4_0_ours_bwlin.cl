// Read-bandwidth probe: grid-stride linear stream of the q bytes, 4 independent uint4 loads in flight per lane.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemv(__global const uint4 *q, __global const half *d, __global const float *x, __global float *y, int K,
                   __local float *red) {
  size_t g = get_global_id(0), T = get_global_size(0), n = T / 64 * 4 * (size_t)(K / 32);
  uint4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (size_t i = g; i + 3 * T < n; i += 4 * T) { a0 ^= q[i]; a1 ^= q[i + T]; a2 ^= q[i + 2 * T]; a3 ^= q[i + 3 * T]; }
  uint4 a = a0 ^ a1 ^ a2 ^ a3;
  y[g & 4095] = (float)(a.x ^ a.y ^ a.z ^ a.w);
}
