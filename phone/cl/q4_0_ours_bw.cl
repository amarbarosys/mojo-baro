// Read-bandwidth probes for the Q4_0 GEMV roofline on Adreno 840: same signature as the ours_* GEMV arms
// (q, d, x, y, K, local) so race.sh drives them; they stream the q buffer (M*K/2 bytes) and write one float per lane.
// gemv: uint4 loads, lane-contiguous 16 B (coalesced 1 KB per 64-lane wave), 4 rows of blocks per group as ours_r4.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemv(__global const uint4 *q, __global const half *d, __global const float *x, __global float *y, int K,
                   __local float *red) {
  int r0 = get_group_id(0) * 4, l = get_local_id(0), nb = K / 32;
  uint4 acc = 0;
  for (int i = 0; i < 4; i++)
    for (int b = l; b < nb; b += 64) acc ^= q[(size_t)(r0 + i) * nb + b];
  y[get_global_id(0) & 4095] = (float)(acc.x ^ acc.y ^ acc.z ^ acc.w);
}
