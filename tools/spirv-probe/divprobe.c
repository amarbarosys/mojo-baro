/* Is float division on the card correctly rounded? Runs x/y, x*(1/y) and a residual-corrected
   quotient for N random pairs plus the q8 tie case, compares bitwise with the CPU's IEEE x/y.
   usage: RUSTICL_ENABLE=radeonsi ./divprobe */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 1000000
static const char *SRC =
    "kernel void d(global const float *x, global const float *y, global float *q, global float *r, global float *c) {"
    "  size_t i = get_global_id(0); q[i] = x[i] / y[i]; r[i] = x[i] * (1.0f / y[i]);"
    "  float t = x[i] / y[i]; c[i] = t + fma(-t, y[i], x[i]) / y[i]; }";
int main(void) {
    cl_platform_id p; cl_device_id devs[8], dev = NULL; cl_uint nd = 0; cl_int err; char name[256];
    if (clGetPlatformIDs(1, &p, NULL) || clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 8, devs, &nd)) { printf("FAIL device\n"); return 1; }
    for (cl_uint i = 0; i < nd; i++) { clGetDeviceInfo(devs[i], CL_DEVICE_NAME, 256, name, NULL); if (strstr(name, "Radeon")) { dev = devs[i]; break; } }
    if (!dev) { printf("FAIL device: no Radeon\n"); return 1; }
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); cl_command_queue cq = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_program pr = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    if (clBuildProgram(pr, 1, &dev, "", NULL, NULL)) { char log[4096]; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 4096, log, NULL); printf("FAIL build: %s\n", log); return 1; }
    cl_kernel k = clCreateKernel(pr, "d", &err);
    float *x = malloc(4 * N), *y = malloc(4 * N), *o[3]; uint32_t s = 99;
    for (int i = 0; i < N; i++) { s = s * 1664525u + 1013904223u; x[i] = (float)(s >> 8) / 65536.0f + 0.001f; s = s * 1664525u + 1013904223u; y[i] = (float)(s >> 8) / 65536.0f + 0.001f; }
    x[0] = 2.453125f; y[0] = 4.90625f / 127.0f;
    cl_mem b[5] = {clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, 4 * N, x, &err), clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, 4 * N, y, &err)};
    for (int j = 2; j < 5; j++) b[j] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * N, NULL, &err);
    for (int j = 0; j < 5; j++) clSetKernelArg(k, j, sizeof(cl_mem), &b[j]);
    size_t g = N; if (clEnqueueNDRangeKernel(cq, k, 1, NULL, &g, NULL, 0, NULL, NULL)) { printf("FAIL enqueue\n"); return 1; }
    const char *lab[3] = {"x/y", "x*(1/y)", "x/y + fma(-q,y,x)/y"};
    for (int j = 0; j < 3; j++) {
        o[j] = malloc(4 * N); if (clEnqueueReadBuffer(cq, b[j + 2], CL_TRUE, 0, 4 * N, o[j], 0, NULL, NULL)) { printf("FAIL read\n"); return 1; }
        long bad = 0; for (int i = 0; i < N; i++) { float w = x[i] / y[i]; bad += memcmp(&w, &o[j][i], 4) != 0; }
        printf("%-22s differs from IEEE in %ld of %d; tie case 2.453125/(4.90625/127) = %.9g (IEEE %.9g)\n", lab[j], bad, N, o[j][0], x[0] / y[0]);
    }
    return 0;
}
