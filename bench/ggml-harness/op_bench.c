// Run one ggml op on llama.cpp's own HIP backend at a traced shape, cache-cold, so the
// reference arm is timed the way ours is: weights/KV rotate across enough copies that the
// working set exceeds 4x the 96 MB Infinity Cache (see bench/coldcache-protocol.md).
// Wall us/iter is printed; the per-kernel device us come from rocprofv3 (bench/ggml-harness/run.sh).
//
//   op_bench mul_mat    TYPE N K M      [iters]   weight TYPE [K,N] x f32 [K,M]  (M=1 -> mmvq/mmvf, M>1 -> mmq/GEMM)
//   op_bench flash_attn NH NHKV HD KV NQ [iters]  f16 K/V cache of KV tokens, NQ query tokens, scale 1/sqrt(HD)
//   op_bench rms_norm   N ROWS          [iters]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define IC_BYTES (96.0 * 1024 * 1024)
#define MAX_ARMS 256

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

static enum ggml_type type_from(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((enum ggml_type) t);
        if (n && strcmp(n, s) == 0) return (enum ggml_type) t;
    }
    fprintf(stderr, "unknown ggml type %s\n", s);
    exit(2);
}

static void fill(struct ggml_tensor * t) {
    size_t n = ggml_nelements(t);
    float * f = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) f[i] = (float) ((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f, 0, n * sizeof(float));
    } else {
        size_t bytes = ggml_nbytes(t);
        void * q = malloc(bytes);
        ggml_quantize_chunk(t->type, f, q, 0, n / t->ne[0], t->ne[0], NULL);
        ggml_backend_tensor_set(t, q, 0, bytes);
        free(q);
    }
    free(f);
}

static int arms_for(double bytes_per_arm) {
    int a = (int) ceil(4.0 * IC_BYTES / bytes_per_arm);
    return a < 1 ? 1 : a > MAX_ARMS ? MAX_ARMS : a;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: see header of op_bench.c\n"); return 2; }
    const char * op = argv[1];
    int iters = 200;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no hip backend\n"); return 1; }

    struct ggml_init_params ip = { .mem_size = ggml_tensor_overhead() * (8 * MAX_ARMS + 8) + ggml_graph_overhead() * MAX_ARMS, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_cgraph * g[MAX_ARMS];
    struct ggml_tensor * cold[MAX_ARMS][2];
    int ncold = 0, arms = 1;
    double arm_bytes = 0, moved = 0;
    char desc[256];

    if (strcmp(op, "mul_mat") == 0 && argc >= 6) {
        enum ggml_type ty = type_from(argv[2]);
        int n = atoi(argv[3]), k = atoi(argv[4]), m = atoi(argv[5]);
        if (argc > 6) iters = atoi(argv[6]);
        arm_bytes = (double) ggml_row_size(ty, k) * n;
        arms = arms_for(arm_bytes);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        for (int a = 0; a < arms; a++) {
            cold[a][0] = ggml_new_tensor_2d(ctx, ty, k, n);
            g[a] = ggml_new_graph(ctx);
            ggml_build_forward_expand(g[a], ggml_mul_mat(ctx, cold[a][0], x));
        }
        ncold = 1;
        moved = arm_bytes + 4.0 * k * m + 4.0 * n * m;
        snprintf(desc, sizeof desc, "mul_mat %s N=%d K=%d M=%d", ggml_type_name(ty), n, k, m);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        fill(x);
        (void) buf;
    } else if (strcmp(op, "flash_attn") == 0 && argc >= 7) {
        int nh = atoi(argv[2]), nhkv = atoi(argv[3]), hd = atoi(argv[4]), kv = atoi(argv[5]), nq = atoi(argv[6]);
        if (argc > 7) iters = atoi(argv[7]);
        arm_bytes = 2.0 * 2.0 * hd * kv * nhkv;
        arms = arms_for(arm_bytes);
        struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nq, nh);
        for (int a = 0; a < arms; a++) {
            cold[a][0] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, kv, nhkv);
            cold[a][1] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, kv, nhkv);
            g[a] = ggml_new_graph(ctx);
            struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, cold[a][0], cold[a][1], NULL, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            ggml_build_forward_expand(g[a], o);
        }
        ncold = 2;
        moved = arm_bytes + 4.0 * hd * nq * nh * 2;
        snprintf(desc, sizeof desc, "flash_attn NH=%d NHKV=%d HD=%d KV=%d NQ=%d f16-kv", nh, nhkv, hd, kv, nq);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        fill(q);
    } else if (strcmp(op, "rms_norm") == 0 && argc >= 4) {
        int n = atoi(argv[2]), rows = atoi(argv[3]);
        if (argc > 4) iters = atoi(argv[4]);
        arm_bytes = 4.0 * n * rows;
        arms = arms_for(arm_bytes);
        for (int a = 0; a < arms; a++) {
            cold[a][0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
            g[a] = ggml_new_graph(ctx);
            ggml_build_forward_expand(g[a], ggml_rms_norm(ctx, cold[a][0], 1e-6f));
        }
        ncold = 1;
        moved = 2.0 * arm_bytes;
        snprintf(desc, sizeof desc, "rms_norm N=%d ROWS=%d", n, rows);
        ggml_backend_alloc_ctx_tensors(ctx, be);
    } else {
        fprintf(stderr, "bad op or args: see header of op_bench.c\n");
        return 2;
    }
    for (int a = 0; a < arms; a++) for (int c = 0; c < ncold; c++) fill(cold[a][c]);

    // read-back: what was actually built, before any timing
    printf("backend %s\n", ggml_backend_name(be));
    printf("arm %s | bytes/arm %.2f MB | arms %d | rotated %.1f MB (%s the 96 MB Infinity Cache x4)\n",
           desc, arm_bytes / 1e6, arms, arms * arm_bytes / 1e6, arms * arm_bytes >= 4 * IC_BYTES ? "exceeds" : "BELOW, cache-warm, INVALID");
    printf("cold tensor[0] type %s ne [%lld %lld %lld]\n", ggml_type_name(cold[0][0]->type),
           (long long) cold[0][0]->ne[0], (long long) cold[0][0]->ne[1], (long long) cold[0][0]->ne[2]);

    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(be, g[i % arms]);
    ggml_backend_synchronize(be);
    double t0 = now_us();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, g[i % arms]);
    ggml_backend_synchronize(be);
    double us = (now_us() - t0) / iters;
    printf("wall %.2f us/iter  moved %.2f MB  %.0f GB/s wall (device time: rocprofv3, run.sh)\n", us, moved / 1e6, moved / us / 1e3);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
