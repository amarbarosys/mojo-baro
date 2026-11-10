// Run ggml's own Q8_0 mul_mat (quantize_q8_1 + mul_mat_vec_q) on the Spark m=1 GEMV shapes
// through the public ggml backend API, so the kernels, geometry and stream are exactly
// llama.cpp's. Wall us/iter printed; per-kernel us come from rocprofv3 around this binary.
// build: bench/ggml-mmvq/build.sh   run: bench/ggml-mmvq/mmvq_shapes [iters]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    struct { const char * name; int n, k; } shapes[] = {
        {"headgate N16 K2560", 16, 2560}, {"o N2560 K4096", 2560, 4096}, {"qkv N6144 K2560", 6144, 2560},
        {"ffn_gate N10240 K2560", 10240, 2560}, {"down N2560 K10240", 2560, 10240}, {"lmhead N131072 K2560", 131072, 2560},
    };
    const int NARMS = 8;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no cuda/hip backend\n"); return 1; }
    printf("backend %s\n", ggml_backend_name(be));
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int n = shapes[s].n, k = shapes[s].k;
        int arms = (n == 131072) ? 1 : NARMS;  // lm-head alone is 357 MB
        struct ggml_init_params ip = { .mem_size = ggml_tensor_overhead() * (4 * NARMS + 4) + ggml_graph_overhead() * NARMS, .mem_buffer = NULL, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w[8]; struct ggml_cgraph * gfs[8];
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
        for (int a = 0; a < arms; a++) {
            w[a] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);
            struct ggml_tensor * y = ggml_mul_mat(ctx, w[a], x);
            gfs[a] = ggml_new_graph(ctx);
            ggml_build_forward_expand(gfs[a], y);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        float * wf = malloc((size_t) n * k * sizeof(float));
        for (size_t i = 0; i < (size_t) n * k; i++) wf[i] = (float) ((i * 2654435761u) % 2001) / 1000.0f - 1.0f;
        size_t qbytes = ggml_row_size(GGML_TYPE_Q8_0, k) * n;
        void * wq = malloc(qbytes);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf, wq, 0, n, k, NULL);
        for (int a = 0; a < arms; a++) ggml_backend_tensor_set(w[a], wq, 0, qbytes);
        float * xf = malloc(k * sizeof(float));
        for (int i = 0; i < k; i++) xf[i] = (float) ((i * 40503u) % 1001) / 500.0f - 1.0f;
        ggml_backend_tensor_set(x, xf, 0, k * sizeof(float));
        for (int i = 0; i < 20; i++) ggml_backend_graph_compute(be, gfs[i % arms]);
        ggml_backend_synchronize(be);
        double t0 = now_us();
        for (int i = 0; i < iters; i++) ggml_backend_graph_compute(be, gfs[i % arms]);
        ggml_backend_synchronize(be);
        double us = (now_us() - t0) / iters;
        double mb = qbytes / 1e6;
        printf("%-24s wall %8.2f us/iter  weight %7.2f MB x%d arms  %6.0f GB/s\n", shapes[s].name, us, mb, arms, mb / us * 1e3);
        free(wf); free(wq); free(xf);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
