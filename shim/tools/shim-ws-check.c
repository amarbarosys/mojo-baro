// Gate for the shim's shared GEMM workspace: the NN and NT entries tune separately but share one buffer.
// Tune a NN shape whose winner needs workspace, tune a NT shape, then run the NN shape again: the buffer
// must still cover the NN winner (amarbaro_workspace_bytes >= what it held) and the result must not change.
// Last line: PASS, FAIL <step>: <reason> (exit 1), or UNEXERCISED (exit 3: no candidate shape needed workspace).
// build: g++ -O2 -x c++ -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -Ishim/include shim/tools/shim-ws-check.c SHIM_SRC.cpp
//        -o OUT -std=c++17 -L/opt/rocm/lib -lhipblaslt -lamdhip64 -Wl,-rpath,/opt/rocm/lib   (SHIM_SRC = shim/src/amarbaro_shim.cpp or a variant)
// run: gpu-wait run --timeout 300 --vram 4 -- OUT
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amarbaro_shim.h"

struct Shape { int m, n, k; };
static const Shape NN[] = {{16, 512, 16384}, {64, 64, 16384}, {128, 128, 8192}, {32, 1024, 8192}, {256, 256, 4096}};
static const Shape NT = {1024, 1024, 1024};
static uint16_t to_h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float from_h(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

static int fail(const char *step, const char *why) { printf("FAIL %s: %s (shim: %s)\n", step, why, amarbaro_last_error()); return 1; }

int main() {
  amarbaro_ctx *ctx = amarbaro_init(0);
  if (!ctx) return fail("init", "amarbaro_init");
  size_t cap = 0;  // elements: the largest operand over all shapes
  for (const Shape &s : NN) for (size_t e : {(size_t)s.m * s.k, (size_t)s.k * s.n, (size_t)s.m * s.n}) cap = e > cap ? e : cap;
  for (size_t e : {(size_t)NT.m * NT.k, (size_t)NT.k * NT.n, (size_t)NT.m * NT.n}) cap = e > cap ? e : cap;
  std::vector<uint16_t> h(cap), out(cap), first(cap);
  srand(7);
  for (auto &v : h) v = to_h((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
  void *a = amarbaro_device_alloc(cap * 2), *b = amarbaro_device_alloc(cap * 2), *c = amarbaro_device_alloc(cap * 2);
  if (!a || !b || !c) return fail("alloc", "amarbaro_device_alloc");
  if (amarbaro_upload(ctx, a, h.data(), cap * 2) || amarbaro_upload(ctx, b, h.data(), cap * 2)) return fail("upload", "amarbaro_upload");

  Shape nn = {0, 0, 0};
  size_t need_nn = 0;
  for (const Shape &s : NN) {
    if (amarbaro_gemm_f16(ctx, s.m, s.n, s.k, a, b, c, 1.0f, 0.0f) || amarbaro_sync(ctx)) return fail("nn tune", "amarbaro_gemm_f16");
    size_t ws = amarbaro_workspace_bytes(ctx);
    printf("nn  %dx%dx%d splitk=%d wgm=%d workspace=%zu\n", s.m, s.n, s.k, amarbaro_splitk(ctx), amarbaro_wgm(ctx), ws);
    if (ws > 0) { nn = s; need_nn = ws; break; }
  }
  if (!need_nn) { printf("UNEXERCISED: no NN candidate shape needed workspace\n"); return 3; }
  if (amarbaro_download(ctx, first.data(), c, (size_t)nn.m * nn.n * 2)) return fail("download", "first NN result");

  if (amarbaro_gemm_f16_nt(ctx, NT.m, NT.n, NT.k, a, b, c, 1.0f, 0.0f) || amarbaro_sync(ctx)) return fail("nt tune", "amarbaro_gemm_f16_nt");
  size_t ws_after_nt = amarbaro_workspace_bytes(ctx);
  printf("nt  %dx%dx%d splitk=%d wgm=%d workspace=%zu\n", NT.m, NT.n, NT.k, amarbaro_splitk(ctx), amarbaro_wgm(ctx), ws_after_nt);
  if (ws_after_nt < need_nn) printf("note: workspace %zu < NN winner's %zu after the NT tune\n", ws_after_nt, need_nn);

  if (amarbaro_gemm_f16(ctx, nn.m, nn.n, nn.k, a, b, c, 1.0f, 0.0f)) return fail("nn rerun", "amarbaro_gemm_f16 on the cached NN shape");
  if (amarbaro_sync(ctx) || amarbaro_download(ctx, out.data(), c, (size_t)nn.m * nn.n * 2)) return fail("nn rerun", "sync or download");
  double worst = 0;
  for (size_t i = 0; i < (size_t)nn.m * nn.n; i++) {
    double x = from_h(out[i]), y = from_h(first[i]);
    double d = fabs(x - y) / (fabs(y) + 1e-3);
    if (!(d <= worst)) worst = d;  // NaN propagates
  }
  printf("nn rerun max rel diff %.3e, workspace %zu (NN winner %zu)\n", worst, amarbaro_workspace_bytes(ctx), need_nn);
  if (amarbaro_workspace_bytes(ctx) < need_nn) return fail("workspace", "buffer smaller than the cached NN winner needs");
  if (!(worst < 1e-2)) return fail("result", "NN result changed after the NT tune");
  printf("PASS\n");
  amarbaro_device_free(a); amarbaro_device_free(b); amarbaro_device_free(c);
  amarbaro_destroy(ctx);
  return 0;
}
