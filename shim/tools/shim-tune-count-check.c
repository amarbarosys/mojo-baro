// Audit row C15: amarbaro_tune_count says when a shape change re-tuned. Same shape twice: one tune; a second shape:
// two; back to the first shape: three (the single cached slot is overwritten, so an alternating caller re-tunes every
// call). Last line: PASS or FAIL <step>: <reason> (exit 1).
// build: g++ -O2 -x c++ -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -Ishim/include shim/tools/shim-tune-count-check.c shim/src/amarbaro_shim.cpp
//        -o OUT -std=c++17 -L/opt/rocm/lib -lhipblaslt -lamdhip64 -Wl,-rpath,/opt/rocm/lib
// run: gpu-wait run --timeout 300 --vram 4 -- OUT
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "amarbaro_shim.h"

static int fail(const char *step, const char *why) { printf("FAIL %s: %s (shim: %s)\n", step, why, amarbaro_last_error()); return 1; }

int main() {
  if (amarbaro_tune_count(nullptr) != -1) return fail("null", "a null context must answer -1");
  amarbaro_ctx *ctx = amarbaro_init(0);
  if (!ctx) return fail("init", "amarbaro_init");
  if (amarbaro_tune_count(ctx) != 0) return fail("fresh", "a fresh context has tuned nothing");
  const size_t cap = (size_t)256 * 4096;  // elements: covers every operand of the two shapes below
  std::vector<unsigned short> h(cap, 0x2c00);
  void *a = amarbaro_device_alloc(cap * 2), *b = amarbaro_device_alloc(cap * 2), *c = amarbaro_device_alloc(cap * 2);
  if (!a || !b || !c) return fail("alloc", "amarbaro_device_alloc");
  if (amarbaro_upload(ctx, a, h.data(), cap * 2) || amarbaro_upload(ctx, b, h.data(), cap * 2)) return fail("upload", "amarbaro_upload");
  struct { int m, n, k; int want; } steps[] = {{64, 64, 1024, 1}, {64, 64, 1024, 1}, {128, 128, 512, 2}, {64, 64, 1024, 3}};
  for (auto &s : steps) {
    if (amarbaro_gemm_f16(ctx, s.m, s.n, s.k, a, b, c, 1.0f, 0.0f) || amarbaro_sync(ctx)) return fail("gemm", "amarbaro_gemm_f16");
    int got = amarbaro_tune_count(ctx);
    printf("%dx%dx%d tune_count=%d (want %d)\n", s.m, s.n, s.k, got, s.want);
    if (got != s.want) return fail("count", "tune_count does not follow the shape changes");
  }
  printf("PASS\n");
  amarbaro_device_free(a); amarbaro_device_free(b); amarbaro_device_free(c);
  amarbaro_destroy(ctx);
  return 0;
}
