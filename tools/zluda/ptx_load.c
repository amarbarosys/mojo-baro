#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
static char *slurp(const char *p) { FILE *f = fopen(p, "rb"); if (!f) exit(2); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f); char *b = malloc(n + 1); fread(b, 1, n, f); b[n] = 0; fclose(f); return b; }
#define R(x) do { CUresult r = (x); const char *s = 0; cuGetErrorName(r, &s); printf("%-34s -> %d %s\n", #x, r, s ? s : "?"); } while (0)
int main(int c, char **v) {
  CUdevice dev; CUcontext ctx; CUmodule m = 0; CUfunction fn;
  setvbuf(stdout, 0, _IONBF, 0);
  R(cuInit(0)); R(cuDeviceGet(&dev, 0)); R(cuDevicePrimaryCtxRetain(&ctx, dev)); R(cuCtxSetCurrent(ctx));
  char *ptx = slurp(v[1]);
  R(cuModuleLoadDataEx(&m, ptx, 0, 0, 0));
  if (!m) return 3;
  R(cuModuleGetFunction(&fn, m, v[2]));
  R(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  R(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
  return 0;
}
