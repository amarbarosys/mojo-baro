#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void *(*init_fn)(void);
typedef void (*fini_fn)(void *);
#define NG 64
static struct { char name[64]; void *val; fini_fn fini; } g[NG];
static int ng;
void *KGEN_CompilerRT_GetOrCreateGlobal(const char *name, size_t len, init_fn init, fini_fn fini) {
  for (int i = 0; i < ng; i++)
    if (strlen(g[i].name) == len && !memcmp(g[i].name, name, len)) return g[i].val;
  if (ng == NG || len >= 64) abort();
  memcpy(g[ng].name, name, len); g[ng].name[len] = 0;
  g[ng].val = init(); g[ng].fini = fini;
  return g[ng++].val;
}
void KGEN_CompilerRT_DestroyGlobals(void) {
  while (ng) { ng--; if (g[ng].fini) g[ng].fini(g[ng].val); }
}
static int argc_; static char **argv_;
void KGEN_CompilerRT_SetArgV(int c, char **v) { argc_ = c; argv_ = v; }
void KGEN_CompilerRT_PrintStackTraceOnFault(void) {}
int KGEN_CompilerRT_fprintf(FILE *f, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt); int r = vfprintf(f, fmt, ap); va_end(ap); return r;
}
void *KGEN_CompilerRT_AlignedAlloc(size_t align, size_t size) {
  void *p = 0; if (align < sizeof(void *)) align = sizeof(void *);
  return posix_memalign(&p, align, size) ? 0 : p;
}
void KGEN_CompilerRT_AlignedFree(void *p) { free(p); }
static char cpu_dev;
void *KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice(long n) { return &cpu_dev; }
void *KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice(void) { return &cpu_dev; }
void KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice(void *d) {}
