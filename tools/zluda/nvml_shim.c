/* Minimal libnvidia-ml.so.1 (NVML) for running MAX CUDA builds under ZLUDA on an AMD GPU.
   One device; PCI location, compute capability and memory come from the loaded libcuda (ZLUDA),
   so MAX's NVML <-> CUDA PCI match holds. Covers exactly the calls in MAX's NVML.inc. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int (*attr_fn)(int *, int, int);
typedef int (*mem_fn)(size_t *, int);
typedef struct { char busIdLegacy[16]; unsigned domain, bus, device, pciDeviceId, pciSubSystemId; char busId[32]; } pci_t;
typedef struct { unsigned long long total, free, used; } mem_t;

static void *cu(void) { static void *h; if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD); if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY); return h ? h : RTLD_DEFAULT; }

static int attr(int a) {
  static attr_fn f;
  if (!f) f = (attr_fn)dlsym(cu(), "cuDeviceGetAttribute");
  int v = 0;
  if (f) f(&v, a, 0);
  return v;
}

int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
const char *nvmlErrorString(int r) { (void)r; return "nvml-shim error"; }
int nvmlSystemGetDriverVersion(char *v, int n) { snprintf(v, n, "zluda-shim"); return 0; }
int nvmlDeviceGetCount_v2(unsigned *c) { *c = 1; return 0; }
int nvmlDeviceGetHandleByIndex(unsigned i, void **d) { if (i) return 2; *d = (void *)1; return 0; }
int nvmlDeviceGetName(void *d, char *s, unsigned n) { (void)d; snprintf(s, n, "ZLUDA AMD GPU"); return 0; }
int nvmlDeviceGetPciInfo_v3(void *d, pci_t *p) {
  (void)d; memset(p, 0, sizeof *p);
  p->domain = attr(50); p->bus = attr(33); p->device = attr(34);
  snprintf(p->busId, 32, "%08x:%02x:%02x.0", p->domain, p->bus, p->device);
  snprintf(p->busIdLegacy, 16, "%04x:%02x:%02x.0", p->domain, p->bus, p->device);
  return 0;
}
int nvmlDeviceGetCudaComputeCapability(void *d, int *ma, int *mi) { (void)d; *ma = attr(75); *mi = attr(76); return 0; }
int nvmlDeviceGetMemoryInfo(void *d, mem_t *m) {
  (void)d; size_t t = 0;
  mem_fn f = (mem_fn)dlsym(cu(), "cuDeviceTotalMem_v2");
  if (f) f(&t, 0);
  m->total = t; m->free = t; m->used = 0;
  return 0;
}
int nvmlDeviceGetCurrentClocksThrottleReasons(void *d, unsigned long long *r) { (void)d; *r = 0; return 0; }
int nvmlDeviceGetTotalEccErrors(void *d, int a, int b, unsigned long long *c) { (void)d; (void)a; (void)b; *c = 0; return 0; }
int nvmlDeviceGetComputeRunningProcesses_v3(void *d, unsigned *n, void *p) { (void)d; (void)p; *n = 0; return 0; }
int nvmlDeviceGetMigMode(void *d, unsigned *c, unsigned *p) { (void)d; *c = 0; *p = 0; return 0; }
