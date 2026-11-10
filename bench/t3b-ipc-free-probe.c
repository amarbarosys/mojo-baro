// T3b leak probe: does hipFree return an allocation that was exported with hipIpcGetMemHandle?
// The T3b gate (job mv5fz2iirds4) saw device VRAM rise one stage size (581 MB) per export and never
// fall, with the engine's live-stage count back at 0. Arms, 5 rounds each, amdgpu sysfs used after:
//   plain   hipMalloc + hipFree
//   ipc     hipMalloc + hipIpcGetMemHandle + hipFree
// argv[1] = rounds (default 5); 45 rounds push 26 GB through the 24 GB card: a real leak fails hipMalloc.
// build: gcc -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include bench/t3b-ipc-free-probe.c -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib -o .work/t3b/ipc-free-probe
#include <hip/hip_runtime_api.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long vram_used(void) {
    glob_t g;
    long long best_total = -1, used = -1;
    if (glob("/sys/class/drm/card[0-9]/device/mem_info_vram_total", 0, NULL, &g) != 0) return -1;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        long long t = 0, u = 0;
        char p[512];
        FILE *f = fopen(g.gl_pathv[i], "r");
        if (!f || fscanf(f, "%lld", &t) != 1) { if (f) fclose(f); continue; }
        fclose(f);
        snprintf(p, sizeof p, "%.*sused", (int)(strlen(g.gl_pathv[i]) - 5), g.gl_pathv[i]);
        f = fopen(p, "r");
        if (!f || fscanf(f, "%lld", &u) != 1) { if (f) fclose(f); continue; }
        fclose(f);
        if (t > best_total) { best_total = t; used = u; }
    }
    globfree(&g);
    return used;
}

int main(int argc, char **argv) {
    const size_t bytes = 581173248;
    const int rounds = argc > 1 ? atoi(argv[1]) : 5;
    if (hipSetDevice(0) != hipSuccess) { printf("FAIL setup: hipSetDevice\n"); return 1; }
    void *warm;
    hipMalloc(&warm, 1 << 20);
    hipFree(warm);
    long long base = vram_used();
    printf("base used=%lld\n", base);
    for (int arm = 0; arm < 2; arm++) {
        long long start = vram_used();
        for (int r = 0; r < rounds; r++) {
            void *p = NULL;
            if (hipMalloc(&p, bytes) != hipSuccess) { printf("FAIL %s round %d: hipMalloc\n", arm ? "ipc" : "plain", r); return 1; }
            if (arm) {
                hipIpcMemHandle_t h;
                if (hipIpcGetMemHandle(&h, p) != hipSuccess) { printf("FAIL ipc round %d: hipIpcGetMemHandle\n", r); return 1; }
            }
            long long held = vram_used();
            if (hipFree(p) != hipSuccess) { printf("FAIL %s round %d: hipFree\n", arm ? "ipc" : "plain", r); return 1; }
            hipDeviceSynchronize();
            long long after = vram_used();
            printf("%s round %d held=%+.3f GB after_free=%+.3f GB\n", arm ? "ipc" : "plain", r, (held - start) / 1e9, (after - start) / 1e9);
        }
        printf("%s arm: net %+.3f GB after %d rounds\n", arm ? "ipc" : "plain", (vram_used() - start) / 1e9, rounds);
    }
    return 0;
}
