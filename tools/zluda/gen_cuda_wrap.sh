#!/usr/bin/env bash
# Generate a pass-through libcuda.so.1 over ZLUDA: every exported cu* symbol jumps to ZLUDA's
# libnvcuda.so; cuFuncSetAttribute(NON_PORTABLE_CLUSTER_SIZE_ALLOWED) reports success (MAX sets it
# after every module load, ZLUDA returns INVALID_VALUE). usage: gen_cuda_wrap.sh ZLUDA_DIR OUT_C
set -euo pipefail
z=$1; out=$2
real=$(readlink -f "$z/libcuda.so.1")
mapfile -t syms < <(nm -D --defined-only "$real" | awk '$2 ~ /[TW]/ && $3 ~ /^cu/ {print $3}' | sort -u | grep -vxE 'cuFuncSetAttribute|cuGetProcAddress|cuGetProcAddress_v2')
[[ ${#syms[@]} -gt 100 ]] || { echo "FAIL gen: only ${#syms[@]} symbols in $real"; exit 1; }
{
  echo '#define _GNU_SOURCE'
  echo '#include <dlfcn.h>'
  echo '#include <stdio.h>'
  echo '#include <stdlib.h>'
  echo '#include <string.h>'
  echo "static const char *REAL = \"$real\";"
  echo "#define N ${#syms[@]}"
  printf 'static const char *names[N] = {'; printf '"%s",' "${syms[@]}"; echo '};'
  echo '__attribute__((visibility("hidden"))) void *slots[N];'
  echo 'static void *h;'
  echo 'static void *real_sym(const char *s) { if (!h) { h = dlopen(REAL, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "cuda-wrap: %s\n", dlerror()); abort(); } } return dlsym(h, s); }'
  echo '__attribute__((constructor)) static void init(void) { for (int i = 0; i < N; i++) slots[i] = real_sym(names[i]); }'
  for i in "${!syms[@]}"; do
    s=${syms[$i]}
    echo "__asm__(\".globl $s\n.type $s,@function\n$s:\n  jmp *slots+$((i * 8))(%rip)\n\");"
  done
  cat <<'C'
typedef int (*fsa_t)(void *, int, int);
int cuFuncSetAttribute(void *f, int attr, int v) {
  static fsa_t r; if (!r) r = (fsa_t)real_sym("cuFuncSetAttribute");
  int rc = r(f, attr, v);
  if (rc == 1 && attr == 14) return 0;
  return rc;
}
typedef int (*gpa_t)(const char *, void **, int, unsigned long long);
typedef int (*gpa2_t)(const char *, void **, int, unsigned long long, int *);
static void fix(const char *s, void **p) { if (*p && strcmp(s, "cuFuncSetAttribute") == 0) *p = (void *)cuFuncSetAttribute; }
int cuGetProcAddress(const char *s, void **p, int ver, unsigned long long fl) {
  static gpa_t r; if (!r) r = (gpa_t)real_sym("cuGetProcAddress");
  int rc = r(s, p, ver, fl); if (!rc) fix(s, p); return rc;
}
int cuGetProcAddress_v2(const char *s, void **p, int ver, unsigned long long fl, int *st) {
  static gpa2_t r; if (!r) r = (gpa2_t)real_sym("cuGetProcAddress_v2");
  int rc = r(s, p, ver, fl, st); if (!rc) fix(s, p); return rc;
}
C
} > "$out"
