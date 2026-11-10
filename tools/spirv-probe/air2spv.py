"""Mojo Metal IR (air64) -> OpenCL-flavoured LLVM IR for `clang --target=spirv64`.

usage: air2spv.py <helpers.ll> <out-dir> <kernel.ll>...
Writes <out-dir>/<amar_name>.ll per input, prints one `name phases=1 barriers=N` line each.
Exits non-zero with `FAIL <kernel>: <construct>` on anything it has no lowering for.

One OpenCL kernel per Mojo kernel, no host-dispatched phases:
  thread position args      -> get_group_id / get_local_id / get_local_size / get_global_size
  thread_index_in_simdgroup -> get_local_id(0) % 32   (the kernels already assume lane = tid % WARP_SIZE)
  addrspace(3) shared mem   -> OpenCL local memory (same address space number, kept as is)
  air.wg.barrier            -> barrier(CLK_LOCAL_MEM_FENCE)
  air.simd_shuffle_xor(v,m) -> store v to local scratch[lid]; barrier; load scratch[lid ^ m]; barrier
  bfloat / half             -> i16 buffers + integer conversion helpers (helpers.c)
  air.<math>, air.convert   -> OpenCL builtins, sitofp / fptosi
  fdiv float                -> q + fma(-q, y, x) / y: the card divides as x * (1/y), 29% of quotients are not
                               IEEE (divprobe.c); one residual step restores correct rounding (0 of 1e6 differ)
"""
import re, struct, sys

KERNELS = ["amar_rmsnorm", "amar_rmsnorm_cast", "amar_rmsnorm_cast2", "amar_swiglu", "amar_rope_rows",
           "amar_softmax_rows", "amar_embed_lookup", "amar_embed_lookup_pos", "amar_argmax_pos",
           "amar_argmax_row", "amar_tok_copy", "amar_tok_remap", "amar_quantize_q8_rows",
           "amar_matmul_skinny_q4rowb", "amar_skinny_reduce"]
WARP = 32
LOCAL_MAX = 256
HEADER = ['target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"',
          'target triple = "spirv64-unknown-unknown"']
IDS = {"threadgroup_position_in_grid": "_Z12get_group_idEj", "thread_position_in_threadgroup": "_Z12get_local_idEj",
       "threads_per_threadgroup": "_Z14get_local_sizeEj", "threads_per_grid": "_Z15get_global_sizeEj"}
MATH = {"exp2": 1, "rsqrt": 1, "cos": 1, "sin": 1, "log2": 1, "sqrt": 1, "fabs": 1, "fmax": 2, "fmin": 2, "fma": 3}
LLTYPE = {"f32": "float", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64"}
NARROW = {"bfloat": "bf16", "half": "f16"}
BARRIER = "  call spir_func void @_Z7barrierj(i32 1)"


class Unsupported(Exception):
    pass


def kernel_name(mangled):
    stem = re.sub(r"^.*?(?=amar_)", "", mangled)
    stem = re.sub(r"_[0-9a-f]{16}$", "", stem)
    stem = re.sub(r"([0-9a-z][A-Z])+$", "", stem)
    stem = re.sub(r"_T[ensor]*$", "", stem)
    if stem in KERNELS:
        return stem
    hits = [k for k in KERNELS if k.startswith(stem)]
    if len(hits) != 1:
        raise Unsupported(f"cannot resolve kernel name from '{mangled}' (stem '{stem}', candidates {hits})")
    return hits[0]


def lltype(t):
    m = re.fullmatch(r"v(\d+)(\w+)", t)
    base = LLTYPE.get(m.group(2) if m else t)
    if not base:
        raise Unsupported(f"type {t}")
    return f"<{m.group(1)} x {base}>" if m else base


def builtin(name, t):
    m = re.fullmatch(r"v(\d+)f32", t)
    sig = (f"Dv{m.group(1)}_f" + "S_" * (MATH[name] - 1)) if m else "f" * MATH[name]
    return f"_Z{len(name)}{name}{sig}"


def f0x(m):
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", struct.unpack(">f", bytes.fromhex(m.group(1)))[0]))[0]


SPECIAL = {"+qnan": "0x7FF8000000000000", "-qnan": "0xFFF8000000000000", "+inf": "0x7FF0000000000000", "-inf": "0xFFF0000000000000"}


def dec(m):
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", struct.unpack("<f", struct.pack("<f", float(m.group(0))))[0]))[0]


def literals(l):
    if not re.search(r"\d\.\d+e[+-]\d+|[+-]qnan|[+-]inf\b|f0x", l):
        return l
    if "double" in l:
        raise Unsupported(f"float literal on a double line: {l.strip()}")
    l = re.sub(r"f0x([0-9A-F]{8})", f0x, l)
    l = re.sub(r"(?<![\w.])-?\d+\.\d+e[+-]\d+", dec, l)
    return re.sub(r"(?<= )[+-](?:qnan|inf)\b", lambda m: SPECIAL[m.group(0)], l)


def convert(src_text):
    fn = re.search(r"^define void @(\S+?)\((.*?)\) local_unnamed_addr #\d+ \{\n(.*?)^\}", src_text, re.S | re.M)
    if not fn:
        raise Unsupported("no kernel define found")
    name = kernel_name(fn.group(1))
    ptr_args = re.findall(r"ptr addrspace\((\d)\)[^%]*%(\d+)", fn.group(2))
    shared = re.findall(r"^(@\S+ = internal addrspace\(3\) global .*)$", src_text, re.M)

    body = []
    for l in fn.group(3).splitlines():
        l = re.sub(r"\s*; preds.*$", "", l)
        l = re.sub(r", !\w+ !\d+", "", l)
        l = re.sub(r" #\d+$", "", l)
        l = re.sub(r"%(\d+)\b", r"%v\1", l)
        l = re.sub(r"^(\d+):", r"v\1:", l)
        body.append(l)

    out, alias, decls, shuf_types, barriers = [], {}, set(), set(), 0
    for l in body:
        m = re.match(r"\s*(%v\d+) = load \{ ptr addrspace\(1\).*\}, ptr addrspace\(1\) (%v\d+)", l)
        if m:
            alias[m.group(1)] = m.group(2)
            continue
        m = re.match(r"\s*(%v\d+) = extractvalue \{ ptr addrspace\(1\).*\} (%v\d+), 0", l)
        if m:
            out.append(f"  {m.group(1)} = getelementptr i8, ptr addrspace(1) {alias[m.group(2)]}, i64 0")
            continue
        if "@air.wg.barrier" in l:
            out.append(BARRIER)
            barriers += 1
            continue
        m = re.match(r"\s*(%v\d+) = call (\w+) @air\.simd_shuffle_xor\.(\w+)\.i16\(\w+ (\S+), i16 (\d+)\)", l)
        if m:
            r, ty, tn, val, mask = m.groups()
            if tn not in LLTYPE or int(mask) >= WARP:
                raise Unsupported(f"simd_shuffle_xor type {tn} mask {mask}")
            shuf_types.add(tn)
            g = f"ptr addrspace(3) @baro.shuf.{tn}"
            out += [f"  {r}.p = getelementptr {ty}, {g}, i64 %baro.lid", f"  store {ty} {val}, ptr addrspace(3) {r}.p", BARRIER,
                    f"  {r}.x = xor i64 %baro.lid, {mask}", f"  {r}.q = getelementptr {ty}, {g}, i64 {r}.x",
                    f"  {r} = load {ty}, ptr addrspace(3) {r}.q", BARRIER]
            barriers += 2
            continue
        m = re.match(r"(\s*%v\d+ = )call [^@]+@air\.convert\.([fsu])\.(\w+)\.([fsu])\.(\w+)\((<[^>]+>|\w+) (\S+)\)", l)
        if m:
            lhs, dk, dt, sk, st, sty, val = m.groups()
            op = {("f", "s"): "sitofp", ("f", "u"): "uitofp", ("s", "f"): "fptosi", ("u", "f"): "fptoui"}.get((dk, sk))
            if not op:
                raise Unsupported(f"air.convert {dk}.{dt} <- {sk}.{st}")
            out.append(f"{lhs}{op} {lltype(st)} {val} to {lltype(dt)}")
            continue
        m = re.match(r"\s*(%v\d+) = fdiv (?:\w+ )*float (\S+), (\S+)$", l)
        if m:
            r, a, b = m.groups()
            decls.add("declare spir_func float @_Z3fmafff(float, float, float)")
            out += [literals(x) for x in [
                f"  {r}.q = fdiv float {a}, {b}", f"  {r}.n = fneg float {r}.q",
                f"  {r}.r = call spir_func float @_Z3fmafff(float {r}.n, float {b}, float {a})",
                f"  {r}.d = fdiv float {r}.r, {b}", f"  {r}.c = fadd float {r}.q, {r}.d",
                f"  {r}.o1 = fcmp one float {r}.c, 0x7FF0000000000000", f"  {r}.o2 = fcmp one float {r}.c, 0xFFF0000000000000",
                f"  {r}.ok = and i1 {r}.o1, {r}.o2", f"  {r} = select i1 {r}.ok, float {r}.c, float {r}.q"]]
            continue
        m = re.search(r"call (?:\w+ )*?(float|<\d+ x float>) @(?:air|llvm)\.(\w+)\.(f32|v\d+f32)\(", l)
        if m and m.group(2) in MATH:
            ty, sym = m.group(1), builtin(m.group(2), m.group(3))
            decls.add(f"declare spir_func {ty} @{sym}({', '.join([ty] * MATH[m.group(2)])})")
            l = l[:m.start()] + f"call spir_func {ty} @{sym}(" + l[m.end():]
        for ty, tag in NARROW.items():
            l = re.sub(rf"(%v\d+) = fpext {ty} (\S+) to float", rf"\1 = call spir_func float @baro_{tag}_to_f32(i16 zeroext \2)", l)
            l = re.sub(rf"(%v\d+) = fptrunc float (\S+) to {ty}", rf"\1 = call spir_func zeroext i16 @baro_f32_to_{tag}(float \2)", l)
            l = re.sub(rf"\b(load|store|getelementptr inbounds|getelementptr) {ty}\b", r"\1 i16", l)
            l = re.sub(rf"\b(load|store|bitcast) <(\d+) x {ty}>", r"\1 <\2 x i16>", l)
        out.append(literals(l))

    text = "\n".join(out)
    left = re.search(r"@air\.[\w.]+|\bbfloat\b|\bhalf\b", text)
    if left:
        raise Unsupported(f"no lowering for '{left.group(0)}'")

    pro = [f"v{len(ptr_args)}:"]
    for var, sym in IDS.items():
        decls.add(f"declare spir_func i64 @{sym}(i32)")
        prev = "zeroinitializer"
        for d in range(3):
            pro.append(f"  %baro.{var}.w{d} = call spir_func i64 @{sym}(i32 {d})")
            pro.append(f"  %baro.{var}.t{d} = trunc i64 %baro.{var}.w{d} to i32")
            nxt = f"%{var}" if d == 2 else f"%baro.{var}.v{d}"
            pro.append(f"  {nxt} = insertelement <3 x i32> {prev}, i32 %baro.{var}.t{d}, i64 {d}")
            prev = nxt
    pro += ["  %baro.lid = call spir_func i64 @_Z12get_local_idEj(i32 0)", f"  %baro.lane = urem i64 %baro.lid, {WARP}",
            "  %thread_index_in_simdgroup = trunc i64 %baro.lane to i32"]
    decls.add("declare spir_func void @_Z7barrierj(i32)")
    globs = shared + [f"@baro.shuf.{t} = internal addrspace(3) global [{LOCAL_MAX} x {LLTYPE[t]}] undef, align 4" for t in sorted(shuf_types)]
    args = ", ".join(f"ptr addrspace({a}) %v{i}" for a, i in ptr_args)
    module = HEADER + globs + sorted(decls) + [f"define spir_kernel void @{name}({args}) {{"] + pro + out + ["}"]
    return name, "\n".join(module) + "\n", barriers


def helper_defs(path):
    src = open(path).read()
    defs = re.findall(r"^define .*?^\}", src, re.S | re.M)
    clean = []
    for d in defs:
        d = re.sub(r" #\d+", "", d)
        d = re.sub(r", !\w+ !\d+", "", d)
        d = re.sub(r"^define (?:dso_local |hidden )*", "define internal ", d)
        clean.append(re.sub(r" local_unnamed_addr", "", d))
    decl = sorted(set(re.findall(r"^declare .*$", src, re.M)))
    return "\n".join(re.sub(r" #\d+", "", x) for x in decl) + "\n" + "\n".join(clean) + "\n"


def main():
    helpers, outdir, failed = helper_defs(sys.argv[1]), sys.argv[2], []
    for path in sys.argv[3:]:
        try:
            name, module, barriers = convert(open(path).read())
        except Unsupported as e:
            print(f"FAIL {path.split('/')[-1]}: {e}")
            failed.append(path)
            continue
        open(f"{outdir}/{name}.ll", "w").write(module + (helpers if "@baro_" in module else ""))
        print(f"{name} phases=1 barriers={barriers}")
    if failed:
        print(f"FAIL {len(failed)}/{len(sys.argv[3:])} convert")
        sys.exit(1)


if __name__ == "__main__":
    main()
