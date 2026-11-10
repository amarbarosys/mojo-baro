"""Inventory of the Metal IR Mojo emits per kernel: one markdown table row each.

usage: inventory.py <kernel.ll>...
"""
import re, sys
from collections import Counter

from air2spv import Unsupported, kernel_name

print("| kernel | IR lines | args (buf as1 / const as2) | air.* and llvm.* calls | simd_shuffle_xor (type: masks) | barriers | shared globals (as3) | loops | narrow types | fdiv | alloca | vector types |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    src = open(path).read()
    fn = re.search(r"^define void @(\S+?)\((.*?)\) local_unnamed_addr #\d+ \{\n(.*?)^\}", src, re.S | re.M)
    body = fn.group(3).splitlines()
    calls = Counter(re.findall(r"@((?:air|llvm)\.[\w.]+)\(", fn.group(3)))
    shuf = {}
    for t, m in re.findall(r"@air\.simd_shuffle_xor\.(\w+)\.i16\(\w+ %\S+, i16 (\d+)\)", fn.group(3)):
        shuf.setdefault(t, []).append(m)
    glob = re.findall(r"^@\S+ = internal addrspace\(3\) global (\[[^\]]+\])", src, re.M)
    loops, cur = 0, 0
    for l in body:
        m = re.match(r"(\d+):", l)
        if m:
            cur = int(m.group(1))
        for tgt in re.findall(r"label %(\d+)", l) if l.strip().startswith("br ") else []:
            loops += int(tgt) <= cur
    other = {k: v for k, v in calls.items() if "simd_shuffle" not in k and "barrier" not in k}
    narrow = sorted(set(re.findall(r"(?:load|store) (bfloat|half|i8|i16)\b", fn.group(3))))
    try:
        kname = kernel_name(fn.group(1))
    except Unsupported:
        kname = re.sub(r"_[0-9a-f]{16}$", "", fn.group(1))
    vec = Counter(re.findall(r"<\d+ x \w+>", fn.group(3)))
    print("| {} | {} | {} / {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        kname, len(body),
        fn.group(2).count("ptr addrspace(1)"), fn.group(2).count("ptr addrspace(2)"),
        ", ".join(f"{k.replace('air.', '')} x{v}" for k, v in sorted(other.items())) or "none",
        "; ".join(f"{t}: {','.join(m)}" for t, m in shuf.items()) or "none",
        calls.get("air.wg.barrier", 0), ", ".join(glob) or "none", loops, ", ".join(narrow) or "none",
        len(re.findall(r"= fdiv ", fn.group(3))), len(re.findall(r"= alloca ", fn.group(3))),
        ", ".join(f"{k} x{v}" for k, v in sorted(vec.items()) if k != "<3 x i32>") or "none"))
