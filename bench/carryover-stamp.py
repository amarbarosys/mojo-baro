#!/usr/bin/env python3
"""In-stream GPU dispatch timing stamps: derive a stamped copy of the engine
that records llvm.readsteadycounter/readcyclecounter around chosen kernel
dispatches, with no host sync and no external tracer.

Promoted from .work/carryover/mkprobe.py (2026-09-15 carry-over probe,
exchange/2026-09-15-carryover-probe.md). Gate-2-style per-kernel timing with
host syncs cannot judge a geometry change on this card (verdict 3 of that
probe); this stamping approach can, because it never leaves the kernel stream.

Writes .work/carryover/src-<arm> for every arm in ARMS (copies of kernels/*.mojo
and serve/*.mojo at that arm's git ref), then patches in the stamp
instrumentation for every site in SITES. Tracked files are never touched.
Every text replacement asserts an exact match count, so a stale anchor string
fails loudly instead of silently patching zero call sites.

Two registries make this reusable for a dispatch site other than the three FFN
GEMVs this probe measured:
  KERNELS: one entry per instrumented kernel function. add_timer() below does
    the actual insertion (read timers, run the body, atomic-accumulate into a
    stamp slot) purely from string anchors, so a new kernel needs a KERNELS
    entry naming its file/def and the same three anchors, not new code.
  SITES: one entry per call site that should route through a stamped kernel.
    A new dispatch point (a different call in window.mojo, or a call
    elsewhere) is a new SITES entry; the harness/window/engine plumbing below
    (the NSLOT stamp buffer, win/layer/site/m addressing, the STAMP print
    line) is already site-count-agnostic.

Usage: bench/carryover-stamp.py [--arm NAME ...]
  With no --arm, builds every arm in DEFAULT_ARMS: C (current checkout) and
  D2 (current checkout, kernels/ssm.mojo + serve/registry.mojo swapped in from
  b31f7b5 with SSM_JSPLIT forced to 2) -- reproduces the 2026-09-15 probe.
"""
import argparse
import os
import shutil
import subprocess

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, ".work/carryover")
# 8192 was enough for the 3-site FFN-only probe (3360 stamps observed on a
# 64-token k=2 decode, ~35 windows). M3's 13-site SITES list fires ~249
# stamps/window (3 FFN + 4 attn + 5 ssm, N_LAYERS/N_ATT/N_SSM = 32/8/24, +1
# head), so the same run would need ~8715 slots -- over capacity, and
# stamp_slot() has no bounds check, so overflow would silently write past
# St's allocation. 32768 gives about 4x headroom over that.
NSLOT = 32768


def git_show(rev, path):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{rev}:{path}"], text=True)


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, f"expected {n} match(es), got {c} for:\n{old[:120]}"
    return text.replace(old, new)


def next_def(t, start):
    i = t.find("\ndef ", start + 1)
    j = t.find("\n@", start + 1)
    c = [x for x in (i, j) if x >= 0]
    return (min(c) + 1) if c else len(t)


# --- KERNELS: reusable per-kernel timer instrumentation -------------------
# Each entry describes one kernel function to clone into a "_st" variant that
# stamps its own wall/cycle span into St[slot]. add_timer() does the actual
# text surgery; everything here is anchor strings, not codegen logic.
KERNELS = {
    "q4rowb": dict(
        file="kernels/matmul_skinny.mojo",
        def_name="amar_matmul_skinny_q4rowb",
        next_def_marker="def amar_matmul_skinny_q8dot[",
        imports=[
            ("from std.sys.intrinsics import llvm_intrinsic\n",
             "from std.sys.intrinsics import llvm_intrinsic\nfrom std.atomic import Atomic\n"),
            ("from std.gpu import block_idx, global_idx, lane_id, thread_idx, WARP_SIZE\n",
             "from std.gpu import block_idx, global_idx, grid_dim, lane_id, thread_idx, WARP_SIZE\n"),
        ],
        sig_anchor="    k_dim: Int32,\n):",
        sig_extra="    k_dim: Int32,\n    St: MutPointer[Scalar[DType.int64], MutAnyOrigin],\n    slot: Int32,\n):",
        entry_anchor="    var M = Int(m)\n",
        exit_anchor=(
            "            if lane == 0:\n"
            "                Cp[part, r, row] = rebind[Cp.ElementType](t[r])\n"
        ),
    ),
}


def add_timer(text, spec):
    """Clone spec's kernel into a '<def_name>_st' variant with a timer stamp
    wrapped around its body: rt0/cy0 read right after entry_anchor, then
    (after the body, right after exit_anchor) an s_waitcnt drain, rt1/cy1
    read, and an atomic accumulate into St[slot*8 : slot*8+8]. The layout
    matches serve/engine.mojo's STAMP print (mn mx sum_rt sum_cyc arrivals
    b0_rt b0_cyc unsafe)."""
    start = text.index(f"def {spec['def_name']}[")
    end = text.index(spec["next_def_marker"])
    body = text[start:end].rstrip() + "\n"
    st = rep(body, f"def {spec['def_name']}[", f"def {spec['def_name']}_st[")
    st = rep(st, spec["sig_anchor"], spec["sig_extra"])
    st = rep(
        st, spec["entry_anchor"],
        "    var rt0 = llvm_intrinsic[\"llvm.readsteadycounter\", Int64]()\n"
        "    var cy0 = llvm_intrinsic[\"llvm.readcyclecounter\", Int64]()\n" + spec["entry_anchor"],
    )
    st = rep(
        st, spec["exit_anchor"],
        spec["exit_anchor"] +
        "    llvm_intrinsic[\"llvm.amdgcn.s.waitcnt\", NoneType](Int32(0))\n"
        "    var rt1 = llvm_intrinsic[\"llvm.readsteadycounter\", Int64]()\n"
        "    var cy1 = llvm_intrinsic[\"llvm.readcyclecounter\", Int64]()\n"
        "    if thread_idx.x == 0:\n"
        "        var sb = Int(slot) * 8\n"
        "        var drt = rt1 - rt0\n"
        "        var dcy = (cy1 - cy0) & Int64(1048575)\n"
        "        if block_idx.x == 0:\n"
        "            St[unsafe_offset=sb] = rt0\n"
        "            St[unsafe_offset=sb + 5] = drt\n"
        "            St[unsafe_offset=sb + 6] = dcy\n"
        "        if drt < Int64(30000):\n"
        "            _ = Atomic[DType.int64, scope=\"agent\"].fetch_add(St.unsafe_offset(sb + 2), drt)\n"
        "            _ = Atomic[DType.int64, scope=\"agent\"].fetch_add(St.unsafe_offset(sb + 3), dcy)\n"
        "        else:\n"
        "            _ = Atomic[DType.int64, scope=\"agent\"].fetch_add(St.unsafe_offset(sb + 7), 1)\n"
        "        var old = Atomic[DType.int64, scope=\"agent\"].fetch_add(St.unsafe_offset(sb + 4), 1)\n"
        "        if old == Int64(grid_dim.x) - 1:\n"
        "            St[unsafe_offset=sb + 1] = rt1\n",
    )
    return text[:end] + st + "\n\n" + text[end:]


def patch_kernel_file(d, kernel_key):
    spec = KERNELS[kernel_key]
    p = os.path.join(d, spec["file"])
    t = open(p).read()
    for old, new in spec["imports"]:
        t = rep(t, old, new)
    t = add_timer(t, spec)
    open(p, "w").write(t)


# --- SITES: which dispatch points route through a stamped kernel ----------
# One entry per gemm_w[...] call in serve/window.mojo's decode-loop window
# (site index is what lands in the STAMP line and in analyze.py's
# --site-names). Extending this probe to a different call site, or a
# different kernel, means adding a KERNELS entry (if it is a new kernel) and
# a SITES entry naming the call to replace with its stamped form;
# patch_window() below loops the list, it does not special-case each one.
#
# The first three (gate/up/down, the shared FFN sub-block, site 0-2) are the
# 2026-09-15 carry-over probe's original set. Sites 3-12 (M3, MSPEC
# precondition 2026-09-15) add every other gemm_w call reached by ONE decode
# window: 4 in the attn sub-block, 5 in the ssm sub-block (mutually
# exclusive per layer -- is_attn(layer) picks one), 1 head/logits GEMV once
# per window. Everything else dispatched in the window (rmsc_k, split_k,
# hrms_q/kv, rope_q/k, append_k, datt_k/att_k, gmul_k, r_add, rgates_k,
# conv_k, l2_k, delta_dispatch, gated_k, r_swiglu, argmax_d/k, tokcp_k,
# embed_k, ~25 kernels total) is NOT stamped and is charged to "gap" by
# carryover-analyze.py -- gap share computed from this SITES set is an UPPER
# BOUND on the true gap share, never a lower one (fewer measured kernels ->
# smaller measured sum_kernel_spans -> larger apparent gap), per the maintainer's
# 2026-09-15 MSPEC decision rule.
SITES = [
    dict(name="gate", kernel="q4rowb", site=0,
         call="gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pg, m)"),
    dict(name="up", kernel="q4rowb", site=1,
         call="gemm_w[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pu, m)"),
    dict(name="down", kernel="q4rowb", site=2,
         call="gemm_w[H, FFN](ctx, FgBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Ph2, m)"),
    dict(name="att_qf", kernel="q4rowb", site=3,
         call="gemm_w[QF, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pqf, m)"),
    dict(name="att_k", kernel="q4rowb", site=4,
         call="gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Pkv, m)"),
    dict(name="att_v", kernel="q4rowb", site=5,
         call="gemm_w[KV, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pkv, m)"),
    dict(name="att_out", kernel="q4rowb", site=6,
         call="gemm_w[H, ATT](ctx, AoBm, b.wbuf, b.off[w + 6], cfg.pack_q4, Ph, m)"),
    dict(name="ssm_qkv", kernel="q4rowb", site=7,
         call="gemm_w[CONV, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pq, m)"),
    dict(name="ssm_z", kernel="q4rowb", site=8,
         call="gemm_w[H, H](ctx, CurBm, b.wbuf, b.off[w + 2], cfg.pack_q4, Ph, m)"),
    dict(name="ssm_a", kernel="q4rowb", site=9,
         call="gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 3], cfg.pack_q4, Pab, m)"),
    dict(name="ssm_b", kernel="q4rowb", site=10,
         call="gemm_w[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 4], cfg.pack_q4, Pab2, m)"),
    dict(name="ssm_out", kernel="q4rowb", site=11,
         call="gemm_w[H, H](ctx, ResBmOld, b.wbuf, b.off[w + 9], cfg.pack_q4, Ph, m)"),
    dict(name="head", kernel="q4rowb", site=12,
         call="gemm_w[VOCAB, H](ctx, CurBm, b.wbuf, b.off[w + 1], cfg.pack_q4, Pv, m)",
         indent=16, layer_expr="N_LAYERS"),
]


def patch_registry(d):
    # gemm_q4 -> gemm_q4_st wraps the one kernel KERNELS currently names
    # (q4rowb). A second kernel entry would need its own gemm_*_st wrapper
    # here, following the same shape.
    p = os.path.join(d, "serve/registry.mojo")
    t = open(p).read()
    t = rep(t, "from matmul_skinny import (\n", "from matmul_skinny import (\n    amar_matmul_skinny_q4rowb_st,\n")
    start = t.index("def gemm_q4[")
    end = next_def(t, start)
    g = t[start:end]
    gs = rep(g, "def gemm_q4[", "def gemm_q4_st[")
    gs = rep(gs, "    m: Int, n: Int, k: Int,\n) raises:", "    m: Int, n: Int, k: Int,\n    mut St: DeviceBuffer[DType.int64], slot: Int,\n) raises:")
    gs = rep(gs, "amar_matmul_skinny_q4rowb[", "amar_matmul_skinny_q4rowb_st[", 5)
    gs = rep(gs, "A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),\n", "A, Wq, Ws, P, Int32(m), Int32(n), Int32(k), St.unsafe_ptr(), Int32(slot),\n", 5)
    t = t[:end] + "\n\n" + gs + t[end:]
    open(p, "w").write(t)


def patch_window(d, sites):
    p = os.path.join(d, "serve/window.mojo")
    t = open(p).read()
    t = rep(t, "    var prof_d: DeviceBuffer[DType.int64]\n", "    var prof_d: DeviceBuffer[DType.int64]\n    var st_d: DeviceBuffer[DType.int64]\n")
    t = rep(
        t, "    var pfx: List[Int]\n\n    def reset(mut self, t0: Int):\n",
        "    var pfx: List[Int]\n    var win: Int\n    var seq: Int\n    var meta: List[Int]\n\n"
        "    def stamp_slot(mut self, layer: Int, site: Int, m: Int) -> Int:\n"
        "        var s = self.seq\n        self.seq += 1\n"
        "        self.meta.append(self.win)\n        self.meta.append(layer)\n        self.meta.append(site)\n        self.meta.append(m)\n"
        "        return s\n\n    def reset(mut self, t0: Int):\n",
    )
    t = rep(t, "        self.pfx = [0, 0, 0, 0]\n", "        self.pfx = [0, 0, 0, 0]\n        self.win = 0\n        self.seq = 0\n        self.meta = List[Int]()\n")
    gw_start = t.index("def gemm_w[")
    gw_end = next_def(t, gw_start)
    gw = t[gw_start:gw_end]
    gws = rep(gw, "def gemm_w[", "def gemm_w_st[")
    gws = rep(gws, "    m: Int,\n) raises:", "    m: Int, mut St: DeviceBuffer[DType.int64], slot: Int,\n) raises:")
    gws = rep(
        gws,
        "        gemm_q4(ctx, A, tens_q4q(ctx, wbuf, o, N * K, row_major[N, K // 2]()), tens_q4s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), P, m, N, K)\n",
        "        gemm_q4_st(ctx, A, tens_q4q(ctx, wbuf, o, N * K, row_major[N, K // 2]()), tens_q4s(ctx, wbuf, o, N * K, row_major[N, K // 32]()), P, m, N, K, St, slot)\n",
    )
    gws = rep(gws, "    else:\n        gemm_q8(", "    else:\n        raise Error(\"stamped FFN path requires the q4 pack\")\n        gemm_q8(")
    t = t[:gw_end] + "\n\n" + gws + t[gw_end:]
    t = rep(
        t, "        for layer in range(0 if (use_mega or use_mega_win) else N_LAYERS):\n",
        "        if not (use_mega or use_mega_win):\n            st.win += 1\n"
        "        for layer in range(0 if (use_mega or use_mega_win) else N_LAYERS):\n",
    )
    for s in sites:
        dims = s["call"][len("gemm_w["):s["call"].index("]")]
        args = s["call"].split("(", 1)[1][:-1]
        layer_expr = s.get("layer_expr", "layer")
        stamped_call = f"gemm_w_st[{dims}]({args}, b.st_d, st.stamp_slot({layer_expr}, {s['site']}, m))"
        indent = " " * s.get("indent", 20)
        t = rep(t, f"{indent}{s['call']}\n", f"{indent}{stamped_call}\n")
    open(p, "w").write(t)


def patch_harness(d):
    p = os.path.join(d, "serve/harness.mojo")
    t = open(p).read()
    t = rep(
        t, "    ctx.enqueue_memset(prof_d, 0)\n",
        "    ctx.enqueue_memset(prof_d, 0)\n"
        f"    var st_d = ctx.enqueue_create_buffer[DType.int64]({NSLOT} * 8)\n"
        f"    var st_h = ctx.enqueue_create_host_buffer[DType.int64]({NSLOT} * 8)\n"
        "    ctx.synchronize()\n"
        f"    for i in range({NSLOT}):\n"
        "        for j in range(8):\n            st_h[8 * i + j] = 0\n"
        "    ctx.enqueue_copy(dst_buf=st_d, src_buf=st_h)\n",
    )
    t = rep(t, "prof_d=prof_d.copy(), dbg_d=", "prof_d=prof_d.copy(), st_d=st_d.copy(), dbg_d=")
    open(p, "w").write(t)


def patch_engine(d):
    p = os.path.join(d, "serve/engine.mojo")
    t = open(p).read()
    t = rep(
        t, "pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0])\n",
        "pc=[0, 0, 0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], win=0, seq=0, meta=List[Int]())\n",
    )
    helper = (
        "def sysfs_probe(tag: String) -> String:\n"
        "    # gpu_metrics v1.3 (120 bytes) + pp_dpm_sclk, read OUTSIDE the timed\n"
        "    # window; both reads are cached by the driver (~20 us), measured before use.\n"
        "    var t0 = perf_counter_ns()\n"
        "    var hexs = String(\"\")\n"
        "    try:\n"
        "        with open(\"/sys/class/drm/card1/device/gpu_metrics\", \"r\") as f:\n"
        "            var raw = f.read_bytes()\n"
        "            for i in range(len(raw)):\n"
        "                hexs += String(Int(raw[i])) + \",\"\n"
        "    except:\n"
        "        hexs = \"ERR\"\n"
        "    var dpm = String(\"\")\n"
        "    try:\n"
        "        with open(\"/sys/class/drm/card1/device/pp_dpm_sclk\", \"r\") as f:\n"
        "            dpm = f.read().replace(\"\\n\", \" \")\n"
        "    except:\n"
        "        dpm = \"ERR\"\n"
        "    var us = Float64(perf_counter_ns() - t0) / 1e3\n"
        "    return String(\"SYSFS \") + tag + \" read_us \" + String(us) + \" dpm \" + dpm + \" gpu_metrics \" + hexs\n\n\n"
    )
    t = rep(t, "def main() raises:", helper + "def main() raises:")
    t = rep(
        t,
        "        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)\n        ctx.synchronize()\n        var t0 = perf_counter_ns()\n        var t_prefill_end = t0\n",
        "        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)\n        ctx.synchronize()\n        print(sysfs_probe(\"pre_run\"))\n        var t0 = perf_counter_ns()\n        var t_prefill_end = t0\n",
    )
    t = rep(
        t,
        "            if not prefill_done and wst.pos >= len(prompt):\n                ctx.synchronize()\n                t_prefill_end = perf_counter_ns()\n",
        "            if not prefill_done and wst.pos >= len(prompt):\n                ctx.synchronize()\n                print(sysfs_probe(\"pre_decode\"))\n                t_prefill_end = perf_counter_ns()\n",
    )
    t = rep(
        t,
        "        if not prefill_done:\n            ctx.synchronize()\n            t_prefill_end = perf_counter_ns()\n",
        "        if not prefill_done:\n            ctx.synchronize()\n            print(sysfs_probe(\"pre_decode\"))\n            t_prefill_end = perf_counter_ns()\n",
    )
    t = rep(
        t, "        print(\"host_enqueue_s:\", t_host, \" gpu_total_s:\", dt)\n",
        "        print(\"host_enqueue_s:\", t_host, \" gpu_total_s:\", dt)\n        print(sysfs_probe(\"post_decode\"))\n",
    )
    t = rep(
        t,
        "        ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)\n        ctx.synchronize()\n        var generated = List[Int]()\n",
        "        ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)\n        ctx.synchronize()\n"
        f"        var sth = ctx.enqueue_create_host_buffer[DType.int64]({NSLOT} * 8)\n"
        "        ctx.enqueue_copy(dst_buf=sth, src_buf=bufs.st_d)\n        ctx.synchronize()\n"
        "        print(\"STAMPS\", wst.seq, \" hz 100000000  cycmask 1048575  layout start end sum_rt sum_cyc arrivals b0_rt b0_cyc unsafe  windows\", wst.win)\n"
        "        for s in range(wst.seq):\n"
        "            var sb = 8 * s\n"
        "            print(\"STAMP\", s, wst.meta[4 * s], wst.meta[4 * s + 1], wst.meta[4 * s + 2], wst.meta[4 * s + 3], sth[sb], sth[sb + 1], sth[sb + 2], sth[sb + 3], sth[sb + 4], sth[sb + 5], sth[sb + 6], sth[sb + 7])\n"
        "        var generated = List[Int]()\n",
    )
    open(p, "w").write(t)


def patch(d, sites):
    kernels_used = sorted({s["kernel"] for s in sites})
    for k in kernels_used:
        patch_kernel_file(d, k)
    patch_registry(d)
    patch_window(d, sites)
    patch_harness(d)
    patch_engine(d)


def build_arm(name, overrides, edits, sites):
    """Base tree is always the current checkout (working tree, kernels/+serve/
    *.mojo). overrides swaps individual files in for a different git ref
    before patching; edits then does small textual tweaks to those swapped
    files (e.g. flipping a comptime knob)."""
    d = os.path.join(OUT, f"src-{name}")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "kernels"))
    os.makedirs(os.path.join(d, "serve"))
    for sub in ("kernels", "serve"):
        for f in sorted(os.listdir(os.path.join(ROOT, sub))):
            if f.endswith(".mojo"):
                shutil.copy(os.path.join(ROOT, sub, f), os.path.join(d, sub, f))
    os.symlink(os.path.join(ROOT, "latentos"), os.path.join(d, "latentos"))
    for path, ref in overrides:
        text = git_show(ref, path)
        for old, new in edits.get(path, []):
            text = rep(text, old, new)
        open(os.path.join(d, path), "w").write(text)
    patch(d, sites)
    print("wrote", d)


# Default arms reproduce the 2026-09-15 carry-over probe exactly: C is the
# current checkout untouched, D2 swaps in ssm.mojo + registry.mojo from the
# (reverted) JSPLIT commit b31f7b5 with SSM_JSPLIT forced to 2, so it differs
# from C only in that one kernel's geometry. A new arm is a new entry here:
# (overrides, edits), both empty for a plain-checkout arm.
DEFAULT_ARMS = {
    "C": ([], {}),
    "D2": (
        [("kernels/ssm.mojo", "b31f7b5"), ("serve/registry.mojo", "b31f7b5")],
        {"serve/registry.mojo": [("comptime SSM_JSPLIT = 1\n", "comptime SSM_JSPLIT = 2\n")]},
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", dest="arms", choices=sorted(DEFAULT_ARMS),
                     help="build only this arm (repeatable); default: all arms in DEFAULT_ARMS")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in (args.arms or sorted(DEFAULT_ARMS)):
        overrides, edits = DEFAULT_ARMS[name]
        build_arm(name, overrides, edits, SITES)


if __name__ == "__main__":
    main()
