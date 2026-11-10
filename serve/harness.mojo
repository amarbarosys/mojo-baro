"""Engine harness: pack loading and buffer allocation shared by serve/engine.mojo
and kernels/test_prefix.mojo. Moved verbatim out of engine.mojo main (M1a);
no decode-path code lives here.
"""
from std.os import getenv
from std.sys import exit
from tune import load_tune
from expert_tier import ExpertTier
from std.collections import Dict
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv, sqrt
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *
from window import *
from cert_hook import CertHook, CERT_LAYER
from prefix import pack_identity_salt
from latent_ipc import weights_import, weights_export, stage_new, host_mapped, _at


struct Pack(Movable):
    # B4 stage 2b: the tier needs the pack directory to read index.txt and
    # experts.bin, and alloc_bufs only ever sees the Pack.
    var packdir: String
    var wbuf: DeviceBuffer[DType.uint8]
    var off: List[Int]
    var kind: List[Int]
    var total: Int
    var q4_off: Int
    var have_q4_draft: Bool
    var pack_q4: Bool
    var pack_kq: Bool
    # A --kq pack stores its non-K-quant 2D weights as q8; the kq megakernel decodes Q4_K / Q6_K only.
    var kq_mixed: Bool
    var e: Int
    var fr_off: Int
    var fr_ids_off: Int
    var fr_k: Int
    var down_q6: Int

    def __init__(out self, packdir: String, var wbuf: DeviceBuffer[DType.uint8], var off: List[Int], var kind: List[Int], total: Int, q4_off: Int, have_q4_draft: Bool, pack_q4: Bool, pack_kq: Bool, fr_off: Int = 0, fr_ids_off: Int = 0, fr_k: Int = 0, down_q6: Int = 0, kq_mixed: Bool = False):
        self.packdir = packdir
        self.wbuf = wbuf^
        self.off = off^
        self.kind = kind^
        self.total = total
        self.q4_off = q4_off
        self.have_q4_draft = have_q4_draft
        self.pack_q4 = pack_q4
        self.pack_kq = pack_kq
        self.kq_mixed = kq_mixed
        self.fr_off = fr_off
        self.fr_ids_off = fr_ids_off
        self.fr_k = fr_k
        self.down_q6 = down_q6
        self.e = len(self.off) - (1 if have_q4_draft else 0) - (2 if fr_k > 0 else 0) - 15


def load_pack(ctx: DeviceContext, packdir: String) raises -> Pack:
    var PACK = packdir + "/pack.bin"
    # --- offset table from the pack index (tools/engine-pack.py order) ----
    var off = List[Int]()
    # Per-entry weight kind beside off: 1 q4, 12 q4_k, 14 q6_k (GT_Q4K / GT_Q6K), 0 anything else.
    var kind = List[Int]()
    var pack_kq = False
    var kq_q8 = False
    var total = 0
    var q4_off = 0
    var have_q4_draft = False
    var pack_q4 = False
    var fr_off = 0
    var fr_ids_off = 0
    var fr_k = 0
    # pack/profile mismatch guard (serve/moe_pack.mojo, bench/moe-loader-protocol.md):
    # a pack built for one H/N_LAYERS read under a different profile's offsets
    # silently walks off its tensors (observed: window.mojo out-of-bounds
    # abort, exit -4) -- checked below, before any offset is used, against two
    # numbers read back from the index itself: H implied by token_embd.weight's
    # own element count, and the real transformer layer count implied by the
    # blk.N groups. A group is excluded from the layer count entirely if ANY
    # of its tensors is a "nextn.*" one: qwen35's optional MTP draft head packs
    # a whole extra "blk.N.*" group (11 ordinary-looking tensor names plus 4
    # nextn.* ones) appended after the real trunk, so a tensor's own name is
    # not enough to tell a real layer from the draft head -- only "does this
    # blk.N group contain a nextn.* tensor at all" is.
    var h_implied = 0
    var n_tier = 0
    var emb_first = False
    var layer_has_nextn = Dict[Int, Bool]()
    # Bit L: layer L's routed down projection is q6_k (the rest are q4_k).
    var down_q6 = 0
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            var name = String(parts[0])
            off.append(Int(parts[2]))
            kind.append(1 if dt == "q4" else (12 if dt == "q4_k" else (14 if dt == "q6_k" else 0)))
            if (dt == "q4_k" or dt == "q6_k") and not name.endswith("_exps.weight"):
                pack_kq = True
            if dt == "q8" and not name.startswith("token_embd") and not name.endswith("_exps.weight") and "nextn" not in name:
                kq_q8 = True
            # B4 stage 2b (bench/moe-tier-protocol.md): a fifth column
            # ("expert", as tools/pack-split-experts.py writes it) marks a
            # tensor that lives in experts.bin rather than in pack.bin.
            # Its entry keeps its position, name, dtype and element count, so
            # window.mojo's positional addressing is unchanged, but it
            # contributes no bytes to the device pack and its offset is an
            # offset into the expert store. serve/expert_tier.mojo resolves
            # those by name; nothing else may dereference them.
            if name.endswith(".ffn_down_exps.weight") and dt == "q6_k":
                down_q6 |= 1 << Int(name.split(".")[1])
            if len(parts) >= 5 and (String(parts[4]) == "expert" or String(parts[4]) == "tier"):
                n_tier += 1
                continue
            if name == "token_embd.weight":
                h_implied = n // VOCAB
                emb_first = len(off) == 1 and dt == "bf16" and off[0] == 0
            var dotparts = name.split(".")
            if len(dotparts) >= 2 and String(dotparts[0]) == "blk":
                var layer_n = Int(dotparts[1])
                var is_nextn = len(dotparts) >= 3 and String(dotparts[2]) == "nextn"
                if layer_n in layer_has_nextn:
                    if is_nextn:
                        layer_has_nextn[layer_n] = True
                else:
                    layer_has_nextn[layer_n] = is_nextn
            if dt == "bf16":
                total += n * B2
            elif dt == "f32":
                total += n * B4
            elif dt == "q8":
                total += n + (n // 32) * 2
            elif dt == "q8_0":
                # raw ggml Q8_0: 32-value blocks, 32 int8 + one f16 scale
                if len(dotparts) >= 3:
                    var cls = String(dotparts[len(dotparts) - 2])
                    if cls == "attn_q" or cls == "attn_k" or cls == "attn_v" or cls == "attn_output" or cls == "attn_qkv" or cls == "attn_gate" or cls == "ssm_out":
                        raise Error("pack " + name + " is raw q8_0; the MoE projections must be in the dense q8 layout (tools/moe-pack-q8d.py, R6.3)")
                total += (n // 32) * 34
            elif dt == "q4_k":
                # raw ggml Q4_K: 256-value superblocks, 144 bytes each
                total += (n // 256) * 144
            elif dt == "q6_k":
                # raw ggml Q6_K: 256-value superblocks, 210 bytes each
                total += (n // 256) * 210
            elif dt == "q4":
                if name == "output.weight.frdraft":
                    # trailing entry (tools/fr-draft.mojo): FR-Spec reduced draft head
                    fr_off = Int(parts[2])
                elif name == "output.weight.q4draft":
                    # trailing entry (tools/engine-pack.py --q4-draft): the draft
                    # head's own q4 copy of output.weight, appended after the
                    # trunk order -- excluded from the blk.32 index math below.
                    q4_off = Int(parts[2])
                    have_q4_draft = True
                else:
                    # --q4 pack: every 2D trunk weight is ggml Q4_0
                    pack_q4 = True
                total += n // 2 + (n // 32) * 2
            elif dt == "i32" and name == "frdraft.ids":
                fr_ids_off = Int(parts[2])
                fr_k = n
                total += n * 4
            else:
                raise Error("unknown pack dtype " + dt)

    if h_implied > 0 and h_implied != H:
        raise Error(
            "pack/profile mismatch: profile H " + String(H)
            + ", pack " + packdir + " implies H " + String(h_implied)
            + " from token_embd.weight"
        )
    var real_layers = 0
    for entry in layer_has_nextn.items():
        if not entry.value:
            real_layers += 1
    if real_layers > 0 and real_layers != N_LAYERS:
        raise Error(
            "pack/profile mismatch: profile N_LAYERS " + String(N_LAYERS)
            + ", pack " + packdir + " implies " + String(real_layers)
            + " real transformer layers (blk.N groups with no nextn.* tensor)"
        )

    # A tier pack read without the tier would hand the expert kernels an
    # offset into a file that is not loaded, and they would gather whatever
    # happens to live there. Refuse instead: this is the failure mode that
    # produces plausible tokens from the wrong weights.
    if n_tier > 0 and getenv("BARO_TIER", "") == "":
        raise Error(
            "pack " + packdir + " has " + String(n_tier)
            + " tier-resident tensors (experts.bin) but BARO_TIER is unset;"
            + " set BARO_TIER=<capacity> or use a full pack"
        )
    if n_tier == 0 and getenv("BARO_TIER", "") != "":
        raise Error(
            "BARO_TIER is set but pack " + packdir + " carries no tier tensors;"
            + " split it first with tools/pack-split-experts.py"
        )

    # --- load pack into one device buffer -----------------------------------
    # T3b step 2: BARO_WEIGHTS_IMPORT maps another engine's copy of this pack (salt and size
    # checked first) instead of reading pack.bin; BARO_WEIGHTS_EXPORT loads into a raw exportable
    # hipMalloc and writes the record. The exporter must outlive every importer.
    var wimp = getenv("BARO_WEIGHTS_IMPORT", "")
    var wexp = getenv("BARO_WEIGHTS_EXPORT", "")
    if wimp != "" and wexp != "":
        raise Error("BARO_WEIGHTS_IMPORT and BARO_WEIGHTS_EXPORT are exclusive")
    # Step 2b: BARO_EMBD_HOST=1 keeps token_embd (the pack's first tensor, one row read per token)
    # in mapped pinned host memory the embed kernels read in place. The device allocation holds
    # pack bytes [emb, total); wbuf's base sits emb bytes before it so every other offset is
    # unchanged, and off[0] is rebased onto the host copy. Nothing may read wbuf below emb.
    var emb = 0
    if getenv("BARO_EMBD_HOST", "0") == "1":
        if n_tier > 0 or not emb_first:
            raise Error("BARO_EMBD_HOST=1 needs a full pack whose first tensor is bf16 token_embd at offset 0")
        emb = total
        for i in range(1, len(off)):
            emb = min(emb, off[i])
        var host = host_mapped(emb)
        var t_emb = perf_counter_ns()
        with open(PACK, "r") as f:
            var fd = f._get_raw_fd()
            var got = 0
            while got < emb:
                var n = external_call["pread", c_ssize_t](fd, _at[DType.uint8](host[0] + got), emb - got, Int64(got))
                if n <= 0:
                    raise Error("short read: token_embd")
                got += Int(n)
        print("token_embd in host memory:", emb, "bytes, read in", Float64(perf_counter_ns() - t_emb) / 1e9, "s")
        off[0] = host[1]
    var dev_bytes = total - emb
    if wimp != "":
        var t_imp = perf_counter_ns()
        var addr = weights_import(wimp, pack_identity_salt(packdir), dev_bytes)
        print("weights imported over IPC:", dev_bytes, "bytes from", wimp, "in", Float64(perf_counter_ns() - t_imp) / 1e9, "s")
        if emb > 0:
            off[0] -= addr - emb
        var shared = DeviceBuffer[DType.uint8](ctx, _at[DType.uint8](addr - emb), total, owning=False)
        return Pack(packdir, shared^, off^, kind^, total, q4_off, have_q4_draft, pack_q4, pack_kq, fr_off, fr_ids_off, fr_k, down_q6, pack_kq and kq_q8)
    print("loading pack:", total, "bytes", " tier tensors:", n_tier, " device bytes:", dev_bytes)
    var whandle = List[UInt8]()
    var wbuf: DeviceBuffer[DType.uint8]
    if wexp != "" or emb > 0:
        var st = stage_new(dev_bytes)
        whandle = st[1].copy()
        wbuf = DeviceBuffer[DType.uint8](ctx, _at[DType.uint8](st[0] - emb), total, owning=False)
        if emb > 0:
            off[0] -= st[0] - emb
    else:
        wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
    with open(PACK, "r") as f:
        var fd = f._get_raw_fd()
        var done = emb
        var flip = False
        while done < total:
            var want = min(CHUNK, total - done)
            var stage = stage1 if flip else stage0
            var sptr = stage.unsafe_ptr()
            var rerr = List[Int64](unsafe_uninit_length=RSPLIT)
            var rerr_ptr = rerr.unsafe_ptr()
            def rchunk(t: Int) {imm fd, imm want, imm done, imm sptr, imm rerr_ptr}:
                var lo = want * t // RSPLIT
                var hi = want * (t + 1) // RSPLIT
                var got = lo
                while got < hi:
                    var n = external_call["pread", c_ssize_t](
                        fd, sptr.unsafe_offset(got), hi - got, Int64(done + got)
                    )
                    if n <= 0:
                        rerr_ptr[unsafe_offset=t] = 1
                        return
                    got += Int(n)
                rerr_ptr[unsafe_offset=t] = 0
            parallelize(rchunk, RSPLIT)
            for t in range(RSPLIT):
                if rerr[t] != 0:
                    raise Error("short read")
            # The in-flight copy sources the OTHER stage: sync only after this
            # read has overlapped it, and always before this stage is enqueued.
            ctx.synchronize()
            var dslice = DeviceBuffer[DType.uint8](
                ctx, wbuf.unsafe_ptr().unsafe_offset(done), want, owning=False
            )
            var hslice = stage.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else stage
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            done += want
            flip = not flip
    ctx.synchronize()
    print("pack loaded in", Float64(perf_counter_ns() - t_load) / 1e9, "s")
    if wexp != "":
        weights_export(wexp, dev_bytes, pack_identity_salt(packdir), whandle)
        print("weights exported over IPC:", dev_bytes, "bytes to", wexp)
    return Pack(packdir, wbuf^, off^, kind^, total, q4_off, have_q4_draft, pack_q4, pack_kq, fr_off, fr_ids_off, fr_k, down_q6, pack_kq and kq_q8)


def kv_pool_seqs() raises -> Int:
    # KV pool size in sequences of tmax; BARO_KV_SHARE=N grows it for shared prefix pages.
    # BARO_KV_POOL=P sets it outright (BARO_SEQS > 2: slots take pages as they decode, so the
    # pool is sized by VRAM, not by SEQ_CAP x tmax).
    var pool = atol(getenv("BARO_KV_POOL", "0"))
    if pool > 0:
        return pool
    return max(SEQ_CAP, atol(getenv("BARO_KV_SHARE", "0")))


def find_drm_dev() -> String:
    var best = String("")
    var max_total: Int = 0
    for i in range(8):
        var dev = "/sys/class/drm/card" + String(i) + "/device/"
        try:
            with open(dev + "mem_info_vram_total", "r") as f:
                var tot = atol(f.read().strip())
                if tot > max_total:
                    max_total = tot
                    best = dev
        except:
            pass
    return best


def find_drm_vram_used() -> String:
    var dev = find_drm_dev()
    return dev + "mem_info_vram_used" if dev != "" else String("")


def gtt_used() -> Int:
    var dev = find_drm_dev()
    if dev == "":
        return -1
    try:
        with open(dev + "mem_info_gtt_used", "r") as f:
            return atol(f.read().strip())
    except:
        return -1


def gtt_check(stage: String, base: Int) -> Int:
    if base < 0:
        return base
    var now = gtt_used()
    if now < 0:
        return base
    var limit = 1073741824
    try:
        limit = atol(getenv("BARO_GTT_WARN_BYTES", "1073741824"))
    except:
        pass
    var grown = now - base
    if grown <= limit:
        return base
    print("WARN gtt: stage", stage, "bytes spilled", grown, "(gtt_used", now, "was", base, "limit", limit, "): a resident pack or buffer may sit in system memory, decode can slow 10x")
    if getenv("BARO_GTT_REFUSE", "0") == "1":
        print("FAIL gtt: BARO_GTT_REFUSE=1 and", grown, "bytes in GTT at", stage)
        exit(3)
    return now


def print_vram_log(stage: String):
    if getenv("BARO_VRAM_LOG", "0") != "1":
        return
    var path = find_drm_vram_used()
    if path == "":
        return
    try:
        with open(path, "r") as f:
            var content = f.read()
            print("VRAM_LOG stage:", stage, "bytes:", atol(content.strip()))
    except:
        pass


def alloc_bufs(ctx: DeviceContext, pack: Pack, tmax: Int) raises -> WindowBufs:
    var wbuf = pack.wbuf
    var off = pack.off.copy()
    var tpages = ceildiv(tmax, KVPAGE)
    var kvpool = tpages * N_ATT * NKVH * KVHSTR
    var kvpool1 = tpages * NKVH * KVHSTR
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var win_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)

    # --- activations / state -------------------------------------------------
    var x_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var curb_d = ctx.enqueue_create_buffer[bf16](MROWS * H)
    var qkv_d = ctx.enqueue_create_buffer[f32](MROWS * max(CONV, QF))
    var z_d = ctx.enqueue_create_buffer[f32](MROWS * max(H, INNER))
    var eg_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](MROWS * CONV)
    var so_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V * SSTATE)
    var resb_d = ctx.enqueue_create_buffer[bf16](MROWS * max(ATT, INNER))
    var qf_d = ctx.enqueue_create_buffer[f32](MROWS * QF)
    var q_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var gate_d = ctx.enqueue_create_buffer[f32](MROWS * ATT)
    var k_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var v_d = ctx.enqueue_create_buffer[f32](MROWS * KV)
    var ao_d = ctx.enqueue_create_buffer[f32](MROWS * NQH * HD)
    var fgb_d = ctx.enqueue_create_buffer[bf16]((2 * 8 * FFN) if IS_MOE else MROWS * FFN)
    var aq_d = ctx.enqueue_create_buffer[DType.int8](MROWS * FFN)
    var asc_d = ctx.enqueue_create_buffer[DType.float16](MROWS * (FFN // 32))
    var logits_d = ctx.enqueue_create_buffer[f32](MROWS * VOCAB)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](SEQ_CAP * tmax)
    var pos3_d = ctx.enqueue_create_buffer[DType.int32](3 * SEQ_CAP * tmax)
    var imgmap_d = ctx.enqueue_create_buffer[DType.int32](SEQ_CAP * tmax)
    var img_d = ctx.enqueue_create_buffer[f32](IMG_CAP * H)
    var hn_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var de_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var hd_d = ctx.enqueue_create_buffer[f32](MROWS * H)
    var mh_d = ctx.enqueue_create_buffer[f32](H)
    var cc_d = ctx.enqueue_create_buffer[bf16](MROWS * QF)
    var dtok_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)

    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * max(H, INNER))
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    # The megakernel parks its split attention partials here: R rows x NQH heads x up to
    # MEGA_G // NKVH splits x (HD + 2) floats, more than SPLITK * SM * FFN for this profile
    # (R = 1 already ran 268 KB into p_ffn2_d; R = 2 ran past both planes and clobbered a
    # neighbour, exchange/mega2-report.md traced P1).
    var p_ffn_d = ctx.enqueue_create_buffer[f32](max(SPLITK * SM * FFN, SEQ_CAP * NQH * (MEGA_G // NKVH) * (HD + 2)))
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * VOCAB)

    var xp_d = ctx.enqueue_create_buffer[f32](CP * H)
    var curbp_d = ctx.enqueue_create_buffer[bf16](CP * H)
    var qkvp_d = ctx.enqueue_create_buffer[f32](CP * max(CONV, QF))
    var zp_d = ctx.enqueue_create_buffer[f32](CP * max(H, INNER))
    var arp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var brp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var egp_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var betap_d = ctx.enqueue_create_buffer[f32](CP * NH_V)
    var convp_d = ctx.enqueue_create_buffer[f32](CP * CONV)
    var sop_d = ctx.enqueue_create_buffer[f32](CP * NH_V * SSTATE)
    var resbp_d = ctx.enqueue_create_buffer[bf16](CP * max(ATT, INNER))
    var qfp_d = ctx.enqueue_create_buffer[f32](CP * QF)
    var qp_d = ctx.enqueue_create_buffer[f32](CP * NQH * HD)
    var gatep_d = ctx.enqueue_create_buffer[f32](CP * ATT)
    var kp_d = ctx.enqueue_create_buffer[f32](CP * KV)
    var vp_d = ctx.enqueue_create_buffer[f32](CP * KV)
    var aop_d = ctx.enqueue_create_buffer[f32](CP * NQH * HD)
    var gp_d = ctx.enqueue_create_buffer[f32](CP * FFN)
    var up_d = ctx.enqueue_create_buffer[f32](CP * FFN)
    var fgbp_d = ctx.enqueue_create_buffer[bf16](CP * FFN)

    var convstate_d = ctx.enqueue_create_buffer[f32](SEQ_CAP * SLOTS * CONV_SLOT)
    var sstate_d = ctx.enqueue_create_buffer[f32](SEQ_CAP * SLOTS * SSM_SLOT)
    var kc_d = ctx.enqueue_create_buffer[KVT](kv_pool_seqs() * kvpool)
    var vc_d = ctx.enqueue_create_buffer[KVT](kv_pool_seqs() * kvpool)
    var kc32_d = ctx.enqueue_create_buffer[KVT](kv_pool_seqs() * kvpool1)
    var vc32_d = ctx.enqueue_create_buffer[KVT](kv_pool_seqs() * kvpool1)
    var kvtab_h = ctx.enqueue_create_host_buffer[DType.int32](SEQ_CAP * tpages)
    for seq in range(SEQ_CAP):
        for i in range(tpages):
            kvtab_h[seq * tpages + i] = Int32(seq * tpages + i)
    var kvtab_d = ctx.enqueue_create_buffer[DType.int32](SEQ_CAP * tpages)
    ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)
    ctx.enqueue_memset(convstate_d, 0)
    ctx.enqueue_memset(sstate_d, 0)
    ctx.enqueue_memset(kc_d, 0)
    ctx.enqueue_memset(vc_d, 0)
    ctx.enqueue_memset(kc32_d, 0)
    ctx.enqueue_memset(vc32_d, 0)
    ctx.synchronize()
    var off_h = ctx.enqueue_create_host_buffer[DType.int64](OFF_CAP)
    ctx.synchronize()
    for i in range(OFF_CAP):
        off_h[i] = Int64(off[i]) if i < len(off) else 0
        # K-quant kind in the top byte: amar_mega_token[KQ=True] splits it off (kq_split); other kinds leave it 0.
        if pack.pack_kq and i < len(off) and (pack.kind[i] == 12 or pack.kind[i] == 14):
            off_h[i] = off_h[i] | (Int64(pack.kind[i]) << 56)
    var off_d = ctx.enqueue_create_buffer[DType.int64](OFF_CAP)
    ctx.enqueue_copy(dst_buf=off_d, src_buf=off_h)
    var araw_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var braw_d = ctx.enqueue_create_buffer[f32](MROWS * NH_V)
    var ctr_d = ctx.enqueue_create_buffer[DType.uint32](3)
    var prof_d = ctx.enqueue_create_buffer[DType.int64](16 * N_LAYERS + 4)
    ctx.enqueue_memset(ctr_d, 0)
    ctx.enqueue_memset(prof_d, 0)
    var dbg_d = ctx.enqueue_create_buffer[f32](2 * N_LAYERS * H)
    var hmax_d = ctx.enqueue_create_buffer[f32](MEGA_MR * MEGA_G_WIN)
    var hidx_d = ctx.enqueue_create_buffer[DType.int32](MEGA_MR * MEGA_G_WIN)
    var dump_h = ctx.enqueue_create_host_buffer[f32](GEN_N * 2 * N_LAYERS * H)
    # BARO_CAPTURE_LAYERS="16,20": layers whose ffn in/out and residual are staged per token (docs/format-capture.md)
    var cap_slot = List[Int](length=N_LAYERS, fill=-1)
    var ncap = 0
    for part in getenv("BARO_CAPTURE_LAYERS", "").split(","):
        if part.byte_length() > 0:
            var cl = atol(String(part))
            if cl < 0 or cl >= N_LAYERS:
                raise Error("BARO_CAPTURE_LAYERS: layer out of range " + String(cl))
            if cap_slot[cl] < 0:
                cap_slot[cl] = ncap
                ncap += 1
    var cap_d = ctx.enqueue_create_buffer[f32](max(ncap, 1) * 3 * H)
    var capp_d = ctx.enqueue_create_buffer[f32](max(ncap, 1) * 3 * PF_CAP_ROWS * H if ncap > 0 else 1)
    var spl_d = ctx.enqueue_create_buffer[f32](SPLICE_MAX * H)  # splice rows, absolute positions 0..SPLICE_MAX-1
    # BARO_IMPLANTS=path: docs/format-implants.md (mojo-neuron). Keys are re-normalised here.
    var imp_set = List[Int](length=N_LAYERS, fill=-1)
    var imp_k = List[Int]()
    var imp_tau = List[Float32]()
    var imp_keys = List[List[Float32]]()
    var imp_gate = List[List[Float32]]()
    var imp_vals = List[List[Float32]]()
    var imp_path = getenv("BARO_IMPLANTS", "")
    if imp_path != "":
        var ib = open(imp_path, "r").read_bytes()
        var ip = ib.unsafe_ptr()
        if len(ib) < 16 or String(unsafe_from_utf8=ib[0:8]) != "NIMP0001":
            raise Error("BARO_IMPLANTS: not an implants.bin v1 file")
        var u32 = ip.bitcast[UInt32]()
        if Int(u32[2]) != H:
            raise Error("BARO_IMPLANTS: hidden size " + String(Int(u32[2])) + " != " + String(H))
        var ns = Int(u32[3])
        var off = 16
        for si in range(ns):
            var layer = Int(ip.bitcast[UInt32]()[off // 4])
            var k = Int(ip.bitcast[UInt32]()[off // 4 + 1])
            var tau = ip.bitcast[Float32]()[off // 4 + 2]
            off += 12
            if layer >= N_LAYERS or imp_set[layer] >= 0 or off + (2 * k * H + k) * 4 > len(ib):
                raise Error("BARO_IMPLANTS: bad set " + String(si))
            imp_set[layer] = si
            imp_k.append(k)
            imp_tau.append(tau)
            var kl = List[Float32](length=k * H, fill=0)
            for i in range(k):
                var nrm: Float32 = 0
                for j in range(H):
                    var v = ip.bitcast[Float32]()[off // 4 + i * H + j]
                    kl[i * H + j] = v
                    nrm += v * v
                nrm = 1.0 / sqrt(nrm) if nrm > 0 else Float32(0)
                for j in range(H):
                    kl[i * H + j] *= nrm
            off += k * H * 4
            var gl = List[Float32](length=k, fill=0)
            for i in range(k):
                gl[i] = ip.bitcast[Float32]()[off // 4 + i]
            off += k * 4
            var vl = List[Float32](length=k * H, fill=0)
            for i in range(k * H):
                vl[i] = ip.bitcast[Float32]()[off // 4 + i]
            off += k * H * 4
            imp_keys.append(kl^)
            imp_gate.append(gl^)
            imp_vals.append(vl^)
        print("BARO_IMPLANTS:", imp_path, ns, "sets")
    var imp_d = ctx.enqueue_create_buffer[f32](H)
    var imp_h = ctx.enqueue_create_host_buffer[f32](H)
    var imp_xb = ctx.enqueue_create_host_buffer[bf16](H)
    ctx.synchronize()






    var pos3_h = ctx.enqueue_create_host_buffer[DType.int32](3 * tmax)
    var imgmap_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var img_h = ctx.enqueue_create_host_buffer[f32](IMG_CAP * H)
    var stream_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    # A1 speculative sampling: two MROWS x VOCAB f32 probability planes (about
    # 8 MB each at this vocab) plus two small id planes. Allocated
    # unconditionally, like every other window buffer, so a request that turns
    # sampling on mid-session does not have to allocate.
    var pt_d = ctx.enqueue_create_buffer[f32](MROWS * VOCAB)
    var pd_d = ctx.enqueue_create_buffer[f32](MROWS * VOCAB)
    # B4 stage 2b: BARO_TIER=<capacity> puts the routed experts in host RAM
    # behind a per-layer LRU. Unset, the tier is inert and the pack is read as
    # before, so this allocation is the only cost of carrying it.
    var tier_cap = getenv("BARO_TIER", "")
    var tier = (
        ExpertTier(ctx, pack.packdir, Int(tier_cap), N_LAYERS) if tier_cap != ""
        else ExpertTier(ctx)
    )
    var etrace_d = ctx.enqueue_create_buffer[DType.int32](TRACE_TOK * N_LAYERS * 2 * TOPK_TRACE)
    var etrace_h = ctx.enqueue_create_host_buffer[DType.int32](TRACE_TOK * N_LAYERS * 2 * TOPK_TRACE)
    var zidx_d = ctx.enqueue_create_buffer[DType.int32](1)
    var dids_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)
    # R4 (bench/moe-persist-protocol.md): the MoE FFN's shared-expert index and
    # the two SSM gate partial planes are zeroed once, here, instead of 100
    # times per token in the window. The window's own per-layer memsets and the
    # per-layer host-buffer copy are gone with them.
    ctx.enqueue_memset(zidx_d, 0)
    ctx.enqueue_memset(etrace_d, -1)
    ctx.enqueue_memset(p_32_d, 0)
    ctx.enqueue_memset(p_32b_d, 0)
    var sout_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)
    var sacc_d = ctx.enqueue_create_buffer[DType.int32](KMAX + 1)
    var sout_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var sacc_h = ctx.enqueue_create_host_buffer[DType.int32](KMAX + 1)
    var pen_hist_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var pen_ids_h = ctx.enqueue_create_host_buffer[DType.int32](MROWS * SAMP_CAP)
    var pen_ids_d = ctx.enqueue_create_buffer[DType.int32](MROWS * SAMP_CAP)
    var pen_cnt_h = ctx.enqueue_create_host_buffer[DType.int32](MROWS * SAMP_CAP)
    var pen_cnt_d = ctx.enqueue_create_buffer[DType.int32](MROWS * SAMP_CAP)
    var pen_npen_h = ctx.enqueue_create_host_buffer[DType.int32](MROWS)
    var pen_npen_d = ctx.enqueue_create_buffer[DType.int32](MROWS)
    var topn_ids_d = ctx.enqueue_create_buffer[DType.int32](MROWS * NTOPLP)
    var topn_ids_h = ctx.enqueue_create_host_buffer[DType.int32](MROWS * NTOPLP)
    var topn_probs_d = ctx.enqueue_create_buffer[f32](MROWS * NTOPLP)
    var topn_probs_h = ctx.enqueue_create_host_buffer[f32](MROWS * NTOPLP)
    var samp_prob_h = ctx.enqueue_create_host_buffer[f32](1)
    # JSON-enforcement item 1: one grammar mask row, (VOCAB+63)//64 u64
    # words (~31 KB at VOCAB 248320) -- trivial next to the buffers above,
    # allocated unconditionally so a request that never sets response_format
    # never waits on an allocation, only a per-process amount of VRAM.
    var gmask_h = ctx.enqueue_create_host_buffer[DType.uint64]((VOCAB + 63) // 64)
    var gmask_d = ctx.enqueue_create_buffer[DType.uint64]((VOCAB + 63) // 64)
    var dump_row_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
    var pf_idx_d = ctx.enqueue_create_buffer[DType.int32](CP * 8)
    var pf_ge_d = ctx.enqueue_create_buffer[DType.int32](CP + 256)
    var pf_gp_d = ctx.enqueue_create_buffer[DType.int32](max((CP + 256) * 8, (CP // 4 + 256) * 32))
    var pf_gs_d = ctx.enqueue_create_buffer[DType.int32](512)
    var pf_dn_d = ctx.enqueue_create_buffer[f32]((CP * 8 * H) if IS_MOE else 1)
    # BARO_MOE_DOT=1: int8 activation rows + f16 scales per 32 for the sdot4 expert kernels (moe_ffn).
    var moe_dot = IS_MOE and getenv("BARO_MOE_DOT", "1") == "1"
    var xq_d = ctx.enqueue_create_buffer[DType.int8]((2 * H) if IS_MOE else 1)
    var xs_d = ctx.enqueue_create_buffer[DType.float16]((2 * H // 32) if IS_MOE else 1)
    var hq_d = ctx.enqueue_create_buffer[DType.int8]((2 * 8 * 512) if IS_MOE else 1)
    var hs_d = ctx.enqueue_create_buffer[DType.float16]((2 * 8 * 512 // 32) if IS_MOE else 1)
    var cert_w = 1
    for l in range(CERT_LAYER):
        cert_w += (7 if is_attn(l) else 10) + 4
    var steer_path = getenv("BARO_STEER", "")
    var steer_on = steer_path != ""
    # BARO_STEER_SERVE=1: allocate the buffer so serve requests can carry steer_file / steer_bank (steer_on is then set per request)
    var sv_d = ctx.enqueue_create_buffer[f32](N_LAYERS * H if (steer_on or getenv("BARO_STEER_SERVE", "0") == "1") else 1)
    if steer_on:
        var sv_b = open(steer_path, "r").read_bytes()
        if len(sv_b) != N_LAYERS * H * 4:
            raise Error("BARO_STEER: wrong file size, want f32[" + String(N_LAYERS) + " * " + String(H) + "]")
        var sv_h = ctx.enqueue_create_host_buffer[f32](N_LAYERS * H)
        for j in range(len(sv_b)):
            sv_h.unsafe_ptr().unsafe_bitcast[UInt8]()[j] = sv_b[j]
        ctx.enqueue_copy(dst_buf=sv_d, src_buf=sv_h)
        ctx.synchronize()
    return WindowBufs(sv_d=sv_d.copy(), steer_on=steer_on, cert=CertHook(ctx, wbuf, off, cert_w), wbuf=wbuf.copy(), off=off.copy(), kind=pack.kind.copy(), dtok_h=dtok_h.copy(), win_h=win_h.copy(), x_d=x_d.copy(), curb_d=curb_d.copy(), qkv_d=qkv_d.copy(), z_d=z_d.copy(), eg_d=eg_d.copy(), beta_d=beta_d.copy(), conv_d=conv_d.copy(), so_d=so_d.copy(), resb_d=resb_d.copy(), qf_d=qf_d.copy(), q_d=q_d.copy(), gate_d=gate_d.copy(), k_d=k_d.copy(), v_d=v_d.copy(), ao_d=ao_d.copy(), fgb_d=fgb_d.copy(), aq_d=aq_d.copy(), asc_d=asc_d.copy(), logits_d=logits_d.copy(), toks_d=toks_d.copy(), pos3_d=pos3_d.copy(), imgmap_d=imgmap_d.copy(), img_d=img_d.copy(), hn_d=hn_d.copy(), de_d=de_d.copy(), hd_d=hd_d.copy(), cc_d=cc_d.copy(), dtok_d=dtok_d.copy(), p_qf_d=p_qf_d.copy(), p_h_d=p_h_d.copy(), p_kv_d=p_kv_d.copy(), p_32_d=p_32_d.copy(), p_32b_d=p_32b_d.copy(), p_ffn_d=p_ffn_d.copy(), p_ffn2_d=p_ffn2_d.copy(), p_v_d=p_v_d.copy(), xp_d=xp_d.copy(), curbp_d=curbp_d.copy(), qkvp_d=qkvp_d.copy(), zp_d=zp_d.copy(), arp_d=arp_d.copy(), brp_d=brp_d.copy(), egp_d=egp_d.copy(), betap_d=betap_d.copy(), convp_d=convp_d.copy(), sop_d=sop_d.copy(), resbp_d=resbp_d.copy(), qfp_d=qfp_d.copy(), qp_d=qp_d.copy(), gatep_d=gatep_d.copy(), kp_d=kp_d.copy(), vp_d=vp_d.copy(), aop_d=aop_d.copy(), gp_d=gp_d.copy(), up_d=up_d.copy(), fgbp_d=fgbp_d.copy(), convstate_d=convstate_d.copy(), sstate_d=sstate_d.copy(), kvpool=kvpool, kc_d=kc_d.copy(), vc_d=vc_d.copy(), kvtab_d=kvtab_d.copy(), kvtab_h=kvtab_h.copy(), tpages=tpages, kc32_d=kc32_d.copy(), vc32_d=vc32_d.copy(), off_d=off_d.copy(), araw_d=araw_d.copy(), braw_d=braw_d.copy(), ctr_d=ctr_d.copy(), prof_d=prof_d.copy(), dbg_d=dbg_d.copy(), hmax_d=hmax_d.copy(), hidx_d=hidx_d.copy(), dump_h=dump_h.copy(), stream_h=stream_h.copy(), pos3_h=pos3_h.copy(), imgmap_h=imgmap_h.copy(), img_h=img_h.copy(), pt_d=pt_d.copy(), pd_d=pd_d.copy(), tier=tier^, etrace_d=etrace_d.copy(), etrace_h=etrace_h.copy(), zidx_d=zidx_d.copy(), dids_d=dids_d.copy(), sout_d=sout_d.copy(), sacc_d=sacc_d.copy(), sout_h=sout_h.copy(), sacc_h=sacc_h.copy(), pen_hist_h=pen_hist_h.copy(), pen_ids_h=pen_ids_h.copy(), pen_ids_d=pen_ids_d.copy(), pen_cnt_h=pen_cnt_h.copy(), pen_cnt_d=pen_cnt_d.copy(), pen_npen_h=pen_npen_h.copy(), pen_npen_d=pen_npen_d.copy(), topn_ids_d=topn_ids_d.copy(), topn_ids_h=topn_ids_h.copy(), topn_probs_d=topn_probs_d.copy(), topn_probs_h=topn_probs_h.copy(), samp_prob_h=samp_prob_h.copy(), dump_row_h=dump_row_h.copy(), gmask_h=gmask_h.copy(), gmask_d=gmask_d.copy(), pf_idx_d=pf_idx_d.copy(), pf_ge_d=pf_ge_d.copy(), pf_gp_d=pf_gp_d.copy(), pf_gs_d=pf_gs_d.copy(), pf_dn_d=pf_dn_d.copy(), xq_d=xq_d.copy(), xs_d=xs_d.copy(), hq_d=hq_d.copy(), hs_d=hs_d.copy(), moe_dot=moe_dot, down_q6=pack.down_q6, tune=load_tune(pack.packdir), cap_d=cap_d.copy(), cap_slot=cap_slot.copy(), capp_d=capp_d.copy(), gain_l=List[Int](), gain_n=List[Int](), gain_k=List[Float32](), spl_layer=-1, spl_zero=False, spl_n=0, spl_d=spl_d.copy(), imp_on=imp_path != "", imp_scale=1.0, imp_lo=0, imp_klim=-1, imp_layer=-1, imp_set=imp_set^, imp_k=imp_k^, imp_tau=imp_tau^, imp_keys=imp_keys^, imp_gate=imp_gate^, imp_vals=imp_vals^, imp_d=imp_d.copy(), imp_h=imp_h.copy(), imp_xb=imp_xb.copy())


def load_steer_vec(ctx: DeviceContext, sv_d: DeviceBuffer[f32], path: String, layer: Int) raises:
    """sv_d = f32[N_LAYERS * H] from path (layer < 0), or zeros with row `layer` = f32[H] from path."""
    var b = open(path, "r").read_bytes()
    if len(b) != (N_LAYERS * H if layer < 0 else H) * 4:
        raise Error("steer file " + path + ": wrong size " + String(len(b)))
    if layer >= N_LAYERS:
        raise Error("steer_layer out of range")
    var h = ctx.enqueue_create_host_buffer[f32](N_LAYERS * H)
    var dst = h.unsafe_ptr().unsafe_bitcast[UInt8]()
    for j in range(N_LAYERS * H * 4):
        dst[j] = 0
    var o = 0 if layer < 0 else layer * H * 4
    for j in range(len(b)):
        dst[o + j] = b[j]
    ctx.enqueue_copy(dst_buf=sv_d, src_buf=h)
    ctx.synchronize()


def load_steer_row(ctx: DeviceContext, sv_d: DeviceBuffer[f32], path: String, layer: Int, row: Int, scale: Float32) raises:
    """sv_d = zeros with row `layer` = scale * (row `row` of the f32[*, H] bank at path)."""
    if layer < 0 or layer >= N_LAYERS or row < 0:
        raise Error("steer_layer / steer_row out of range")
    var f = open(path, "r")
    _ = f.seek(row * H * 4)
    var b = f.read_bytes(H * 4)
    if len(b) != H * 4:
        raise Error("steer_bank " + path + ": row " + String(row) + " past the end")
    var h = ctx.enqueue_create_host_buffer[f32](N_LAYERS * H)
    var hb = h.unsafe_ptr().unsafe_bitcast[UInt8]()
    for j in range(N_LAYERS * H * 4):
        hb[j] = 0
    for j in range(H * 4):  # copy by index: a pointer taken from `b` would not keep it alive past its last named use
        hb[layer * H * 4 + j] = b[j]
    for j in range(H):
        h[layer * H + j] = h[layer * H + j] * scale
    ctx.enqueue_copy(dst_buf=sv_d, src_buf=h)
    ctx.synchronize()


def sequence_bufs(ctx: DeviceContext, b: WindowBufs, seq: Int, tmax: Int, share: Bool = False) raises -> WindowBufs:
    var v = b.copy()
    var state_span = SLOTS * CONV_SLOT
    var ssm_span = SLOTS * SSM_SLOT
    var kv_span = b.kvpool // SEQ_CAP
    v.convstate_d = DeviceBuffer[f32](ctx, b.convstate_d.unsafe_ptr().unsafe_offset(seq * state_span), state_span, owning=False)
    v.sstate_d = DeviceBuffer[f32](ctx, b.sstate_d.unsafe_ptr().unsafe_offset(seq * ssm_span), ssm_span, owning=False)
    if not share:
        v.kc_d = DeviceBuffer[KVT](ctx, b.kc_d.unsafe_ptr().unsafe_offset(seq * kv_span), kv_span, owning=False)
        v.vc_d = DeviceBuffer[KVT](ctx, b.vc_d.unsafe_ptr().unsafe_offset(seq * kv_span), kv_span, owning=False)
        v.kvpool = kv_span
    v.kvtab_d = DeviceBuffer[DType.int32](ctx, b.kvtab_d.unsafe_ptr().unsafe_offset(seq * b.tpages), b.tpages, owning=False)
    v.kvtab_h = b.kvtab_h.create_sub_buffer[DType.int32](seq * b.tpages, b.tpages)
    v.toks_d = DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(seq * tmax), tmax, owning=False)
    v.pos3_d = DeviceBuffer[DType.int32](ctx, b.pos3_d.unsafe_ptr().unsafe_offset(seq * 3 * tmax), 3 * tmax, owning=False)
    v.imgmap_d = DeviceBuffer[DType.int32](ctx, b.imgmap_d.unsafe_ptr().unsafe_offset(seq * tmax), tmax, owning=False)
    return v^


def capture_meta(dir: String, model: String, fields: String, b: WindowBufs) raises:
    """BARO_CAPTURE=DIR: write DIR/meta.json (docs/format-capture.md); records are appended per token by capture_write."""
    var ls = String("")
    for l in range(N_LAYERS):
        if b.cap_slot[l] >= 0:
            ls += ("," if ls.byte_length() > 0 else "") + String(l)
    with open(dir + "/meta.json", "w") as f:
        f.write(String("{\"model\":\"") + model + "\",\"dtype\":\"f16\",\"h\":" + String(H) + ",\"n_layers\":" + String(N_LAYERS) + ",\"fields\":\"" + fields + "\",\"layers\":[" + ls + "]}\n")


def capture_write(ctx: DeviceContext, b: WindowBufs, dir: String, fields: String, pos: Int, req_id: Int) raises:
    """Append position `pos` to DIR: per captured layer one f16 record of the `fields` rows (x ffn in, y ffn out, r residual
    after the layer) in L<l>.f16, plus tokens.i32 (the token fed at pos) and req.i32. Values past f16 range saturate and are counted."""
    var ncap = 0
    for l in range(N_LAYERS):
        if b.cap_slot[l] >= 0:
            ncap += 1
    var h = ctx.enqueue_create_host_buffer[f32](ncap * 3 * H)
    var tk = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_buf=h, src_buf=DeviceBuffer[f32](ctx, b.cap_d.unsafe_ptr(), ncap * 3 * H, owning=False))
    ctx.enqueue_copy(dst_buf=tk.create_sub_buffer[DType.int32](0, 1), src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(pos), 1, owning=False))
    ctx.synchronize()
    tk[1] = Int32(req_id)
    var sat = 0
    var fb = fields.as_bytes()
    var nf = fields.byte_length()
    for l in range(N_LAYERS):
        var s = b.cap_slot[l]
        if s < 0:
            continue
        var o = List[Scalar[DType.float16]](length=nf * H, fill=0)
        for fi in range(nf):
            var row = (s * 3 + (0 if fb[fi] == UInt8(ord("x")) else 1 if fb[fi] == UInt8(ord("y")) else 2)) * H
            for j in range(H):
                var v = h[row + j]
                if v > 65504.0 or v < -65504.0:
                    sat += 1
                    v = 65504.0 if v > 0 else -65504.0
                o[fi * H + j] = Scalar[DType.float16](v)
        with open(dir + "/L" + String(l) + ".f16", "a") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=o.unsafe_ptr().bitcast[UInt8](), length=nf * H * 2))
    with open(dir + "/tokens.i32", "a") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=tk.unsafe_ptr().bitcast[UInt8](), length=4))
    with open(dir + "/req.i32", "a") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=tk.unsafe_ptr().unsafe_offset(1).bitcast[UInt8](), length=4))
    if sat > 0:
        print("capture: pos", pos, "saturated", sat, "values to f16 range")


def capture_write_chunk(ctx: DeviceContext, b: WindowBufs, dir: String, fields: String, pos: Int, n: Int, req_id: Int) raises:
    """Batched-prefill capture: positions pos .. pos+n-1 of one chunk, same files and record layout as capture_write (docs/format-capture.md). y = residual after the FFN add minus before it."""
    var ncap = 0
    for l in range(N_LAYERS):
        if b.cap_slot[l] >= 0:
            ncap += 1
    var h = ctx.enqueue_create_host_buffer[f32](ncap * 3 * PF_CAP_ROWS * H)
    var tk = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=h, src_buf=DeviceBuffer[f32](ctx, b.capp_d.unsafe_ptr(), ncap * 3 * PF_CAP_ROWS * H, owning=False))
    ctx.enqueue_copy(dst_buf=tk, src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(pos), n, owning=False))
    ctx.synchronize()
    var fb = fields.as_bytes()
    var nf = fields.byte_length()
    var sat = 0
    for l in range(N_LAYERS):
        var s = b.cap_slot[l]
        if s < 0:
            continue
        var o = List[Scalar[DType.float16]](length=n * nf * H, fill=0)
        for r in range(n):
            var px = ((s * 3) * PF_CAP_ROWS + r) * H
            var pm = ((s * 3 + 1) * PF_CAP_ROWS + r) * H
            var pp = ((s * 3 + 2) * PF_CAP_ROWS + r) * H
            for fi in range(nf):
                for j in range(H):
                    var v = h[px + j] if fb[fi] == UInt8(ord("x")) else (h[pp + j] - h[pm + j] if fb[fi] == UInt8(ord("y")) else h[pp + j])
                    if v > 65504.0 or v < -65504.0:
                        sat += 1
                        v = 65504.0 if v > 0 else -65504.0
                    o[(r * nf + fi) * H + j] = Scalar[DType.float16](v)
        with open(dir + "/L" + String(l) + ".f16", "a") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=o.unsafe_ptr().bitcast[UInt8](), length=n * nf * H * 2))
    with open(dir + "/tokens.i32", "a") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=tk.unsafe_ptr().bitcast[UInt8](), length=4 * n))
    var rq = List[Int32](length=n, fill=Int32(req_id))
    with open(dir + "/req.i32", "a") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=rq.unsafe_ptr().bitcast[UInt8](), length=4 * n))
    if sat > 0:
        print("capture: chunk at pos", pos, "saturated", sat, "values to f16 range")
