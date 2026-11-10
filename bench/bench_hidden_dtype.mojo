# bench_hidden_dtype.mojo — E9 HIDDEN Precision Probe (f32 vs bf16)
# Tests whether rounding the 8 HIDDEN vectors to bf16 before ingest leaves
# the receiver's 64 generated tokens identical on 20/20 prompts.

from std.math import ceildiv
from std.time import perf_counter_ns
from std.sys import has_accelerator

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major
from registry import *
from window import *
from harness import load_pack, alloc_bufs, Pack
from realign import final_norm_hidden
from serve_proto import default_sample_params

def step_latent_raw(
    ctx: DeviceContext,
    mut b: WindowBufs,
    cfg: WindowCfg,
    mut st: WindowState,
    latent_dev_vec: DeviceBuffer[f32],
) raises:
    # Inject latent vector directly into b.x_d (bypassing embed_k)
    var x_slice = DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False)
    ctx.enqueue_copy(dst_buf=x_slice, src_buf=latent_dev_vec)

    # Run megakernel with m=1
    var Xm = TileTensor(b.x_d, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var Toks = TileTensor(b.toks_d, toks_layout)
    var ConvStateAll = TileTensor(b.convstate_d, csall_layout)
    var SStateAll = TileTensor(b.sstate_d, ssall_layout)
    var Hnm0 = TileTensor(b.hn_d, xm_layout)
    var Dtok0 = TileTensor(b.dtok_d, dtok_layout)

    ctx.enqueue_function[mega_token_q4_k](
        b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
        TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
        TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
        TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
        TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
        TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
        TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(0), Int32(1), Int32(cfg.att_split),
        b.sv_d.unsafe_ptr(), Int32(0),
        grid_dim=MEGA_G, block_dim=ROW_THREADS,
    )
    st.ring = (st.ring + 1) % SLOTS
    st.pos_prev = st.pos
    st.pos += 1

def reset_buffers(ctx: DeviceContext, mut b: WindowBufs, prompt: List[Int], tmax: Int) raises:
    ctx.enqueue_memset(b.convstate_d, 0)
    ctx.enqueue_memset(b.sstate_d, 0)
    ctx.enqueue_memset(b.kc_d, 0)
    ctx.enqueue_memset(b.vc_d, 0)
    ctx.enqueue_memset(b.kc32_d, 0)
    ctx.enqueue_memset(b.vc32_d, 0)
    ctx.enqueue_memset(b.ctr_d, 0)
    ctx.enqueue_memset(b.prof_d, 0)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.synchronize()
    for i in range(tmax):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr(), tmax, owning=False), src_buf=toks_h)
    ctx.synchronize()

def read_prompt(path: String) raises -> List[Int]:
    var prompt = List[Int]()
    with open(path, "r") as f:
        var data = f.read_bytes()
        var val = 0
        var have = False
        for i in range(len(data)):
            var b = Int(data[i])
            if b >= 48 and b <= 57:
                val = val * 10 + (b - 48)
                have = True
            else:
                if have:
                    prompt.append(val)
                val = 0
                have = False
        if have:
            prompt.append(val)
    return prompt^

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var packdir = ".work/engine-pack-q4"
    var pack = load_pack(ctx, packdir)
    comptime tmax = 1088
    var bufs = alloc_bufs(ctx, pack, tmax)
    var wst = WindowState(
        pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0,
        tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0,
        fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0],
        grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False,
    )

    comptime K_LATENT = 8
    comptime GEN_STEPS = 64
    var latent_h = ctx.enqueue_create_host_buffer[f32](K_LATENT * H)
    var latent_bf16 = ctx.enqueue_create_host_buffer[f32](K_LATENT * H)
    var latent_dev = ctx.enqueue_create_buffer[f32](H)
    var toks_host = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.synchronize()

    var prompt_names = List[String]()
    prompt_names.append("p01-water")
    prompt_names.append("p02-python-fib")
    prompt_names.append("p03-story")
    prompt_names.append("p04-list-planets")
    prompt_names.append("p05-math")
    prompt_names.append("p06-translate")
    prompt_names.append("p07-json")
    prompt_names.append("p08-sql")
    prompt_names.append("p09-explain-gpu")
    prompt_names.append("p10-recipe")
    prompt_names.append("p11-email")
    prompt_names.append("p12-rust")
    prompt_names.append("p13-haiku")
    prompt_names.append("p14-history")
    prompt_names.append("p15-bash")
    prompt_names.append("p16-chat")
    prompt_names.append("p17-summarize")
    prompt_names.append("p18-regex")
    prompt_names.append("p19-numbers")
    prompt_names.append("p20-dialog")

    print("=========================================================================================")
    print("  E9 — HIDDEN Precision Probe: f32 vs bf16 (20 Prompts, K=8, 64 Gen Tokens)")
    print("=========================================================================================")
    print("Prompt | Prompt Toks | Match (64/64) | Differing | First Diff (f32 vs bf16) | Status")
    print("-------+-------------+---------------+-----------+--------------------------+---------")

    var pass_count = 0

    for p_idx in range(len(prompt_names)):
        var pname = prompt_names[p_idx]
        var ppath = "bench/mtp-prompts/" + pname + ".tokens"
        var prompt = read_prompt(ppath)
        var plen = len(prompt)

        # ---------------------------------------------------------------------
        # 1. Producer generates 8 latent steps
        # ---------------------------------------------------------------------
        reset_buffers(ctx, bufs, prompt, tmax)
        var pf_rows = 0
        var pf_tail = 0
        if plen - 1 >= PF_MIN:
            pf_rows = plen - 1
            pf_tail = pf_rows % MROWS
            if pf_tail == 0:
                pf_tail = MROWS

        var cfg_prompt = WindowCfg(
            pack_q4=pack.pack_q4, draft_q4=False, q4_off=pack.q4_off, e=pack.e, kcfg=2,
            spec=False, expert_trace=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
            dump=False, dump4=False, dump_layer=0, mega=True, att_split=1088, mega_win=False, dot3=False, pf_chunk=1024,
            pf_rows=pf_rows, pf_tail=pf_tail, n_total=plen + 1, fr_k=0, fr_off=0, fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=plen,
            sample=default_sample_params(), dump_pen=False,
        )
        wst.reset(perf_counter_ns())
        while wst.pos < plen:
            step_window(ctx, bufs, cfg_prompt, wst)
        ctx.synchronize()

        for s in range(K_LATENT):
            # bufs.hn_d is never written under mega=True (kernels/mega.mojo:1182
            # gates the write on fold_head==2; every launch here passes 1), so
            # reading it ships a stale earlier-chunk vector -- this is what
            # invalidated the 2026-09-09 E9 run. final_norm_hidden re-derives the
            # post-final-norm hidden from b.x_d instead (serve/realign.mojo).
            final_norm_hidden(ctx, bufs, latent_dev)
            var host_dst = latent_h.create_sub_buffer[f32](s * H, H)
            ctx.enqueue_copy(dst_buf=host_dst, src_buf=latent_dev)
            ctx.synchronize()
            step_latent_raw(ctx, bufs, cfg_prompt, wst, latent_dev)
        ctx.synchronize()

        # ---------------------------------------------------------------------
        # 2. Receiver Condition 1 (f32)
        # ---------------------------------------------------------------------
        reset_buffers(ctx, bufs, prompt, tmax)
        wst.reset(perf_counter_ns())
        while wst.pos < plen:
            step_window(ctx, bufs, cfg_prompt, wst)
        ctx.synchronize()

        for s in range(K_LATENT):
            var host_src = latent_h.create_sub_buffer[f32](s * H, H)
            ctx.enqueue_copy(dst_buf=latent_dev, src_buf=host_src)
            ctx.synchronize()
            step_latent_raw(ctx, bufs, cfg_prompt, wst, latent_dev)
        ctx.synchronize()

        var total_target = plen + K_LATENT + GEN_STEPS
        var cfg_gen = WindowCfg(
            pack_q4=pack.pack_q4, draft_q4=False, q4_off=pack.q4_off, e=pack.e, kcfg=2,
            spec=False, expert_trace=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
            dump=False, dump4=False, dump_layer=0, mega=True, att_split=1088, mega_win=False, dot3=False, pf_chunk=1024,
            pf_rows=0, pf_tail=0, n_total=total_target, fr_k=0, fr_off=0, fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=plen + K_LATENT,
            sample=default_sample_params(), dump_pen=False,
        )
        while wst.pos < total_target - 1:
            step_window(ctx, bufs, cfg_gen, wst)
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=toks_host, src_buf=bufs.toks_d)
        ctx.synchronize()
        var gen_f32 = List[Int]()
        for i in range(plen + K_LATENT, total_target):
            gen_f32.append(Int(toks_host[i]))

        # ---------------------------------------------------------------------
        # 3. Receiver Condition 2 (bf16)
        # ---------------------------------------------------------------------
        for i in range(K_LATENT * H):
            var x = latent_h[i]
            var bf = Scalar[DType.bfloat16](x)
            latent_bf16[i] = Float32(bf)

        reset_buffers(ctx, bufs, prompt, tmax)
        wst.reset(perf_counter_ns())
        while wst.pos < plen:
            step_window(ctx, bufs, cfg_prompt, wst)
        ctx.synchronize()

        for s in range(K_LATENT):
            var host_src = latent_bf16.create_sub_buffer[f32](s * H, H)
            ctx.enqueue_copy(dst_buf=latent_dev, src_buf=host_src)
            ctx.synchronize()
            step_latent_raw(ctx, bufs, cfg_prompt, wst, latent_dev)
        ctx.synchronize()

        while wst.pos < total_target - 1:
            step_window(ctx, bufs, cfg_gen, wst)
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=toks_host, src_buf=bufs.toks_d)
        ctx.synchronize()
        var gen_bf16 = List[Int]()
        for i in range(plen + K_LATENT, total_target):
            gen_bf16.append(Int(toks_host[i]))

        # ---------------------------------------------------------------------
        # 4. Compare f32 vs bf16
        # ---------------------------------------------------------------------
        var matches = 0
        var first_diff = -1
        for i in range(GEN_STEPS):
            if gen_f32[i] == gen_bf16[i]:
                matches += 1
            elif first_diff < 0:
                first_diff = i

        var pshort = pname.split("-")[0]
        var line = pshort + "    | "
        line += String(plen) + ("   " if plen < 10 else "  ") + "| "
        line += String(matches) + "/" + String(GEN_STEPS) + "       | "
        line += String(GEN_STEPS - matches) + ("        " if (GEN_STEPS - matches) < 10 else "       ") + "| "
        if matches == GEN_STEPS:
            line += "—                        | PASS"
            pass_count += 1
        else:
            line += "tok[" + String(first_diff) + "]: " + String(gen_f32[first_diff]) + " vs " + String(gen_bf16[first_diff]) + " | FAIL"
        print(line)

    print("-----------------------------------------------------------------------------------------")
    print("Summary:")
    print("  Exact Token Matches (64/64):", pass_count, "/ 20 Prompts")
    print("=========================================================================================")
    if pass_count == 20:
        print("VERDICT: E9 PASS — dtype=bf16 admitted for HIDDEN (payload halved to 64 KiB).")
    else:
        print("VERDICT: E9 FAIL — bf16 diverged; f32 mandatory for HIDDEN (128 KiB payload).")
