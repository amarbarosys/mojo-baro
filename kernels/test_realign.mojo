"""REALIGN live test (round 3): for 5 real prompts on .work/engine-pack-q4,
prefill through the prompt under mega=True (the E9/HARNESS configuration --
bench_latent_handoff.mojo runs every latent step through the megakernel),
call realign_expected_embedding and final_norm_hidden on the final row, and
dump the row's pre-final-norm hidden (b.x_d), e, and the f32 post-final-norm
hidden (h, round 3) to .work/realign-dump/ for tools/realign_oracle.py
(numpy) to recompute independently off the dumped hidden state and the
pack's own weights -- never against a Mojo-side logits/hn_d dump (see
serve/realign.mojo's docstring for why b.logits_d/b.hn_d are not usable
here).
"""
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major
from registry import *
from window import *
from harness import load_pack, alloc_bufs
from serve_proto import default_sample_params
from realign import realign_expected_embedding, final_norm_hidden
from realign_kernels import amar_realign_gather, amar_realign_reduce, REALIGN_SPLIT


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
    ctx.enqueue_copy(dst_buf=b.toks_d, src_buf=toks_h)
    ctx.synchronize()


def dump_f32(path: String, host: HostBuffer[f32], n: Int) raises:
    with open(path, "w") as f:
        var p = host.unsafe_ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=n * 4))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var packdir = ".work/engine-pack-q4"
    var pack = load_pack(ctx, packdir)
    var pack_q4 = pack.pack_q4
    var q4_off = pack.q4_off
    var e = pack.e

    comptime tmax = TMAX
    var bufs = alloc_bufs(ctx, pack, tmax)
    var wst = WindowState(
        pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0,
        tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0,
        fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0],
        grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False,
    )

    var prompt_names = List[String]()
    prompt_names.append("p01-water")
    prompt_names.append("p02-python-fib")
    prompt_names.append("p03-story")
    prompt_names.append("p04-list-planets")
    prompt_names.append("p05-math")

    var e_dev = ctx.enqueue_create_buffer[f32](H)
    var h_dev = ctx.enqueue_create_buffer[f32](H)
    var x0_h = ctx.enqueue_create_host_buffer[f32](H)
    var e_h = ctx.enqueue_create_host_buffer[f32](H)
    var h_h = ctx.enqueue_create_host_buffer[f32](H)

    print("REALIGN_SPLIT (gather row-split over VOCAB):", REALIGN_SPLIT)
    print("prompt | prefill_toks | realign_us")
    print("-------+--------------+-----------")

    for p_idx in range(len(prompt_names)):
        var pname = prompt_names[p_idx]
        var prompt = read_prompt("bench/mtp-prompts/" + pname + ".tokens")
        var plen = len(prompt)

        reset_buffers(ctx, bufs, prompt, tmax)
        var cfg_prompt = WindowCfg(
            pack_q4=pack_q4, draft_q4=False, q4_off=q4_off, e=e, kcfg=2,
            spec=False, expert_trace=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
            dump=False, dump4=False, dump_layer=-1, mega=True, att_split=TMAX, mega_win=False,
            dot3=False, pf_chunk=1024, pf_rows=0, pf_tail=0, n_total=plen, fr_k=0, fr_off=0,
            fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=plen, sample=default_sample_params(), dump_pen=False,
        )
        wst.reset(perf_counter_ns())
        while wst.pos < plen:
            step_window(ctx, bufs, cfg_prompt, wst)
        ctx.synchronize()

        ctx.synchronize()
        var t0 = perf_counter_ns()
        realign_expected_embedding(ctx, bufs, e_dev, pack_q4)
        final_norm_hidden(ctx, bufs, h_dev)
        ctx.synchronize()
        var dt_us = Float64(perf_counter_ns() - t0) / 1e3

        ctx.enqueue_copy(
            dst_buf=x0_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.x_d.unsafe_ptr(), H, owning=False),
        )
        ctx.enqueue_copy(dst_buf=e_h, src_buf=e_dev)
        ctx.enqueue_copy(dst_buf=h_h, src_buf=h_dev)
        ctx.synchronize()

        dump_f32(".work/realign-dump/" + pname + "-x0.f32", x0_h, H)
        dump_f32(".work/realign-dump/" + pname + "-e.f32", e_h, H)
        dump_f32(".work/realign-dump/" + pname + "-h.f32", h_h, H)

        var line = pname + " | " + String(plen) + " | " + String(dt_us)
        print(line)


    print("REALIGN test: dump complete -> .work/realign-dump/")
