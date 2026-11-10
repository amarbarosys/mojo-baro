"""baro REALIGN (round 2): training-free expected embedding
e = softmax(logits) @ W_emb, computed entirely from state this function
owns -- it never reads b.logits_d or b.hn_d.

Round 1 read b.logits_d, populated only by the non-mega "launch path"
(serve/window.mojo:943-944); under mega=True (HARNESS's actual caller,
bench_latent_handoff.mojo, exactly E9's step_latent_raw loop) that
buffer is never written at all, so round 1 only worked because its own
test happened to force mega=False.

b.hn_d looked like the fix (the launch path's "post-final-norm hidden,
pre-head" buffer, serve/window.mojo:927-934), but kernels/mega.mojo's
only write to it (`Hn_`, amar_mega_token's `rms_f32_phase` call) is
gated on the internal `fold_head == 2` argument -- and every call site
in this repo (serve/window.mojo's three mega/mega_win launches, and
bench_hidden_dtype.mojo's step_latent_raw) passes fold_head in {0, 1}.
So b.hn_d is not stale under mega=True, it is simply never written by
the fused kernel at all; whatever is in it is whatever a *prior*,
unrelated non-mega chunk last left there (confirmed empirically:
kernels/test_realign.mojo with mega=True read hn_d row 0 and got a
real-looking, prompt-varying vector on every one of 5 prompts, but its
norm and argmax did not match the same prompt's actual final row --
it was an earlier prefill chunk's row 0, not the current position).

The one buffer every mega/non-mega/spec path keeps genuinely current
is b.x_d (the pre-final-norm residual stream: every layer of every
call, mega or not, writes through it). So this function re-derives the
post-final-norm hidden itself from b.x_d row 0, with the exact
formula and weight offset the launch path uses for the head
(serve/window.mojo:936-944: rmsc_k against output_norm.weight, matmul
against output.weight, off[w]/off[w+1] where
w = 1 + N_SSM*10 + N_ATT*7 + N_LAYERS*4 -- fixed, independent of the
q4-draft/pack_q4 bookkeeping `e` carries, verified against
.work/engine-pack-q4/index.txt line 426/427 == output_norm.weight /
output.weight), reusing b.p_v_d as the same GEMM partial scratch the
launch path uses (no fresh multi-hundred-MB allocation per call).

Final signature (HARNESS adopts verbatim):
    realign_expected_embedding(ctx, mut b: WindowBufs, mut e_dev: DeviceBuffer[f32], pack_q4: Bool) raises

Round 3: HARNESS's L8-raw arm (and E9 before it) reads b.hn_d directly for
the post-final-norm hidden and ships it as f32 -- also dead under mega=True
(same fold_head==2 gate, kernels/mega.mojo:1182). final_norm_hidden below
reuses step 1's b.x_d row read but through rms_h2 (amar_rmsnorm, f32 in/out,
registry.mojo:181) instead of rmsc_h2 (amar_rmsnorm_cast, casts to bf16) --
the raw arm wants the un-rounded value, so no bf16 anywhere in this path:
    final_norm_hidden(ctx, mut b: WindowBufs, mut h_dev: DeviceBuffer[f32]) raises
"""
from std.math import ceildiv

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from registry import *
from window import WindowBufs, tens_bf16, row_f32, tens_f32, gemm_w
from elementwise import amar_softmax_rows
from realign_kernels import amar_realign_gather, amar_realign_reduce, REALIGN_SPLIT

comptime HEAD_NORM_IDX = 1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4

comptime probs_layout = row_major[1, VOCAB]()
comptime part_layout = row_major[REALIGN_SPLIT, H]()

comptime realign_head_reduce_k = amar_skinny_reduce[type_of(p_v), type_of(probs_layout), 1]
comptime realign_softmax_k = amar_softmax_rows[type_of(probs_layout)]
comptime realign_gather_k = amar_realign_gather[type_of(emb_layout), type_of(probs_layout), type_of(part_layout)]
comptime realign_reduce_k = amar_realign_reduce[type_of(part_layout), type_of(h_layout)]


def realign_expected_embedding(
    ctx: DeviceContext,
    mut b: WindowBufs,
    mut e_dev: DeviceBuffer[f32],
    pack_q4: Bool,
) raises:
    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    ctx.enqueue_function[rmsc_h2](
        row_f32(ctx, b.x_d, 0, H, h2_layout),
        tens_f32(ctx, b.wbuf, b.off[HEAD_NORM_IDX], H, h_layout),
        TileTensor(curb_d, h2_layout),
        Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
    )

    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    gemm_w[VOCAB, H](
        ctx, TileTensor(curb_d, h2_layout), b.wbuf, b.off[HEAD_NORM_IDX + 1], b.kind[HEAD_NORM_IDX + 1],
        TileTensor(b.p_v_d, p_v), 1,
    )
    ctx.enqueue_function[realign_head_reduce_k](
        TileTensor(b.p_v_d, p_v), TileTensor(logits_d, probs_layout), Int32(1), Int32(VOCAB),
        grid_dim=ceildiv(VOCAB, 256), block_dim=256,
    )

    ctx.enqueue_function[realign_softmax_k](
        TileTensor(logits_d, probs_layout), Int32(VOCAB), grid_dim=1, block_dim=256,
    )

    var part_d = ctx.enqueue_create_buffer[f32](REALIGN_SPLIT * H)
    var Embd = tens_bf16(ctx, b.wbuf, b.off[0], VOCAB * H, emb_layout)
    ctx.enqueue_function[realign_gather_k](
        Embd, TileTensor(logits_d, probs_layout), TileTensor(part_d, part_layout), Int32(VOCAB), Int32(H),
        grid_dim=(ceildiv(H, 256), REALIGN_SPLIT), block_dim=256,
    )

    ctx.enqueue_function[realign_reduce_k](
        TileTensor(part_d, part_layout), TileTensor(e_dev, h_layout), Int32(H),
        grid_dim=ceildiv(H, 256), block_dim=256,
    )


def final_norm_hidden(
    ctx: DeviceContext,
    mut b: WindowBufs,
    mut h_dev: DeviceBuffer[f32],
) raises:
    ctx.enqueue_function[rms_h2](
        row_f32(ctx, b.x_d, 0, H, h2_layout),
        tens_f32(ctx, b.wbuf, b.off[HEAD_NORM_IDX], H, h_layout),
        TileTensor(h_dev, h2_layout),
        Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
    )


def head_logits(
    ctx: DeviceContext,
    mut b: WindowBufs,
    pack_q4: Bool,
    mut logits_dev: DeviceBuffer[f32],
) raises:
    """The trunk's own VOCAB logits at the current position (row 0 of
    `b.x_d`, same convention `final_norm_hidden` uses). Identical to
    `realign_expected_embedding`'s norm+GEMM prefix, stopping at the raw
    logits instead of continuing into its own softmax and expected-embedding
    reduction -- P5a (`bench/p5a-smoke-protocol.md`) needs the distribution
    itself, not an embedding-space summary of it."""
    var curb_d = ctx.enqueue_create_buffer[bf16](H)
    ctx.enqueue_function[rmsc_h2](
        row_f32(ctx, b.x_d, 0, H, h2_layout),
        tens_f32(ctx, b.wbuf, b.off[HEAD_NORM_IDX], H, h_layout),
        TileTensor(curb_d, h2_layout),
        Int32(H), Float32(1e-6), grid_dim=1, block_dim=256,
    )
    gemm_w[VOCAB, H](
        ctx, TileTensor(curb_d, h2_layout), b.wbuf, b.off[HEAD_NORM_IDX + 1], b.kind[HEAD_NORM_IDX + 1],
        TileTensor(b.p_v_d, p_v), 1,
    )
    ctx.enqueue_function[realign_head_reduce_k](
        TileTensor(b.p_v_d, p_v), TileTensor(logits_dev, probs_layout), Int32(1), Int32(VOCAB),
        grid_dim=ceildiv(VOCAB, 256), block_dim=256,
    )
