"""Parity: the dense pair megakernel row bodies (kernels/mega_rows.mojo) at R=2 against serial m=1, BIT-EXACT per phase on real Q4 pack weights (needs .work/bake27/verify3/pack)."""
from std.math import ceildiv
from std.sys import has_accelerator, size_of
from std.utils import StaticTuple
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import EW_THREADS
from matmul_skinny import ROW_WAVES, ROW_THREADS, ROW_VEC, q4_dot_blocks
from ssm import CONV, KDIM, NH_K, NH_V, SSTATE, SSM_EPS
from attn import HD, NQH, NKVH, KVT, KVQ8, TCAP, kv_off, kv_tab_off, NROT, attn_head_span
from model import H, FFN, QF, KV, N_LAYERS
from mega import grid_barrier, stamp, rope_cs, MEGA_G, RMS_EPS, ATT_SCALE, DATT_NLD
from mega_rows import rms_rows, amar_mega_rows

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i8 = DType.int8
comptime f16 = DType.float16

comptime R = 2
comptime T = R
comptime INNER = NH_V * SSTATE
comptime ATT_DIM = NQH * HD

struct SplitMix64:
    var s: UInt64
    def __init__(out self, seed: UInt64):
        self.s = seed
    def next(mut self) -> UInt64:
        self.s += 0x9E3779B97F4A7C15
        var z = self.s
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

def differ(name: String, a: MutPointer[UInt8, MutUntrackedOrigin], b: MutPointer[UInt8, MutUntrackedOrigin], n: Int) raises:
    var bad = 0
    var nz = 0
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
        if a[unsafe_offset=i] != 0:
            nz += 1
    print(name, "bytes", n, "differ", bad, "nonzero", nz)
    if bad != 0:
        var first = -1
        for i in range(n):
            if a[unsafe_offset=i] != b[unsafe_offset=i]:
                first = i
                break
        raise Error("FAIL parity diff on " + name + " first mismatch at byte " + String(first))

@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def test_rms_kernel[R_val: Int](
    xp: MutPointer[Scalar[f32], MutAnyOrigin],
    gp: MutPointer[Scalar[f32], MutAnyOrigin],
    cb: MutPointer[Scalar[bf16], MutAnyOrigin],
    xg: MutPointer[Scalar[f32], MutAnyOrigin],
):
    rms_rows[R_val](xp, gp, cb, xg)

@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def test_q4_proj_kernel[R_val: Int, OutDim: Int, KDim: Int](
    xp: MutPointer[Scalar[bf16], MutAnyOrigin],
    wp: MutPointer[Scalar[u8], MutAnyOrigin],
    op: MutPointer[Scalar[f32], MutAnyOrigin],
):
    var X = TileTensor(xp, row_major[R_val, KDim]())
    var Waq = TileTensor(wp, row_major[OutDim, KDim // 2]())
    var Was = TileTensor(wp.unsafe_offset(OutDim * KDim // 2).unsafe_bitcast[Scalar[f16]](), row_major[OutDim, KDim // 32]())
    var O = TileTensor(op, row_major[R_val, OutDim]())
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave = tid // WARP_SIZE
    var bid = Int(block_idx.x)
    var nwaves = Int(grid_dim.x) * ROW_WAVES
    var g = bid * ROW_WAVES + wave
    while g < OutDim:
        var t = q4_dot_blocks[R_val, 2](X, Waq, Was, g, lane, 0, KDim // 32, R_val)
        if lane == 0:
            comptime for r in range(R_val):
                O[r, g] = rebind[O.ElementType](t[r])
        g += nwaves

def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var g = SplitMix64(0x123456789ABCDEF0)

    # 1. Host inputs
    var x_f32_h = ctx.enqueue_create_host_buffer[f32](T * H)
    var x_bf16_h = ctx.enqueue_create_host_buffer[bf16](T * H)
    var gamma_h = ctx.enqueue_create_host_buffer[f32](H)
    for i in range(T * H):
        var v = Float32(Int(g.next() % 2001) - 1000) / 500.0
        x_f32_h[i] = v
        x_bf16_h[i] = BFloat16(v)
    for i in range(H):
        gamma_h[i] = Float32(Int(g.next() % 1000) + 1) / 1000.0

    var atto_in_h = ctx.enqueue_create_host_buffer[bf16](T * ATT_DIM)
    for i in range(T * ATT_DIM):
        atto_in_h[i] = BFloat16(Float32(Int(g.next() % 2001) - 1000) / 1000.0)

    var ffn_in_h = ctx.enqueue_create_host_buffer[bf16](T * FFN)
    for i in range(T * FFN):
        ffn_in_h[i] = BFloat16(Float32(Int(g.next() % 2001) - 1000) / 1000.0)

    var inner_in_h = ctx.enqueue_create_host_buffer[bf16](T * INNER)
    for i in range(T * INNER):
        inner_in_h[i] = BFloat16(Float32(Int(g.next() % 2001) - 1000) / 1000.0)

    var pack_file = open(".work/bake27/verify3/pack/pack.bin", "r")
    
    def load_pack_tensor(offset: Int, bytes_len: Int) raises {mut ctx, mut pack_file} -> DeviceBuffer[u8]:
        var hb = ctx.enqueue_create_host_buffer[u8](bytes_len)
        _ = pack_file.seek(offset)
        var n_read = pack_file.read(Span[UInt8](unsafe_ptr=hb.unsafe_ptr(), length=bytes_len))
        if n_read != bytes_len:
            raise Error("Failed to read pack tensor bytes")
        var db = ctx.enqueue_create_buffer[u8](bytes_len)
        ctx.enqueue_copy(dst_buf=db, src_buf=hb)
        return db^

    # Layer 0 (SSM) tensors
    var w_conv_d = load_pack_tensor(2542817280, CONV * H // 2 + CONV * H // 32 * 2)
    var w_z_d = load_pack_tensor(2572308480, INNER * H // 2 + INNER * H // 32 * 2)
    var w_alpha_d = load_pack_tensor(2590003200, NH_V * H // 2 + NH_V * H // 32 * 2)
    var w_beta_d = load_pack_tensor(2590141440, NH_V * H // 2 + NH_V * H // 32 * 2)
    var w_ssmo_d = load_pack_tensor(2590444416, H * INNER // 2 + H * INNER // 32 * 2)
    var w_ffng_d = load_pack_tensor(2608159616, FFN * H // 2 + FFN * H // 32 * 2)
    var w_ffnu_d = load_pack_tensor(2658294656, FFN * H // 2 + FFN * H // 32 * 2)
    var w_ffnd_d = load_pack_tensor(2708429696, H * FFN // 2 + H * FFN // 32 * 2)

    # Layer 3 (ATT) tensors
    var w_attq_d = load_pack_tensor(3190121088, QF * H // 2 + QF * H // 32 * 2)
    var w_attk_d = load_pack_tensor(3225510528, KV * H // 2 + KV * H // 32 * 2)
    var w_attv_d = load_pack_tensor(3228459648, KV * H // 2 + KV * H // 32 * 2)
    var w_atto_d = load_pack_tensor(3231410816, H * ATT_DIM // 2 + H * ATT_DIM // 32 * 2)

    # Device buffers for inputs
    var x_f32_d = ctx.enqueue_create_buffer[f32](T * H)
    var x_bf16_d = ctx.enqueue_create_buffer[bf16](T * H)
    var gamma_d = ctx.enqueue_create_buffer[f32](H)
    var atto_in_d = ctx.enqueue_create_buffer[bf16](T * ATT_DIM)
    var ffn_in_d = ctx.enqueue_create_buffer[bf16](T * FFN)
    var inner_in_d = ctx.enqueue_create_buffer[bf16](T * INNER)

    ctx.enqueue_copy(dst_buf=x_f32_d, src_buf=x_f32_h)
    ctx.enqueue_copy(dst_buf=x_bf16_d, src_buf=x_bf16_h)
    ctx.enqueue_copy(dst_buf=gamma_d, src_buf=gamma_h)
    ctx.enqueue_copy(dst_buf=atto_in_d, src_buf=atto_in_h)
    ctx.enqueue_copy(dst_buf=ffn_in_d, src_buf=ffn_in_h)
    ctx.enqueue_copy(dst_buf=inner_in_d, src_buf=inner_in_h)

    # Output buffers
    var cb_R_d = ctx.enqueue_create_buffer[bf16](T * H)
    var cb_M_d = ctx.enqueue_create_buffer[bf16](T * H)
    var fp_R_d = ctx.enqueue_create_buffer[f32](T * H)
    var fp_M_d = ctx.enqueue_create_buffer[f32](T * H)

    # 1. Test rms_rows: R=2 vs 2 x R=1
    ctx.enqueue_function[test_rms_kernel[2]](
        x_f32_d.unsafe_ptr(), gamma_d.unsafe_ptr(), cb_R_d.unsafe_ptr(), fp_R_d.unsafe_ptr(),
        grid_dim=MEGA_G, block_dim=ROW_THREADS,
    )
    for t in range(T):
        ctx.enqueue_function[test_rms_kernel[1]](
            x_f32_d.unsafe_ptr().unsafe_offset(t * H), gamma_d.unsafe_ptr(),
            cb_M_d.unsafe_ptr().unsafe_offset(t * H), fp_M_d.unsafe_ptr().unsafe_offset(t * H),
            grid_dim=MEGA_G, block_dim=ROW_THREADS,
        )

    # Projections helper test
    def test_pair_vs_serial[OutDim: Int, KDim: Int](
        inp: DeviceBuffer[bf16],
        weights: DeviceBuffer[u8],
        out_R: DeviceBuffer[f32],
        out_M: DeviceBuffer[f32],
    ) raises {imm ctx}:
        ctx.enqueue_function[test_q4_proj_kernel[2, OutDim, KDim]](
            inp.unsafe_ptr(), weights.unsafe_ptr(), out_R.unsafe_ptr(),
            grid_dim=MEGA_G, block_dim=ROW_THREADS,
        )
        for t in range(T):
            ctx.enqueue_function[test_q4_proj_kernel[1, OutDim, KDim]](
                inp.unsafe_ptr().unsafe_offset(t * KDim), weights.unsafe_ptr(),
                out_M.unsafe_ptr().unsafe_offset(t * OutDim),
                grid_dim=MEGA_G, block_dim=ROW_THREADS,
            )

    # Layer 0 SSM output buffers
    var conv_R_d = ctx.enqueue_create_buffer[f32](T * CONV)
    var conv_M_d = ctx.enqueue_create_buffer[f32](T * CONV)
    test_pair_vs_serial[CONV, H](x_bf16_d, w_conv_d, conv_R_d, conv_M_d)

    var z_R_d = ctx.enqueue_create_buffer[f32](T * INNER)
    var z_M_d = ctx.enqueue_create_buffer[f32](T * INNER)
    test_pair_vs_serial[INNER, H](x_bf16_d, w_z_d, z_R_d, z_M_d)

    var alpha_R_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    var alpha_M_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    test_pair_vs_serial[NH_V, H](x_bf16_d, w_alpha_d, alpha_R_d, alpha_M_d)

    var beta_R_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    var beta_M_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    test_pair_vs_serial[NH_V, H](x_bf16_d, w_beta_d, beta_R_d, beta_M_d)

    var ssmo_R_d = ctx.enqueue_create_buffer[f32](T * H)
    var ssmo_M_d = ctx.enqueue_create_buffer[f32](T * H)
    test_pair_vs_serial[H, INNER](inner_in_d, w_ssmo_d, ssmo_R_d, ssmo_M_d)

    # FFN output buffers
    var ffng_R_d = ctx.enqueue_create_buffer[f32](T * FFN)
    var ffng_M_d = ctx.enqueue_create_buffer[f32](T * FFN)
    test_pair_vs_serial[FFN, H](x_bf16_d, w_ffng_d, ffng_R_d, ffng_M_d)

    var ffnu_R_d = ctx.enqueue_create_buffer[f32](T * FFN)
    var ffnu_M_d = ctx.enqueue_create_buffer[f32](T * FFN)
    test_pair_vs_serial[FFN, H](x_bf16_d, w_ffnu_d, ffnu_R_d, ffnu_M_d)

    var ffnd_R_d = ctx.enqueue_create_buffer[f32](T * H)
    var ffnd_M_d = ctx.enqueue_create_buffer[f32](T * H)
    test_pair_vs_serial[H, FFN](ffn_in_d, w_ffnd_d, ffnd_R_d, ffnd_M_d)

    # Layer 3 Attention output buffers
    var attq_R_d = ctx.enqueue_create_buffer[f32](T * QF)
    var attq_M_d = ctx.enqueue_create_buffer[f32](T * QF)
    test_pair_vs_serial[QF, H](x_bf16_d, w_attq_d, attq_R_d, attq_M_d)

    var attk_R_d = ctx.enqueue_create_buffer[f32](T * KV)
    var attk_M_d = ctx.enqueue_create_buffer[f32](T * KV)
    test_pair_vs_serial[KV, H](x_bf16_d, w_attk_d, attk_R_d, attk_M_d)

    var attv_R_d = ctx.enqueue_create_buffer[f32](T * KV)
    var attv_M_d = ctx.enqueue_create_buffer[f32](T * KV)
    test_pair_vs_serial[KV, H](x_bf16_d, w_attv_d, attv_R_d, attv_M_d)

    var atto_R_d = ctx.enqueue_create_buffer[f32](T * H)
    var atto_M_d = ctx.enqueue_create_buffer[f32](T * H)
    test_pair_vs_serial[H, ATT_DIM](atto_in_d, w_atto_d, atto_R_d, atto_M_d)

    ctx.synchronize()

    # Comparisons
    def cmp[dt: DType](name: String, r: DeviceBuffer[dt], m: DeviceBuffer[dt], n: Int) raises {imm ctx}:
        var rh = ctx.enqueue_create_host_buffer[dt](n)
        var mh = ctx.enqueue_create_host_buffer[dt](n)
        ctx.enqueue_copy(dst_buf=rh, src_buf=r)
        ctx.enqueue_copy(dst_buf=mh, src_buf=m)
        ctx.synchronize()
        differ(name, rh.unsafe_ptr().unsafe_bitcast[UInt8](), mh.unsafe_ptr().unsafe_bitcast[UInt8](), n * size_of[Scalar[dt]]())

    cmp("rms_bf16", cb_R_d, cb_M_d, T * H)
    cmp("rms_f32", fp_R_d, fp_M_d, T * H)
    cmp("q4_conv", conv_R_d, conv_M_d, T * CONV)
    cmp("q4_z", z_R_d, z_M_d, T * INNER)
    cmp("q4_alpha", alpha_R_d, alpha_M_d, T * NH_V)
    cmp("q4_beta", beta_R_d, beta_M_d, T * NH_V)
    cmp("q4_ssmo", ssmo_R_d, ssmo_M_d, T * H)
    cmp("q4_ffng", ffng_R_d, ffng_M_d, T * FFN)
    cmp("q4_ffnu", ffnu_R_d, ffnu_M_d, T * FFN)
    cmp("q4_ffnd", ffnd_R_d, ffnd_M_d, T * H)
    cmp("q4_attq", attq_R_d, attq_M_d, T * QF)
    cmp("q4_attk", attk_R_d, attk_M_d, T * KV)
    cmp("q4_attv", attv_R_d, attv_M_d, T * KV)
    cmp("q4_atto", atto_R_d, atto_M_d, T * H)

    print("PASS: all 14 dense row kernel phases bit-identical across R=2 vs serial m=1")
