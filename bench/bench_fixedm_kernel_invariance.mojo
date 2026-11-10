# bench_fixedm_kernel_invariance.mojo — E10 Skinny GEMM position invariance probe
# Tests whether row 0 output is bit-identical when rows 1..3 or 1..7 hold filler set F1 vs F2.

from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.ffi import external_call

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from matmul_skinny import (
    amar_matmul_skinny_q4rowb, amar_matmul_skinny_q8row,
    SM, SPLITK, ROW_WAVES, ROW_THREADS,
)
from tio import load_into

comptime MAXM = 8
comptime K = 4096
comptime N = 12288
comptime QBYTES4 = N * (K // 2) + N * (K // 32) * 2

comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()
comptime p_layout = row_major[SPLITK, SM, N]()
comptime a_layout = row_major[MAXM, K]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8

comptime GR = ceildiv(N, ROW_WAVES)

def q4_tensors(
    ctx: DeviceContext, q4_dev: DeviceBuffer[u8]
) raises -> Tuple[TileTensor[u8, type_of(q4_layout), MutAnyOrigin], TileTensor[f16, type_of(s_layout), MutAnyOrigin]]:
    var qdp = q4_dev.unsafe_ptr()
    var qb = DeviceBuffer[u8](ctx, qdp, N * (K // 2), owning=False)
    var sb = DeviceBuffer[f16](ctx, (qdp + N * (K // 2)).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    return (TileTensor(qb, q4_layout), TileTensor(sb, s_layout))

def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime base = ".work/gguf/blk_0_ffn_gate_weight"

    var q4_host = ctx.enqueue_create_host_buffer[u8](QBYTES4)
    ctx.synchronize()
    load_into(base + ".q4.bin", q4_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * (K // 2))
    load_into(base + ".q4scales.bin", (q4_host.unsafe_ptr() + N * (K // 2)).unsafe_bitcast[UInt8](), N * (K // 32) * 2)

    var q4_dev = ctx.enqueue_create_buffer[u8](QBYTES4)
    ctx.enqueue_copy(dst_buf=q4_dev, src_buf=q4_host)
    ctx.synchronize()

    var a_dev = ctx.enqueue_create_buffer[bf16](MAXM * K)
    var a_host = ctx.enqueue_create_host_buffer[bf16](MAXM * K)

    var cp_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var cp_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()

    var A = TileTensor(a_dev, a_layout)
    var Cp = TileTensor(cp_dev, p_layout)
    var qtw = q4_tensors(ctx, q4_dev)
    var Q = qtw[0]
    var S = qtw[1]

    comptime q4rowb_1 = amar_matmul_skinny_q4rowb[2, 1, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(p_layout)]
    comptime q4rowb_4 = amar_matmul_skinny_q4rowb[2, 4, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(p_layout)]
    comptime q4rowb_8 = amar_matmul_skinny_q4rowb[2, 8, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(p_layout)]

    print("=========================================================================================")
    print("  E10 — Skinny GEMM FIXED(M) Position Invariance Probe (20 Prompts)")
    print("=========================================================================================")
    print("Prompt | M1 (5x control) | F(4, F1) == F(4, F2) | F(4, F1) == M1 | F(8, F1) == F(4, F1)")
    print("-------+-----------------+---------------------+----------------+----------------------")

    var pass_control = 0
    var pass_fixedm = 0
    var match_m1_fixedm = 0
    var match_m4_m8 = 0

    for p in range(20):
        # 1. Fill row 0 with a distinct deterministic pseudo-random prompt pattern
        var seed0: UInt64 = 10007 + UInt64(p) * 65537
        for k in range(K):
            seed0 = (seed0 * 6364136223846793005 + 1442695040888963407)
            var val_f = Float32(Int((seed0 >> 32) & 0xFFFF) - 32768) / 32768.0
            a_host[k] = Scalar[bf16](val_f)

        # 2. Control: M1 repeated 5 times
        var m1_control_ok = True
        var first_m1_val = List[Float32]()
        for rep in range(5):
            # Clear other rows to 0
            for r in range(1, MAXM):
                for k in range(K):
                    a_host[r * K + k] = 0
            ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
            ctx.enqueue_memset(cp_dev, 0)
            ctx.enqueue_function[q4rowb_1](A, Q, S, Cp, Int32(1), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.enqueue_copy(dst_buf=cp_host, src_buf=cp_dev)
            ctx.synchronize()

            if rep == 0:
                for col in range(N):
                    first_m1_val.append(Float32(cp_host[col]))
            else:
                for col in range(N):
                    if Float32(cp_host[col]) != first_m1_val[col]:
                        m1_control_ok = False
                        break

        if m1_control_ok:
            pass_control += 1

        # 3. Arm F(4, F1): Fill rows 1..3 with Filler Set 1 (pattern A)
        var seed1: UInt64 = 0xDEADBEEF + UInt64(p) * 1337
        for r in range(1, 4):
            for k in range(K):
                seed1 = (seed1 * 6364136223846793005 + 1)
                var val_f = Float32(Int((seed1 >> 32) & 0xFFFF) - 32768) / 32768.0
                a_host[r * K + k] = Scalar[bf16](val_f)
        for r in range(4, MAXM):
            for k in range(K):
                a_host[r * K + k] = 0

        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_memset(cp_dev, 0)
        ctx.enqueue_function[q4rowb_4](A, Q, S, Cp, Int32(4), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=cp_host, src_buf=cp_dev)
        ctx.synchronize()

        var f4_f1_val = List[Float32]()
        for col in range(N):
            f4_f1_val.append(Float32(cp_host[col]))

        # 4. Arm F(4, F2): Fill rows 1..3 with Filler Set 2 (COMPLETELY DIFFERENT pattern B)
        var seed2: UInt64 = 0xCAFEBABE + UInt64(p) * 99991
        for r in range(1, 4):
            for k in range(K):
                seed2 = (seed2 * 6364136223846793005 + 7)
                var val_f = Float32(Int((seed2 >> 32) & 0xFFFF) - 32768) / 32768.0
                a_host[r * K + k] = Scalar[bf16](val_f)
        for r in range(4, MAXM):
            for k in range(K):
                a_host[r * K + k] = 0

        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_memset(cp_dev, 0)
        ctx.enqueue_function[q4rowb_4](A, Q, S, Cp, Int32(4), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=cp_host, src_buf=cp_dev)
        ctx.synchronize()

        var f4_f2_match = True
        for col in range(N):
            if Float32(cp_host[col]) != f4_f1_val[col]:
                f4_f2_match = False
                break
        if f4_f2_match:
            pass_fixedm += 1

        # Check bonus: does F(4, F1) match M1?
        var m1_match = True
        for col in range(N):
            if f4_f1_val[col] != first_m1_val[col]:
                m1_match = False
                break
        if m1_match:
            match_m1_fixedm += 1

        # 5. Arm F(8, F1): 8 rows
        var seed3: UInt64 = 0x12345678 + UInt64(p) * 4321
        for r in range(4, 8):
            for k in range(K):
                seed3 = (seed3 * 6364136223846793005 + 3)
                var val_f = Float32(Int((seed3 >> 32) & 0xFFFF) - 32768) / 32768.0
                a_host[r * K + k] = Scalar[bf16](val_f)

        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_memset(cp_dev, 0)
        ctx.enqueue_function[q4rowb_8](A, Q, S, Cp, Int32(8), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=cp_host, src_buf=cp_dev)
        ctx.synchronize()

        var m8_match = True
        for col in range(N):
            if Float32(cp_host[col]) != f4_f1_val[col]:
                m8_match = False
                break
        if m8_match:
            match_m4_m8 += 1

        var s_ctrl = "5/5 PASS" if m1_control_ok else "FAIL"
        var s_fixedm = "PASS (exact)" if f4_f2_match else "FAIL (diverged)"
        var s_m1 = "PASS (exact)" if m1_match else "differs"
        var s_m8 = "PASS (exact)" if m8_match else "differs"

        var p_idx = String(p + 1)
        if p_idx.byte_length() == 1:
            p_idx = "0" + p_idx
        print("p" + p_idx + "     | " + s_ctrl + "    | " + s_fixedm + "       | " + s_m1 + "   | " + s_m8)

    print("-----------------------------------------------------------------------------------------")
    print("Summary:")
    print("  Control M1 (5/5 exact):         " + String(pass_control) + " / 20 PASS")
    print("  FIXED(4) Invariance (F1 == F2):  " + String(pass_fixedm) + " / 20 PASS")
    print("  M1 vs FIXED(4) Cross-match:     " + String(match_m1_fixedm) + " / 20 MATCH")
    print("  FIXED(4) vs FIXED(8) Match:     " + String(match_m4_m8) + " / 20 MATCH")
    print("=========================================================================================")

    if pass_fixedm == 20:
        print("\nVERDICT: E10 PASS — Skinny GEMM kernels are 100% position-invariant under FIXED(M).")
        print("         Row 0 output is bit-identical regardless of filler co-tenants (20/20).")
    else:
        print("\nVERDICT: E10 KILL — Co-tenant divergence detected; FIXED(M) is DYN.")
