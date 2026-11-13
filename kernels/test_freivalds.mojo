"""Gate for kernels/freivalds.mojo (exchange/freivalds-prereg-2026-10-12.md, frozen).

1. ref_eq_gpu   CPU reference (per-element mod-p, different summation order) == GPU u, v, w and
                verdict on 50 random shapes m, n, k in [1, 4096] (3 pinned shapes, 47 random).
2. correct_pass 1000/1000 PASS on a correct C at 4096^3.
   entry_fail   1000/1000 FAIL with one entry changed (random position, nonzero int32 delta;
                runs 0,1 mod 4 use +-(2^31-1), 2 mod 4 a small delta, 3 mod 4 a random one).
   row_fail / col_fail  100 each, one whole row / column shifted by a random nonzero delta.
   wrap_fail    100, one entry changed by INT32_MIN (sign-bit flip: the change is a multiple of
                2^31, invisible to any verifier working modulo 2^32 with an even coefficient).
3. speed_floor  full check (draw r, upload, 3 kernels, device compare, 16-byte copy, sync) at
                4096^3 <= 1.5x the time to read A + B + C at the measured D2D bandwidth.
   speed_gemm   full check < 5% of the int8 WMMA GEMM that produced C.
C comes from the int8 x int8 -> int32 WMMA GEMM in this file, itself checked exactly against a
host GEMM on four small shapes (the Freivalds PASS at every other shape is the GEMM's check).
Prints `PASS N/N` or `FAIL k/N <names>`. `--no-speed` (run-tests.sh) prints the timings but judges only 1-2: the
suite checks correctness, the frozen speed bars live in bench/freivalds-gate.sh (8/9, speed_gemm missed).
"""
from std.math import ceildiv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation
from arch import mma, acc_row

from freivalds import Freivalds, freivalds_check, fv_add, fv_term, FV_P, u64, i64, i32, UP

comptime i8 = DType.int8
comptime u8 = DType.uint8
comptime BP = MutPointer[Scalar[i8], MutAnyOrigin]
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]
comptime DIM = 4096
comptime GM_THREADS = 256


def gemm_i8(A: BP, B: BP, C: IP, m: Int32, n: Int32, k_dim: Int32):
    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var wm = wave // 2
    var wn = wave % 2
    var block_row = Int(block_idx.y) * 128
    var block_col = Int(block_idx.x) * 128
    var row0 = block_row + wm * 32
    var col0 = block_col + wn * 64
    var sa = stack_allocation[i8, address_space=AddressSpace.SHARED](row_major[128, 32]())
    var sb = stack_allocation[i8, address_space=AddressSpace.SHARED](row_major[128, 32]())
    var sav = sa.vectorize[1, 16]()
    var sbv = sb.vectorize[1, 16]()
    var acc = Array[SIMD[i32, 8], 8](fill=SIMD[i32, 8](0))
    var ar = tid // 2
    var ac = (tid % 2) * 16
    var bk = tid // 8
    var bc = (tid % 8) * 16
    for step in range(ceildiv(K, 32)):
        var k0 = step * 32
        var av = SIMD[i8, 16](0)
        var gr = block_row + ar
        if gr < M:
            if k0 + ac + 16 <= K:
                av = A.unsafe_offset(gr * K + k0 + ac).unsafe_load[width=16]()
            else:
                for e in range(16):
                    if k0 + ac + e < K:
                        av[e] = A[unsafe_offset=gr * K + k0 + ac + e]
        var bv = SIMD[i8, 16](0)
        var gk = k0 + bk
        var gc = block_col + bc
        if gk < K:
            if gc + 16 <= N:
                bv = B.unsafe_offset(gk * N + gc).unsafe_load[width=16]()
            else:
                for e in range(16):
                    if gc + e < N:
                        bv[e] = B[unsafe_offset=gk * N + gc + e]
        barrier()
        sav[ar, ac // 16] = rebind[sav.ElementType](av)
        comptime for e in range(16):
            sb[bc + e, bk] = rebind[sb.ElementType](bv[e])
        barrier()
        comptime for kk in range(2):
            var afr = Array[SIMD[i8, 16], 2](uninitialized=True)
            comptime for tm in range(2):
                afr[tm] = rebind[SIMD[i8, 16]](sav[wm * 32 + tm * 16 + h, kk])
            comptime for tn in range(4):
                var bfr = rebind[SIMD[i8, 16]](sbv[wn * 64 + tn * 16 + h, kk])
                comptime for tm in range(2):
                    var d = SIMD[i32, 8](0)
                    mma(d, afr[tm], bfr, acc[tm * 4 + tn])
                    acc[tm * 4 + tn] = d
    comptime for tm in range(2):
        comptime for tn in range(4):
            var c = col0 + tn * 16 + h
            if c < N:
                comptime for i in range(8):
                    var r = acc_row(row0 + tm * 16, i, half)
                    if r < M:
                        C[unsafe_offset=r * N + c] = acc[tm * 4 + tn][i]


def read_sum(src: UP, words: Int32, dst: UP):
    var gid = Int(block_idx.x) * GM_THREADS + Int(thread_idx.x)
    var stride = Int(block_idx.x) * 0 + 4096 * GM_THREADS
    var acc = SIMD[u64, 2](0)
    for j in range(gid * 2, Int(words), stride * 2):
        acc += src.unsafe_offset(j).unsafe_load[width=2]()
    dst[unsafe_offset=gid] = acc[0] + acc[1]


def poke(C: IP, start: Int32, stride: Int32, count: Int32, delta: Int32):
    var i = Int(block_idx.x) * GM_THREADS + Int(thread_idx.x)
    if i < Int(count):
        var at = Int(start) + i * Int(stride)
        C[unsafe_offset=at] = C[unsafe_offset=at] + delta


def run_gemm(ctx: DeviceContext, A: DeviceBuffer[i8], B: DeviceBuffer[i8], C: DeviceBuffer[i32], m: Int, n: Int, k: Int) raises:
    ctx.enqueue_function[gemm_i8](
        A.unsafe_ptr(), B.unsafe_ptr(), C.unsafe_ptr(), Int32(m), Int32(n), Int32(k),
        grid_dim=(ceildiv(n, 128), ceildiv(m, 128)), block_dim=GM_THREADS,
    )


def run_poke(ctx: DeviceContext, C: DeviceBuffer[i32], start: Int, stride: Int, count: Int, delta: Int32) raises:
    ctx.enqueue_function[poke](
        C.unsafe_ptr(), Int32(start), Int32(stride), Int32(count), delta,
        grid_dim=ceildiv(count, GM_THREADS), block_dim=GM_THREADS,
    )


def lcg(mut st: UInt64) -> UInt64:
    st = st * 6364136223846793005 + 1442695040888963407
    return st >> 24


def fill_i8(buf: HostBuffer[i8], count: Int, mut st: UInt64):
    for i in range(count):
        buf[i] = Int8(Int(lcg(st) & 0xFF) - 128)


def cpu_rowdot[T: DType](
    Mp: MutPointer[Scalar[T], MutUntrackedOrigin], x: MutPointer[Scalar[u64], MutUntrackedOrigin],
    dst: MutPointer[Scalar[u64], MutUntrackedOrigin], rows: Int, cols: Int
):
    def one(r: Int) {imm Mp, imm x, imm dst, imm cols}:
        var acc = UInt64(0)
        for j in range(cols):
            acc = fv_add(acc, fv_term(Mp[unsafe_offset=r * cols + j].cast[i64](), x[unsafe_offset=j]))
        dst[unsafe_offset=r] = acc
    parallelize(one, rows)


def cpu_gemm_eq(
    a: HostBuffer[i8], b: HostBuffer[i8], c: HostBuffer[i32], m: Int, n: Int, k: Int
) -> Int:
    var bad = 0
    for i in range(m):
        for j in range(n):
            var s = Int32(0)
            for t in range(k):
                s += Int32(a[i * k + t]) * Int32(b[t * n + j])
            if s != c[i * n + j]:
                bad += 1
    return bad


def median(v: List[Float64]) -> Float64:
    var s = v.copy()
    sort(s)
    return s[len(s) // 2]


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var a_h = ctx.enqueue_create_host_buffer[i8](DIM * DIM)
    var b_h = ctx.enqueue_create_host_buffer[i8](DIM * DIM)
    var c_h = ctx.enqueue_create_host_buffer[i32](DIM * DIM)
    var a_d = ctx.enqueue_create_buffer[i8](DIM * DIM)
    var b_d = ctx.enqueue_create_buffer[i8](DIM * DIM)
    var c_d = ctx.enqueue_create_buffer[i32](DIM * DIM)
    var u_ref = ctx.enqueue_create_host_buffer[u64](DIM)
    var v_ref = ctx.enqueue_create_host_buffer[u64](DIM)
    var w_ref = ctx.enqueue_create_host_buffer[u64](DIM)
    ctx.synchronize()
    var st = UInt64(20261012)
    var names = String("")
    var fails = 0
    var total = 0
    print("p = 2^61-1 =", FV_P, " r source /dev/urandom  operand int8  C int32")

    var gemm_bad = 0
    comptime small = [(17, 33, 65), (128, 128, 128), (1, 1, 1), (3, 200, 1)]
    comptime for s in range(4):
        comptime m = small[s][0]
        comptime n = small[s][1]
        comptime k = small[s][2]
        fill_i8(a_h, m * k, st)
        fill_i8(b_h, k * n, st)
        ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
        ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
        run_gemm(ctx, a_d, b_d, c_d, m, n, k)
        ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
        ctx.synchronize()
        var bad = cpu_gemm_eq(a_h, b_h, c_h, m, n, k)
        print("gemm_i8 vs host", m, n, k, "mismatches", bad)
        gemm_bad += bad
    total += 1
    if gemm_bad != 0:
        fails += 1
        names += " gemm_host"

    var ref_bad = 0
    for s in range(50):
        var m = 1 + Int(lcg(st) % DIM)
        var n = 1 + Int(lcg(st) % DIM)
        var k = 1 + Int(lcg(st) % DIM)
        if s == 0:
            m = DIM; n = DIM; k = DIM
        elif s == 1:
            m = 1; n = 1; k = 1
        elif s == 2:
            m = 17; n = 4095; k = 33
        fill_i8(a_h, m * k, st)
        fill_i8(b_h, k * n, st)
        ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
        ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
        run_gemm(ctx, a_d, b_d, c_d, m, n, k)
        ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
        ctx.synchronize()
        var fv = Freivalds(ctx, m, n, k)
        var ok = fv.run[i8](ctx, a_d, b_d, c_d)
        fv.read_back(ctx)
        cpu_rowdot[i8](b_h.unsafe_ptr(), fv.r_last_h.unsafe_ptr(), u_ref.unsafe_ptr(), k, n)
        cpu_rowdot[i8](a_h.unsafe_ptr(), u_ref.unsafe_ptr(), v_ref.unsafe_ptr(), m, k)
        cpu_rowdot[i32](c_h.unsafe_ptr(), fv.r_last_h.unsafe_ptr(), w_ref.unsafe_ptr(), m, n)
        var bad = 0
        for i in range(k):
            if u_ref[i] != fv.u_h[i]:
                bad += 1
        for i in range(m):
            if v_ref[i] != fv.v_h[i]:
                bad += 1
            if w_ref[i] != fv.w_h[i]:
                bad += 1
            if v_ref[i] != w_ref[i]:
                bad += 1
        if not ok:
            bad += 1
        if bad != 0 or s < 3:
            print("shape", m, n, k, "r0", fv.r_last_h[0], "ref-vs-gpu mismatches", bad, "verdict", ok)
        ref_bad += bad
    total += 1
    if ref_bad != 0:
        fails += 1
        names += " ref_eq_gpu"

    fill_i8(a_h, DIM * DIM, st)
    fill_i8(b_h, DIM * DIM, st)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
    run_gemm(ctx, a_d, b_d, c_d, DIM, DIM, DIM)
    ctx.synchronize()
    var fv = Freivalds(ctx, DIM, DIM, DIM)

    var passes = 0
    for _ in range(1000):
        if fv.run[i8](ctx, a_d, b_d, c_d):
            passes += 1
    print("correct C: PASS", passes, "/ 1000")
    total += 1
    if passes != 1000:
        fails += 1
        names += " correct_pass"

    var caught = 0
    for t in range(1000):
        var at = Int(lcg(st) % (DIM * DIM))
        var delta: Int32
        if t % 4 == 0:
            delta = Int32(2147483647)
        elif t % 4 == 1:
            delta = Int32(-2147483647)
        elif t % 4 == 2:
            delta = Int32(1 + Int(lcg(st) % 255)) * (Int32(1) if lcg(st) % 2 == 0 else Int32(-1))
        else:
            delta = Int32(0)
            while delta == 0:
                delta = Int32(Int(lcg(st) & 0xFFFFFFFF) - 2147483648)
        run_poke(ctx, c_d, at, 1, 1, delta)
        var ok = fv.run[i8](ctx, a_d, b_d, c_d)
        run_poke(ctx, c_d, at, 1, 1, -delta)
        if not ok:
            caught += 1
    print("one entry corrupted: FAIL", caught, "/ 1000")
    total += 1
    if caught != 1000:
        fails += 1
        names += " entry_fail"

    var row_caught = 0
    var col_caught = 0
    var wrap_caught = 0
    for t in range(100):
        var delta = Int32(0)
        while delta == 0:
            delta = Int32(Int(lcg(st) & 0xFFFFFFFF) - 2147483648)
        var i = Int(lcg(st) % DIM)
        var j = Int(lcg(st) % DIM)
        run_poke(ctx, c_d, i * DIM, 1, DIM, delta)
        if not fv.run[i8](ctx, a_d, b_d, c_d):
            row_caught += 1
        run_poke(ctx, c_d, i * DIM, 1, DIM, -delta)
        run_poke(ctx, c_d, j, DIM, DIM, delta)
        if not fv.run[i8](ctx, a_d, b_d, c_d):
            col_caught += 1
        run_poke(ctx, c_d, j, DIM, DIM, -delta)
        var at = Int(lcg(st) % (DIM * DIM))
        run_poke(ctx, c_d, at, 1, 1, Int32(-2147483648))
        if not fv.run[i8](ctx, a_d, b_d, c_d):
            wrap_caught += 1
        run_poke(ctx, c_d, at, 1, 1, Int32(-2147483648))
    print("row corrupted: FAIL", row_caught, "/ 100  column: FAIL", col_caught, "/ 100  INT32_MIN wrap: FAIL", wrap_caught, "/ 100")
    total += 3
    if row_caught != 100:
        fails += 1
        names += " row_fail"
    if col_caught != 100:
        fails += 1
        names += " col_fail"
    if wrap_caught != 100:
        fails += 1
        names += " wrap_fail"
    if not fv.run[i8](ctx, a_d, b_d, c_d):
        raise Error("C not restored after corruption rounds")

    comptime FLUSH = 256 * 1024 * 1024
    var copy_src = ctx.enqueue_create_buffer[u8](FLUSH)
    var copy_dst = ctx.enqueue_create_buffer[u8](FLUSH)
    var sum_d = ctx.enqueue_create_buffer[u64](4096 * GM_THREADS)
    ctx.synchronize()
    var bw = List[Float64]()
    var rbw = List[Float64]()
    for it in range(14):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=copy_dst, src_buf=copy_src)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[read_sum](copy_src.unsafe_ptr().unsafe_bitcast[Scalar[u64]](), Int32(FLUSH // 8), sum_d.unsafe_ptr(), grid_dim=4096, block_dim=GM_THREADS)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        if it >= 3:
            bw.append(2.0 * Float64(FLUSH) / Float64(t1 - t0))
            rbw.append(Float64(FLUSH) / Float64(t2 - t1))
    var gbps = median(bw)
    var rgbps = median(rbw)
    var bytes = Float64(4 * DIM * DIM + 2 * DIM * DIM)
    var floor_ms = bytes / (max(gbps, rgbps) * 1.0e6)

    var gemm_t = List[Float64]()
    for it in range(14):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        run_gemm(ctx, a_d, b_d, c_d, DIM, DIM, DIM)
        ctx.synchronize()
        if it >= 3:
            gemm_t.append(Float64(perf_counter_ns() - t0) / 1.0e6)
    var gemm_ms = median(gemm_t)

    var warm_t = List[Float64]()
    var cold_t = List[Float64]()
    var kern_t = List[Float64]()
    for it in range(24):
        run_gemm(ctx, a_d, b_d, c_d, DIM, DIM, DIM)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var ok = fv.run[i8](ctx, a_d, b_d, c_d)
        var t1 = perf_counter_ns()
        if not ok:
            raise Error("timed check failed on a correct C")
        ctx.enqueue_function[read_sum](copy_src.unsafe_ptr().unsafe_bitcast[Scalar[u64]](), Int32(FLUSH // 8), sum_d.unsafe_ptr(), grid_dim=4096, block_dim=GM_THREADS)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        ok = fv.run[i8](ctx, a_d, b_d, c_d)
        var t3 = perf_counter_ns()
        if not ok:
            raise Error("timed check failed on a correct C")
        ctx.enqueue_function[read_sum](copy_src.unsafe_ptr().unsafe_bitcast[Scalar[u64]](), Int32(FLUSH // 8), sum_d.unsafe_ptr(), grid_dim=4096, block_dim=GM_THREADS)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        fv.launch[i8](ctx, a_d, b_d, c_d)
        ctx.synchronize()
        var t5 = perf_counter_ns()
        if it >= 3:
            warm_t.append(Float64(t1 - t0) / 1.0e6)
            cold_t.append(Float64(t3 - t2) / 1.0e6)
            kern_t.append(Float64(t5 - t4) / 1.0e6)
    var warm_ms = median(warm_t)
    var cold_ms = median(cold_t)
    var kern_ms = median(kern_t)
    var api_t = List[Float64]()
    for it in range(8):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var ok = freivalds_check[i8](ctx, a_d, b_d, c_d, DIM, DIM, DIM)
        if not ok:
            raise Error("freivalds_check failed on a correct C")
        if it >= 3:
            api_t.append(Float64(perf_counter_ns() - t0) / 1.0e6)
    print("4096^3: D2D copy GB/s", gbps, " read-only GB/s", rgbps, " floor(A+B+C bytes at the best of the two) ms", floor_ms, " bytes", bytes)
    print("4096^3: gemm_i8 ms", gemm_ms, " check cold ms", cold_ms, " check warm (right after the GEMM) ms", warm_ms,
          " kernels-only cold ms", kern_ms, " one-shot freivalds_check (alloc+draw+check) ms", median(api_t))
    print("ratios: cold/floor", cold_ms / floor_ms, " warm/floor", warm_ms / floor_ms, " cold/gemm", cold_ms / gemm_ms,
          " warm/gemm", warm_ms / gemm_ms, " C stream GB/s (cold, kernels)", 4.0 * DIM * DIM / (kern_ms * 1.0e6))
    if len(argv()) > 1 and String(argv()[1]) == "--no-speed":
        print("speed rows not judged (--no-speed); frozen bars: bench/freivalds-gate.sh")
    else:
        total += 2
        if cold_ms > 1.5 * floor_ms:
            fails += 1
            names += " speed_floor"
        if cold_ms > 0.05 * gemm_ms:
            fails += 1
            names += " speed_gemm"

    if fails == 0:
        print("PASS", total, "/", total)
    else:
        print("FAIL", fails, "/", total, names)
        raise Error("gate failed")
