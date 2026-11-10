"""cpu_worker (Round 5): the pthread that runs expert_ffn_cpu for the tier. Gates: output equals the
direct call bitwise on 40 layers, flag2 raised by the worker, an n=0 job (zero + flag2 + done) round
trips in under 10 us, submit never blocks and main runs independent work while the worker computes,
a real n=2 job costs under 10 us more than the direct call. CPU only, no pack.

Build: ./.venv/bin/mojo build serve/test_cpu_worker.mojo -I serve -o .work/test_cpu_worker
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import alloc
from std.sys import exit, llvm_intrinsic
from std.time import perf_counter_ns
from expert_cpu import expert_ffn_cpu, Q4K, Q4K_BYTES
from cpu_worker import CpuJob, cpu_worker_start, cpu_worker_submit, cpu_worker_done

comptime H = 2048
comptime FFN = 512
comptime NE = 8
comptime LAYERS = 40
comptime EB = FFN * (H // Q4K) * Q4K_BYTES
comptime EBD = H * (FFN // Q4K) * Q4K_BYTES

comptime P8 = MutPointer[UInt8, MutUntrackedOrigin]


def fill(p: P8, n: Int, mut s: UInt64):
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        p[i] = UInt8(s >> 56)
    for b in range(n // Q4K_BYTES):
        var o = b * Q4K_BYTES
        p[o] = 0x1F
        p[o + 1] = 0x21
        p[o + 2] = 0x1F
        p[o + 3] = 0x1D


def median(mut v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def main() raises:
    var gate = alloc[UInt8](LAYERS * NE * EB)
    var up = alloc[UInt8](LAYERS * NE * EB)
    var down = alloc[UInt8](LAYERS * NE * EBD)
    var s = UInt64(7)
    fill(gate, LAYERS * NE * EB, s)
    fill(up, LAYERS * NE * EB, s)
    fill(down, LAYERS * NE * EBD, s)
    var xb = alloc[BFloat16](H)
    var x = alloc[Float32](H)
    for i in range(H):
        xb[i] = (Float32((i * 37) % 101 - 50) / 64).cast[DType.bfloat16]()
        x[i] = xb[i].cast[DType.float32]()
    var ids = alloc[Int32](NE)
    var wts = alloc[Float32](NE)
    for j in range(NE):
        ids[j] = Int32(NE - 1 - j)
        wts[j] = Float32(j + 1) / 36
    var hbuf = alloc[Float32](NE * FFN)
    var xq = alloc[Int8](H + NE * FFN)
    var aux = alloc[Float32](2 * (H + NE * FFN) // 32)
    var want = alloc[Float32](H)
    var got = alloc[Float32](H)
    var xw = alloc[Float32](H)
    var hw = alloc[Float32](NE * FFN)
    var qw = alloc[Int8](H + NE * FFN)
    var aw = alloc[Float32](2 * (H + NE * FFN) // 32)
    var f2 = alloc[Int32](1)
    f2[0] = 0

    var job = alloc[CpuJob](1)
    job[].gate = gate
    job[].up = up
    job[].down = down
    job[].ids = ids
    job[].wts = wts
    job[].x_bf = xb
    job[].x_f = xw
    job[].hbuf = hw
    job[].xq = qw
    job[].aux = aw
    job[].dst = got
    job[].flag2 = f2
    _ = cpu_worker_start[H, FFN](job)

    for _ in range(20):
        expert_ffn_cpu[H, FFN](gate, up, down, ids, wts, 2, x, hbuf, xq, aux, want)

    var seq = Int32(0)
    var bad = 0
    for l in range(LAYERS):
        expert_ffn_cpu[H, FFN](gate + l * NE * EB, up + l * NE * EB, down + l * NE * EBD, ids, wts, 2, x, hbuf, xq, aux, want)
        for c in range(H):
            got[c] = Float32(-1)
        job[].gate = gate + l * NE * EB
        job[].up = up + l * NE * EB
        job[].down = down + l * NE * EBD
        job[].n = 2
        seq += 1
        job[].flag2_val = seq
        cpu_worker_submit(job, seq)
        while not cpu_worker_done(job, seq):
            llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()
        if f2[0] != seq:
            print("FAIL flag2: worker left", f2[0], "want", seq)
            exit(1)
        for c in range(H):
            if got[c] != want[c]:
                bad += 1
    print("correctness: mismatching outputs", bad, "of", LAYERS * H)
    if bad != 0:
        print("FAIL correctness: worker output differs from the direct call")
        exit(1)

    var rt = List[Float64]()
    for _ in range(2000):
        job[].n = 0
        var t0 = perf_counter_ns()
        seq += 1
        job[].flag2_val = seq
        cpu_worker_submit(job, seq)
        while not cpu_worker_done(job, seq):
            llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()
        rt.append(Float64(perf_counter_ns() - t0) / 1e3)
    var rt_med = median(rt)
    print("n=0 job (zero dst, flag2, done) round trip us: median", rt_med, "p99", rt[len(rt) * 99 // 100], "max", rt[len(rt) - 1])

    var direct = List[Float64]()
    var viaw = List[Float64]()
    var submit_us = List[Float64]()
    var work_during = List[Float64]()
    var acc = Float32(0)
    for rep in range(300):
        var l = rep % LAYERS
        var t0 = perf_counter_ns()
        expert_ffn_cpu[H, FFN](gate + l * NE * EB, up + l * NE * EB, down + l * NE * EBD, ids, wts, 2, x, hbuf, xq, aux, want)
        direct.append(Float64(perf_counter_ns() - t0) / 1e3)
    for rep in range(300):
        var l = rep % LAYERS
        job[].gate = gate + l * NE * EB
        job[].up = up + l * NE * EB
        job[].down = down + l * NE * EBD
        job[].n = 2
        var t0 = perf_counter_ns()
        seq += 1
        job[].flag2_val = seq
        cpu_worker_submit(job, seq)
        var t1 = perf_counter_ns()
        submit_us.append(Float64(t1 - t0) / 1e3)
        var chunks = 0
        while not cpu_worker_done(job, seq):
            for k in range(64):
                acc += x[(chunks * 64 + k) % H] * 1.0001
            chunks += 1
        viaw.append(Float64(perf_counter_ns() - t0) / 1e3)
        work_during.append(Float64(chunks))
    var d_med = median(direct)
    var w_med = median(viaw)
    var s_med = median(submit_us)
    var c_med = median(work_during)
    print("direct expert_ffn_cpu n=2 us: median", d_med)
    print("via worker submit->done us: median", w_med, "(handoff cost", w_med - d_med, ")")
    print("submit call us: median", s_med, " independent 64-FMA chunks main ran meanwhile: median", c_med, "(acc", acc, ")")

    var wake = List[Float64]()
    for _ in range(20):
        _ = external_call["usleep", Int32](Int32(5000))
        job[].n = 0
        var t0 = perf_counter_ns()
        seq += 1
        job[].flag2_val = seq
        cpu_worker_submit(job, seq)
        while not cpu_worker_done(job, seq):
            llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()
        wake.append(Float64(perf_counter_ns() - t0) / 1e3)
    var wake_med = median(wake)
    print("wake from the idle sleep (5 ms gap) us: median", wake_med, "max", wake[len(wake) - 1])

    Atomic[Int32].store[ordering=Ordering.RELEASE](job[].stop, 1)

    var ok = True
    if rt_med >= 10:
        print("FAIL handoff: n=0 round trip", rt_med, "us >= 10")
        ok = False
    if s_med >= 2:
        print("FAIL submit blocks: submit call", s_med, "us")
        ok = False
    if c_med < 10:
        print("FAIL main not free: ran", c_med, "chunks while the worker computed")
        ok = False
    if w_med - d_med >= 10:
        print("FAIL job overhead: worker path", w_med - d_med, "us over the direct call")
        ok = False
    if wake[len(wake) - 1] >= 500:
        print("FAIL idle wake: max", wake[len(wake) - 1], "us >= 500")
        ok = False
    if not ok:
        exit(1)
    print("PASS cpu_worker")
