"""BARO_TIER_CPU worker (Round 5): one pthread that owns expert_ffn_cpu so the host thread only plans.

The host fills a CpuJob and bumps `seq`; the worker converts x, computes the CPU misses into `dst`
(coherent memory the GPU's wait kernel reads), raises `flag2 = flag2_val`, then bumps `done`. It spins
while jobs arrive and sleeps in 100 us steps after 2 ms without one, so an idle server burns no core.
"""
from std.atomic import Atomic, Ordering, fence
from std.ffi import external_call
from std.memory import alloc
from std.sys import llvm_intrinsic
from std.time import perf_counter_ns
from expert_cpu import expert_ffn_cpu

comptime IDLE_NS = 2_000_000


struct CpuJob:
    var seq: MutPointer[Int32, MutUntrackedOrigin]
    var done: MutPointer[Int32, MutUntrackedOrigin]
    var stop: MutPointer[Int32, MutUntrackedOrigin]
    var gate: MutPointer[UInt8, MutUntrackedOrigin]
    var up: MutPointer[UInt8, MutUntrackedOrigin]
    var down: MutPointer[UInt8, MutUntrackedOrigin]
    var ids: MutPointer[Int32, MutUntrackedOrigin]
    var wts: MutPointer[Float32, MutUntrackedOrigin]
    var n: Int
    var x_bf: MutPointer[BFloat16, MutUntrackedOrigin]
    var x_f: MutPointer[Float32, MutUntrackedOrigin]
    var hbuf: MutPointer[Float32, MutUntrackedOrigin]
    var xq: MutPointer[Int8, MutUntrackedOrigin]
    var aux: MutPointer[Float32, MutUntrackedOrigin]
    var dst: MutPointer[Float32, MutUntrackedOrigin]
    var flag2: MutPointer[Int32, MutUntrackedOrigin]
    var flag2_val: Int32
    var ns: Int


def cpu_worker_main[H: Int, FFN: Int](arg: Int) abi("C") -> Int:
    var j = MutPointer[CpuJob, MutUntrackedOrigin](unsafe_from_address=arg)
    var seen = Int32(0)
    var spins = 0
    var idle = False
    var t_idle = perf_counter_ns()
    while True:
        var s = Atomic[Int32].load[ordering=Ordering.ACQUIRE](j[].seq)
        if s == seen:
            if Atomic[Int32].load[ordering=Ordering.ACQUIRE](j[].stop) != 0:
                break
            if idle:
                _ = external_call["usleep", Int32](Int32(100))
            else:
                llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()
                spins += 1
                if spins & 255 == 0 and perf_counter_ns() - t_idle > IDLE_NS:
                    idle = True
            continue
        seen = s
        var t0 = perf_counter_ns()
        if j[].n > 0:
            for i in range(H):
                j[].x_f[i] = j[].x_bf[i].cast[DType.float32]()
            expert_ffn_cpu[H, FFN](j[].gate, j[].up, j[].down, j[].ids, j[].wts, j[].n, j[].x_f, j[].hbuf, j[].xq, j[].aux, j[].dst)
        else:
            for i in range(H):
                j[].dst[i] = 0
        fence[Ordering.RELEASE]()
        j[].flag2.unsafe_store[volatile=True](j[].flag2_val)
        j[].ns = Int(perf_counter_ns() - t0)
        Atomic[Int32].store[ordering=Ordering.RELEASE](j[].done, s)
        idle = False
        spins = 0
        t_idle = perf_counter_ns()
    return 0


def cpu_worker_start[H: Int, FFN: Int](job: MutPointer[CpuJob, MutUntrackedOrigin]) raises -> Int:
    """Zero the counters, then spawn the pthread. Returns its id."""
    var ctl = alloc[Int32](48)
    for i in range(48):
        ctl[i] = 0
    job[].seq = ctl
    job[].done = ctl + 16
    job[].stop = ctl + 32
    job[].n = 0
    job[].ns = 0
    var tid = alloc[UInt64](1)
    var rc = external_call["pthread_create", Int32](tid, Int(0), cpu_worker_main[H, FFN], Int(job))
    if rc != 0:
        raise Error("cpu worker: pthread_create rc " + String(rc))
    return Int(tid[0])


@always_inline
def cpu_worker_submit(job: MutPointer[CpuJob, MutUntrackedOrigin], seq: Int32):
    Atomic[Int32].store[ordering=Ordering.RELEASE](job[].seq, seq)


@always_inline
def cpu_worker_done(job: MutPointer[CpuJob, MutUntrackedOrigin], seq: Int32) -> Bool:
    return Atomic[Int32].load[ordering=Ordering.ACQUIRE](job[].done) == seq
