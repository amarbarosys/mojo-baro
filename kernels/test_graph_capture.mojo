"""Graph-capture probe: MAX's DeviceContext stream is reachable as a raw hipStream_t through
libAsyncRTMojoBindings (AsyncRT_DeviceContext_stream, AsyncRT_DeviceStream_hip_stream), so HIP stream
capture can record ordinary ctx.enqueue_function launches into a hipGraph and replay them.
Checks: N captured launches of an add-one kernel replay to the right value, R times; prints the per-launch
cost of plain enqueue vs one graph launch.
"""
from std.memory import UnsafePointer
from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns
from std.sys import exit
from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime N = 1000
comptime REPS = 5
comptime ASYNCRT = ".venv/lib/python3.12/site-packages/modular/lib/libAsyncRTMojoBindings.so"


def add_one(p: UnsafePointer[Int32, MutAnyOrigin]):
    if thread_idx.x == 0:
        p[0] = p[0] + 1


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.int32](1)
    buf.enqueue_fill(0)
    ctx.synchronize()
    var p = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))
    var rt = OwnedDLHandle(ASYNCRT)
    var hip = OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")
    var ds = Int(0)
    var err = rt.get_function[Int]("AsyncRT_DeviceContext_stream")(UnsafePointer(to=ds), ctx._handle)
    if err != 0 or ds == 0:
        print("FAIL graph-capture: AsyncRT_DeviceContext_stream err", err)
        exit(1)
    var hs = Int(0)
    err = rt.get_function[Int]("AsyncRT_DeviceStream_hip_stream")(UnsafePointer(to=hs), ds)
    if err != 0 or hs == 0:
        print("FAIL graph-capture: AsyncRT_DeviceStream_hip_stream err", err)
        exit(1)
    print("hip stream", hs)
    ctx.enqueue_function[add_one](p, grid_dim=1, block_dim=64)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N):
        ctx.enqueue_function[add_one](p, grid_dim=1, block_dim=64)
    ctx.synchronize()
    var plain_us = Float64(perf_counter_ns() - t0) / 1e3 / N
    var rc = hip.get_function[Int32]("hipStreamBeginCapture")(hs, Int32(2))
    if rc != 0:
        print("FAIL graph-capture: hipStreamBeginCapture rc", rc)
        exit(1)
    for _ in range(N):
        ctx.enqueue_function[add_one](p, grid_dim=1, block_dim=64)
    var graph = Int(0)
    rc = hip.get_function[Int32]("hipStreamEndCapture")(hs, UnsafePointer(to=graph))
    if rc != 0 or graph == 0:
        print("FAIL graph-capture: hipStreamEndCapture rc", rc)
        exit(1)
    var nn = Int(0)
    _ = hip.get_function[Int32]("hipGraphGetNodes")(graph, Int(0), UnsafePointer(to=nn))
    var gx = Int(0)
    rc = hip.get_function[Int32]("hipGraphInstantiate")(UnsafePointer(to=gx), graph, Int(0), Int(0), Int(0))
    if rc != 0:
        print("FAIL graph-capture: hipGraphInstantiate rc", rc)
        exit(1)
    var t1 = perf_counter_ns()
    for _ in range(REPS):
        rc = hip.get_function[Int32]("hipGraphLaunch")(gx, hs)
        if rc != 0:
            print("FAIL graph-capture: hipGraphLaunch rc", rc)
            exit(1)
    rc = hip.get_function[Int32]("hipStreamSynchronize")(hs)
    var graph_us = Float64(perf_counter_ns() - t1) / 1e3 / Float64(N * REPS)
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=host, src_buf=buf)
    ctx.synchronize()
    var got = Int(host.unsafe_ptr()[0])
    var want = 1 + N + N * REPS
    print("graph nodes", nn, "value", got, "want", want, "| us/launch plain", plain_us, "graph", graph_us)
    if got != want or nn != N:
        print("FAIL graph-capture: value or node count")
        exit(1)
    print("PASS graph-capture")
