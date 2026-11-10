from std.ffi import external_call
from std.memory import UnsafePointer, alloc

def worker(arg: OpaquePointer[MutAnyOrigin]) abi("C") -> OpaquePointer[MutAnyOrigin]:
    var p = arg.bitcast[Int]()
    p[0] = p[0] * 10
    return arg

def main():
    comptime N = 8
    var vals = alloc[Int](N)
    var tids = alloc[UInt64](N)
    for i in range(N):
        vals[i] = i + 1
        _ = external_call["pthread_create", Int32](tids + i, Int(0), worker, (vals + i).bitcast[NoneType]())
    for i in range(N):
        _ = external_call["pthread_join", Int32](tids[i], Int(0))
    var s = 0
    for i in range(N):
        s += vals[i]
    print("pthread sum", s, "expect 360")
