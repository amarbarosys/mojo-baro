from std.sys.info import is_amd_gpu
from std.sys import llvm_intrinsic
from max.gpu.primitives import warp
from layout.tensor_core import mma as _amd_mma


@always_inline
def mma(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    comptime if is_amd_gpu():
        _amd_mma(d, a, b, c)
    else:
        d = type_of(d)(0)


comptime GPU_SCOPE = "agent" if is_amd_gpu() else "device"


@always_inline
def _bytes(v: Int32, signed: Bool) -> SIMD[DType.int32, 4]:
    var r = SIMD[DType.int32, 4](0)
    comptime for i in range(4):
        var b = (v >> Int32(8 * i)) & Int32(0xFF)
        r[i] = b - Int32(256) if signed and b >= Int32(128) else b
    return r


@always_inline
def sdot4(a: Int32, b: Int32, c: Int32, clamp: Bool) -> Int32:
    comptime if is_amd_gpu():
        return llvm_intrinsic["llvm.amdgcn.sdot4", Int32](a, b, c, clamp)
    else:
        return c + (_bytes(a, True) * _bytes(b, True)).reduce_add()


@always_inline
def sudot4(asign: Bool, a: Int32, bsign: Bool, b: Int32, c: Int32, clamp: Bool) -> Int32:
    comptime if is_amd_gpu():
        return llvm_intrinsic["llvm.amdgcn.sudot4", Int32](asign, a, bsign, b, c, clamp)
    else:
        return c + (_bytes(a, asign) * _bytes(b, bsign)).reduce_add()


@always_inline
def perm(hi: UInt32, lo: UInt32, sel: UInt32) -> UInt32:
    comptime if is_amd_gpu():
        return llvm_intrinsic["llvm.amdgcn.perm", UInt32](hi, lo, sel)
    else:
        var both = (UInt64(hi) << 32) | UInt64(lo)
        var r = UInt32(0)
        comptime for i in range(4):
            var s = (sel >> UInt32(8 * i)) & UInt32(0xFF)
            r |= UInt32((both >> UInt64(8 * s)) & UInt64(0xFF)) << UInt32(8 * i)
        return r


@always_inline
def setprio[p: Int]():
    comptime if is_amd_gpu():
        llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](Int16(p))


@always_inline
def sleep1():
    comptime if is_amd_gpu():
        llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))


@always_inline
def realtime() -> Int64:
    comptime if is_amd_gpu():
        return llvm_intrinsic["llvm.amdgcn.s.sendmsg.rtn", Int64](Int32(131))
    else:
        return 0


@always_inline
def readlane(p: Float32, lane: Int32) -> Float32:
    comptime if is_amd_gpu():
        return llvm_intrinsic["llvm.amdgcn.readlane", Float32](p, lane)
    else:
        return warp.shuffle_idx(p, UInt32(lane))
