from std.sys.info import is_amd_gpu, _is_amd_rdna4
from std.sys import llvm_intrinsic, get_defined_bool
from max.gpu import lane_id
from max.gpu.primitives import warp
from layout.tensor_core import mma as _amd_mma

comptime EMU_GFX12_ROWS = is_amd_gpu() and not _is_amd_rdna4() and get_defined_bool["BARO_EMU_GFX12_ROWS", False]()
comptime GFX12_ROWS = _is_amd_rdna4() or EMU_GFX12_ROWS


@always_inline
def _row[G12: Bool](i: Int, half: Int) -> Int:
    comptime if G12:
        return i + 8 * half
    else:
        return 2 * i + half


@always_inline
def acc_row(base: Int, i: Int, half: Int) -> Int:
    comptime if GFX12_ROWS:
        return base + i + 8 * half
    else:
        return base + 2 * i + half


@always_inline
def _relayout[T: DType, //, TO12: Bool](x: SIMD[T, 8]) -> SIMD[T, 8]:
    var half = Int(lane_id()) // 16
    var oth = SIMD[T, 8](0)
    comptime for i in range(8):
        oth[i] = warp.shuffle_xor(x[i], UInt32(16))
    var r = SIMD[T, 8](0)
    comptime for i in range(8):
        var v0 = x[0]
        var v1 = x[0]
        comptime for th in range(2):
            comptime for sh in range(2):
                comptime for si in range(8):
                    comptime if _row[not TO12](si, sh) == _row[TO12](i, th):
                        var v = x[si]
                        comptime if sh != th:
                            v = oth[si]
                        comptime if th == 0:
                            v0 = v
                        else:
                            v1 = v
        r[i] = v0 if half == 0 else v1
    return r


@always_inline
def gfx11_rows[T: DType, //](x: SIMD[T, 8]) -> SIMD[T, 8]:
    comptime if GFX12_ROWS:
        return _relayout[False](x)
    else:
        return x


@always_inline
def mma[T: DType, //](mut d: SIMD[T, 8], a: SIMD, b: SIMD, c: SIMD[T, 8]):
    comptime if is_amd_gpu():
        comptime if EMU_GFX12_ROWS:
            _amd_mma(d, a, b, _relayout[False](c))
            d = _relayout[True](d)
        else:
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
