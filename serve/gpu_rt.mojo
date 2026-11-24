"""Host GPU runtime by vendor: HIP (libamdhip64) on AMD builds, the CUDA runtime on NVIDIA builds.

Callers keep HIP names and HIP flag values; `rt_name` and `rt_flags` translate them for CUDA.
The target is fixed at build time by `--target-accelerator` (gfx* -> HIP, sm_* -> CUDA).
Covered: hipMalloc, hipFree, hipHostMalloc, hipHostFree, hipHostRegister, hipHostUnregister,
hipHostGetDevicePointer, hipIpcGetMemHandle, hipIpcCloseMemHandle (same signatures in CUDA;
register flags Mapped=2, ReadOnly=8 and alloc flag Mapped=2 are equal in both)."""
from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys.info import _accelerator_arch

comptime RT_CUDA = _accelerator_arch().startswith("sm_") or _accelerator_arch().startswith("nvidia")
comptime HIP_COHERENT: UInt32 = 0x40000000


def rt_lib() raises -> OwnedDLHandle:
    comptime if RT_CUDA:
        return OwnedDLHandle(getenv("BARO_CUDART_LIB", "libcudart.so"))
    else:
        return OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")


def rt_name(hip: String) -> String:
    comptime if RT_CUDA:
        if hip == "hipHostMalloc":
            return "cudaHostAlloc"
        if hip == "hipHostFree":
            return "cudaFreeHost"
        return "cuda" + String(hip[byte=3:])
    else:
        return hip


def rt_flags(hip_flags: UInt32) -> UInt32:
    comptime if RT_CUDA:
        return hip_flags & ~HIP_COHERENT
    else:
        return hip_flags
