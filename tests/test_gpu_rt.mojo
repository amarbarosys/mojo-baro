"""gpu_rt maps HIP names and flags to CUDA on sm_* builds and leaves them unchanged on gfx* builds.
Build twice: --target-accelerator gfx1100 and --target-accelerator sm_80 (no GPU needed to run)."""
from gpu_rt import RT_CUDA, rt_name, rt_flags


def check(got: String, want: String) raises:
    if got != want:
        raise Error("FAIL gpu_rt: got " + got + " want " + want)


def main() raises:
    var hip: List[String] = ["hipMalloc", "hipFree", "hipHostMalloc", "hipHostFree", "hipHostRegister",
        "hipHostUnregister", "hipHostGetDevicePointer", "hipIpcGetMemHandle", "hipIpcCloseMemHandle"]
    var cuda: List[String] = ["cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaFreeHost", "cudaHostRegister",
        "cudaHostUnregister", "cudaHostGetDevicePointer", "cudaIpcGetMemHandle", "cudaIpcCloseMemHandle"]
    for i in range(9):
        check(rt_name(hip[i]), cuda[i] if RT_CUDA else hip[i])
    check(String(rt_flags(UInt32(0x40000002))), "2" if RT_CUDA else String(UInt32(0x40000002)))
    check(String(rt_flags(UInt32(0x0A))), "10")
    print("GPU_RT OK", "cuda" if RT_CUDA else "hip")
