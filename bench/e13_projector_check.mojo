# E13 piece 2 CHECK: on random inputs, this Mojo host projector (e13_projector.mojo)
# must match torch's forward pass on the identical weights within 1e-5 max
# abs diff. tools/e13-projector-oracle.py (run first, AMDHQ .venv, torch as
# oracle only) writes the weights, a random k*H input, and torch's output for
# that input; this binary re-applies the same weights in Mojo and diffs.
# Pure host/CPU on both sides -- no GPU, no gpu-wait needed.
from std.os import getenv

from e13_projector import load_projector, apply_projector_k, bytes_to_f32_list


def main() raises:
    var h = atol(getenv("E13_PROJ_H", "4096"))
    var k = atol(getenv("E13_PROJ_K", "8"))
    var work_dir = getenv("E13_WORK_DIR", ".work/e13")

    var proj = load_projector(work_dir + "/projector-check-weights.bin", h)

    var input_bytes: List[UInt8]
    with open(work_dir + "/projector-check-input.bin", "r") as f:
        input_bytes = f.read_bytes(k * h * 4)
    var xs = bytes_to_f32_list(input_bytes, k * h)

    var expected_bytes: List[UInt8]
    with open(work_dir + "/projector-check-expected.bin", "r") as f:
        expected_bytes = f.read_bytes(k * h * 4)
    var expected = bytes_to_f32_list(expected_bytes, k * h)

    var got = apply_projector_k(proj, xs, k)

    var max_diff: Float64 = 0.0
    var worst_i = -1
    for i in range(k * h):
        var d = abs(Float64(got[i]) - Float64(expected[i]))
        if d > max_diff:
            max_diff = d
            worst_i = i

    print("CHECK k=", k, " h=", h, " max_abs_diff=", max_diff, " worst_idx=", worst_i)
    comptime threshold = 1e-5
    if max_diff < threshold:
        print("CHECK: PASS (< 1e-5)")
    else:
        print("CHECK: FAIL (>= 1e-5)")
        raise Error("projector max abs diff " + String(max_diff) + " >= 1e-5")
