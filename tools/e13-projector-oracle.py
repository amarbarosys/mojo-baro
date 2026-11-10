"""E13 piece 2 oracle (torch as verifier only, per repo CLAUDE.md - never the
Mojo projector's implementation). Writes fixed-seed random weights for
Linear(H,H) -> GELU(tanh-approx) -> Linear(H,H), a random k*H input, and
torch's f32 CPU forward output for that input, in the flat binary format
bench/e13_projector.mojo documents and reads. Run before
bench/e13_projector_check.mojo; run with the AMDHQ .venv (torch/transformers
only live there), e.g.:
    ~/AMDHQ/.venv/bin/python tools/e13-projector-oracle.py
"""
import os

import torch
import torch.nn as nn

H = int(os.environ.get("E13_PROJ_H", "4096"))
K = int(os.environ.get("E13_PROJ_K", "8"))
WORK_DIR = os.environ.get("E13_WORK_DIR", ".work/e13")
SEED = int(os.environ.get("E13_PROJ_SEED", "1234"))


def f32_bytes(tensor):
    return tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()


def write_f32(path, tensor):
    with open(path, "wb") as f:
        f.write(f32_bytes(tensor))


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    fc1 = nn.Linear(H, H, dtype=torch.float32)
    fc2 = nn.Linear(H, H, dtype=torch.float32)
    gelu = nn.GELU(approximate="tanh")

    x = torch.randn(K, H, dtype=torch.float32)

    with torch.no_grad():
        y = fc2(gelu(fc1(x)))

    weights_path = os.path.join(WORK_DIR, "projector-check-weights.bin")
    with open(weights_path, "wb") as f:
        for t in (fc1.weight, fc1.bias, fc2.weight, fc2.bias):
            f.write(f32_bytes(t))

    write_f32(os.path.join(WORK_DIR, "projector-check-input.bin"), x)
    write_f32(os.path.join(WORK_DIR, "projector-check-expected.bin"), y)

    print(f"wrote {weights_path} and input/expected for k={K} h={H} seed={SEED}")
    print(f"torch output stats: mean={y.mean().item():.6f} std={y.std().item():.6f}")


if __name__ == "__main__":
    main()
