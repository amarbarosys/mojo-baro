#!/usr/bin/env python3
"""Tiny GGUFs that drive every branch of tools/engine-pack.{py,mojo}. Fixture for tools/engine-pack-check.sh only;
tensor names are the real ones, sizes are not. Matrices cycle through BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K (random
blocks with finite f16 scales, some all-zero rows for d = 0), with F16 and F32 2D tensors where the order allows.

Usage: tools/engine-pack-fixture.py OUT.gguf KIND [SEED]
  KIND  trunk (qwen35 engine order + blk.32 head) | dense-llama (tied, split Q/K/V; layer 1 Q4_K/Q4_K/Q6_K, the fused-QKV split) |
        dense-qwen2 (bias, attn_gate, untied, one fused attn_qkv layer) | dense-bf16 | moe (qwen35moe)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
from gguf import GGUFWriter, GGMLQuantizationType as Q  # noqa: E402

BLK = {Q.Q4_0: (32, 18, [0]), Q.Q8_0: (32, 34, [0]), Q.Q4_K: (256, 144, [0, 2]), Q.Q5_K: (256, 176, [0, 2]),
       Q.Q6_K: (256, 210, [208])}
NE0 = 256


def blocks(rng, qt, rows, ne0):
    if qt == Q.BF16:
        x = rng.standard_normal((rows, ne0)).astype(np.float32) * 0.05
        x[0, :32] = 0
        return (x.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(rows, -1)
    per, nb, f16_at = BLK[qt]
    b = rng.integers(0, 256, size=(rows, ne0 // per, nb), dtype=np.uint8)
    for o in f16_at:
        b[:, :, o:o + 2] = rng.uniform(-0.05, 0.05, size=(rows, ne0 // per)).astype(np.float16).view(np.uint8).reshape(rows, ne0 // per, 2)
    b[0, 0, :] = 0
    return b.reshape(rows, -1)


class Fix:
    def __init__(self, out, arch, seed):
        self.rng = np.random.default_rng(seed)
        self.w = GGUFWriter(out, arch)
        self.k = seed

    def mat(self, name, rows, qt=None, ne0=NE0):
        cyc = [Q.BF16, Q.Q4_0, Q.Q8_0, Q.Q4_K, Q.Q5_K, Q.Q6_K]
        qt = qt if qt is not None else cyc[self.k % len(cyc)]
        self.k += 1
        self.w.add_tensor(name, blocks(self.rng, qt, rows, ne0), raw_dtype=qt)

    def f32(self, name, shape):
        self.w.add_tensor(name, self.rng.standard_normal(shape).astype(np.float32))

    def f16(self, name, shape):
        self.w.add_tensor(name, (self.rng.standard_normal(shape) * 0.05).astype(np.float16))

    def close(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


def trunk(f, seed):
    f.w.add_block_count(33)
    f.w.add_uint32("qwen35.nextn_predict_layers", 1)
    emb = [Q.BF16, Q.Q4_0, Q.Q8_0, Q.Q4_K, Q.Q6_K][seed % 5]
    f.mat("token_embd.weight", 8, emb)
    for i in range(33):
        b = f"blk.{i}."
        f.f32(b + "attn_norm.weight", NE0)
        if i == 32 or (i + 1) % 4 == 0:
            for n in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                f.mat(b + n, 2)
            f.f32(b + "attn_q_norm.weight", 32)
            f.f32(b + "attn_k_norm.weight", 32)
            f.mat(b + "attn_output.weight", 2)
        else:
            f.mat(b + "attn_qkv.weight", 3)
            f.mat(b + "attn_gate.weight", 2)
            f.f16(b + "ssm_alpha.weight", (2, NE0)) if i == 1 else f.mat(b + "ssm_alpha.weight", 2)
            f.mat(b + "ssm_beta.weight", 2)
            f.f32(b + "ssm_conv1d.weight", (3, 4))
            f.f32(b + "ssm_a", 4)
            f.f32(b + "ssm_dt.bias", 4)
            f.f32(b + "ssm_norm.weight", 32)
            f.mat(b + "ssm_out.weight", 2)
        f.f32(b + "post_attention_norm.weight", NE0)
        for n in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            f.mat(b + n, 2)
        if i == 32:
            f.mat(b + "nextn.eh_proj.weight", 2)
            for n in ("nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"):
                f.f32(b + n, NE0)
    f.f32("output_norm.weight", NE0)
    f.mat("output.weight", 8, [Q.Q6_K, Q.BF16, Q.Q4_0][seed % 3])


def dense(f, kind, seed):
    f.w.add_block_count(2)
    bf = kind == "dense-bf16"
    qt = (lambda: Q.BF16) if bf else (lambda: None)
    f.mat("token_embd.weight", 8, Q.BF16 if bf else [Q.Q4_K, Q.Q6_K, Q.Q8_0, Q.Q4_0][seed % 4])
    for i in range(2):
        b = f"blk.{i}."
        f.f32(b + "attn_norm.weight", NE0)
        if kind == "dense-qwen2" and i == 1:
            f.mat(b + "attn_qkv.weight", 4)
        elif kind == "dense-llama" and i == 1:
            f.mat(b + "attn_q.weight", 2, Q.Q4_K)
            f.mat(b + "attn_k.weight", 1, Q.Q4_K)
            f.mat(b + "attn_v.weight", 1, Q.Q6_K)
        else:
            f.mat(b + "attn_q.weight", 2, qt())
            f.mat(b + "attn_k.weight", 1, qt())
            f.f16(b + "attn_v.weight", (1, NE0)) if bf else f.mat(b + "attn_v.weight", 1)
        if kind == "dense-qwen2":
            f.f32(b + "attn_q.bias", 2 * 4)
            f.f32(b + "attn_k.bias", 4)
            f.f32(b + "attn_v.bias", 4)
            f.mat(b + "attn_gate.weight", 2)
        f.mat(b + "attn_output.weight", 2, qt())
        f.f32(b + "ffn_norm.weight", NE0)
        for n in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            f.mat(b + n, 2, qt())
    f.f32("output_norm.weight", NE0)
    if kind == "dense-qwen2":
        f.mat("output.weight", 8, Q.Q6_K)


def moe(f, seed):
    f.mat("token_embd.weight", 8, Q.Q4_K)
    for i in range(2):
        b = f"blk.{i}."
        f.f32(b + "attn_norm.weight", NE0)
        for n in ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight", "attn_output.weight"):
            f.mat(b + n, 2, Q.Q8_0)
        f.mat(b + "ffn_up_shexp.weight", 2, Q.Q8_0)
        f.f32(b + "ffn_gate_inp.weight", (4, NE0))
        f.f32(b + "ssm_a", 4)
        for n, qt in (("ffn_gate_exps.weight", Q.Q4_K), ("ffn_up_exps.weight", Q.Q5_K), ("ffn_down_exps.weight", Q.Q6_K)):
            f.w.add_tensor(b + n, blocks(f.rng, qt, 4 * 2, NE0).reshape(4, 2, -1), raw_dtype=qt)
    f.f32("output_norm.weight", NE0)
    f.mat("output.weight", 8, [Q.Q6_K, Q.Q8_0][seed % 2])


def main():
    out, kind = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    arch = {"trunk": "qwen35", "dense-llama": "llama", "dense-qwen2": "qwen2", "dense-bf16": "granite",
            "moe": "qwen35moe"}[kind]
    f = Fix(out, arch, seed)
    if kind == "trunk":
        trunk(f, seed)
    elif kind == "moe":
        moe(f, seed)
    else:
        dense(f, kind, seed)
    f.close()


if __name__ == "__main__":
    main()
