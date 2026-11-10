#!/usr/bin/env python3
"""usage: bench/tests/state-diff-fixture.py OUTDIR -- writes synthetic MoE-shaped state files for the state_diff differential test:
ref.st (BAROST01 f32), drift.st (ref + 1% noise), i8.st (BAROST03 KVG 256), i8g32.st (BAROST03 KVG 32),
short.st (other pos). Random but seeded; layer-dependent scale so per-layer rows differ."""
import pathlib, sys, numpy as np
KVPAGE, HD, NATT, NKVH = 128, 256, 10, 2
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
def head(magic, pos, conv, ssm, kvn, salt, toks):
    return magic + np.array([pos, conv.size, ssm.size, kvn], "<i8").tobytes() + salt + toks.astype("<i4").tobytes()
def f32(path, pos, conv, ssm, k, v, salt, toks):
    path.write_bytes(head(b"BAROST01", pos, conv, ssm, k.size, salt, toks) + conv.astype("<f4").tobytes() + ssm.astype("<f4").tobytes() + k.astype("<f4").tobytes() + v.astype("<f4").tobytes())
def q8(x, kvg):
    b = x.reshape(-1, KVPAGE, HD // kvg, kvg); sc = np.abs(b).max(-1, keepdims=True) / 127; sc[sc == 0] = 1
    q = np.clip(np.round(b / sc), -127, 127).astype(np.int8)
    return np.concatenate([q.reshape(-1, KVPAGE * HD).view(np.uint8), sc.astype("<f4").reshape(-1, KVPAGE * (HD // kvg)).view(np.uint8)], 1).tobytes()
def i8(path, pos, conv, ssm, k, v, salt, toks, kvg):
    kb, vb = q8(k, kvg), q8(v, kvg)
    path.write_bytes(head(b"BAROST03", pos, conv, ssm, len(kb), salt, toks) + conv.astype("<f4").tobytes() + ssm.astype("<f4").tobytes() + kb + vb)
pos = 300; pages = -(-pos // KVPAGE); salt = bytes(range(32)); toks = rng.integers(0, 151000, pos)
conv, ssm = rng.standard_normal(1536), rng.standard_normal(4096)
lay = (1 + np.arange(NATT))[None, :, None, None, None]
k = (rng.standard_normal((pages, NATT, NKVH, KVPAGE, HD)) * lay).reshape(-1); v = (rng.standard_normal((pages, NATT, NKVH, KVPAGE, HD)) * lay).reshape(-1)
f32(out / "ref.st", pos, conv, ssm, k, v, salt, toks)
f32(out / "drift.st", pos, conv * 1.01, ssm, k + 0.01 * rng.standard_normal(k.size), v, salt, toks)
i8(out / "i8.st", pos, conv, ssm, k, v, salt, toks, 256)
i8(out / "i8g32.st", pos, conv, ssm, k, v, salt, toks, 32)
f32(out / "short.st", 200, conv, ssm, k[: 2 * NATT * NKVH * KVPAGE * HD], v[: 2 * NATT * NKVH * KVPAGE * HD], salt, toks[:200])
