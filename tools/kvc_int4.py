#!/usr/bin/env python3
"""BAROKV4 / BAROKV5: the int8 pool's rows requantised to 4 bits, nibble-packed (own codec, tier 2a).
usage: kvc_int4.py BAROST03_IN PAYLOAD_OUT RECON_BAROST03_OUT [--kv5]
Payload = the file's fixed part (header with magic BAROKV4, ids, conv, SSM) | for K then V:
  per page tail scales as in BAROST03 (128 f32 per (page, layer, head)) | rows packed 2 values
  per byte (low nibble first), q = round(v * 7 / 127) in [-7, 7] stored as q + 8.
--kv5 (C21): magic BAROKV5, conv and SSM stored as f16, each page tail as 128 f16 scales.
RECON = a BAROST03 file whose rows hold round(q * 127 / 7) (what the decoder writes into the
pool) and, for kv5, whose conv/SSM/scales are the f16-rounded values; for the quality gate on the
int8 engine: fork(RECON) vs fork(original) is the arm.
Prints bytes and the mean |error| relative to the row scale.
"""
import sys
import numpy as np
NAT, NKVH, HD, KVPAGE = 8, 4, 256, 128
TAIL = KVPAGE * 4
KVHSTR = KVPAGE * HD + TAIL
MX = 7

def main():
    src, pay_out, recon_out = sys.argv[1:4]
    kv5 = "--kv5" in sys.argv[4:]
    raw = np.fromfile(src, dtype=np.uint8)
    assert raw[:8].tobytes() == b"BAROST03", raw[:8].tobytes()
    pos, conv_n, ssm_n, kv_n = (int(x) for x in raw[8:40].view(np.int64))
    fixed = 72 + 4 * pos + 4 * (conv_n + ssm_n)
    pages = -(-pos // KVPAGE)
    assert pages * NAT * NKVH * KVHSTR == kv_n and raw.size == fixed + 2 * kv_n
    head = raw[:fixed].copy(); head[:8] = np.frombuffer(b"BAROKV5" if kv5 else b"BAROKV4", dtype=np.uint8).tolist() + [0]
    recon = raw.copy()
    if kv5:
        st = raw[72 + 4 * pos: fixed].view(np.float32).astype(np.float16)
        recon[72 + 4 * pos: fixed] = st.astype(np.float32).view(np.uint8)
        head = np.concatenate([head[:72 + 4 * pos], st.view(np.uint8)])
    payload = [head.tobytes()]
    err_sum = 0.0; n_vals = 0
    for kv in range(2):
        pool = raw[fixed + kv * kv_n: fixed + (kv + 1) * kv_n].reshape(pages * NAT * NKVH, KVHSTR)
        rows = pool[:, :KVPAGE * HD].view(np.int8).astype(np.int32)      # [blocks, 128*256]
        tails = pool[:, KVPAGE * HD:]
        q = np.rint(rows * MX / 127).astype(np.int32)                     # [-7, 7]
        back = np.rint(q * 127 / MX).astype(np.int32)
        err_sum += np.abs(back - rows).sum(); n_vals += rows.size
        u = (q + 8).astype(np.uint8)                                      # 1..15
        packed = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)        # [blocks, 128*256/2]
        rp = recon[fixed + kv * kv_n: fixed + (kv + 1) * kv_n].reshape(pages * NAT * NKVH, KVHSTR)
        rp[:, :KVPAGE * HD] = back.astype(np.int8).view(np.uint8)
        if kv5:
            t16 = tails.copy().view(np.float32).astype(np.float16)
            rp[:, KVPAGE * HD:] = t16.astype(np.float32).view(np.uint8)
            tails = t16.view(np.uint8)
        payload.append(tails.tobytes()); payload.append(packed.tobytes())
    pay = b"".join(payload)
    open(pay_out, "wb").write(pay); recon.tofile(recon_out)
    print(f"{head[:7].tobytes().decode()} payload {len(pay)} B ({raw.size / len(pay):.2f}x of BAROST03 {raw.size}), recon {recon.size} B, mean |err| {err_sum / n_vals / 127:.4f} of row scale, pos {pos}")
main()
