#!/usr/bin/env python3
"""tools/pack-split-experts.py: split a MoE pack's routed expert tensors out
into their own file, so the engine can load a pack that does not contain
them (B4 stage 2b, docs/NEXT-PLAN.md B4 step 2: the trunk stays in VRAM,
routed experts move to host RAM).

serve/window.mojo addresses tensors by POSITION in index.txt, not by name,
so index.txt's line order and line count cannot change; only each line's
OFFSET moves (into pack.bin for non-expert tensors, into a new experts.bin
for routed expert tensors, both compacted with no holes) and expert lines
gain a fifth column ("expert") so the loader can tell the two apart --
existing readers split on spaces and use only the first four columns, so
this is invisible to them.

Byte-size-per-dtype formulas are transcribed from serve/harness.mojo's
load_pack (the one place they are allowed to drift from): bf16 = n*2,
f32 = n*4, q8 = n + (n//32)*2, q8_0 = (n//32)*34, q4_k = (n//256)*144,
q6_k = (n//256)*210, q4 = n//2 + (n//32)*2, i32 = n*4 (frdraft.ids only,
harness.mojo's own trailing-entry case). Any other dtype is a hard error,
never a guess.

Usage:
  tools/pack-split-experts.py SRCPACKDIR DSTDIR [--verify] [--verify-only]

--verify: after writing DSTDIR, sample at least 20 tensors spanning every
  dtype and both classes (expert, non-expert) present in the source, sha256-
  compare each one's bytes at its new offset against its bytes at its old
  offset, and assert the size arithmetic: sum(pack.bin sizes) +
  sum(experts.bin sizes) == source pack.bin size, and both new files'
  on-disk sizes match those sums exactly.
--verify-only: skip writing DSTDIR (it must already exist from a prior run)
  and just run the --verify check, so it can be re-run later without
  re-copying a multi-GB pack.
"""
import argparse
import os
import hashlib
import random
import re
from pathlib import Path

EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")


def is_expert(name):
    return bool(EXPERT_RE.match(name))


def dtype_size(dt, n):
    if dt == "bf16":
        return n * 2
    if dt == "f32":
        return n * 4
    if dt == "q8":
        return n + (n // 32) * 2
    if dt == "q8_0":
        return (n // 32) * 34
    if dt == "q4_k":
        return (n // 256) * 144
    if dt == "q6_k":
        return (n // 256) * 210
    if dt == "q4":
        return n // 2 + (n // 32) * 2
    if dt == "i32":
        return n * 4
    raise SystemExit(f"unknown pack dtype {dt!r} (name/offset omitted): "
                      f"refusing to guess its byte size, add it to "
                      f"serve/harness.mojo:load_pack first")


def parse_index(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split(" ")
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    return rows


def do_split(src, dst):
    rows = parse_index(src / "index.txt")
    src_pack_size = (src / "pack.bin").stat().st_size

    trunk_off = 0
    expert_off = 0
    trunk_total = 0
    expert_total = 0
    out_lines = []

    with open(src / "pack.bin", "rb") as fsrc, \
         open(dst / "pack.bin", "wb") as ftrunk, \
         open(dst / "experts.bin", "wb") as fexp:
        for name, dt, off, n in rows:
            size = dtype_size(dt, n)
            dest_f = fexp if is_expert(name) else ftrunk
            at = expert_off if is_expert(name) else trunk_off
            done = 0
            while done < size:
                got = os.copy_file_range(fsrc.fileno(), dest_f.fileno(), size - done,
                                         off + done, at + done)
                if got == 0:
                    raise SystemExit(f"short read for {name}: wanted {size} "
                                      f"bytes at {off}, source ended early")
                done += got
            if is_expert(name):
                out_lines.append(f"{name} {dt} {expert_off} {n} expert")
                expert_off += size
                expert_total += size
            else:
                out_lines.append(f"{name} {dt} {trunk_off} {n}")
                trunk_off += size
                trunk_total += size

    (dst / "index.txt").write_text("\n".join(out_lines) + "\n")

    assert trunk_total + expert_total == src_pack_size, (
        f"size mismatch: trunk {trunk_total} + experts {expert_total} = "
        f"{trunk_total + expert_total}, source pack.bin was {src_pack_size}"
    )
    dst_pack_size = (dst / "pack.bin").stat().st_size
    dst_experts_size = (dst / "experts.bin").stat().st_size
    assert dst_pack_size == trunk_total, (
        f"pack.bin on-disk size {dst_pack_size} != computed {trunk_total}")
    assert dst_experts_size == expert_total, (
        f"experts.bin on-disk size {dst_experts_size} != computed {expert_total}")

    return trunk_total, expert_total, src_pack_size, len(rows)


def do_verify(src, dst, min_samples=20, seed=0):
    src_rows = parse_index(src / "index.txt")
    dst_rows = parse_index(dst / "index.txt")
    if len(src_rows) != len(dst_rows):
        raise SystemExit(f"line count mismatch: src {len(src_rows)} lines, "
                          f"dst {len(dst_rows)} lines")

    src_pack_size = (src / "pack.bin").stat().st_size
    dst_pack_size = (dst / "pack.bin").stat().st_size
    dst_experts_size = (dst / "experts.bin").stat().st_size

    trunk_total = 0
    expert_total = 0
    for (sname, sdt, soff, sn), (dname, ddt, doff, dn) in zip(src_rows, dst_rows):
        if sname != dname or sdt != ddt or sn != dn:
            raise SystemExit(f"index row mismatch: src ({sname} {sdt} {soff} {sn}) "
                              f"vs dst ({dname} {ddt} {doff} {dn})")
        size = dtype_size(ddt, dn)
        if is_expert(dname):
            expert_total += size
        else:
            trunk_total += size

    assert trunk_total + expert_total == src_pack_size, (
        f"size mismatch: trunk {trunk_total} + experts {expert_total} = "
        f"{trunk_total + expert_total}, source pack.bin was {src_pack_size}"
    )
    assert dst_pack_size == trunk_total, (
        f"pack.bin on-disk size {dst_pack_size} != computed {trunk_total}")
    assert dst_experts_size == expert_total, (
        f"experts.bin on-disk size {dst_experts_size} != computed {expert_total}")

    groups = {}
    for i, (name, dt, off, n) in enumerate(src_rows):
        groups.setdefault((dt, is_expert(name)), []).append(i)

    rng = random.Random(seed)
    per_group = max(1, -(-min_samples // len(groups)))
    sample_idx = set()
    for key, idxs in groups.items():
        k = min(per_group, len(idxs))
        sample_idx.update(rng.sample(idxs, k))
    if len(sample_idx) < min_samples:
        remaining = [i for i in range(len(src_rows)) if i not in sample_idx]
        need = min_samples - len(sample_idx)
        sample_idx.update(rng.sample(remaining, min(need, len(remaining))))
    sample_idx = sorted(sample_idx)

    checked = 0
    dtypes_seen = set()
    classes_seen = set()
    with open(src / "pack.bin", "rb") as fsrc, \
         open(dst / "pack.bin", "rb") as ftrunk, \
         open(dst / "experts.bin", "rb") as fexp:
        for i in sample_idx:
            name, dt, old_off, n = src_rows[i]
            _, _, new_off, _ = dst_rows[i]
            size = dtype_size(dt, n)
            expert = is_expert(name)
            fsrc.seek(old_off)
            old_bytes = fsrc.read(size)
            dest_f = fexp if expert else ftrunk
            dest_f.seek(new_off)
            new_bytes = dest_f.read(size)
            old_h = hashlib.sha256(old_bytes).hexdigest()
            new_h = hashlib.sha256(new_bytes).hexdigest()
            if old_h != new_h:
                raise SystemExit(
                    f"BYTE MISMATCH: {name} ({dt}, "
                    f"{'expert' if expert else 'trunk'}): old offset {old_off} "
                    f"sha256 {old_h[:16]} != new offset {new_off} sha256 {new_h[:16]}"
                )
            checked += 1
            dtypes_seen.add(dt)
            classes_seen.add(expert)

    print(f"verify: {checked} tensors byte-exact, dtypes covered: {sorted(dtypes_seen)}, "
          f"classes covered: expert={True in classes_seen} non-expert={False in classes_seen}")
    print(f"verify: size OK, trunk {trunk_total} + experts {expert_total} "
          f"== source pack.bin {src_pack_size}")
    print(f"verify: pack.bin on-disk {dst_pack_size} bytes, "
          f"experts.bin on-disk {dst_experts_size} bytes")
    return checked, dtypes_seen, classes_seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srcpackdir")
    ap.add_argument("dstdir")
    ap.add_argument("--verify", action="store_true",
                     help="run the byte-exactness and size check after splitting")
    ap.add_argument("--verify-only", action="store_true",
                     help="skip splitting, verify an already-split DSTDIR")
    args = ap.parse_args()

    src = Path(args.srcpackdir)
    dst = Path(args.dstdir)

    if not args.verify_only:
        dst.mkdir(parents=True, exist_ok=True)
        trunk_total, expert_total, src_pack_size, n_lines = do_split(src, dst)
        print(f"split: {n_lines} index lines, pack.bin {trunk_total} bytes, "
              f"experts.bin {expert_total} bytes, source pack.bin was {src_pack_size} bytes")

    if args.verify or args.verify_only:
        do_verify(src, dst)


if __name__ == "__main__":
    main()
