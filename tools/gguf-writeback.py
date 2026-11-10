#!/usr/bin/env python3
"""Inspect and byte-bounded-patch tensors in an existing GGUF.

The writer never serializes GGUF metadata.  It copies the source byte-for-byte
and replaces only the data ranges reported by GGUFReader for the explicitly
named tensors.  Patch arrays in the NPZ are in GGUF's reported shape.  BF16
arrays may be raw uint16 bits or floating point values, which are converted
with round-to-nearest-even.

Examples:
  python tools/gguf-writeback.py inventory --source model.gguf \
    --tensor blk.24.ffn_down.weight --tensor blk.25.ffn_down.weight \
    --report inventory.json
  python tools/gguf-writeback.py write --source base.gguf --dest patched.gguf \
    --patch-npz merged.npz --tensor blk.24.ffn_down.weight --report write.json
  python tools/gguf-writeback.py verify --source base.gguf --dest patched.gguf \
    --tensor blk.24.ffn_down.weight --report verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def reader_for(path: Path):
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit(
            "gguf Python package is required; set PYTHONPATH to llama.cpp/gguf-py"
        ) from exc
    return GGUFReader(str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while chunk := src.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def requested_names(args: argparse.Namespace) -> list[str]:
    names = list(dict.fromkeys(args.tensor or []))
    if not names:
        raise SystemExit("at least one --tensor is required")
    return names


def tensor_table(path: Path, names: list[str]) -> list[dict]:
    reader = reader_for(path)
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"missing tensors in {path}: {', '.join(missing)}")
    size = path.stat().st_size
    rows = []
    for name in names:
        tensor = by_name[name]
        start = int(tensor.data_offset)
        n_bytes = int(tensor.n_bytes)
        end = start + n_bytes
        if start < 0 or end > size or end < start:
            raise SystemExit(f"invalid byte range for {name}: {start}:{end} of {size}")
        rows.append(
            {
                "name": name,
                # GGUF metadata lists dims fastest-first (reversed from a
                # numpy/torch row-major shape); GGUFReader.shape passes that
                # convention through unchanged. The on-disk bytes -- and
                # every patch array this tool accepts -- are in the REVERSED,
                # numpy-natural shape (verified against an HF-loaded
                # nn.Linear.weight for a non-square tensor: only the
                # reversed order round-trips as an exact byte match).
                # Reporting tensor.shape here directly, as an earlier
                # version did, silently accepted a transposed patch array
                # for any non-square tensor and wrote scrambled bytes.
                "shape": [int(dim) for dim in reversed(tensor.shape)],
                "dtype": tensor.tensor_type.name,
                "offset": start,
                "n_bytes": n_bytes,
                "end": end,
            }
        )
    ordered = sorted(rows, key=lambda row: row["offset"])
    for previous, current in zip(ordered, ordered[1:]):
        if current["offset"] < previous["end"]:
            raise SystemExit(
                f"overlapping requested ranges: {previous['name']} and {current['name']}"
            )
    return rows


def ensure_same_layout(source: Path, dest: Path, names: list[str]) -> None:
    src_rows = tensor_table(source, names)
    dst_rows = tensor_table(dest, names)
    for src, dst in zip(src_rows, dst_rows):
        for field in ("shape", "dtype", "offset", "n_bytes", "end"):
            if src[field] != dst[field]:
                raise SystemExit(
                    f"GGUF layout changed for {src['name']} field {field}: "
                    f"source={src[field]} dest={dst[field]}"
                )


def npz_arrays(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        wanted = set(names)
        missing = sorted(wanted - keys)
        extra = sorted(keys - wanted)
        if missing or extra:
            raise SystemExit(
                f"patch NPZ keys must exactly match --tensor names; missing={missing} extra={extra}"
            )
        return {name: np.array(archive[name], copy=True) for name in names}


def f32_to_bf16(values: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(values, dtype="<f4")
    bits = f32.view("<u4")
    # Round to nearest, ties to even, matching the usual BF16 cast.
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype("<u2", copy=False).tobytes(order="C")


def patch_bytes(array: np.ndarray, row: dict) -> bytes:
    expected_shape = tuple(row["shape"])
    if tuple(array.shape) != expected_shape:
        raise SystemExit(
            f"{row['name']}: patch shape {tuple(array.shape)} != GGUF {expected_shape}"
        )
    dtype = row["dtype"]
    if dtype == "BF16":
        if array.dtype == np.dtype("uint16"):
            raw = np.ascontiguousarray(array, dtype="<u2").tobytes(order="C")
        else:
            raw = f32_to_bf16(array)
    elif dtype == "F32":
        raw = np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")
    elif dtype == "F16":
        raw = np.ascontiguousarray(array, dtype="<f2").tobytes(order="C")
    else:
        raise SystemExit(
            f"{row['name']}: dtype {dtype} is not supported by the safe NPZ writer"
        )
    if len(raw) != row["n_bytes"]:
        raise SystemExit(
            f"{row['name']}: patch byte length {len(raw)} != GGUF {row['n_bytes']}"
        )
    return raw


def diff_bytes(left: np.ndarray, right: np.ndarray, start: int, end: int) -> int:
    total = 0
    chunk = 64 * 1024 * 1024
    for pos in range(start, end, chunk):
        stop = min(pos + chunk, end)
        total += int(np.count_nonzero(left[pos:stop] != right[pos:stop]))
    return total


def diff_report(source: Path, dest: Path, rows: list[dict]) -> dict:
    if source.stat().st_size != dest.stat().st_size:
        raise SystemExit(
            f"file length changed: source={source.stat().st_size} dest={dest.stat().st_size}"
        )
    ensure_same_layout(source, dest, [row["name"] for row in rows])
    source_map = np.memmap(source, dtype=np.uint8, mode="r")
    dest_map = np.memmap(dest, dtype=np.uint8, mode="r")
    size = source.stat().st_size
    outside = 0
    inside = {}
    cursor = 0
    for row in sorted(rows, key=lambda item: item["offset"]):
        start, end = row["offset"], row["end"]
        outside += diff_bytes(source_map, dest_map, cursor, start)
        inside[row["name"]] = diff_bytes(source_map, dest_map, start, end)
        cursor = end
    outside += diff_bytes(source_map, dest_map, cursor, size)
    del source_map
    del dest_map
    return {
        "source": str(source),
        "dest": str(dest),
        "source_bytes": size,
        "tensor_ranges": rows,
        "outside_range_diff_bytes": outside,
        "inside_range_diff_bytes": inside,
        "all_named_ranges_changed": all(value > 0 for value in inside.values()),
        "pass": outside == 0 and all(value > 0 for value in inside.values()),
        "source_sha256": file_sha256(source),
        "dest_sha256": file_sha256(dest),
    }


def write_report(report_path: str | None, report: dict) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded)


def cmd_inventory(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    names = requested_names(args)
    rows = tensor_table(source, names)
    report = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": file_sha256(source),
        "tensor_ranges": rows,
    }
    write_report(args.report, report)


def cmd_write(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    if source == dest:
        raise SystemExit("source and destination must differ; base GGUF is immutable")
    if dest.exists():
        raise SystemExit(f"destination already exists: {dest}")
    names = requested_names(args)
    rows = tensor_table(source, names)
    arrays = npz_arrays(Path(args.patch_npz), names)
    patches = {row["name"]: patch_bytes(arrays[row["name"]], row) for row in rows}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=dest.name + ".", suffix=".partial", dir=dest.parent, delete=False
        ) as temp:
            temp_path = Path(temp.name)
        shutil.copyfile(source, temp_path)
        with temp_path.open("r+b") as output:
            for row in rows:
                output.seek(row["offset"])
                output.write(patches[row["name"]])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, dest)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    report = diff_report(source, dest, rows)
    report["patch_npz"] = str(Path(args.patch_npz).resolve())
    write_report(args.report, report)
    if not report["pass"]:
        raise SystemExit("writeback receipt FAIL")


def cmd_verify(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    names = requested_names(args)
    report = diff_report(source, dest, tensor_table(source, names))
    write_report(args.report, report)
    if not report["pass"]:
        raise SystemExit("writeback receipt FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "write", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--source", required=True)
        child.add_argument("--tensor", action="append", required=True)
        child.add_argument("--report")
        if command != "inventory":
            child.add_argument("--dest", required=True)
        if command == "write":
            child.add_argument("--patch-npz", required=True)
        child.set_defaults(func=globals()[f"cmd_{command}"])
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
