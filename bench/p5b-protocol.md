# P5b protocol: bounded GGUF write-back

The base source is immutable:

`~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-04867e2.gguf`

The source SHA256 and the selected tensor metadata are recorded by the CPU
inventory command before any patch is produced.  P5b selects exactly
`blk.24.ffn_down.weight` through `blk.31.ffn_down.weight`.  The writer copies
the source byte-for-byte and patches only the `data_offset:n_bytes` ranges
reported by the source GGUF reader.  It does not use a GGUF writer or rebuild
metadata.

## CPU preflight

Run from the lane worktree with the llama.cpp GGUF package visible:

```sh
export PYTHONPATH=~/llama.cpp/gguf-py
python tools/gguf-writeback.py inventory \
  --source ~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16-BARO-04867e2.gguf \
  --tensor blk.24.ffn_down.weight --tensor blk.25.ffn_down.weight \
  --tensor blk.26.ffn_down.weight --tensor blk.27.ffn_down.weight \
  --tensor blk.28.ffn_down.weight --tensor blk.29.ffn_down.weight \
  --tensor blk.30.ffn_down.weight --tensor blk.31.ffn_down.weight \
  --report .work/team-B/codex/p5b/inventory/source.json
```

The synthetic receipt uses a tiny GGUF and an NPZ with the same reported
shapes.  It must show `outside_range_diff_bytes: 0`, every selected range
changed, equal source/destination file sizes, and a successful independent
`verify` invocation.  A source hash mismatch or any outside-range byte is a
hard failure.

The training-side artifact must provide one NPZ key for each selected GGUF
name, with the exact GGUF shape.  For BF16, uint16 values are interpreted as
raw little-endian BF16 bits; float arrays are converted with round-to-nearest-
even.  Extra or missing keys, shape mismatch, unsupported dtype, and byte-size
mismatch fail before a destination is created.

## GPU and acceptance order

Only after the CPU receipt is committed may Sonnet submit the single training
job through `gpu-wait` with priority 10, preemptible, `--vram 22`, a 3600 s
timeout, and a one GPU-hour cap.  The target remains the dense BF16 champion;
the LoRA rank is 16 and alpha is 32; only the eight selected `ffn_down` tensors
are merged.  The base bake and pack are never modified.

After training, the order is: bounded write-back receipt; repack with the
unmodified `tools/engine-pack.py`; forced agreement against llama.cpp on the
patched file, minimum 90 percent over the frozen 20 prompts; quality table on
the quality lane's same rows; and 20/20 identity on unpatched prompts.  Any
outside-range difference, forced agreement below 90 percent after one repair,
quality regression without a written accepted trade-off, or unpatched
identity miss kills P5b and leaves the base bake in place.
