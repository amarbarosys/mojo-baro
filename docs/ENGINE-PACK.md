# Engine weight pack

`tools/engine-pack.mojo` is the production packer for qwen35 trunk packs. It streams tensor rows from GGUF shards and writes `pack.bin`, `index.txt` and `identity.json`. `tools/engine-pack.py` stays as the byte-identical oracle.

The trunk supports `--q8`, `--q4`, `--kq`, `--q2b3`, `--tq1` and `--tq2`; the ternary codecs are mutually exclusive with `--q8`, `--q4` and `--kq`. Ternary payloads use per-block `amax` scaling and follow the oracle exactly, codec bytes first, then the f16 scales.

```sh
./.venv/bin/mojo build --fp-mode contract=off tools/engine-pack.mojo -I serve -I tools -o .work/engine-pack
./.work/engine-pack MODEL.gguf OUTDIR --tq1
./tools/engine-pack-check.sh
```

`tools/engine-pack-check.sh` compares full pack bytes, index, identity and summary lines against the oracle on tiny fixtures, including all three ternary codecs.
