# tools/itools

Symlinks to the `~/iTools/bin` shims this repo calls, so the set is tracked in git and usable as `tools/itools/<name>`
from scripts. The tools themselves live in `~/iTools` (reusable across repos, CLAUDE.md §11). `tools/ci-checks.sh`
fails on a link whose target is missing or not executable.

| link | used for |
|---|---|
| gate-dryrun, refcache, pass-cache, seq-decide | gate harness: CPU dry run, reference cache, test pass cache, sequential A/B stop rule |
| lane-prep, lane-merge | worktree lanes |
| isa-loops, isa-diff | ISA receipts for kernels (megakernel fingerprint, spills) |
| pack-fit | VRAM budget of a pack + KV against MAX's pool |
| model-import, hf-verify, llm-scan, publish-purge | model import / bake verification, publish copies |
| gguf-kv, gguf-sweep, gguf-experts, ggml-harvest | GGUF metadata, tensor sweeps, expert stats, llama.cpp kernel census |
