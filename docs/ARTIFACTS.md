# Disk artifacts index

Two of the three largest artifacts are live packs filed under dead experiment names: the RegesCore tier pack sits
in `AiTrainer/latent-spec`, the RegesCore resident pack in `.work/moe-ngram`. A housekeeping pass on 2026-10-14
marked the first for deletion before a reference check caught it. This index records what each large untracked
file is, what reads it, and whether it is still needed.

Snapshot 2026-10-14 at `87b9c4b`. Sizes are exclusive bytes (`sudo btrfs filesystem du -s`, links followed);
"read by" is `git grep -F <path>` and `git grep -F <basename>` outside `exchange/` and `docs/releases/`, plus the same basename search over `~/iTools`; a reader that takes the path only as an argument does not show up. Nothing listed here was moved or
deleted. Refresh: rerun those two commands; `exp-track --report` adds an age/status guess (mtime based, so it
calls a pack ACTIVE when it was merely opened).

Status words: **live** (a tracked script, profile or gate reads it), **record** (results already copied into
`exchange/` or Brain, the bytes are regenerable), **open** (belongs to an unfinished card or lane),
**idle** (nothing reads it, no open card).

## Packs (model weights in engine format)

| path | GiB | what | read by | status |
|---|---:|---|---|---|
| `.work/latent-spec/pack-tier/` (link into `~/Projects/AiTrainer/latent-spec`) | 19.6 | RegesCore-35B tier pack: `pack.bin` trunk in VRAM, `experts.bin` routed experts in host RAM (`BARO_TIER=64`). Built 2026-09-19; the only copy since its twins were deleted 2026-09-24 | `profiles/regescore-builder.toml`, `bench/regescore-blender-ab.sh`, `bench/vision-smoke.sh` (tokenizer) | live |
| `.work/moe-ngram/pack/` | 19.6 (rebuilt 2026-11-09 from RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf with tools/engine-pack.py; pack_sha256 4006579559612d7b equals the old e340ee1 bake) | RegesCore-35B resident pack. Named after the ngram-spec round, but it is the default MoE pack of the A/B harnesses | `bench/ab-resident.sh`, `bench/decode-arms.sh`, `bench/moe-ngram-gate-sweep.sh` | live |
| `mojo-baro-lanes/moe-miss/.work/moe-miss/{pack,tier-pack}/` | 0 exclusive (42.0 shared) | Byte-identical twins of `.work/moe-ngram/pack/pack.bin` and `latent-spec/pack-tier/{experts,pack}.bin` (fclones hash, 2026-10-21). Reflinked with `fclones dedupe` 2026-10-21: /home 81 -> 120 GB free, no file removed. Deleting either side now frees nothing; the live copies are the ones above | the moe-miss lane (merged into main) | twin, reflinked |
| `.work/lanes-archive/{moe-miss,sparkq4k,open3}/` | 0.1 | Text receipts (<2 MB .txt/.md/.json/.tsv/.csv/.log) and `exchange/` copied out of those three lane worktrees 2026-10-21 (6125 / 1334 / 206 files, counts verified, `INDEX.txt`). All three lanes' code is in main | nothing | record, keep (small) |
| `mojo-baro-lanes/sparkq4k/.work/sparkq4k/packs/` (= `.work/sparkq4k`, a link) | 20.6 | Spark K-quant / q8 packs (qwen7b, minicpm5, vibethinker) of the sparkq4k lane. Main's `.work/sparkq4k` is a SYMLINK into this lane worktree: the only copy. Removing the worktree breaks it | `bench/sparkq4k-ab.sh`, `bench/sparkf32-ab.sh` | live, lives in a lane |
| `mojo-baro-lanes/open3/.work/open3/verify/out/pack/` | 0 (deleted 2026-10-21, 17.2 GB) | Qwen3.8-27B-OBLITERATED verify pack (sha d7a23527...), rebuild from `~/Models/qwen3.8-27b-obliterated-q4_K_M/...-BARO-36e440b.gguf`; identity in `.work/lanes-archive/open3/verify-pack-identity.json` | nothing | deleted, rebuildable |
| `mojo-baro-lanes/*/.work/run-tests-turn/*.baro` | 0 (deleted 2026-10-21, 5.2 GB) | 9 lane copies, hash-identical (8a5c2a20) to the real `.work/run-tests-turn/` here | nothing | deleted, twin kept |
| `.work/e11b-qwythos-pack/` (= `.work/engine-pack-q4`, a link) | 6.2 | Qwythos-9B q4 pack, the dense champion | `CLAUDE.md`, `README.md`, LatentOS runs | live |
| `.work/spark-pack-mojo/` | 0 | Spark-X2.5-4B pack from the v0.5.3 grammar gate; deleted 2026-10-14, rebuild from `~/Models/spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M-BARO-cd53675.gguf` | release note v0.5.3 only | deleted |
| `.work/regescore-latent/receiver-pack/` (link into AiTrainer) | 2.2 | RegesCore receiver pack from the 2026-09-25..27 semantic-latent gates | `bench/semantic-latent-gate.sh`, `bench/vision-latent-gate.sh`, one quarantined Gemini probe | idle (gates of the Gemini window) |

## Dumps and experiment data

| path | GiB | what | read by | status |
|---|---:|---|---|---|
| `.work/latent-9b/dump/data.bin` | 0 (deleted 2026-10-14, freed 13.0 GiB on disk) | Qwythos-9B hidden-state dump for the latent-wa probe, 2026-10-12. Result in `exchange/receipts/latent-wa/qwythos9b-probe.txt`, method to rebuild it in `qwythos9b-dump-method.md` beside it (CPU, 2 min); `index.txt`, the small `wa.*` and `probe.txt` remain | nothing | deleted, rebuildable |
| `.work/latent-spec/agreement/base.logits` | 0.9 | Full-RegesCore wikitext logits, the reference for the 2026-09-19 shared-weight drafter table (top-3-bit view 84.9%, 4-of-8 experts 76.1%, Q2_K copy 64.6%). Table in Brain `2026-09-19-moe-ngram-spec-falsified` | `bench/draft-agreement.sh` (rebuilds it when missing) | record |
| `.work/latent-spec/agreement*/`, `overlap/` | <0.01 | Logs of that table and the expert-overlap receipt (39% of experts shared by adjacent tokens) | Brain note | record, keep (small) |
| `.work/mtp-head/` (link into AiTrainer) | 3.4 | Draft-head training data and frozen `lm_head`/`embed` arrays, 2026-09-19 (A4 / P5a, parked) | `tools/mtp-head/*.py` take it as an argument; no default path points here | idle |
| `.work/a2-rec/` | 2.4 | A2 gate reference cache (`ref-<sha>` per prompt set) | `bench/a2-gate.sh`, its selftest | live |
| `.work/gguf/` | 1.9 | Extracted GGUF tensors (`moe_w*.bin` and others) for kernel benches | `bench/bench_coldcache.mojo`, `bench/attn-protocol.md`, amdhq converter | live |
| `.work/miniagi/` | 1.2 | miniagi runs and checkpoints (published weights live in `~/Models/miniagi/`) | `bench/miniagi-*.sh`, `miniagi/README.md` | live |
| `.work/research-paper-corpus/` (link into AiTrainer), `.work/research-text/` | 1.0 + 0.2 | Paper corpus for the 2026-09-29 hidden-state-base investigation | `tools/jlens_multihop.py:24` (hard-coded AiTrainer path) | open (card waits on the maintainer: teacher and route) |
| `.work/p6/` | 0.7 | q4x kernel fixtures `gt11`..`gt30` | `kernels/test_iq4nl.mojo`, `kernels/test_q2_0.mojo`, `docs/KERNELS.md` | live |
| `.work/loop/` | 0.3 | Loop-gate iterations 008..011 | `bench/loop-protocol.md`, `tools/loop-embed-winner.sh` | live |
| `.work/kvg/` | 0.2 | lane-kvg engines (MoE int8 KV, group 32/64/256) | nothing; `lane-kvg` is 1 commit ahead of main | open |
| `.work/p7/` | 0.1 | P7 megakernel dumps, 2026-09-24 | nothing | idle |
| `.work/k3-opt/` | 0.1 | K3 hyperopt arms and hot-expert lists, 2026-09-24 | `briefs/2026-09-24-k3-hyperopt-plan.md` | idle |
| `.work/ckpt-s4-test/` | 0.1 | Checkpoint budget tests (f32 vs i8), 2026-09-27 | nothing | idle |
| `.work/latent-gate/` (link into AiTrainer) | 0.1 | L1 LatentOS sidecar gate output | `tools/latent-gate.sh` | live |
| `.work/leakd-engine/`, `.work/live-latent-check-2026-09-30/` | 0.3 | Engine leak builds (2026-10-09), LatentOS IPC check (2026-09-30) | nothing | idle |
| `.work/jlens/` | 0.1 | Jacobian-lens runs on Qwen2.5-0.5B; receipts in `exchange/receipts/jlens-qwen05/` | nothing | record |

## Environments and build output

| path | GiB | what | status |
|---|---:|---|---|
| `.work/oracle-venv/` | 0.7 | torch venv for the Python oracles (`miniagi/latent/dcheck.sh`) | live |
| `.work/build-cache/` | 0.7 | `tools/build-many.sh` cache | live |
| `.venv/`, `.venv-prev/` | 0.5 + 0.15 | Mojo 1.2 toolchain; the previous toolchain (1.1), kept since the upgrade | live / idle |
| `amdhq/.venv-vllm/` | 0 | vLLM 0.20.2rc1 ROCm venv, never used for a recorded run; deleted 2026-10-14 | deleted |
| `amdhq/.venv-mlc/`, `amdhq/.venv/` | 1.6 + 1.4 | MLC and main AMDHQ venvs | idle / live |
| `android-baro/llama/{.cxx,build}` | 0 | Android on-device llama.cpp build output in `~/Android/baro`; deleted 2026-10-14, the next Gradle build recreates it | deleted |

## Lab VMs and images (`amdhq/labiso/`)

| path | GiB | what | status |
|---|---:|---|---|
| `vm/aarch64/` | 0 | aarch64 lab VM from the 2026-09-17 router check; image deleted 2026-10-14, `boot.sh` kept, `build-aarch64.sh` rebuilds | deleted |
| `vm/.work/` | 3.3 | A3 two-VM disks (`a3-vmA/B.img`, 40 GB sparse each) | open (Link 5, two-VM path) |
| `out/*.iso` | 0 | Lab install ISO 2026-08-28; deleted 2026-10-14, `build.sh` rebuilds it | deleted |

## Lanes

| path | GiB | status |
|---|---:|---|
| `../mojo-baro-lanes/latent-bridge-p4` | 6.0 | open: holds the uncommitted real-data P4 receipts |
| `../mojo-baro-lanes/latent-bridge` | 0.06 | open |

## Stray files

`.work/dupe.hash` (0.13 GiB, duperemove hash database for `~/Models`) moved to `~/.cache/housekeeping/dupe.hash` 2026-10-14.

What this index cannot tell: whether an **idle** artifact will be wanted again. That is a decision per row,
not a property of the bytes.

## 2026-11-08: pack binaries removed for disk, small files kept

`pack.bin`/`experts.bin` (and the mtp-head / mojo-gguf l1 binaries) were deleted to free 64 GB; identity.json,
index.txt, meta.json and the token files stay. sha256 of every removed file: `~/Brain/OS/2026-11-08-disk-packs-binaries-removed.md`.
Rebuild a pack with the Python oracle (the Mojo tool needs its closure dir, `tools/pack-run.sh`), then compare the hash:

    .venv/bin/python3 -I tools/engine-pack.py <source.gguf from identity.json> <dir> --q4     # dense q4 trunk
    .venv/bin/python3 -I tools/engine-pack.py <source.gguf> <dir> --arch qwen35moe            # RegesCore MoE

Measured: `.work/e11b-qwythos-pack` (= `.work/engine-pack-q4`, the dense champion) was rebuilt the same day in 90 s
CPU / 1.27 GB peak, sha256 715d4c34 identical, and is back. Still removed: `.work/moe-ngram/pack` (source now the
rebake `RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf`), `.work/latent-spec/pack-tier` (split it after moe-ngram with
`tools/pack-split-experts.py`), `.work/bake27/verify3/pack` (source 27B gguf to identify by hash), `.work/mtp-head`
binaries (parked for good).
