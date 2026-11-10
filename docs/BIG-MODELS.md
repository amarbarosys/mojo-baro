# Big models: tiered weights across VRAM, RAM and NVMe (GPU and CPU)

**On this box a weight held in RAM is computed faster by the CPU (q4_k GEMV 51 GB/s) than it can be shipped to the
card (28.5 GB/s pinned PCIe at best, 15.5 at 1 MB pieces), and the NVMe moves only 1.4-2.8 GB/s at expert-sized random
reads.** So the tier plan puts the re-read-every-token weights in VRAM first, computes RAM-held weights on the CPU
where the planner says it pays, and keeps the disk for bytes read rarely, in large contiguous pieces.

The plan (2026-10-12, approved): one weight store for MoE experts and dense trunks over GGUF directly, a planner that
splits VRAM / RAM / NVMe by these numbers, GPU engines and a CPU-only forward on top. Method from kimi-k3-in-c
(`~/Brain/mojo/mojo-baro/2026-10-12-kimi-k3-in-c-analysis.md`): residency never changes output, budget goes first to
what is re-read every token, counters that cannot flatter. Phases: P0 measure (below) · P1 WeightStore
`serve/wstore.mojo` · P2 planner (a tools/ planner, not written yet) · P3 GPU engines on the store (MoE past RAM, dense trunk
streaming) · P4 trace-sized caches · P5 CPU kernels + hybrid · P6 CPU-only forward. Proving ladder, all on disk:
Llama-3.2-1B, RegesCore-35B MoE (20 GB), Qwen3.8-27B / OrcaSAQ-27B dense (16 / 15 GB), Flash-Next GSQ Q2_0 / UD-IQ1_M
(66 / 75 GB, larger than the 64 GB of RAM), with budgets forced small.

## P0: what this box moves (2026-10-12)
7800X3D 8C/16T (AVX-512 f/bw/vnni/bf16), 64 GB DDR5, RX 7900 XTX, Kingston NV3 2 TB NVMe. Three runs per row;
spread = (max - min) / mean. Receipts: `exchange/receipts/big-models-p0/`.

| path | piece | GB/s (median of 3) | spread | instrument |
|---|---|---|---|---|
| CPU RAM read, 16 threads | 2.3 GB | 54.0 | 2% | `bench/cpu-gemv.mojo roof 16` |
| CPU RAM read, 1 thread | 2.3 GB | 48.8 | 10% | `roof 1` |
| CPU q4_k GEMV (AVX-512 VNNI), 16 threads | 2.3 GB | 51.2 | 4% | `bench/cpu-gemv.mojo q4k 16` |
| CPU q4_k GEMV, 1 thread | 2.3 GB | 22.7 | 2% | `q4k 1` |
| host pinned -> VRAM | 1 / 4 / 16 / 64 MB / 1 GB | 15.5 / 23.8 / 27.4 / 28.5 / 28.8 | 0-1% | iTools `hipbw` |
| host pageable -> VRAM | 1 / 4 / 16 / 64 MB / 1 GB | 0.6 / 2.4 / 7.5 / 15.1 / 26.9 | 3-32% | `hipbw` |
| GPU kernel reading host memory (zero-copy) | 1 / 4 / 64 MB / 1 GB | 3.4 / 10.4 / 25.6 / 28.6 | 0-8% (46% at 16 MB) | `hipbw` |
| NVMe O_DIRECT sequential | 64 MB | 6.1 | 3% | iTools `tier-bandwidth` |
| NVMe page-cached re-read | 64 MB | 13.2 | 17% | `tier-bandwidth` |
| NVMe O_DIRECT random, QD 1 / 4 / 16 | 860 KB | 1.39 / 1.94 / 1.99 | 25-43% | `bench/nvme-rand.mojo` |
| same | 1.35 MB | 1.89 / 2.43 / 2.14 | 21-124% | same |
| same | 17 MB | 2.99 / 3.56 / 4.66 | 16-145% | same |
| same | 64 MB | 4.08 / 2.82 / 5.67 | 4-48% | same |

NVMe random tails: the worst single read took 50-660 ms (max/mean 3-68x at QD 1), so one slow expert can stall a layer.

## Hybrid placement is a bandwidth win, not a guess
CPU q4_k GEMV at 16 threads runs at 95% of the RAM read roof (51 of 54 GB/s), 1.8x what PCIe can deliver. The earlier
`BARO_TIER_CPU` loss on the 35B (small experts, one sync per miss) is a latency problem P5 has to remove; the
bandwidth is there.

## Pageable memory is never shipped
0.6 GB/s at 1 MB pieces: every host tier is pinned or registered (`hipHostRegister` on an mmap, as the expert tier and
q4x already do).

## The disk serves only rare bytes, in large pieces
At expert sizes the NVMe is 20-35x slower than RAM. Pieces of 17-64 MB read 2-3x faster than 1 MB ones, so the store
reads a layer's experts as contiguous runs in disk-offset order. Per-token byte budgets at 20 tok/s: NVMe <= 0.1-0.3
GB, PCIe <= 1.4 GB, CPU RAM <= 2.6 GB, VRAM ~48 GB. The planner (P2) assigns by these.

## Disk-tier speed claims need repeats
Spreads of 16-145% on NVMe rows mean a single disk-tier number decides nothing; receipts carry per-arm device
throughput (kimi measured 2.17x from the device alone on identical work).
