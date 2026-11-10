# neuron hops: prediction, frozen before the timed grid

Code: nlook.mojo (lookup rows), neuron.mojo (driver), tools/neuron-data.mojo (subset), tools/neuron_oracle.py (gradient oracle).
Checks already run (before this file): fp32 path = float64 torch oracle, every block <= 1e-4 rel, same-input runs byte-identical.
bf16 path vs oracle: 7% rel at b=64, 16% at b=3 (no-neuron control 1%): UNVERIFIED, so the grid is judged as a bf16 run and a
bf16 gradient claim is not made. One 60-update smoke (real, bf16 and fp32, 1.4 s and 9.9 s) ran for timing only.

Task: 2020 enwiki articles (most-linked 2048 of wikinet p10, induced graph, min 4 links), 4 link slots, 16 k-means clusters of bge.
Question: start article (bge vector) + 2 or 3 slot tokens; answer = cluster of the article reached. Starts split 80/10/10 by hash.
No-lookup ceiling (neuron-data, H=3, test starts): majority 28.8%, start cluster + slots 31.8%, chance 6.25%.
Memory: one row per link (8080), key = unit(bge_start + slot code), value = bge of the linked article. W_a, W_b trained.

Arms: real | shuffled (values permuted across rows, keys intact) | none (pause rows, no lookup, ENT row kept).
Model 1.59M (d 256, 2 layers, +W_a/W_b 0.2M). Seeds 110, 111, 112. b 256, 2/3 hops mixed per update.
Grid: 1500 updates, lr 1e-3 linear decay, aux 0.5 (readout loss on the true link, real and shuffled), wd 0.01, clip 1.
Score: test-start accuracy, mean of the 2-hop and 3-hop accuracy, 4096 questions each, untouched stream 20000+seed.

BAR (frozen): real beats shuffled AND none on at least 2 of 3 seeds, a win meaning real - max(shuffled, none) >= 2.0 pp on that seed.
If real misses the bar: before calling it capacity, rerun all three arms on seed 110 at 3x updates (4500) and the next lr down (3e-4);
if real then wins on seed 110, run seeds 111, 112 with that setting. If it still misses, the answer is plain: real does not win.

Prediction: none = 29-33% (ceiling 31.8%), shuffled = 29-33% (values carry no signal), real = 55-95% if the lookup trains in
1500 updates, else 30-40%. P(bar met in the first grid) = 0.45, P(met after the protocol rerun) = 0.7.
Failure shapes I expect: aux readout accuracy rises while answer accuracy stays low (W_b does not carry the article through the
two-layer model), or addressing collapses to hub articles (majority-cluster answers).
