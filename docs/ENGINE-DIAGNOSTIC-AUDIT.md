# Engine questions: evidence audit, 2026-09-05

Source baseline: `7cae375`. This is a source/receipt audit, not a new GPU
benchmark. Existing timings retain their original workloads and limitations.

## 1. The 1.04x versus 1.06–1.11x discrepancy is not established

`bench/mrow-gemm-protocol.md` M0 records 88.3/92.2 us for m=1/2.
But `.work/briefs/status-mrowB.md` also records repeat pairs
88.370/91.551 and 80.791/91.501 us: ratios 1.036 and 1.133.
The m=2 time is stable while the denominator moves. These receipts do not
support treating 1.04 as a reproducible kernel constant.

NBUF=8 rotates 427,819,008 bytes of weights/scales, comfortably above the
96 MB Infinity Cache. Both this and a multi-GB trunk stream target nonresident
weights. They differ in reuse distance, preceding kernels, clock state,
activation residency, allocation/layout and host scheduling; that does not
establish a fundamentally different weight-cache regime.

The bench times enqueue plus synchronize per invocation. PROFILE=4 similarly
serializes stages, includes host view construction/dispatch work, and aggregates
32 layers. Gate/up use N=12288,K=4096; down uses N=4096,K=12288, so the 1.11x
down figure is not the same shape. Compile-time activation/output layouts also
differ between fixtures and engine. Same template name is not proof of identical
machine code. The current bench dispatches MR=4 directly; the engine uses MR=5
for runtime m=4. MR=8 additionally changes QV from 16 to 8.

Next discriminating measurement: repeated interleaved m=1/2 pairs, same binary,
engine layouts and all three FFN shapes; device timestamps plus host wall time;
rotate-only versus actual preceding-kernel stream. Record clocks and spread.
Only then attribute any remaining delta to cache, code generation or scheduling.
The bench remains useful for parity, ablations and candidate screening; adopting
a change requires unprofiled engine results on the prompt set.

## 2. Per-call latency and useful-row throughput have different optima

M0 gives times [88.3,92.2,112.9,201.2] us for m=[1,2,4,8]. Per row these are
[88.3,46.1,28.2,25.2] us. Weight amortization already wins versus m separate
calls. No measured m beats m=1 in absolute call latency; no global optimum over
all m/configurations has been established. m>8 is unsupported by this kernel's
current buffers, and extrapolation across the QV change is invalid.

M1 ablations support arithmetic as the main incremental cost: removing extra
FMAs nearly removes scaling; removing extra activation traffic does not. This
does not prove every shape/m is compute-bound. For speculative decode minimize
window time per accepted output, not time per computed row. Wider windows can
lose because additional rows are rejected and draft steps cost time.

## 3. Roof percentages used an inconsistent byte model

`tools/audit-engine-bytes.py .work/engine-pack-q8` validates every index offset
and the final pack size without reading tensor payloads:

| Component | Bytes |
|---|---:|
| Entire q8 pack | 10,728,640,512 |
| bf16 embedding table | 2,034,237,440 |
| Separate draft layer | 258,557,952 |
| Trunk and output weights | 8,435,845,120 |
| Above plus one bf16 embedding row | 8,435,853,312 |

The q8d pack adds 572,129,280 bytes of q4 draft output weights; no-spec logical
weight demand is unchanged. A whole embedding table and unused draft layer are
not read per no-spec token. At 68.77 tok/s the logical unique-weight rate is
580.1 GB/s, or 64.46% using the question's assumed 900 GB/s denominator. Applying
our same logical byte count to 74.14 tok/s gives 625.4 GB/s, 69.49%: a 5.03-point
gap under this convention, not the original six. Repo protocols instead use the
card's nominal 960 GB/s roof, yielding 60.43% and 65.15%. Neither denominator is
a newly measured sustained ceiling, and neither percentage measures physical
memory utilization. The hardware is GDDR6, although prior notes call its roof HBM.

The 7.8% throughput gap remains; the claimed six utilization points are not a
diagnosis. Logical bytes exclude state, KV, intermediates, repeated loads and
cache effects. llama.cpp uses quantized activations and the recorded comparison
uses q8 KV, while this engine uses bf16 GEMM activations and f32 KV. Audit its
actual tensor types/accesses separately before asserting equal physical demand.
Use matched unprofiled timings and a device timeline/traffic counters to divide
the residual into useful weight service, extra traffic, kernel work and idle gaps.

## 4. Draft width and acceptance

`bench/mtp-protocol.md` measured k=1/2/3/4 medians 98.2/100.7/94.7/92.4 tok/s;
k=2 is the best tested fixed width on that set. It means two proposed tokens
and a three-row verification window. It is not an optimum derivable from 0.69
alone, nor has the width sweep been repeated with the q4 draft head.

For conditional prefix survival probabilities p_i, expected outputs in a full
window are E_k = 1 + sum(i=1..k, product(j=1..i,p_j)). Minimize T_k/E_k,
including draft, verification, acceptance/rollback and boundary costs. Widen
when marginal expected outputs per marginal time exceed E_k/T_k. Aggregate
accepted/drafted is not a per-position conditional p_i. An iid p=0.69 toy model
would give E_2=2.1661, but is not a fit to these receipts.

`bench/draft-q4-protocol.md` measured 0.6992 -> 0.6918 acceptance and
102.53 -> 104.18 tok/s (+1.61%). Numerics can move acceptance; 0.69 is not an
immutable head property. The observed 0.74 percentage-point change has no
reported uncertainty and is not a measured ceiling on recoverable acceptance.
Measure per-depth agreement, margins and time on identical trunk histories to
separate quantization error from head prediction error. A more accurate head
only helps if recovered outputs pay for its extra cost. Preserve exact target
verification; changing its acceptance rule is a different correctness contract.

## 5. Prefill already uses chunks and causal attention

`serve/engine.mojo` already chooses m=min(MROWS,remaining prompt), MROWS=8.
`amar_attn_decode` launches one block per query head/row and sets
T=t_len+row, providing the causal boundary after appending the chunk's KV.
Thus M5 item 3 describes work that is partly already implemented.

Keep chunked projections as the incremental baseline. For longer prompts use
a tiled causal prefill attention path with online softmax, integrating queries
across the chunk and attending all prior KV. A single whole-prompt launch can
still be tiled internally; one launch versus chunks is not the key algorithmic
distinction. SSM recurrence remains ordered even if projections are batched.
The right chunk size needs prompt-length/TTFT measurements, not a decode ratio.

MAX_T=1024 in attn.mojo sizes shared score storage; it is not a chunk size or
supported context promise. Registry TMAX=128 sizes actual KV/token buffers.
Long-context work must reconcile these and enforce prompt+generation capacity
before writes. That guard now exists: `serve/engine.mojo` rejects a request
whose prompt plus generation exceeds `tmax` before any write, and checks
`n_total` again in the decode loop. Changing a constant alone
is not a prefill implementation. Larger score arrays also increase shared-memory
pressure. Prefill affects TTFT/total latency, not the decode-only tok/s_gen target.

## 6. Launch overhead remains unisolated by host_enqueue_s

host_enqueue_s is wall time since t0 before the final synchronize. It includes
prefill synchronization, speculative copies/waits, and any profiling syncs.
gpu_total_s is the same wall interval after the final drain, not GPU busy time.
Their difference measures the final outstanding tail, not total GPU idle time.

The measured 2.57 us empty-launch floor and 646 launches imply 1.660 ms. The
6.9% figure divides by the old 24 ms bf16 token; on 1/68.77 s it is 11.4%,
assuming unchanged count/floor. Neither is a rigorous recoverable-overhead bound:
empty launches and real kernels have different overlap, service and ramp costs.
The actual five-stage/fused-shape receipt (24 vs 16 us, 24 layers) offers only
about 0.192 ms, or 1.3% of a q8 token, before parity/occupancy constraints.
Use a device timeline with host correlation to expose gaps and queue starvation.
Existing evidence rejects a large generic fusion win, not every launch mechanism.

## 7. FFN fusion cannot remove dense weight reads

FFN(x)=Wd[SiLU(Wg*x) elementwise-multiplied by Wu*x]. Gate/up can share x;
down needs the nonlinear intermediate and separate weights. At H=4096,F=12288,
the three q8 matrices contain 160,432,128 bytes including scales. Input x is
8 KiB in bf16; the down input is 24 KiB. Eliminating unique activation transfers
barely changes logical bytes at m=1; repeated cache accesses are a separate cost.
Keeping intermediates on chip can reduce launches/cache traffic, not make the
weight stream disappear. All weights cannot reside in the 96 MB cache. The
nonlinearity also prevents collapsing these into one fixed linear matrix.

## 8. q4 trunk is a live precision experiment, not an established win

The same-run fixture recorded q4/q8 time 0.6636 (58.621/88.331 us), about 1.51x
kernel speed. Q4_0/Q8_0 bytes are 18/34=0.5294, so the measured kernel already
falls short of ideal byte scaling. For eligible time fraction f, unchanged
remainder gives speedup 1/(1-0.3364*f), not automatically 1.51x end-to-end.

Trunk quantization changes target outputs, unlike draft-only quantization.
Separate quantizer/kernel parity against the same q4 weights from model-quality
loss against q8/bf16. Compare against llama.cpp at matched quantization/quality;
beating its q8 arm with a q4 trunk is a different claim. The current pack CLI
implements q8 and q4-draft, not the full --q4 trunk proposed in q4-protocol.md.

## 9. Structured edits are a cleaner next proposer interface

loop-protocol.md records 0/3 then 0/4 survivors. Reading the actual
`.work/loop/002/cand-*.diff` adds a stronger result than that protocol's summary:
all four have malformed hunk counts; candidates 0, 1 and 3 reference undefined
fusion symbols without defining them; candidate 2 replaces a line with itself.
Candidate 1 also removes the up projection. Repairing transport alone would not
rescue these four proposals. This establishes both interface and proposal-quality
failures, not general inability of a 9B model to optimize.
patch --fuzz tolerates missing context; it does not generally repair malformed
hunk syntax/counts. A deterministic hunk recount can isolate that experiment.

Prefer schema-validated edits carrying exact path, base digest, a unique symbol
signature or exact old-text anchor, and replacement text. The controller creates
the diff and rejects stale/ambiguous targets and no-op edits. Function name alone
is insufficient for overloads. Keep file/scope, compile, parity and performance gates outside
proposer control. This is a proposed interface change, not implemented here.
Do not count transport failures as optimization trials or widen context because
of them. First obtain valid compilable edits, then measure candidate quality.

## 10. A falsifier must bind the comparison and the allowed levers

The 20-prompt median paired ratio 0.78 needs about 28.2% speedup to reach 1.0
under that aggregation. Ratio of medians 100.7/123.5 is a different statistic.
Recovering the no-spec 74.14/68.77 gap alone would take 0.78 only to about 0.84,
even if it carried through to all speculative work. The llama.cpp MTP receipt
also reports disagreement with its own greedy outputs on 4/20 prompts: retain
that correctness asymmetry when defining the competitive target.

No receipt establishes a Mojo-language or gfx1100 structural impossibility.
Define prompt/tokenization, precision/quality, exactness, generation length,
decode versus total latency and aggregation first. Then build an optimistic
lower bound on time/output for each allowed design from compulsory bytes at a
valid measured bandwidth ceiling, arithmetic and serial dependencies, avoiding
double-counting overlapped work. Stop that design when even its optimistic
paired-ratio ceiling is <=1.0 against the fixed competitor. With bandwidth B,
bytes/output must be below B/R_target as a necessary bandwidth-only condition.
Current pack-size percentages cannot supply this bound, especially under MTP
where trunk weights are shared across accepted outputs. A quality failure can
independently falsify the q4 path; transport failure cannot falsify kernel search.
