# Independent Codex audit, 2026-11-16

Pane: a lane pane. Scope: read-only source audit; no kernel edits, builds, GPU runs, or commits.

Independent verdict: no MUST defect found for the shipped configurations, positive context length, valid page tables and finite model-range inputs. Four findings: 0 MUST, 1 SHOULD, 3 NOTE. GPU execution and emitted resource usage are UNVERIFIED by this audit.

Read codebase-memory skill and code-review protocol. Applied session-loaded prefix-cache-discipline and memory-optimization. Consulted rdna3-isa and inference-kernels relevant sections. The implementation brief's pre-edit skill checklist does not apply to this read-only audit. Brain recall found no prior audit verdict. Ledger guidance: source-level live-value counts do not establish emitted VGPR usage; isolated tests do not establish engine parity.

## Evidence

Source hashes at audit:
- `kernels/gemma4_attn.mojo`: `aed0643a138cf7d6b4a31dabdd6dc7b555a1216748c70620f0d39b294c33529c`
- `kernels/test_gemma4_attn.mojo`: `05f8a8780ca79a0c6b78e5974bcb67fd701aa4926fa30864a77994e9b76f5696`

Graph lookup returned no target symbols. Coverage reported both targets not tracked, so all 217 kernel lines and 415 test lines were read directly. Helper definitions were read directly after symbol outlines. Coverage for dattn, arch and bexp reported metadata_match, which is not a proof of completeness. The external llama.cpp reference was inspected directly, outside this project's graph.

CPU arithmetic simulation in the audit tool call passed 64 lane-result checks for NLD 4/8 and 672 boundary/split combinations. It checked unique complete attended-token coverage and that valid rows stay on the chunk's page. Contexts included 127/128/129, 191/192, 511/512/513/514/519 and 16384; windows 0/511/512/513; splits 1/2/3/7/16/64. This tests the index and reduction formulas, not compiled Mojo or GPU behavior.

## Findings formed before reading the earlier review

1. **SHOULD**, `kernels/test_gemma4_attn.mojo:232`: scale sensitivity has no explicit negative-control launch. The unscaled f64 oracle and large random scores should reject an accidental 1/sqrt(HD), but this particular mutation has not been demonstrated by the test's mutant checks. Concrete fixture: T=2, q=e0, K0=0, K1=e0, V0=0, V1=1 gives 0.7310586 at scale 1 versus 0.5156199 at scale 1/16 (HD=256). Fix: add a scale-mutated launch or equivalent rescaled-query negative control and require it to fail BAR. This is a test-strength recommendation, not a demonstrated false PASS of the current suite.

2. **NOTE**, `kernels/gemma4_attn.mojo:125` and `:179`: numerical domain and empty-context preconditions are implicit. T=0 has zero denominator; a live +inf score gives exp(inf-inf); finite Q/K large enough to overflow their f32 dot also produces non-finite output. A finite live score equal to NEG gives masked tail lanes nonzero probability. Example: HD=256, WINDOW=512, NLD=8, T=2, q0=1, K0[0]=K1[0]=-3.4e38, other K/Q components zero, V0=0 and V1=1 yields 7/8 rather than 1/2 because six masked elements replicate V1 with probability 1. Fix: document T>=1 and finite, bounded scores, or explicitly mask probabilities and define a non-finite policy if the kernel must support these inputs. These are outside normal RMS-normalized Gemma4 inference, so they are not release blockers.

3. **NOTE**, `kernels/gemma4_attn.mojo:67`: generic launch assumptions are not fully asserted. BT must contain complete waves and the launched block must match BT. Example: BT=48 on wave32 leaves wave=1 indexing arrays sized NW=1; current assertions can accept this. Fix: assert BT % WARP_SIZE == 0 and document matching launch dimensions. Shipped BT=256/1024 are valid. NLD divides wave32, hence divides the 128-token page on gfx1100; no cross-page chunk defect exists for the shipped NLD values.

4. **NOTE**, `kernels/gemma4_attn.mojo:87` and `:137`: resource pressure needs an emitted-code receipt before any occupancy claim. Concrete shapes: HD256/W512/GB1/NLD8/BT1024 holds source arrays q+o+kr+vr+part totaling 152 f32 elements per lane before temporaries; HD512/W0/GB2/NLD4/BT256 totals 200. These are source live-value estimates, not VGPR counts. Fix: record VGPR, scratch/spills and LDS from the compiled production specializations. No source-level barrier divergence was found.

## Correctness assessment

- Scale is exactly 1: dot products flow directly into softmax. No attention softcap is present. llama.cpp `src/models/gemma4.cpp:12`, `:15-17`, `:299-306` confirms scale 1 and source-layer KV reuse. Caller-selected att_i expresses sharing; integration must map source layers to slots in the correct HD pool.
- Window is [max(0,T-WINDOW), T), matching inclusive positions max(0,t-511)..t when T=t+1. Splits partition aligned NLD chunks, then live masks remove window/tail excess.
- Loads clamp invalid chunk elements to live rows before loading K/V. This avoids masked NaN padding entering a zero-times-NaN FMA. For shipped NLD=4/8 and 128-token pages, a chunk cannot cross a page boundary. `dkv_tab_off` includes physical page, NAT and slot, then the token and dimension offset.
- Butterfly reductions select token floor(lane/DUP) and reduce all 32 lane contributions. The denominator counts one duplicate per token. `readlane(p,i*DUP)` selects that token's full dot probability; all lanes of an active wave execute this call with a uniform lane selector.
- Online state is (m,l,o), with l=sum exp(score-m), o=sum exp(score-m)*V. Both wave and split combines rescale l/o by exp(old_m-global_m) before dividing once. This is the correct stable merge.
- An empty wave or split retains finite NEG, zero l and zero o. Empty-only partial blocks write zeros with NEG. With normal nonempty scores, their merge weight underflows to zero; with all empty waves, exp(NEG-NEG)=1 remains finite until final zero-denominator normalization. Positive T plus any valid split count ensures some nonempty split.
- Shared arrays are written before uniform barriers. SLAB writes disjoint regions then synchronizes. The fallback serializes wave accumulation with a barrier after each wave; no thread exits around these barriers.
- Source LDS on wave32: HD256/W512 is 33024 bytes (32768 acc + 256 stats), HD256/W0 and HD512/W0 are 33024 and 32896 bytes. HD512/W512 takes the fallback: 2304 bytes and 33 block barriers, versus two barriers in SLAB. The large 1024-thread sliding block and repeated fallback barriers can constrain performance; no measured occupancy or speed conclusion follows from source alone.

## Test assessment

The f64 oracle is structurally independent of the GPU reductions and softmax merge. All four HD/window combinations, both query scales, long T=16384, and deliberately empty splits are covered. Output poisoning detects missing stores; explicit NaN rejection prevents a poisoned result silently passing.

Wrong KV slot reads NaN because slot 0 is never populated. Ignoring the reversed page table fails multi-page cases. A larger window reads poisoned rows; a smaller window removes a planted dominant oldest row. A shortened context removes a planted newest row. Actual window +/-1 mutants run for HD256; context-length mutants run across shapes. Wrong scaling should fail the oracle for multi-token cases, but there is no explicit scale mutation receipt. T=1 alone cannot detect scaling, as expected.

`two_pools` tests coexisting storage and interleaved launches, not concurrent streams or full-engine shared-layer routing. `run-tests.sh:101` registers the test. The existing executable is not evidence of a run on the audited hashes; this audit did not execute it.

## Earlier-review comparison

Read only after the independent findings were saved.

| Earlier point | Verdict against current source |
| --- | --- |
| Online softmax and partial merge | AGREE. Same derivation above. |
| Finite NEG prevents empty-partial NaNs | AGREE for empty partials merged into positive-length normal input. T=0 still has undefined normalization. |
| Window includes the current token | AGREE. Header line 6 now states the T convention. |
| Butterfly/readlane mapping | AGREE; CPU lane arithmetic independently checked. |
| Scale 1, int8 refusal, att_i and Q/O layout | AGREE. |
| MUST separate HD pools and add live-pool test | AGREE with the integration requirement; DISAGREE that this remains an unaddressed kernel MUST. Header lines 3-5 and two_pools now provide the requested contract and test. Engine wiring is outside this audit. |
| HD512/NLD8 register risk | AGREE for that specialization. Current g4_nld selects 4 for HD512. Emitted VGPR/spill evidence is still needed for performance claims. |
| Eight serial wave-combine barriers | DISAGREE as a description of current shipped full-attention configurations: these take SLAB and two block barriers. HD512/window512 uses a 32-wave fallback with 33 total barriers. |
| Merge has no bounds check | DISAGREE with current source: lines 202-203 return for c>=HD_. Too few threads still leave output unwritten; launch contract at line 8 and test wrapper use HD/8. |
| Test large-magnitude scores for bexp | AGREE; current qs=1/4 fixtures include planted scores reported as 43-341. This does not certify a BARO_SHARED_EXP build that was not run. |
| Required shape matrix, empty splits, window mutant, two pools | AGREE; all are present in current test source. |

NEW: explicit scale-mutant recommendation; precise numeric-domain limitations; incomplete BT specialization assertions. Resource finding updates the earlier risk to the current configurations.

Next action: owner adds the scale negative control and supplies the current-hash GPU correctness and emitted-resource receipts before claiming runtime validation.
