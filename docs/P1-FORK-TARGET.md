# P1 cross-node fork: two live nodes fit at 4k, not at 32k

**Correction, 2026-09-17 afternoon. This document's first version said the rig "cannot be built on
this card" and that MAX's memory cap "does not govern the hold". Both were wrong as general claims.**
I had tested the 9B at `BARO_TMAX=33024` only, varied the cap, never the context length, and
generalised from one operating point. the maintainer asked whether 1B models could be tried; that question
is what exposed it. Measured since, `bench/fork-cap-probe.sh`, both nodes answering with identical
ids and exiting clean:

| engines | TMAX | cap | node A | node B | total | |
|---|---|---|---|---|---|---|
| 2 x Llama-3.2-1B (spark) | 4096 | 45 | 11.3 GB | 6.2 GB | 19.4 GB | PASS `.work/fork/cap-probe-1b` |
| 2 x 9B dense q4 (engine) | 4096 | 10 | 10.7 GB | 10.7 GB | 23.5 GB | PASS `.work/fork/cap-probe-9b-t4096` |
| 2 x 9B dense q4 (engine) | 33024 | 45, 25, 10 | 19.5, 21.5, 18.9 GB | OOM | | FAIL, table below |

So gate 2 splits. **Its identity half (20 short prompts, three link rates) has a literal rig: two
live 9B nodes at TMAX 4096,** with about 0.5 GB of headroom, so nothing else may hold VRAM. Only the
9B matters here: `serve/engine.mojo` can import state, `serve/spark.mojo` (every 1B we serve) can
export but not import. **Its 32k timing half still has no literal rig,** and for that half the rig
table below still applies and the choice is still the coordinator's or the maintainer's.

Lane FORK, `briefs/2026-09-17-latentos-cross-node-fork.md`, 2026-09-17. Nothing here is a gate result.

## At 32k one engine needs about 19 GB, whatever the cap says

The whiteboard listed `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` as "untested" for the
engine (the GEMM benches run it at 10). `bench/fork-cap-probe.sh` starts node A, reads VRAM, starts
node B beside it. Dense q4 pack, `BARO_TMAX=33024`, engine `aca4c9e0ac5f72c5`, receipts
`.work/fork/cap-probe*/probe.log`:

| percent | node A VRAM delta | node B |
|---|---|---|
| 45 | 19,503 MiB | `hipErrorOutOfMemory` |
| 25 | 21,486 MiB | OOM on the 6.72 GB pack allocation |
| 10 | 18,852 MiB | OOM on the 6.72 GB pack allocation |

25 percent held MORE than 45 percent. I read that as "the knob does not govern the hold". The
narrower truth: at this context length the engine's real need (pack, 32k of f32 KV, prefill
scratch) is about 19 GB, so it grows past any pool the cap sets. The 1B holds above (11.3 GB, then
6.2 GB) are consistent with the cap acting as a share of FREE memory at startup; that mechanism is
inferred from two points and not tested.

The first run of that probe cost more than it should have. My script started its servers inside a
command substitution, lost their pids, and hung after node B's OOM with node A alive: the job sat
in the GPU queue for 9 minutes in front of a priority-90 job until I cancelled it by hand. Fixed in
`ebb406d` with a CPU preflight of the failure path; ledger `gpuwaitingroom.md` 2026-09-17.

## Four rigs can run; two of them cannot fail, so they are not gates

| rig | what it measures | what it cannot judge |
|---|---|---|
| A. Sequential XTX, the rig already frozen in `docs/P1-STATE-API.md` build step 3: node A exports and exits, the LAT1 stream crosses the shaped veth, node B (other pack path, cold) imports and continues | identity from an imported state on a COLD target; payload against a fair recipe arm, because the same GPU re-prefills | the live holder-to-target socket in `fork_target.rs`: it drives export and import, never `/v1/fork` with `target` |
| B. Self-target through a peer forwarder: one node, `target` is a `socat` in the peer netns forwarding back to the same node, tbf on `veth-b4a` egress only so the state crosses the shaped hop once | the shipped `/v1/fork` `target` path end to end at 100 Mbit, 1 Gbit, 10 Gbit, 32k wall clock included | identity from the imported state: the target just exported that prefix, so a hit may come from its resident checkpoint |
| C. Time-sliced engines under live nodes: both `baro-serve` processes stay up, each engine child holds the GPU only while it works | the literal API between two nodes | a fair payload time. The target pays an engine cold start (pack load) no two-GPU deployment pays, so the 1 Gbit verdict would rest on a subtracted term. Also new serving behaviour, L-sized |
| D. Target on the iGPU (P4 precedent) | the live path between two real engines | the kill line. The recipe arm re-prefills 32k on an iGPU, so payload cannot lose. The iGPU carries the open p06 request-state bleed (`exchange/lane-P4-report.md`), so a miss would be ambiguous, and its reference ids come from another GPU architecture |

## I would run A plus B, and say plainly what neither proves

A carries both of gate 2's claims on a cold target with a recipe arm that can win. B proves the code
this lane shipped moves a 32k state over a shaped link and answers. Neither verdict rests on a
subtracted term. C buys the literal API shape at the price of an unfair clock; D cannot fire the
kill line at all.

This recommendation now covers the 32k timing half only. I first wrote here that two DISTINCT live
engines of ours talking to each other "needs a second GPU, which this machine does not have". That
was wrong: at TMAX 4096 two live 9B nodes fit, so the live path and the identity half can run on
the literal rig. What stays out of reach on this machine is two live engines at 32k.

## The API that landed (`8ba8294`), unverified end to end

`POST /v1/fork` accepts `"target":"HOST:PORT"` (an optional `http://` is stripped; anything else is a
400). This amends CONTRACT 3 of `docs/P1-STATE-API.md`, which answered 501
`fork_target_needs_router`: the brief moves the fork itself onto `baro-serve` and leaves PLACEMENT
(node id to address, sticky routing) to the router after P0b. `serve/src/bin/router.rs` is untouched.

1. The holder tokenizes the prompt and exports it through `state.rs`'s own LAT1 writer
   (`export_lat1_file`, split out of `export_stream`: one writer, two callers), at the prompt end
   minus one, the only position `save_state` writes. Format is the holder's startup format
   (`BARO_STATE_INT8=1` for int8 pages).
2. Header and body stream to the target's `POST /v1/state/import` in 8 MiB chunks. The target's
   CONTRACT 2 checks run unchanged; a refusal (409 `state_identity`, 501 `kvq_state_open`) is
   relayed with the target's own status and body.
3. The fork body, `target` removed and `prompt` replaced by the holder's token ids, goes to the
   target's `/v1/fork`. The answer is the target's plus a `target` object: `node`, `pos`,
   `prefix_hash`, `format`, `state_bytes`, `export_s`, `import_s`, `answer_s`, `import`.

Checked: `cargo nextest` 75/75 (7 new: a mock node receiving header and file with the exact
`Content-Length`, the 409 relay, an unreachable target), clippy clean. Not checked: any live run.

One defect found while reading and left alone, because the file is team A's: `export` reports `pos`
from the request or the role-boundary default, but the engine always saves at the prompt end minus
one. With `messages` and a boundary before the end, the LAT1 `pos_hi` and `prefix_hash` describe a
position the body does not hold. The fork path always uses the end minus one and is not affected.

## Open

Which rig judges gate 2? Until that is answered the lane works on gate 4, which does not depend on
it: our engine and llama-server are time-sliced by construction (export to a file, stop one, start
the other). If the answer is "buy the literal shape", C is the only way, and its payload number
should be published with the cold start inside it, not subtracted.
