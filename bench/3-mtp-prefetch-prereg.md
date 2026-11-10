# MTP draft expert prefetch falsifier (2026-10-21)

Before any route sample, freeze this diagnostic. No kernel edits or performance timing.

## Measurement

For each speculative verify window, record the MTP draft router's selected expert IDs by draft step, the verify router's selected IDs by token and target layer, and the verify pass's actual cache misses with exact expert byte sizes. The draft step is aligned to the corresponding verify token index. The draft's selected ID set is treated as a candidate for every verify layer, since it has no layer-specific prediction.

Report two measures:

1. Route overlap: matching draft IDs / all verify routed expert selections, preserving token position and comparing the expert ID value across target layers.
2. Miss-byte coverage: bytes from actual verify misses whose `(layer, expert ID)` is selected by the draft at a verify token that routes to that key, divided by all actual verify miss bytes in that window.

Deduplicate actual misses by `(layer, expert ID)`, as the engine cache does. Weight miss-byte coverage by the exact gate, up, and down tensor byte count. Include per-window and aggregate values.

## Frozen kill line

STOP if route overlap is below 50% or miss-byte coverage is below 10%. Continue to design prefetch only if both are at or above the thresholds. The first falsifier uses one short prompt with at least four verify windows; no speed result is collected.

The output trace is opt-in (`Q4X_MTP_TRACE`) and reuses the existing selection host buffer. With the flag unset, the decode path is unchanged.
