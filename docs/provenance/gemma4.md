# Provenance: gemma4 (Forge-1 E2B) in mojo-baro

The idea: run the Gemma 4 E2B decoder in our own Mojo engine on the 7900 XTX. The architecture facts (what the
model computes) are a specification, read from the reference implementation and the model card, and written into
our own code from that list; no code is copied or translated.

Primary sources: the Gemma 4 model (google/gemma-4-E2B config.json, `~/Models/gemma-4-e2b-base`), the GGUF keys of
the Forge-1 file. Where we saw the computation spelled out: llama.cpp `src/models/gemma4.cpp` (956591f), MAX
`pipelines/architectures/gemma4`.

The spec, as we use it (per token, layer il):
- x = token_embd[tok] * sqrt(H). PLE input: e = per_layer_token_embd[tok] * sqrt(PLE) / sqrt(2);
  p = per_layer_model_proj . x, each PLE-wide slice RMS-normed with eps * H (the 1/sqrt(H) projection scale
  folded into eps), times per_layer_proj_norm / sqrt(2); ple = p + e.
- attention: h = rmsnorm(x) * attn_norm; q = W_q h, per-head rmsnorm * q_norm, NEOX rope; own-KV layers:
  k = W_k h, rmsnorm * k_norm, rope; v = W_v h, rmsnorm without weight. Scale 1.0. Sliding layers attend the
  last WIN positions (base 1e4, rotary 256), full layers all (base 1e6, rotary 512, theta / rope_freqs[j]).
  Layers past n_layer - shared_kv_layers have no K/V and attend the cache of the last own-KV layer of their kind.
  x = x + rmsnorm(W_o a) * post_attention_norm.
- MLP: x = x + rmsnorm(W_down (gelu_tanh(W_gate g) * W_up g)) * post_ffw_norm, g = rmsnorm(x) * ffn_norm;
  width per layer from feed_forward_length (double width on the shared-KV layers).
- PLE: x = x + rmsnorm(W_proj (gelu_tanh(W_inp_gate x) * ple[il])) * post_norm; then x *= layer_output_scale.
- out: logits = token_embd . (rmsnorm(x) * output_norm); softcap 30 * tanh(l / 30).

Measured limits of the references (2026-11-16): llama.cpp 99 tok/s decode F16 here; MAX nightly correct but
0.95 tok/s decode. Our bar: correctness by teacher-forced agreement vs llama.cpp, decode speed against its 99.
