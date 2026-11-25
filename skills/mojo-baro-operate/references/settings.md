# Settings reference

Source for every row: docs/settings.toml. Basic settings are those with `level = "basic"` (settings.toml, per `[[setting]]`). The face shows them by default (face/settings.mojo:159).

## Basic settings

One line each: key, group, default, restart needed, summary (docs/settings.toml).

- `sampling.temperature`: sampling, default 0.0, no restart. Randomness of the draw; 0 is greedy.
- `sampling.top_p`: sampling, default unset, no restart. Keep the smallest token set whose probability sums to p.
- `sampling.top_k`: sampling, default unset, no restart. Sample only from the k most likely tokens.
- `sampling.max_tokens`: sampling, default 16000, no restart. Most tokens one answer may generate.
- `chat_template_kwargs.enable_thinking`: sampling, default true, no restart. Let the model think before it answers; the chat shows the thinking block folded.
- `BARO_DEFAULT_TOP_K`: sampling, default unset, restart. Top-k for requests that leave it out.
- `BARO_DEFAULT_TOP_P`: sampling, default unset, restart. Top-p for requests that leave it out.
- `BARO_DEFAULT_MIN_P`: sampling, default unset, restart. Min-p for requests that leave it out.
- `BARO_KVQ`: engine, default f32, restart. KV cache dtype: f32, or int8 (dense only).
- `BARO_TMAX`: engine, default 16384, restart. Context window the KV pool is sized for, in tokens.
- `BARO_MEGA`: engine, default true on dense (0 on MoE), restart. Persistent megakernel.
- `BARO_PREFILL`: engine, default true, restart. Batched prefill, up to `BARO_PREFILL_C` rows per chunk.
- `BARO_SPEC`: engine, default true on dense, restart. Speculative decoding with the model's MTP draft head.
- `BARO_TIER`: engine, default unset, restart. Experts in host RAM with this many cached per layer in VRAM (64 is 5.2 GB).
- `BARO_SEQS`: engine, default 1, restart. Sequences in flight on one engine.
- `BARO_PACK`: model, default `.work/moe-w1/pack`, restart. Weight pack directory the engine loads.
- `BARO_CACHE`: model, default `~/.cache/baro`, restart. Directory for compiled engines, manifests and packs.
- `BARO_POOL`: front, default 1, restart. Number of engine processes `baro-serve` spawns.

Source: docs/settings.toml basic rows.

## Presets

Eight presets, three groups. Each line gives the name, group and the values it stages (docs/settings.toml `[[preset]]`).

- **Fast** (speed): `BARO_SPEC` on, `BARO_SPEC_K` 2, `BARO_MEGA` on, `BARO_PREFILL` on, `BARO_TMAX` 16384, temperature 0.0.
- **Balanced** (speed): startup default. `BARO_SPEC` on, `BARO_SPEC_K` 2, `BARO_KVQ` f32, `BARO_MEGA` on, `BARO_PREFILL` on, `BARO_TMAX` 32768, temperature 0.7, top_p 0.9.
- **Quality** (speed): exact path. `BARO_SPEC` off, `BARO_KVQ` f32, `BARO_TMAX` 32768, temperature 0.0. Quality gain unmeasured.
- **Chat** (task): temperature 0.7, top_p 0.9, max_tokens 16000. Values are a convention, unmeasured.
- **Code** (task): temperature 0.2, top_p 0.95, max_tokens 16000. Values are a convention, unmeasured.
- **Reasoning** (task): thinking on, temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, max_tokens 16000.
- **Low VRAM** (fit): `BARO_KVQ` int8, `BARO_TMAX` 16384. Check with `pack-fit` before applying.
- **Long context** (fit): `BARO_KVQ` int8, `BARO_TMAX` 131072, `BARO_SPEC` on, `BARO_MEGA` on.

Caution for both fit presets: `BARO_KVQ` is a build flag (`-D`). The GUI Apply path passes it to `tools/baro serve` as an environment variable, and `tools/baro` only turns a profile's `[build]` section into `-D` flags (the profile block of `tools/baro`, from the comment "A profile (profiles/NAME.toml"; a static read, not run). After applying, confirm the KV type another way, or use the profile route (`profiles/turbo27b-coder.toml`). `GET /health` `limits.tmax` does show the window.

## Do not enable

Each line gives the reason and where it is stated. The full list with sources is in SKILL.md under "What NOT to turn on".

- q4x MTP verify window (FAILS, docs/CAPABILITIES.md, "q4x MTP verify window").
- `BARO_PF_ATT`, `BARO_PF_SSM` on MoE prefill (FAIL identity, docs/CAPABILITIES.md, "WMMA attention and the dense chunk scan in MoE prefill").
- A requantised q4 pack of a Q4_K_M model (FAILS, 92.97% agreement, README.md, "Decode against llama.cpp").
- `BARO_NGRAM` on MoE (slower, docs/CAPABILITIES.md, "MoE prompt-lookup (ngram) speculation").
- `BARO_MEGA_WIN`, `BARO_DOT` (default off, docs/CAPABILITIES.md, the `BARO_MEGA_WIN` and `BARO_DOT` entries).
- `presence_penalty`, `frequency_penalty` (inert; docs/operate-capabilities.toml, penalties).
- Mojo front (experimental; docs/operate-capabilities.toml, mojo-front).
- `BARO_CERT` as a default (decode ratio 1.023; docs/operate-capabilities.toml, cert).
- `BARO_SHARED_EXP` (3.6x exp cost; docs/CAPABILITIES.md, "Shared exp/log/rsqrt/sin/cos").
- `tool_choice: auto` as a tool filter (does not restrict names; docs/operate-capabilities.toml, tools).
- CPU engine for speed (6 to 7 tok/s; docs/operate-capabilities.toml, cpu).
