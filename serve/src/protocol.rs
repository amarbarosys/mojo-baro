//! The stdin/stdout line protocol between this server and `serve/engine.mojo`
//! running with `BARO_SERVE=1`. Documented in `serve/PROTOCOL.md`.
//!
//! Every function here is total: a malformed or unexpected line is returned
//! as [`EngineMsg::Log`] and never panics or is trusted.

use serde::Serialize;
use serde_json::Value;

/// C3 sampler control block (bench/chat-protocol.md): parsed by the engine
/// (`serve/engine.mojo::SampleParams`) but not yet acted on there -- the
/// live decode loop still always takes the greedy/MTP path. `#[serde(skip_serializing_if)]`
/// on every field means a request that sets none of them serialises
/// byte-for-byte as it did before this struct existed.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct SampleParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_topk: Option<u32>,
    /// neuron v2 (E1): engine steering and capture fields, forwarded as the engine's own request keys. `gain` is "L:N:k,..."
    /// (FFN activation N of layer L times k); steer_file / steer_bank (+ layer, row, scale) add a vector; `capture` names a
    /// directory under the engine's BARO_CAPTURE (docs/format-capture.md in mojo-neuron). Absent = byte-identical request line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_bank: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_layer: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_row: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<String>,
    /// Tool-name constraint (`main.rs::tool_name_regexes`): a draw constraint, so it rides with the
    /// sampler fields. The engine arms `tool_re` after each `<tool_call>`, `tool_re0` at the first draw.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_re: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_re0: Option<String>,
}

/// T3b (AMDHQ 06-experiments `## T3b`): HIP IPC checkpoint handoff. `ipc_export`,
/// `ipc_release`, `ipc_sha` and `ipc_stats` are engine ops answered by one `ipc_op` line;
/// `ipc_ingest` + `ipc_prefix` ride on an ordinary request and land the exported state first.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct IpcFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_export: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_release: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_sha: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_ph: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_stats: Option<bool>,
    /// The 320-byte message (256-byte LatentHeader + 64-byte handle) as hex.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_ingest: Option<String>,
    /// The checkpoint's prefix ids as hex little-endian u32.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipc_prefix: Option<String>,
}

/// One request line, serialised exactly as the engine's parser expects.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub n: u32,
    pub spec: bool,
    /// Token-id sequences; generation stops as soon as the tail of the
    /// generated tokens equals any of them (EOS ids included, as length-1
    /// sequences -- `serve/src/main.rs` builds this list).
    pub stop: Vec<Vec<u32>>,
    /// M1b role-boundary checkpoint hint positions (`Text::role_boundaries`),
    /// ascending; empty when the request has no message list.
    pub ckpt: Vec<u32>,
    /// Checkpoint API (LatentOS plan 10 sec 8): BAROST01 state file the
    /// engine writes after prefill / loads before the chain lookup. Absent
    /// keys, so an engine without the feature ignores them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_save: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_load: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_fork: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_resume: Option<String>,
    /// P2b: `prompt` holds only the suffix; the engine prepends the token list
    /// stored in the `state_load` file's header, so a fork's request line is
    /// O(suffix), not O(P). Absent when false.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub state_prefix: bool,
    #[serde(flatten)]
    pub sample: SampleParams,
    /// JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
    /// `response_format.json_schema.schema`, verbatim -- `serve/grammar_rt.mojo`
    /// compiles it. Absent for every request that does not set
    /// `response_format`, which is every request before this lane.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    /// Item 2: whether the model is reasoning for this request (from
    /// `chat_template_kwargs.enable_thinking`, default true). Only read by
    /// the engine when `schema` is also set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// P0a-e (`serve/PROTOCOL.md`): ask for the last-prompt-token hidden
    /// state as an extra `{"id","embed":[...]}` line between the first
    /// token and `done`. Absent (not `Some(false)`) on every request that
    /// does not ask, which is every request before this lane. The caller
    /// still sends `n:1, spec:false`; the engine does not infer them from
    /// this flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<bool>,
    /// Vision (docs/vision.md): one `.bemb` file per run of `<|image_pad|>`
    /// ids in the prompt, in order, written by `tools/mtmd-embed`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub embeds: Vec<String>,
    #[serde(flatten)]
    pub ipc: IpcFields,
}

/// The line that cancels the request currently decoding, if its id matches.
/// Written to the same stdin the request line uses; the engine polls for it
/// once per decode window.
pub fn cancel_line(id: u64) -> String {
    format!("{{\"cancel\":{id}}}\n")
}

/// What a request line asks for, decided by which Option fields are set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Generation,
    /// One engine op (`ipc_export`, `ipc_release`, `ipc_sha`, `ipc_stats`), answered by one `ipc_op` line.
    IpcOp,
    /// A generation that first lands an exported state (`ipc_ingest` + `ipc_prefix`).
    IpcIngest,
}

impl Request {
    /// The line's kind, or why it is malformed (two ops in one line, an op beside a prompt or an ingest, half an
    /// ingest). The serialized bytes do not change.
    pub fn kind(&self) -> Result<Kind, &'static str> {
        let i = &self.ipc;
        let ops = [i.ipc_export.is_some(), i.ipc_release.is_some(), i.ipc_sha.is_some(), i.ipc_stats.is_some()].iter().filter(|b| **b).count();
        let ingest = i.ipc_ingest.is_some() || i.ipc_prefix.is_some();
        if ops > 1 {
            Err("malformed request: two IPC ops in one line")
        } else if ops == 1 && (ingest || !self.prompt.is_empty()) {
            Err("malformed request: an IPC op shares its line with a prompt or an ingest")
        } else if ingest && (i.ipc_ingest.is_none() || i.ipc_prefix.is_none()) {
            Err("malformed request: ipc_ingest and ipc_prefix come together")
        } else if ops == 1 {
            Ok(Kind::IpcOp)
        } else if ingest {
            Ok(Kind::IpcIngest)
        } else {
            Ok(Kind::Generation)
        }
    }

    /// The wire form: one JSON object plus a trailing newline.
    pub fn line(&self) -> String {
        // A struct of integers, a vector and a bool cannot fail to serialise.
        let mut s = serde_json::to_string(self).unwrap_or_default();
        s.push('\n');
        s
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DoneStats {
    pub n: u32,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub tok_s: f64,
    pub drafted: Option<u64>,
    pub accepted: Option<u64>,
    pub cached: Option<u64>,
    pub prefill_rows: Option<u64>,
    pub restore_s: Option<f64>,
    /// state-file load for a checkpoint fork, and engine wall from request read to done
    pub load_s: Option<f64>,
    /// load_s split: file read, header/token parse, host dequantise or copy, host-to-device
    pub load_read_s: Option<f64>,
    pub load_parse_s: Option<f64>,
    pub load_pin_s: Option<f64>,
    pub load_host_s: Option<f64>,
    pub load_h2d_s: Option<f64>,
    pub wall_s: Option<f64>,
    /// "length" | "stop" | "cancelled"; absent on an older engine that has
    /// no concept of an early stop, in which case the caller falls back to
    /// its own client-side EOS check.
    pub finish: Option<String>,
    /// Wire seam (docs/lanes/2026-11-09-fsm-fixes.md): words the engine adds to the done line; absent on an
    /// older engine. `fail` is the grid-barrier fail word (non-zero = the tokens are invalid). `path` is an
    /// opaque string (mega, mega_win, spec, launch, prefill, replay, pair, seqs, ...): never matched on.
    pub fail: Option<u64>,
    pub path: Option<String>,
    pub grammar: Option<Value>,
    pub gdraws: Option<u64>,
    pub gaccepted: Option<u64>,
    pub grejected: Option<u64>,
    pub captured: Option<u64>,
    /// Where a checkpoint fork's state came from: "hot" or "file".
    pub load_tier: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadyData {
    pub tmax: u32,
    pub mrows: u32,
    pub kmax: u32,
    pub spec_k: u32,
    pub pack: String,
    /// The engine's KV dtype from its ready line ("int8", "bf16", "f32"). The engine fixes it at compile
    /// time (int8 is the MoE default since f3220ad), so baro-serve's own env cannot know it.
    pub kv: String,
    /// Wire seam: the expert-tier mode word and the number of concurrent sequences; absent on an older engine.
    pub tier: Option<String>,
    pub seqs: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatentResponseData {
    pub id: u64,
    pub latent_id: String,
    pub tokens: Option<u32>,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineMsg {
    /// Printed once after the pack is loaded and every buffer is allocated.
    Ready(Box<ReadyData>),
    /// One generated token id for request `id`, in generation order.
    /// `logprob`/`top_logprobs` are present only when the engine actually
    /// sampled with `top_logprobs > 0` for this token (item 4,
    /// briefs/2026-09-16-sampling-all-models-lane.md); absent on every
    /// other line, including every line before this feature existed.
    Tok { id: u64, tok: u32, logprob: Option<f64>, top_logprobs: Vec<(u32, f64)> },
    /// P0a-e: the requested embedding vector for request `id`, sent once
    /// between the first `Tok` line and `Done`. Does not end the request.
    Embed { id: u64, vector: Vec<f32> },
    Hidden { id: u64, vector: Vec<f32> },
    LogitsTopK { id: u64, values: Vec<(u32, f64)> },
    /// Request `id` finished; no more `Tok` lines follow for it.
    Done { id: u64, stats: Box<DoneStats> },
    /// Request `id` was rejected before any token was produced.
    Error { id: u64, error: String },
    /// Latent fork/resume response: engine printed a latent_id (and optional
    /// tokens/bytes for fork). This is emitted before generation continues
    /// for resume, or as the sole response for fork.
    LatentResponse(Box<LatentResponseData>),
    /// T3b: the one reply to an IPC op (`ipc_op` set); ends request `id`.
    Ipc { id: u64, value: Box<Value> },
    /// Anything else: engine diagnostics, or a line that failed validation.
    Log(String),
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FastTokLine {
    id: u64,
    tok: u32,
    #[serde(default)]
    logprob: Option<f64>,
}

/// The per-token line when the engine sampled with `top_logprobs > 0`. Typed so the hot path
/// does not build a `Value` tree per entry; any line this rejects (a null list, a malformed
/// entry, extra keys) takes the lenient `Value` path below, which keeps its old result.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LpTokLine {
    id: u64,
    tok: u32,
    #[serde(default)]
    logprob: Option<f64>,
    top_logprobs: Vec<TopLp>,
}

#[derive(serde::Deserialize)]
struct TopLp {
    id: u32,
    logprob: f64,
}

fn get_u64(v: &Value, key: &str) -> Option<u64> {
    v.get(key)?.as_u64()
}

fn get_u32(v: &Value, key: &str) -> Option<u32> {
    u32::try_from(get_u64(v, key)?).ok()
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    let f = v.get(key)?.as_f64()?;
    f.is_finite().then_some(f)
}

/// Classify one engine stdout line. Never trusts a malformed line.
pub fn parse_line(line: &str) -> EngineMsg {
    let raw = line.trim_end_matches(['\r', '\n']);
    if !raw.starts_with('{') {
        return EngineMsg::Log(raw.to_string());
    }
    if let Ok(t) = serde_json::from_str::<FastTokLine>(raw) {
        return EngineMsg::Tok {
            id: t.id,
            tok: t.tok,
            logprob: t.logprob.filter(|f| f.is_finite()),
            top_logprobs: Vec::new(),
        };
    }
    if let Ok(t) = serde_json::from_str::<LpTokLine>(raw) {
        return EngineMsg::Tok {
            id: t.id,
            tok: t.tok,
            logprob: t.logprob.filter(|f| f.is_finite()),
            top_logprobs: t.top_logprobs.into_iter().map(|e| (e.id, e.logprob)).collect(),
        };
    }
    let v: Value = match serde_json::from_str(raw) {
        Ok(Value::Object(m)) => Value::Object(m),
        _ => return EngineMsg::Log(raw.to_string()),
    };
    if v.get("ready").and_then(Value::as_bool) == Some(true) {
        if let (Some(tmax), Some(mrows), Some(kmax), Some(spec_k), Some(pack)) = (
            get_u32(&v, "tmax"),
            get_u32(&v, "mrows"),
            get_u32(&v, "kmax"),
            get_u32(&v, "spec_k"),
            v.get("pack").and_then(Value::as_str),
        ) {
            return EngineMsg::Ready(Box::new(ReadyData {
                tmax,
                mrows,
                kmax,
                spec_k,
                pack: pack.to_string(),
                kv: match v.get("kv").and_then(Value::as_str) {
                    Some("int8") => "int8",
                    Some("bfloat16") | Some("bf16") => "bf16",
                    _ => "f32",
                }
                .to_string(),
                tier: v.get("tier").and_then(Value::as_str).map(str::to_string),
                seqs: get_u32(&v, "seqs"),
            }));
        }
        return EngineMsg::Log(raw.to_string());
    }
    let Some(id) = get_u64(&v, "id") else {
        return EngineMsg::Log(raw.to_string());
    };
    if let Some(err) = v.get("error") {
        return match err.as_str() {
            Some(s) => EngineMsg::Error {
                id,
                error: s.to_string(),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    // Latent fork/resume response: {"latent_id":"...","tokens":N,"bytes":M}
    // tokens and bytes are optional (present for fork, absent for resume)
    if v.get("ipc_op").is_some() {
        return EngineMsg::Ipc { id, value: Box::new(v) };
    }
    if v.get("latent_id").is_some() {
        let latent_id = v.get("latent_id").and_then(Value::as_str).unwrap_or("").to_string();
        let tokens = v.get("tokens").and_then(Value::as_u64).map(|x| x as u32);
        let bytes = v.get("bytes").and_then(Value::as_u64);
        return EngineMsg::LatentResponse(Box::new(LatentResponseData { id, latent_id, tokens, bytes }));
    }
    if v.get("done").and_then(Value::as_bool) == Some(true) {
        if let (Some(n), Some(prefill_s), Some(decode_s), Some(tok_s)) = (
            get_u32(&v, "n"),
            get_f64(&v, "prefill_s"),
            get_f64(&v, "decode_s"),
            get_f64(&v, "tok_s"),
        ) {
            return EngineMsg::Done {
                id,
                stats: Box::new(DoneStats {
                    n,
                    prefill_s,
                    decode_s,
                    tok_s,
                    drafted: get_u64(&v, "drafted"),
                    accepted: get_u64(&v, "accepted"),
                    cached: get_u64(&v, "cached"),
                    prefill_rows: get_u64(&v, "prefill_rows"),
                    restore_s: get_f64(&v, "restore_s"),
                    load_s: get_f64(&v, "load_s"),
                    load_read_s: get_f64(&v, "load_read_s"),
                    load_parse_s: get_f64(&v, "load_parse_s"),
                    load_pin_s: get_f64(&v, "load_pin_s"),
                    load_host_s: get_f64(&v, "load_host_s"),
                    load_h2d_s: get_f64(&v, "load_h2d_s"),
                    wall_s: get_f64(&v, "wall_s"),
                    finish: v.get("finish").and_then(Value::as_str).map(str::to_string),
                    fail: get_u64(&v, "fail"),
                    path: v.get("path").and_then(Value::as_str).map(str::to_string),
                    grammar: v.get("grammar").cloned(),
                    gdraws: get_u64(&v, "gdraws"),
                    gaccepted: get_u64(&v, "gaccepted"),
                    grejected: get_u64(&v, "grejected"),
                    captured: get_u64(&v, "captured"),
                    load_tier: v.get("load_tier").and_then(Value::as_str).map(str::to_string),
                }),
            };
        }
        return EngineMsg::Log(raw.to_string());
    }
    if let Some(vector) = v.get("embed") {
        return match vector.as_array() {
            Some(arr) => match arr.iter().map(|e| e.as_f64()).collect::<Option<Vec<f64>>>() {
                Some(floats) if floats.iter().all(|f| f.is_finite()) => {
                    EngineMsg::Embed { id, vector: floats.into_iter().map(|f| f as f32).collect() }
                }
                _ => EngineMsg::Log(raw.to_string()),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    if let Some(vector) = v.get("hidden") {
        return match vector.as_array() {
            Some(arr) => match arr.iter().map(|e| e.as_f64()).collect::<Option<Vec<f64>>>() {
                Some(floats) if floats.iter().all(|f| f.is_finite()) => {
                    EngineMsg::Hidden { id, vector: floats.into_iter().map(|f| f as f32).collect() }
                }
                _ => EngineMsg::Log(raw.to_string()),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    if let Some(values) = v.get("logits_topk") {
        return match values.as_array() {
            Some(arr) => EngineMsg::LogitsTopK {
                id,
                values: arr.iter().filter_map(|e| Some((get_u32(e, "id")?, get_f64(e, "logit")?))).collect(),
            },
            None => EngineMsg::Log(raw.to_string()),
        };
    }
    if let Some(tok) = get_u32(&v, "tok") {
        let logprob = get_f64(&v, "logprob");
        let top_logprobs = v
            .get("top_logprobs")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| Some((get_u32(e, "id")?, get_f64(e, "logprob")?)))
                    .collect()
            })
            .unwrap_or_default();
        return EngineMsg::Tok { id, tok, logprob, top_logprobs };
    }
    EngineMsg::Log(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One sampled-token line with `top_logprobs`, the shape `window.mojo` prints per token.
    /// `BARO_PARSE_N` repeats the parse for a timing run.
    #[test]
    fn parse_line_tok_with_top_logprobs() {
        let top = (0..20).map(|i| format!("{{\"id\":{},\"logprob\":-{}.25}}", 1000 + i, i)).collect::<Vec<_>>().join(",");
        let line = format!("{{\"id\":9,\"tok\":42,\"logprob\":-0.5,\"top_logprobs\":[{top}]}}\n");
        let want = EngineMsg::Tok { id: 9, tok: 42, logprob: Some(-0.5), top_logprobs: (0..20).map(|i| (1000 + i, -(i as f64) - 0.25)).collect() };
        let n: usize = std::env::var("BARO_PARSE_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            assert_eq!(std::hint::black_box(parse_line(std::hint::black_box(&line))), want);
        }
        if n > 1 {
            eprintln!("parse_line top_logprobs: {:.0} ns/line", t0.elapsed().as_nanos() as f64 / n as f64);
        }
        // A malformed entry is dropped, the rest kept (the Value path's contract).
        let bad = "{\"id\":9,\"tok\":42,\"logprob\":-0.5,\"top_logprobs\":[{\"id\":1,\"logprob\":-1.0},{\"id\":2},{\"id\":3,\"logprob\":-3.0}]}";
        assert_eq!(parse_line(bad), EngineMsg::Tok { id: 9, tok: 42, logprob: Some(-0.5), top_logprobs: vec![(1, -1.0), (3, -3.0)] });
    }

    #[test]
    fn engine_msg_size_is_boxed() {
        assert!(std::mem::size_of::<EngineMsg>() <= 56);
    }

    #[test]
    fn request_line_is_the_engine_shape() {
        let r = Request {
            id: 7,
            prompt: vec![760, 6511, 314],
            n: 64,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(r.line(), "{\"id\":7,\"prompt\":[760,6511,314],\"n\":64,\"spec\":false,\"stop\":[],\"ckpt\":[]}\n");
    }

    #[test]
    fn request_line_carries_stop_sequences() {
        let r = Request {
            id: 1,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![vec![151645], vec![9707, 11]],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":1,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[[151645],[9707,11]],\"ckpt\":[]}\n"
        );
    }

    #[test]
    fn request_line_carries_ckpt_hints() {
        let r = Request {
            id: 2,
            prompt: vec![1, 2, 3],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![7914, 8020],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":2,\"prompt\":[1,2,3],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[7914,8020]}\n"
        );
    }

    #[test]
    fn request_line_carries_sample_params_only_when_set() {
        let r = Request {
            id: 3,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams {
                temperature: Some(0.8),
                top_p: Some(0.9),
                top_k: Some(40),
                min_p: Some(0.05),
                seed: Some(42),
                presence_penalty: None,
                frequency_penalty: None,
                top_logprobs: None,
                ..Default::default()
            },
            schema: None,
            reasoning: None,
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":3,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[],\"temperature\":0.8,\"top_p\":0.9,\"top_k\":40,\"min_p\":0.05,\"seed\":42}\n"
        );
    }

    #[test]
    fn request_line_carries_schema_and_reasoning_only_when_set() {
        let r = Request {
            id: 4,
            prompt: vec![1],
            n: 8,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams {
                temperature: Some(0.7),
                ..SampleParams::default()
            },
            schema: Some(json!({"type": "object", "properties": {"a": {"type": "string"}}})),
            reasoning: Some(false),
            embed: None,
        };
        assert_eq!(
            r.line(),
            "{\"id\":4,\"prompt\":[1],\"n\":8,\"spec\":false,\"stop\":[],\"ckpt\":[],\"temperature\":0.7,\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}},\"reasoning\":false}\n"
        );
    }

    #[test]
    fn request_line_carries_embed_only_when_set() {
        let r = Request {
            id: 6,
            prompt: vec![1],
            n: 1,
            spec: false,
            stop: vec![],
            ckpt: vec![],
            state_save: None,
            state_load: None,
            latent_fork: None,
            latent_resume: None,
            ipc: Default::default(),
            state_prefix: false,
            embeds: vec![],
            sample: SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: Some(true),
        };
        assert_eq!(r.line(), "{\"id\":6,\"prompt\":[1],\"n\":1,\"spec\":false,\"stop\":[],\"ckpt\":[],\"embed\":true}\n");
    }

    /// Wire seam: the ready line's `tier`/`seqs` and the done line's added words reach the front; an older
    /// engine's lines (none of them) parse to None.
    #[test]
    fn wire_seam_words_are_parsed_and_absent_means_none() {
        let EngineMsg::Ready(r) = parse_line("{\"ready\":true,\"tmax\":1,\"mrows\":1,\"kmax\":1,\"spec_k\":0,\"pack\":\"p\",\"tier\":\"zero_copy\",\"seqs\":4}") else { panic!("not ready") };
        assert_eq!((r.tier.as_deref(), r.seqs), (Some("zero_copy"), Some(4)));
        let EngineMsg::Done { stats, .. } = parse_line("{\"id\":1,\"done\":true,\"n\":2,\"prefill_s\":0.1,\"decode_s\":0.2,\"tok_s\":3.0,\"fail\":2,\"path\":\"some_future_word\",\"grammar\":\"json\",\"gdraws\":9,\"gaccepted\":7,\"grejected\":2,\"captured\":5,\"load_tier\":\"file\"}") else { panic!("not done") };
        assert_eq!((stats.fail, stats.path.as_deref(), stats.load_tier.as_deref()), (Some(2), Some("some_future_word"), Some("file")));
        assert_eq!((stats.gdraws, stats.gaccepted, stats.grejected, stats.captured), (Some(9), Some(7), Some(2), Some(5)));
        assert_eq!(stats.grammar, Some(json!("json")));
        let EngineMsg::Done { stats, .. } = parse_line("{\"id\":1,\"done\":true,\"n\":2,\"prefill_s\":0.1,\"decode_s\":0.2,\"tok_s\":3.0}") else { panic!("not done") };
        assert_eq!((stats.fail, stats.path, stats.grammar, stats.captured, stats.load_tier), (None, None, None, None, None));
    }

    /// A-M5: two ops (or an op plus a prompt) in one line used to dispatch silently on whichever the engine parsed first.
    #[test]
    fn a_request_line_has_one_kind_or_a_named_reason() {
        let ipc = |f: &dyn Fn(&mut IpcFields)| { let mut r = Request { n: 1, ..Default::default() }; f(&mut r.ipc); r };
        assert_eq!(ipc(&|i| i.ipc_stats = Some(true)).kind(), Ok(Kind::IpcOp));
        assert_eq!(ipc(&|_| {}).kind(), Ok(Kind::Generation));
        let both = ipc(&|i| { i.ipc_stats = Some(true); i.ipc_release = Some(3); });
        assert!(both.kind().unwrap_err().contains("two IPC ops"));
        let mut with_prompt = ipc(&|i| i.ipc_sha = Some(1));
        with_prompt.prompt = vec![5];
        assert!(with_prompt.kind().unwrap_err().contains("shares its line"));
        let mut ingest = ipc(&|i| { i.ipc_ingest = Some("ab".into()); i.ipc_prefix = Some("cd".into()); });
        ingest.prompt = vec![5];
        assert_eq!(ingest.kind(), Ok(Kind::IpcIngest));
        assert!(ipc(&|i| i.ipc_ingest = Some("ab".into())).kind().unwrap_err().contains("come together"));
    }

    #[test]
    fn cancel_line_is_a_bare_id() {
        assert_eq!(cancel_line(42), "{\"cancel\":42}\n");
    }

    #[test]
    fn ready_line() {
        let m = parse_line(
            "{\"ready\":true,\"tmax\":128,\"mrows\":8,\"kmax\":8,\"spec_k\":2,\"pack\":\".work/engine-pack-q4\"}\n",
        );
        assert_eq!(
            m,
            EngineMsg::Ready(Box::new(ReadyData {
                tmax: 128,
                mrows: 8,
                kmax: 8,
                spec_k: 2,
                pack: ".work/engine-pack-q4".into(),
                kv: "f32".into(),
                tier: None,
                seqs: None,
            }))
        );
        let EngineMsg::Ready(r) = parse_line("{\"ready\":true,\"tmax\":128,\"mrows\":8,\"kmax\":8,\"spec_k\":0,\"kv\":\"int8\",\"pack\":\"p\"}") else {
            panic!("not a ready line")
        };
        assert_eq!(r.kv, "int8");
        let EngineMsg::Ready(r) = parse_line("{\"ready\":true,\"tmax\":128,\"mrows\":8,\"kmax\":8,\"spec_k\":0,\"kv\":\"float32\",\"pack\":\"p\"}") else {
            panic!("not a ready line")
        };
        assert_eq!(r.kv, "f32");
    }

    #[test]
    fn tok_and_done_lines() {
        assert_eq!(parse_line("{\"id\":3,\"tok\":11751}"), EngineMsg::Tok { id: 3, tok: 11751, logprob: None, top_logprobs: vec![] });
        let m = parse_line(
            "{\"id\":3,\"done\":true,\"n\":64,\"prefill_s\":0.01,\"decode_s\":0.5,\"tok_s\":126.0,\"drafted\":40,\"accepted\":28,\"k\":2,\"cached\":7913,\"prefill_rows\":41,\"restore_s\":0.002,\"load_s\":1.85,\"wall_s\":2.4}",
        );
        match m {
            EngineMsg::Done { id, stats } => {
                assert_eq!(id, 3);
                assert_eq!(stats.n, 64);
                assert_eq!(stats.drafted, Some(40));
                assert_eq!(stats.accepted, Some(28));
                assert_eq!(stats.tok_s, 126.0);
                assert_eq!(stats.cached, Some(7913));
                assert_eq!(stats.prefill_rows, Some(41));
                assert_eq!(stats.restore_s, Some(0.002));
                assert_eq!(stats.load_s, Some(1.85));
                assert_eq!(stats.wall_s, Some(2.4));
                assert_eq!(stats.finish, None);
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let m = parse_line("{\"id\":4,\"done\":true,\"n\":1,\"prefill_s\":0.01,\"decode_s\":0.0,\"tok_s\":0.0}");
        assert!(matches!(m, EngineMsg::Done { id: 4, .. }));
    }

    #[test]
    fn done_line_carries_finish_reason() {
        for (finish, n) in [("stop", 9u32), ("cancelled", 3u32)] {
            let m = parse_line(&format!(
                "{{\"id\":5,\"done\":true,\"n\":{n},\"prefill_s\":0.01,\"decode_s\":0.1,\"tok_s\":1.0,\"finish\":\"{finish}\"}}"
            ));
            match m {
                EngineMsg::Done { stats, .. } => {
                    assert_eq!(stats.n, n);
                    assert_eq!(stats.finish, Some(finish.to_string()));
                }
                other => panic!("expected Done, got {other:?}"),
            }
        }
    }

    #[test]
    fn embed_line() {
        assert_eq!(
            parse_line("{\"id\":6,\"embed\":[0.1,0.2,-0.3]}"),
            EngineMsg::Embed { id: 6, vector: vec![0.1, 0.2, -0.3] }
        );
        assert_eq!(parse_line("{\"id\":6,\"embed\":[]}"), EngineMsg::Embed { id: 6, vector: vec![] });
    }

    #[test]
    fn latent_stream_lines() {
        assert_eq!(
            parse_line("{\"id\":6,\"hidden\":[0.1,-0.2]}"),
            EngineMsg::Hidden { id: 6, vector: vec![0.1, -0.2] }
        );
        assert_eq!(
            parse_line("{\"id\":6,\"logits_topk\":[{\"id\":3,\"logit\":1.5}]}"),
            EngineMsg::LogitsTopK { id: 6, values: vec![(3, 1.5)] }
        );
    }

    #[test]
    fn error_line() {
        assert_eq!(
            parse_line("{\"id\":9,\"error\":\"prompt+n exceeds TMAX 128\"}"),
            EngineMsg::Error {
                id: 9,
                error: "prompt+n exceeds TMAX 128".into()
            }
        );
    }

    #[test]
    fn diagnostics_are_logs() {
        assert_eq!(parse_line("tok/s_gen: 126.5\n"), EngineMsg::Log("tok/s_gen: 126.5".into()));
        assert_eq!(parse_line("GENERATED: 1 2 3 "), EngineMsg::Log("GENERATED: 1 2 3 ".into()));
        assert_eq!(parse_line(""), EngineMsg::Log(String::new()));
    }

    #[test]
    fn malformed_lines_are_never_trusted() {
        for bad in [
            "{\"id\":",                              // truncated
            "{\"id\":\"x\",\"tok\":5}",              // id not an integer
            "{\"id\":1,\"tok\":-5}",                 // negative token
            "{\"id\":1,\"tok\":4294967296}",         // token above u32
            "{\"id\":1,\"tok\":\"5\"}",              // token as string
            "{\"tok\":5}",                           // no id
            "{\"id\":1,\"done\":true}",              // done without stats
            "{\"id\":1,\"done\":true,\"n\":1,\"prefill_s\":\"a\",\"decode_s\":0,\"tok_s\":0}",
            "{\"id\":1,\"error\":5}",                // error not a string
            "{\"ready\":true}",                      // ready without limits
            "[1,2,3]",                               // not an object
            "{\"id\":1,\"done\":true,\"n\":1,\"prefill_s\":1e999,\"decode_s\":0,\"tok_s\":0}",
            "{\"id\":1,\"embed\":\"x\"}",             // embed not an array
            "{\"id\":1,\"embed\":[\"x\"]}",           // embed element not a number
        ] {
            assert!(matches!(parse_line(bad), EngineMsg::Log(_)), "{bad} must not parse");
        }
    }

    #[test]
    fn extra_fields_are_ignored() {
        assert_eq!(
            parse_line("{\"id\":1,\"tok\":2,\"extra\":{\"x\":[1]}}"),
            EngineMsg::Tok { id: 1, tok: 2, logprob: None, top_logprobs: vec![] }
        );
    }
}
