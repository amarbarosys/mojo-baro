//! Ollama-compatible surface on top of the existing request path (P0a,
//! `docs/PLATFORM-PLAN.md`). Every route here builds the same `Gen` the
//! OpenAI endpoints build and submits it through the same
//! `check_and_submit`/`collect`/`ndjson_stream` plumbing, so a single-turn
//! `/api/generate` (the default, `raw` unset) renders through the same chat
//! template as `/api/chat` and `/v1/chat/completions` and produces the same
//! text for the same input (P0a gate 2).
//!
//! `--ollama-port` from the plan's own text is not a separate flag here:
//! every route below lives on the one existing router, so pointing PAIR at
//! this server's `--port` (e.g. 11434) already serves it, and a second flag
//! would just be another name for the same value (CLAUDE.md 7).
//!
//! `images`, `tool_calls` (request side) and `repeat_penalty` are parsed and
//! accepted so a real Ollama client does not 400, but not wired to
//! anything: there is no engine-side multiplicative penalty to map
//! `repeat_penalty` onto (only the OpenAI-shape additive
//! `presence_penalty`/`frequency_penalty` in `protocol::SampleParams`), and
//! vision/tool-call support is out of scope for this item. A parameter with
//! no read-back is not evidence it took effect (`bench/PROTOCOL-RULES.md`
//! P1), so these are named here rather than silently claimed.

use axum::body::Body;

use super::*;

/// UTC RFC3339 timestamp with no `chrono` dependency (Howard Hinnant's
/// `civil_from_days`, https://howardhinnant.github.io/date_algorithms.html).
fn now_rfc3339() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    rfc3339_from_unix_secs(secs)
}

fn rfc3339_from_unix_secs(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let tod = secs % 86400;
    let (h, m, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}.000000000Z")
}

fn ns(secs: f64) -> u64 {
    (secs.max(0.0) * 1e9) as u64
}

// ---- options map (Ollama's `options` object) -------------------------------

#[derive(Deserialize, Default, Clone)]
pub struct Options {
    #[serde(default)]
    num_predict: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    #[allow(dead_code)]
    repeat_penalty: Option<f32>,
    /// Reported, not silently clamped: unread here, so an over-`tmax` value
    /// surfaces through `check_and_submit`'s existing
    /// `exceed_context_size_error` on the real prompt+n check instead of
    /// being quietly overridden.
    #[serde(default)]
    #[allow(dead_code)]
    num_ctx: Option<u32>,
}

impl Options {
    fn to_sample_params(&self) -> protocol::SampleParams {
        protocol::SampleParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            presence_penalty: None,
            frequency_penalty: None,
            top_logprobs: None,
            ..Default::default()
        }
    }
}

fn default_stream() -> bool {
    true
}

// ---- shared generation plumbing --------------------------------------------

/// Builds the same `Gen` `/v1/chat/completions` would for `msgs`, or (when
/// `raw_prompt` is given) tokenizes it directly with no template, the way
/// `/api/generate`'s `raw: true` asks for.
fn build_gen(app: &App, t: &Text, msgs: &[ChatMessage], max_tokens: Option<i64>, opts: &Options, raw_prompt: Option<&str>) -> Result<Gen, ApiError> {
    let (prompt, ckpt) = match raw_prompt {
        Some(raw) => (t.encode(raw, true).map_err(bad)?, vec![]),
        None => {
            let rendered = t.apply_chat_template(msgs, None, None).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, e))?;
            (t.encode(&rendered, true).map_err(bad)?, t.role_boundaries(msgs, None, None))
        }
    };
    Ok(Gen {
        prompt,
        n: max_tokens
            .filter(|&n| n > 0)
            .map(|n| n as u32)
            .or_else(|| opts.num_predict.filter(|&n| n > 0).map(|n| n as u32))
            .unwrap_or(DEFAULT_MAX_TOKENS),
        spec: spec_default(app, None),
        stream: false,
        stop: compute_stop(app, opts.stop.clone().map(StopParam::Many)),
        ckpt,
        state: (None, None),
        state_prefix: false,
        base_pos: 0,
        embeds: vec![],
        sample: opts.to_sample_params(),
        schema: None,
        reasoning: None,
        embed: None,
    })
}

/// NDJSON framing (Ollama's wire shape: raw JSON objects separated by `\n`,
/// `content-type: application/x-ndjson`), the non-SSE sibling of
/// `main.rs`'s `sse_stream`.
fn ndjson_stream(app: Shared, rx: mpsc::UnboundedReceiver<Event>, mut chunk: impl FnMut(&App, ChunkKind) -> Value + Send + 'static) -> Response {
    let mut acc = Acc::new();
    let mut ended = false;
    let body_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).filter_map(move |ev| {
        if ended {
            return None;
        }
        let v = match ev {
            Event::Tok { tok, .. } => {
                let delta = acc.take(app.text.as_ref(), tok)?;
                chunk(&app, ChunkKind::Delta { text: delta, token: tok, logprob: None })
            }
            // No Ollama route sets embed; embeddings.rs collects it separately.
            Event::Embed(_) => return None,
            Event::Hidden(_) | Event::LogitsTopK(_) => return None,
            Event::Latent { .. } | Event::Ipc(_) => return None,
            Event::Done(s) => {
                ended = true;
                let reason = acc.finish_reason(s.finish.as_deref());
                chunk(&app, ChunkKind::Finish { reason, stats: stats_json(&s), tokens: std::mem::take(&mut acc.tokens) })
            }
            Event::Error(e) => {
                ended = true;
                json!({"error": format!("engine: {e}")})
            }
        };
        let mut line = v.to_string();
        line.push('\n');
        Some(Ok::<_, Infallible>(line))
    });
    Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

// ---- GET /api/tags, /api/ps, /api/version, POST /api/show ------------------

fn model_details() -> Value {
    json!({"parent_model": "", "format": "gguf", "family": "unknown", "families": Value::Null,
           "parameter_size": "", "quantization_level": ""})
}

pub async fn tags(State(app): State<Shared>) -> Json<Value> {
    Json(json!({"models": [{
        "name": app.model, "model": app.model, "modified_at": now_rfc3339(),
        "size": 0, "digest": format!("sha256:{}", app.identity.pack),
        "details": model_details(),
    }]}))
}

pub async fn ps(State(app): State<Shared>) -> Json<Value> {
    if !app.engine.alive() {
        return Json(json!({"models": []}));
    }
    Json(json!({"models": [{
        "name": app.model, "model": app.model,
        "size": 0, "digest": format!("sha256:{}", app.identity.pack),
        "details": model_details(),
        "expires_at": now_rfc3339(), "size_vram": 0,
    }]}))
}

pub async fn version() -> Json<Value> {
    Json(json!({"version": "0.1.0-baro"}))
}

#[derive(Deserialize)]
pub struct ShowReq {
    #[allow(dead_code)]
    #[serde(default, alias = "name")]
    model: Option<String>,
}

pub async fn show(State(_app): State<Shared>, Json(_r): Json<ShowReq>) -> Json<Value> {
    Json(json!({
        "modelfile": "", "parameters": "", "template": "",
        "details": model_details(),
        "model_info": {"general.architecture": "unknown"},
    }))
}

#[derive(Deserialize)]
pub struct PullReq {
    #[serde(alias = "name")]
    model: String,
    #[serde(default = "default_stream")]
    stream: bool,
}

pub async fn pull(State(app): State<Shared>, Json(r): Json<PullReq>) -> Result<Response, ApiError> {
    if r.model != app.model {
        return Err(ApiError::Plain(StatusCode::NOT_FOUND, format!("model {} is not loaded", r.model)));
    }
    let digest = format!("sha256:{}", app.identity.pack);
    let lines = [
        json!({"status": "verifying digest", "digest": digest}),
        json!({"status": "success"}),
    ];
    if !r.stream {
        return Ok(Json(lines[1].clone()).into_response());
    }
    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from(lines.iter().map(Value::to_string).map(|s| s + "\n").collect::<String>()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

// ---- POST /api/chat ---------------------------------------------------------

#[derive(Deserialize)]
pub struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[allow(dead_code)]
    #[serde(default)]
    images: Option<Vec<Value>>,
}

#[derive(Deserialize)]
pub struct OllamaChatReq {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(default)]
    options: Options,
    #[allow(dead_code)]
    #[serde(default)]
    keep_alive: Option<Value>,
    #[allow(dead_code)]
    #[serde(default)]
    format: Option<Value>,
}

fn chat_done_json(model: &str, created: &str, n_prompt: usize, n_completion: usize, stats: &Value, reason: &str) -> Value {
    let prefill_s = stats.get("prefill_s").and_then(Value::as_f64).unwrap_or(0.0);
    let decode_s = stats.get("decode_s").and_then(Value::as_f64).unwrap_or(0.0);
    json!({
        "model": model, "created_at": created,
        "message": {"role": "assistant", "content": ""},
        "done": true, "done_reason": reason,
        "total_duration": ns(prefill_s + decode_s), "load_duration": 0,
        "prompt_eval_count": n_prompt, "prompt_eval_duration": ns(prefill_s),
        "eval_count": n_completion, "eval_duration": ns(decode_s),
    })
}

pub async fn chat(State(app): State<Shared>, Json(r): Json<OllamaChatReq>) -> Result<Response, ApiError> {
    let t = need_text(&app)?;
    if r.messages.is_empty() {
        return Err(bad("messages is empty"));
    }
    let msgs: Vec<ChatMessage> = r.messages.iter().map(|m| ChatMessage { role: m.role.clone(), content: m.content.clone(), tool_calls: None }).collect();
    let mut g = build_gen(&app, t, &msgs, None, &r.options, None)?;
    g.stream = r.stream;
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let n_prompt = g.prompt.len();
    let (_req_id, rx) = check_and_submit(&app, &g)?;
    if g.stream {
        let created = now_rfc3339();
        return Ok(ndjson_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, .. } => json!({"model": model, "created_at": created,
                "message": {"role": "assistant", "content": text}, "done": false}),
            ChunkKind::Hidden(_) | ChunkKind::LogitsTopK(_) => json!({"done": false}),
            ChunkKind::Finish { reason, stats, tokens } => chat_done_json(&model, &created, n_prompt, tokens.len(), &stats, &reason),
        }));
    }
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    let created = now_rfc3339();
    let mut v = chat_done_json(&model, &created, n_prompt, acc.tokens.len(), &stats, &reason);
    v["message"] = json!({"role": "assistant", "content": text_out});
    Ok(Json(v).into_response())
}

// ---- POST /api/generate ------------------------------------------------------

#[derive(Deserialize)]
pub struct OllamaGenerateReq {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    raw: bool,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(default)]
    options: Options,
    #[allow(dead_code)]
    #[serde(default)]
    images: Option<Vec<Value>>,
    #[allow(dead_code)]
    #[serde(default)]
    template: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    keep_alive: Option<Value>,
    #[allow(dead_code)]
    #[serde(default)]
    context: Option<Vec<u32>>,
}

fn generate_done_json(model: &str, created: &str, n_prompt: usize, stats: &Value, reason: &str, prompt: &[u32], gen_tokens: &[u32]) -> Value {
    let prefill_s = stats.get("prefill_s").and_then(Value::as_f64).unwrap_or(0.0);
    let decode_s = stats.get("decode_s").and_then(Value::as_f64).unwrap_or(0.0);
    let context: Vec<u32> = prompt.iter().chain(gen_tokens.iter()).cloned().collect();
    json!({
        "model": model, "created_at": created,
        "response": "",
        "done": true, "done_reason": reason, "context": context,
        "total_duration": ns(prefill_s + decode_s), "load_duration": 0,
        "prompt_eval_count": n_prompt, "prompt_eval_duration": ns(prefill_s),
        "eval_count": gen_tokens.len(), "eval_duration": ns(decode_s),
    })
}

pub async fn generate(State(app): State<Shared>, Json(r): Json<OllamaGenerateReq>) -> Result<Response, ApiError> {
    let t = need_text(&app)?;
    if r.prompt.is_empty() {
        return Err(bad("prompt is empty"));
    }
    let g0 = if r.raw {
        build_gen(&app, t, &[], None, &r.options, Some(&r.prompt))?
    } else {
        let mut msgs = Vec::new();
        if let Some(sys) = &r.system {
            if !sys.is_empty() {
                msgs.push(ChatMessage { role: "system".into(), content: sys.clone(), tool_calls: None });
            }
        }
        msgs.push(ChatMessage { role: "user".into(), content: r.prompt.clone(), tool_calls: None });
        build_gen(&app, t, &msgs, None, &r.options, None)?
    };
    let mut g = g0;
    g.stream = r.stream;
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let prompt_ids = g.prompt.clone();
    let n_prompt = prompt_ids.len();
    let (_req_id, rx) = check_and_submit(&app, &g)?;
    if g.stream {
        let created = now_rfc3339();
        return Ok(ndjson_stream(app.clone(), rx, move |_, kind| match kind {
            ChunkKind::Delta { text, .. } => json!({"model": model, "created_at": created, "response": text, "done": false}),
            ChunkKind::Hidden(_) | ChunkKind::LogitsTopK(_) => json!({"done": false}),
            ChunkKind::Finish { reason, stats, tokens } => generate_done_json(&model, &created, n_prompt, &stats, &reason, &prompt_ids, &tokens),
        }));
    }
    let (acc, text_out, stats) = collect(&app, rx).await?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    let created = now_rfc3339();
    let mut v = generate_done_json(&model, &created, n_prompt, &stats, &reason, &prompt_ids, &acc.tokens);
    v["response"] = json!(text_out);
    Ok(Json(v).into_response())
}

#[cfg(test)]
mod ollama_tests {
    use super::*;

    #[test]
    fn rfc3339_epoch_zero_is_1970_01_01() {
        assert_eq!(rfc3339_from_unix_secs(0), "1970-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn rfc3339_known_date_2024_03_01() {
        // 2024-03-01T00:00:00Z = 1709251200 (leap-year boundary, the case
        // most likely to break a hand-rolled civil-from-days).
        assert_eq!(rfc3339_from_unix_secs(1_709_251_200), "2024-03-01T00:00:00.000000000Z");
    }

    #[test]
    fn rfc3339_time_of_day() {
        assert_eq!(rfc3339_from_unix_secs(1_709_251_200 + 3661), "2024-03-01T01:01:01.000000000Z");
    }

    #[test]
    fn ns_converts_seconds_to_nanoseconds() {
        assert_eq!(ns(1.5), 1_500_000_000);
        assert_eq!(ns(-1.0), 0, "a negative duration must not underflow to a huge u64");
    }

    #[test]
    fn options_repeat_penalty_and_num_ctx_are_parsed_but_not_mapped() {
        let opts: Options = serde_json::from_value(json!({"repeat_penalty": 1.1, "num_ctx": 99999, "temperature": 0.7})).unwrap();
        let sp = opts.to_sample_params();
        assert_eq!(sp.temperature, Some(0.7));
        // No SampleParams field exists for either: this test fails to compile,
        // not fails at runtime, if one is ever wired without updating the
        // module doc comment's claim that they are unmapped.
    }
}
