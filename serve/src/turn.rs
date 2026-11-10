//! QS batch (AMDHQ 06 "QS batch", 2026-09-24): the agent-turn endpoint, the
//! checkpoint file pull/import pair, and `/metrics`.
//!
//! `POST /v1/turn` is a fork with the numbers an agent runtime needs: `ttft_s`
//! measured in the front (request in -> first generated token out) and the
//! `tier` the prefix came from (`hot`: the engine still held it, `load_s` under
//! 2 ms; `file`: the state file was read). `GET /v1/checkpoints/{id}/file`
//! serves the raw BAROST0x bytes with `X-Baro-Sha256`; `POST
//! /v1/checkpoints/import` takes a multipart of the sidecar JSON (`meta`) and
//! the file (`file`) and registers them, so a second host pulls a checkpoint
//! over HTTP instead of `nc` (the B4 rig's transport). `/metrics` is Prometheus
//! text: forks by tier, TTFT histogram, live checkpoints, bytes on disk.
use super::*;
use axum::extract::{Multipart, Path as UrlPath};
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// TTFT histogram upper bounds, milliseconds; the last bucket is +Inf.
pub const TTFT_BUCKETS_MS: [u64; 8] = [25, 50, 75, 100, 150, 250, 500, u64::MAX];

#[derive(Default)]
pub struct Metrics {
    pub forks: AtomicU64,
    pub forks_hot: AtomicU64,
    pub forks_file: AtomicU64,
    pub forks_unknown: AtomicU64,
    pub forks_ipc: AtomicU64,
    pub turns: AtomicU64,
    pub imports: AtomicU64,
    pub file_pulls: AtomicU64,
    pub ttft_sum_ms: AtomicU64,
    pub ttft_count: AtomicU64,
    pub ttft_buckets: [AtomicU64; 8],
}

impl Metrics {
    /// `(tier, tier_source)`: the engine's own `load_tier` word when the done line has one ("engine"); else the
    /// old guess from `load_s` under 2 ms ("inferred"); with neither, "unknown" -- a missing number is not "hot".
    pub fn record_fork(&self, stats: &Value) -> (&'static str, &'static str) {
        self.forks.fetch_add(1, Ordering::Relaxed);
        let (tier, source) = match stats.get("load_tier").and_then(Value::as_str) {
            Some("hot") => ("hot", "engine"),
            Some("file") => ("file", "engine"),
            Some("ipc") => ("ipc", "engine"),
            _ => match stats.get("load_s").and_then(Value::as_f64) {
                Some(l) if l < 0.002 => ("hot", "inferred"),
                Some(_) => ("file", "inferred"),
                None => ("unknown", "inferred"),
            },
        };
        match tier {
            "hot" => self.forks_hot.fetch_add(1, Ordering::Relaxed),
            "file" => self.forks_file.fetch_add(1, Ordering::Relaxed),
            "ipc" => self.forks_ipc.fetch_add(1, Ordering::Relaxed),
            _ => self.forks_unknown.fetch_add(1, Ordering::Relaxed),
        };
        (tier, source)
    }

    pub fn record_ttft(&self, ttft_s: f64) {
        let ms = (ttft_s * 1000.0).round().max(0.0) as u64;
        self.ttft_sum_ms.fetch_add(ms, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
        for (i, ub) in TTFT_BUCKETS_MS.iter().enumerate() {
            if ms <= *ub {
                self.ttft_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Deserialize)]
pub struct TurnReq {
    checkpoint: String,
    /// Text or token ids appended after the checkpointed prompt.
    suffix: Value,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(default)]
    model: Option<String>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

pub async fn turn(State(app): State<Shared>, Json(r): Json<TurnReq>) -> Result<Response, ApiError> {
    let t0 = Instant::now();
    // The lease keeps the sweep from deleting the state file before the engine has loaded it.
    let (c, _lease) = app.ckpts.lease(&r.checkpoint, now()).ok_or_else(|| checkpoints::not_found(&r.checkpoint))?;
    if c.identity.pack != app.identity.pack {
        return Err(ApiError::Mismatch("pack".into()));
    }
    let suffix = match &r.suffix {
        Value::String(s) => need_text(&app)?.encode(s, false).map_err(bad)?,
        other => prompt_ids(&app, other)?,
    };
    let mut prompt: Vec<u32> = c.tokens[c.pos..].to_vec();
    prompt.extend(suffix.iter().copied());
    let n = r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let n_prompt = c.pos + prompt.len();
    let g = Gen {
        prompt,
        n,
        spec: spec_default(&app, r.spec),
        stream: false,
        stop: compute_stop(&app, r.stop),
        ckpt: vec![],
        state: (None, Some(c.path.to_string_lossy().into_owned())),
        state_prefix: true,
        base_pos: c.pos,
        embeds: vec![],
        sample: r.sampler.to_sample_params(None),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let (req_id, rx) = check_and_submit(&app, &g)?;
    let (acc, text_out, stats) = collect(&app, rx).await.map_err(|e| checkpoints::expired_if_gone(e, &c))?;
    let ttft_s = acc.first_tok_at.map(|t| (t - t0).as_secs_f64());
    let (tier, tier_source) = app.metrics.record_fork(&stats);
    app.metrics.turns.fetch_add(1, Ordering::Relaxed);
    if let Some(t) = ttft_s {
        app.metrics.record_ttft(t);
    }
    let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0) as usize;
    let model = r.model.unwrap_or_else(|| app.model.clone());
    Ok(Json(json!({
        "object": "turn", "id": format!("turn-{req_id}"), "model": model, "checkpoint": c.id, "pos": c.pos,
        "text": text_out, "tokens": acc.tokens, "finish_reason": acc.finish_reason(stats.get("finish").and_then(Value::as_str)),
        "restored": cached >= c.pos, "cached_tokens": cached, "suffix_tokens": suffix.len(),
        "tier": tier, "tier_source": tier_source, "state_source": checkpoints::StateSource::Explicit.word(), "ttft_s": ttft_s, "wall_s": t0.elapsed().as_secs_f64(),
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats,
    })).into_response())
}

pub async fn file(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Result<Response, ApiError> {
    let c = app.ckpts.get(&id, now()).ok_or_else(|| checkpoints::not_found(&id))?;
    let bytes = tokio::fs::read(&c.path).await
        .map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", c.path.display())))?;
    let sha = checkpoints::hex(&checkpoints::sha256(&bytes));
    // The header carries the sidecar WITHOUT the token ids: at 32k they are 142 KB, above
    // curl's 100 KB header line cap (exit 100). The full sidecar is GET .../meta.
    let mut slim = checkpoints::DiskCkpt::from(&c);
    slim.tokens.clear();
    let meta = serde_json::to_string(&slim).unwrap_or_default();
    app.metrics.file_pulls.fetch_add(1, Ordering::Relaxed);
    Ok((
        [
            (axum::http::header::CONTENT_TYPE, "application/octet-stream".to_string()),
            (axum::http::HeaderName::from_static("x-baro-sha256"), sha),
            (axum::http::HeaderName::from_static("x-baro-checkpoint"), c.id.clone()),
            (axum::http::HeaderName::from_static("x-baro-meta"), meta),
        ],
        bytes,
    ).into_response())
}

/// `GET /v1/checkpoints/{id}/meta`: the sidecar JSON with the token ids (the multipart
/// `meta` part of an import, verbatim).
pub async fn meta(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Result<Json<checkpoints::DiskCkpt>, ApiError> {
    let c = app.ckpts.get(&id, now()).ok_or_else(|| checkpoints::not_found(&id))?;
    Ok(Json(checkpoints::DiskCkpt::from(&c)))
}

/// Multipart parts: `meta` (the sidecar JSON as `GET .../meta` returns it, or `.../file`'s
/// `X-Baro-Meta`), `file` (the raw state file). The identity in `meta` must
/// match this server's (409 field otherwise); the file's sha must match
/// `sha256` if that part is present.
pub async fn import(State(app): State<Shared>, mut mp: Multipart) -> Result<Response, ApiError> {
    let mut meta: Option<checkpoints::DiskCkpt> = None;
    let mut file: Option<Vec<u8>> = None;
    let mut sha_want: Option<String> = None;
    while let Some(field) = mp.next_field().await.map_err(|e| bad(format!("multipart: {e}")))? {
        let name = field.name().unwrap_or("").to_string();
        let data = field.bytes().await.map_err(|e| bad(format!("multipart {name}: {e}")))?;
        match name.as_str() {
            "meta" => meta = Some(serde_json::from_slice(&data).map_err(|e| bad(format!("meta: {e}")))?),
            "file" => file = Some(data.to_vec()),
            "sha256" => sha_want = Some(String::from_utf8_lossy(&data).trim().to_string()),
            _ => return Err(bad(format!("unknown part {name}"))),
        }
    }
    let (meta, file) = match (meta, file) {
        (Some(m), Some(f)) => (m, f),
        _ => return Err(bad("import needs the parts meta and file")),
    };
    if let Some(field) = app.identity.first_mismatch(&serde_json::to_value(&meta.identity).unwrap_or(Value::Null)) {
        return Err(ApiError::Mismatch(field));
    }
    let sha = checkpoints::hex(&checkpoints::sha256(&file));
    if let Some(w) = &sha_want {
        if w != &sha {
            return Err(bad(format!("sha256 mismatch: body {sha}, expected {w}")));
        }
    }
    if meta.tokens.len() < 2 || meta.pos + 1 != meta.tokens.len() || checkpoints::Registry::id_for(&meta.tokens) != meta.id {
        return Err(bad("meta does not describe its own tokens (id, pos)"));
    }
    if file.len() as u64 != meta.bytes {
        return Err(bad(format!("file is {} bytes, meta says {}", file.len(), meta.bytes)));
    }
    let t_now = now();
    if let Some(c) = app.ckpts.get(&meta.id, t_now) {
        return Ok(Json(json!({"created": false, "checkpoint": c.json(), "sha256": sha})).into_response());
    }
    std::fs::create_dir_all(&app.ckpts.dir).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", app.ckpts.dir.display())))?;
    let path = app.ckpts.path_for(&meta.id);
    let abs = std::fs::canonicalize(&app.ckpts.dir).unwrap_or_else(|_| app.ckpts.dir.clone()).join(path.file_name().unwrap_or_default());
    std::fs::write(&abs, &file).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", abs.display())))?;
    let ttl_s = meta.ttl_s.max(1);
    let c = checkpoints::Ckpt {
        id: meta.id.clone(), pos: meta.pos, tokens: meta.tokens.clone(), path: abs.clone(), bytes: meta.bytes,
        created: t_now, ttl_s, expires_at: t_now + ttl_s, identity: app.identity.clone(),
    };
    let cj = c.json();
    if let Err(err) = app.ckpts.insert(c, t_now) {
        let _ = std::fs::remove_file(&abs);
        return Err(checkpoints::insert_error(err, app.ckpts.cap));
    }
    app.metrics.imports.fetch_add(1, Ordering::Relaxed);
    Ok(Json(json!({"created": true, "checkpoint": cj, "sha256": sha})).into_response())
}

pub async fn metrics(State(app): State<Shared>) -> Response {
    let m = &app.metrics;
    let t_now = now();
    let live = app.ckpts.live(t_now);
    let bytes: u64 = live.iter().map(|c| c.bytes).sum();
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let mut s = String::new();
    s.push_str("# TYPE baro_checkpoint_forks_total counter\n");
    s.push_str(&format!("baro_checkpoint_forks_total{{tier=\"hot\"}} {}\n", g(&m.forks_hot)));
    s.push_str(&format!("baro_checkpoint_forks_total{{tier=\"file\"}} {}\n", g(&m.forks_file)));
    s.push_str(&format!("baro_checkpoint_forks_total{{tier=\"ipc\"}} {}\n", g(&m.forks_ipc)));
    s.push_str(&format!("baro_checkpoint_forks_total{{tier=\"unknown\"}} {}\n", g(&m.forks_unknown)));
    s.push_str(&format!("# TYPE baro_turns_total counter\nbaro_turns_total {}\n", g(&m.turns)));
    s.push_str(&format!("# TYPE baro_checkpoint_imports_total counter\nbaro_checkpoint_imports_total {}\n", g(&m.imports)));
    s.push_str(&format!("# TYPE baro_checkpoint_file_pulls_total counter\nbaro_checkpoint_file_pulls_total {}\n", g(&m.file_pulls)));
    s.push_str(&format!("# TYPE baro_checkpoints_live gauge\nbaro_checkpoints_live {}\n", live.len()));
    s.push_str(&format!("# TYPE baro_checkpoint_bytes gauge\nbaro_checkpoint_bytes {bytes}\n"));
    s.push_str("# TYPE baro_turn_ttft_seconds histogram\n");
    for (i, ub) in TTFT_BUCKETS_MS.iter().enumerate() {
        let le = if *ub == u64::MAX { "+Inf".to_string() } else { format!("{}", *ub as f64 / 1000.0) };
        s.push_str(&format!("baro_turn_ttft_seconds_bucket{{le=\"{le}\"}} {}\n", g(&m.ttft_buckets[i])));
    }
    s.push_str(&format!("baro_turn_ttft_seconds_sum {}\n", g(&m.ttft_sum_ms) as f64 / 1000.0));
    s.push_str(&format!("baro_turn_ttft_seconds_count {}\n", g(&m.ttft_count)));
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], s).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A-M15: a slow hot restore (over 2 ms) was reported as `file`, a fast file read as `hot`, and a missing
    /// `load_s` counted as hot. The engine's `load_tier` word decides now; the old guess is labelled inferred.
    #[test]
    fn the_engine_load_tier_beats_the_load_s_guess() {
        let m = Metrics::default();
        let got = |v: Value| format!("{:?}", m.record_fork(&v));
        assert!(got(json!({"load_tier": "file", "load_s": 0.0001})).contains("\"file\""), "engine file tier read as hot");
        assert!(got(json!({"load_tier": "hot", "load_s": 0.5})).contains("\"hot\""), "engine hot tier read as file");
        assert!(got(json!({"load_tier": "file"})).contains("\"engine\""));
        assert!(got(json!({"load_tier": "ipc", "load_s": 0.5})).contains("\"ipc\""), "the engine's ipc tier is its own word");
        assert!(got(json!({"load_s": 0.0001})).contains("\"inferred\""));
        assert!(got(json!({})).contains("\"unknown\""), "a missing load_s must not count as hot");
    }
}
