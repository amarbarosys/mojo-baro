//! T3b (AMDHQ `docs/design/latent-os/06-experiments.md` `## T3b`): two engines on one GPU
//! share a checkpoint over HIP IPC, the front relays 320 bytes and the ack.
//!
//! `POST /v1/ipc/fork` on node A `{checkpoint, target, suffix, max_tokens, ...sampler}`:
//! engine A stages the checkpoint (`ipc_export`, a 256-byte LatentHeader + 64-byte handle),
//! A posts it with the prefix ids to B's `POST /v1/ipc/turn`, B's engine checks the header
//! against its own pack and those ids (refusal = 409, no HIP call), ingests and answers as a
//! fork; then A frees the stage (`ipc_release`), whatever B answered. `ttft_s` is A's request
//! arrival to B's first token, both read from the same host's wall clock.
//! `POST /v1/ipc/sha {pos, prefix_hash}` and `GET /v1/ipc/stats` are the gate's instruments.
//! `tamper` (`"pack"` | `"prefix_hash"`) flips one header byte before the relay: the 409 check.

use super::*;
use crate::protocol::{IpcFields, Request};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn unix_s() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// `submit_req` refuses a malformed line by name (a bug in the caller: 500); anything else is the engine not taking it (503).
fn submit_err(e: String) -> ApiError {
    ApiError::Plain(if e.starts_with("malformed request") { StatusCode::INTERNAL_SERVER_ERROR } else { StatusCode::SERVICE_UNAVAILABLE }, e)
}

fn gateway(msg: String) -> ApiError {
    ApiError::Plain(StatusCode::BAD_GATEWAY, msg)
}

/// One IPC op line to engine 0; its single `ipc_op` reply.
async fn op(app: &Shared, ipc: IpcFields) -> Result<Value, ApiError> {
    let (_, mut rx) = app.engine.submit_req(Request { n: 1, ipc, ..Default::default() })
        .map_err(submit_err)?;
    match rx.recv().await {
        Some(Event::Ipc(v)) => Ok(v),
        Some(Event::Error(e)) => Err(gateway(format!("engine: {e}"))),
        other => Err(gateway(format!("engine: unexpected answer to an IPC op: {other:?}"))),
    }
}

/// Little-endian u32 ids as hex, the engine's `ipc_prefix`.
pub fn ids_hex(ids: &[u32]) -> String {
    let mut s = String::with_capacity(ids.len() * 8);
    for t in ids {
        for b in t.to_le_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
    }
    s
}

/// Flip one byte of the 320-byte message (hex): role_sha[0] (offset 24) for "pack",
/// prefix_hash[0] (offset 176) for "prefix_hash".
pub fn tamper(msg: &str, what: &str) -> Result<String, ApiError> {
    let off = match what {
        "pack" => 24,
        "prefix_hash" => 176,
        _ => return Err(bad(format!("tamper must be \"pack\" or \"prefix_hash\", got {what:?}"))),
    };
    if msg.len() != 640 || !msg.is_ascii() {
        return Err(gateway(format!("engine export message is {} hex digits, not 640", msg.len())));
    }
    let b = u8::from_str_radix(&msg[2 * off..2 * off + 2], 16).map_err(|e| gateway(e.to_string()))? ^ 0x01;
    Ok(format!("{}{b:02x}{}", &msg[..2 * off], &msg[2 * off + 2..]))
}

/// Frees an engine-side stage when dropped without an explicit release (the handler future dropped by a client
/// reset between export and release): the stage holds engine resources until released.
struct StageGuard {
    app: Shared,
    stage: u64,
    released: bool,
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let (app, stage) = (self.app.clone(), self.stage);
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                let _ = op(&app, IpcFields { ipc_release: Some(stage), ..Default::default() }).await;
            });
        }
    }
}

pub async fn fork(State(app): State<Shared>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let t0 = Instant::now();
    let t0_unix = unix_s();
    let id = body.get("checkpoint").and_then(Value::as_str).ok_or_else(|| bad("checkpoint (id) is required"))?.to_string();
    let target = body.get("target").and_then(Value::as_str).ok_or_else(|| bad("target HOST:PORT is required"))?;
    let addr = fork_target::parse_target(target)?;
    let (c, _lease) = app.ckpts.lease(&id, now()).ok_or_else(|| checkpoints::not_found(&id))?;
    let suffix = match body.get("suffix") {
        Some(Value::String(s)) => need_text(&app)?.encode(s, false).map_err(bad)?,
        Some(other) => prompt_ids(&app, other)?,
        None => vec![],
    };
    let mut prompt: Vec<u32> = c.tokens[c.pos..].to_vec();
    prompt.extend(suffix.iter().copied());

    let ex = op(&app, IpcFields { ipc_export: Some(c.path.to_string_lossy().into_owned()), ..Default::default() }).await?;
    let stage = ex.get("ipc").and_then(Value::as_u64).ok_or_else(|| gateway(format!("engine export answered {ex}")))?;
    let export_s = t0.elapsed().as_secs_f64();
    let mut guard = StageGuard { app: app.clone(), stage, released: false };
    let relayed = async {
        let mut msg = ex.get("msg").and_then(Value::as_str).unwrap_or("").to_string();
        if let Some(t) = body.get("tamper").and_then(Value::as_str) {
            msg = tamper(&msg, t)?;
        }
        let mut fwd = body.clone();
        if let Some(o) = fwd.as_object_mut() {
            for k in ["checkpoint", "target", "suffix", "tamper"] {
                o.remove(k);
            }
            o.insert("msg".into(), json!(msg));
            o.insert("prefix".into(), json!(ids_hex(&c.tokens[..c.pos])));
            o.insert("prompt".into(), json!(prompt));
        }
        fork_target::post_json(&addr, "/v1/ipc/turn", &fwd).await.map_err(gateway)
    }
    .await;
    // The ack: B has answered (or failed), so its ingest is over; A frees the stage either way.
    // A failed release must not eat the relay's result: it is reported next to the answer.
    let rel = op(&app, IpcFields { ipc_release: Some(stage), ..Default::default() }).await;
    guard.released = true;
    let (status, b_body) = relayed?;
    if !status.is_success() {
        return Ok(fork_target::relay(status, b_body));
    }
    let mut answer: Value = serde_json::from_slice(&b_body).map_err(|e| gateway(format!("target answered non-JSON: {e}")))?;
    let first = answer.get("first_tok_unix").and_then(Value::as_f64);
    if let Some(o) = answer.as_object_mut() {
        o.insert("checkpoint".into(), json!(c.id));
        o.insert("pos".into(), json!(c.pos));
        o.insert("tier".into(), json!("ipc"));
        o.insert("state_source".into(), json!("explicit"));
        o.insert("ttft_s".into(), json!(first.map(|f| f - t0_unix)));
        o.insert("wall_s".into(), json!(t0.elapsed().as_secs_f64()));
        o.insert("ipc".into(), json!({"node": addr, "stage": stage, "bytes": ex.get("bytes"), "prefix_hash": ex.get("prefix_hash"),
            "export_s": export_s, "engine_export_s": ex.get("export_s"), "live_after_release": rel.as_ref().ok().and_then(|r| r.get("live")),
            "release_error": rel.as_ref().err().map(|_| "the stage release failed; ipc/stats shows the leak")}));
    }
    Ok(Json(answer).into_response())
}

#[derive(Deserialize)]
pub struct IpcTurnReq {
    msg: String,
    prefix: String,
    prompt: Vec<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

pub async fn turn(State(app): State<Shared>, Json(r): Json<IpcTurnReq>) -> Result<Response, ApiError> {
    let t0 = Instant::now();
    let n_prompt = r.prefix.len() / 8 + r.prompt.len();
    let n = fit_n(r.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS), n_prompt, app.engine.limits.tmax as u64);
    if r.prompt.is_empty() || n == 0 {
        return Err(bad("an IPC turn needs a non-empty prompt and max_tokens >= 1"));
    }
    if n_prompt as u64 + n as u64 > app.engine.limits.tmax as u64 {
        return Err(ApiError::exceed_context(n_prompt as u64, app.engine.limits.tmax as u64));
    }
    let req = Request {
        prompt: r.prompt,
        n,
        spec: spec_default(&app, r.spec),
        stop: compute_stop(&app, r.stop),
        sample: r.sampler.to_sample_params(None),
        ipc: IpcFields { ipc_ingest: Some(r.msg), ipc_prefix: Some(r.prefix), ..Default::default() },
        ..Default::default()
    };
    let (req_id, rx) = app.engine.submit_req(req).map_err(submit_err)?;
    let (acc, text_out, stats) = collect(&app, rx).await.map_err(|e| match e {
        ApiError::Plain(_, m) if m.contains("IPC refused") => ApiError::Plain(StatusCode::CONFLICT, m),
        e => e,
    })?;
    let first_tok_unix = acc.first_tok_at.map(|t| unix_s() - t.elapsed().as_secs_f64());
    let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0);
    Ok(Json(json!({
        "object": "ipc.turn", "id": format!("ipc-{req_id}"), "text": text_out, "tokens": acc.tokens,
        "finish_reason": acc.finish_reason(stats.get("finish").and_then(Value::as_str)),
        "cached_tokens": cached, "ingest_s": stats.get("load_s"),
        "b_ttft_s": acc.first_tok_at.map(|t| (t - t0).as_secs_f64()), "first_tok_unix": first_tok_unix,
        "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats,
    })).into_response())
}

#[derive(Deserialize)]
pub struct ShaReq {
    pos: u64,
    /// Empty matches the one chain item at `pos` (A before its first export).
    #[serde(default)]
    prefix_hash: String,
}

pub async fn sha(State(app): State<Shared>, Json(r): Json<ShaReq>) -> Result<Json<Value>, ApiError> {
    Ok(Json(op(&app, IpcFields { ipc_sha: Some(r.pos), ipc_ph: Some(r.prefix_hash), ..Default::default() }).await?))
}

pub async fn stats(State(app): State<Shared>) -> Result<Json<Value>, ApiError> {
    Ok(Json(op(&app, IpcFields { ipc_stats: Some(true), ..Default::default() }).await?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_hex_is_little_endian_u32() {
        assert_eq!(ids_hex(&[1, 0x0403_0201, 248044]), "0100000001020304ecc80300");
    }

    #[test]
    fn tamper_flips_exactly_one_header_byte() {
        let msg = "0".repeat(640);
        let p = tamper(&msg, "pack").ok().unwrap();
        let h = tamper(&msg, "prefix_hash").ok().unwrap();
        assert_eq!(&p[48..50], "01");
        assert_eq!(&h[352..354], "01");
        assert_eq!(p.chars().filter(|&c| c != '0').count(), 1);
        assert_eq!(h.chars().filter(|&c| c != '0').count(), 1);
        assert!(tamper(&msg, "other").is_err());
        assert!(tamper("00", "pack").is_err());
    }

    #[test]
    fn an_ipc_op_reply_is_its_own_message() {
        match crate::protocol::parse_line("{\"id\":3,\"ipc_op\":\"release\",\"ipc\":1,\"live\":0}\n") {
            crate::protocol::EngineMsg::Ipc { id, value } => {
                assert_eq!(id, 3);
                assert_eq!(value["live"], 0);
            }
            _ => panic!("ipc_op line not parsed as Ipc"),
        }
    }

    #[test]
    fn the_ingest_fields_serialise_onto_the_request_line() {
        let r = Request { id: 9, prompt: vec![5], n: 2, ipc: IpcFields { ipc_ingest: Some("ab".into()), ipc_prefix: Some("cd".into()), ..Default::default() }, ..Default::default() };
        let line = r.line();
        assert!(line.contains("\"ipc_ingest\":\"ab\"") && line.contains("\"ipc_prefix\":\"cd\""), "{line}");
        assert!(!line.contains("ipc_export") && !line.contains("ipc_release"), "{line}");
    }

    /// A-M8: a handler future dropped between export and release (client reset) left the engine-side stage
    /// held for good; the guard sends the release.
    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_stage_guard_sends_the_release() {
        let stub = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stub_engine.sh");
        let log = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.work/fix-front-test/a-m8.log");
        std::fs::create_dir_all(log.parent().unwrap()).unwrap();
        let _ = std::fs::remove_file(&log);
        let env = vec![("STUB_MODE".into(), "ok".into()), ("STUB_LOG".into(), log.display().to_string())];
        let pool = crate::engine::EnginePool::spawn(&stub, std::path::Path::new("none"), 1, &env).await.expect("stub pool");
        let mut app = Arc::try_unwrap(crate::sse_tests::test_app()).ok().expect("fresh app");
        app.engine = pool;
        let app: Shared = Arc::new(app);
        drop(StageGuard { app: app.clone(), stage: 5, released: false });
        for _ in 0..100 {
            if std::fs::read_to_string(&log).unwrap_or_default().contains("\"ipc_release\":5") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(std::fs::read_to_string(&log).unwrap_or_default().contains("\"ipc_release\":5"), "no release line reached the engine");
        // An explicit release disarms it.
        let before = std::fs::read_to_string(&log).unwrap_or_default().matches("ipc_release").count();
        drop(StageGuard { app: app.clone(), stage: 6, released: true });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(std::fs::read_to_string(&log).unwrap_or_default().matches("ipc_release").count(), before);
        app.engine.shutdown().await;
    }

    /// A-M8: a cross-node fork failure names the phase it reached.
    #[tokio::test(flavor = "current_thread")]
    async fn a_phase_error_body_names_the_phase() {
        let resp = ApiError::Phase(StatusCode::BAD_GATEWAY, "import", "target down".into()).into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!((v["error"]["phase"].as_str(), v["error"]["type"].as_str()), (Some("import"), Some("fork_failed")));
    }
}
