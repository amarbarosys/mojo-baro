//! `/api/embeddings` and `/v1/embeddings` (P0a gate 4, landed once P0a-e's
//! engine wire merged to main). One engine request per input string: `n:1`,
//! `spec:false`, `embed:true`. The vector is the engine's last-prompt-token
//! post-final-norm hidden state, already L2-normalized on the engine side
//! (`serve/PROTOCOL.md`); this layer does not normalize it again.

use super::*;

fn input_strings(r: &EmbeddingsReq) -> Result<Vec<String>, ApiError> {
    let v = r.input.clone().or_else(|| r.prompt.clone()).ok_or_else(|| bad("input is required"))?;
    match v {
        Value::String(s) => Ok(vec![s]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(s) => Ok(s),
                _ => Err(bad("input array must contain only strings (token-id embeddings are not supported by this item)")),
            })
            .collect(),
        _ => Err(bad("input must be a string or an array of strings")),
    }
}

/// Collects one embed request to its vector. `Embed` arrives before `Done`
/// (`serve/PROTOCOL.md`); a `Tok` line is expected too (the head runs
/// unfolded for this request) and carries nothing we need. An `Error` line
/// is the engine's documented refusal (spark, or `BARO_SEQS > 1`), mapped to
/// 501, not a generic failure code.
async fn collect_embed(mut rx: mpsc::UnboundedReceiver<Event>) -> Result<Vec<f32>, ApiError> {
    let mut vector: Option<Vec<f32>> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Tok { .. } => {}
            Event::Embed(v) => vector = Some(v),
            Event::Hidden(_) | Event::LogitsTopK(_) | Event::Latent { .. } | Event::Ipc(_) => {}
            Event::Done(_) => {
                return vector.ok_or_else(|| {
                    ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, "engine finished an embed:true request with no embed line".into())
                });
            }
            Event::Error(e) => {
                return Err(ApiError::Plain(StatusCode::NOT_IMPLEMENTED, format!("embeddings_pending: {e} (spark engine or BARO_SEQS > 1)")));
            }
        }
    }
    Err(ApiError::Plain(StatusCode::BAD_GATEWAY, "engine closed the connection before finishing the embed request".into()))
}

#[derive(Deserialize)]
pub struct EmbeddingsReq {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    /// Some clients (and the plan's own wording) call it `prompt`; accepted
    /// as an alias for `input` when `input` is absent.
    #[serde(default)]
    prompt: Option<Value>,
    /// OpenAI `encoding_format`: `float` (default) or `base64` (little-endian
    /// f32 bytes, standard alphabet with padding).
    #[serde(default)]
    encoding_format: Option<String>,
}

fn base64_f32(v: &[f32]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

pub async fn embeddings(State(app): State<Shared>, Json(r): Json<EmbeddingsReq>) -> Result<Json<Value>, ApiError> {
    let t = need_text(&app)?;
    let inputs = input_strings(&r)?;
    if inputs.is_empty() {
        return Err(bad("input must not be empty"));
    }
    let as_base64 = match r.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return Err(bad(format!("encoding_format must be float or base64, got {other}"))),
    };
    let model = r.model.clone().unwrap_or_else(|| app.model.clone());
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0usize;
    for (i, text) in inputs.iter().enumerate() {
        let prompt = t.encode(text, true).map_err(bad)?;
        if prompt.is_empty() {
            return Err(bad("input tokenizes to zero tokens"));
        }
        total_prompt_tokens += prompt.len();
        let g = Gen {
            prompt,
            n: 1,
            spec: false,
            stream: false,
            stop: vec![],
            ckpt: vec![],
            state: (None, None),
            state_prefix: false,
            base_pos: 0,
            embeds: vec![],
            sample: protocol::SampleParams::default(),
            schema: None,
            reasoning: None,
            embed: Some(true),
        };
        let (_req_id, rx) = check_and_submit(&app, &g)?;
        let vector = collect_embed(rx).await?;
        let embedding = if as_base64 { json!(base64_f32(&vector)) } else { json!(vector) };
        data.push(json!({"object": "embedding", "index": i, "embedding": embedding}));
    }
    Ok(Json(json!({
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": total_prompt_tokens, "total_tokens": total_prompt_tokens},
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ApiError` has no `Debug` impl (main.rs), so `.unwrap()` does not
    /// compile on a `Result<_, ApiError>`; this is the local substitute.
    fn ok<T>(r: Result<T, ApiError>) -> T {
        match r {
            Ok(v) => v,
            Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn base64_f32_matches_known_vectors() {
        assert_eq!(base64_f32(&[]), "");
        assert_eq!(base64_f32(&[1.0]), "AACAPw==");
        assert_eq!(base64_f32(&[1.0, -2.5]), "AACAPwAAIMA=");
        assert_eq!(base64_f32(&[0.0, 1.0, 2.0]), "AAAAAAAAgD8AAABA");
    }

    #[test]
    fn input_strings_accepts_a_single_string() {
        let r = EmbeddingsReq { model: None, input: Some(json!("hello")), prompt: None, encoding_format: None };
        assert_eq!(ok(input_strings(&r)), vec!["hello".to_string()]);
    }

    #[test]
    fn input_strings_accepts_an_array() {
        let r = EmbeddingsReq { model: None, input: Some(json!(["a", "b"])), prompt: None, encoding_format: None };
        assert_eq!(ok(input_strings(&r)), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn input_strings_falls_back_to_prompt() {
        let r = EmbeddingsReq { model: None, input: None, prompt: Some(json!("via prompt")), encoding_format: None };
        assert_eq!(ok(input_strings(&r)), vec!["via prompt".to_string()]);
    }

    #[test]
    fn input_strings_rejects_token_id_arrays() {
        let r = EmbeddingsReq { model: None, input: Some(json!([1, 2, 3])), prompt: None, encoding_format: None };
        assert!(input_strings(&r).is_err());
    }

    #[test]
    fn input_strings_requires_one_of_input_or_prompt() {
        let r = EmbeddingsReq { model: None, input: None, prompt: None, encoding_format: None };
        assert!(input_strings(&r).is_err());
    }

    #[tokio::test]
    async fn collect_embed_returns_the_vector_sent_before_done() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Tok { tok: 5, logprob: None, top_logprobs: vec![] }).unwrap();
        tx.send(Event::Embed(vec![0.1, 0.2, 0.3])).unwrap();
        tx.send(Event::Done(Box::new(protocol::DoneStats {
            n: 1,
            prefill_s: 0.0,
            decode_s: 0.0,
            tok_s: 0.0,
            drafted: None,
            accepted: None,
            cached: None,
            prefill_rows: None,
            restore_s: None,
            load_s: None,
            load_read_s: None,
            load_parse_s: None,
            load_pin_s: None,
            load_host_s: None,
            load_h2d_s: None,
            wall_s: None,
            finish: None,
            ..Default::default()
        })))
        .unwrap();
        drop(tx);
        assert_eq!(ok(collect_embed(rx).await), vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn collect_embed_maps_an_engine_refusal_to_501() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Error("embeddings not supported on this engine".into())).unwrap();
        drop(tx);
        match collect_embed(rx).await {
            Err(ApiError::Plain(code, msg)) => {
                assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
                assert!(msg.contains("embeddings_pending"), "{msg}");
            }
            Err(_) => panic!("expected ApiError::Plain"),
            Ok(_) => panic!("expected an error"),
        }
    }

    #[tokio::test]
    async fn collect_embed_errors_when_done_arrives_with_no_vector() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Event::Done(Box::new(protocol::DoneStats {
            n: 1,
            prefill_s: 0.0,
            decode_s: 0.0,
            tok_s: 0.0,
            drafted: None,
            accepted: None,
            cached: None,
            prefill_rows: None,
            restore_s: None,
            load_s: None,
            load_read_s: None,
            load_parse_s: None,
            load_pin_s: None,
            load_host_s: None,
            load_h2d_s: None,
            wall_s: None,
            finish: None,
            ..Default::default()
        })))
        .unwrap();
        drop(tx);
        assert!(collect_embed(rx).await.is_err());
    }
}
