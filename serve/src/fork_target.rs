//! `/v1/fork` with `target` (P1 design item 2, `docs/P1-FORK-TARGET.md`):
//! export on this node, stream to the target node's `/v1/state/import`,
//! answer from the target. The target is a node ADDRESS (`host:port`) in
//! this lane; placement by node id is the router's job once P0b merges.
//!
//! The wire is `state.rs`'s own LAT1 stream, byte for byte, so the target's
//! CONTRACT 2 identity checks (the 409 path) judge a forked state exactly
//! as they judge any other import. No HTTP client dependency: one request
//! per connection with `Connection: close`, the same shape `audio.rs` uses
//! for its sidecar, because a fork moves one state and then one JSON body.

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const CHUNK_LEN: usize = 8 * 1024 * 1024;
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

fn gateway(msg: String) -> ApiError {
    ApiError::Plain(StatusCode::BAD_GATEWAY, msg)
}

/// `host:port`, with an optional `http://` prefix and trailing slash
/// stripped. Anything else is a 400: a fork never guesses a node.
pub fn parse_target(target: &str) -> Result<String, ApiError> {
    let t = target.strip_prefix("http://").unwrap_or(target).trim_end_matches('/');
    let ok = match t.rsplit_once(':') {
        Some((host, port)) => !host.is_empty() && !host.contains('/') && port.parse::<u16>().is_ok(),
        None => false,
    };
    if !ok {
        return Err(bad(format!("target must be a node address HOST:PORT, got {target:?}")));
    }
    Ok(t.to_string())
}

fn parse_response(resp: &[u8], what: &str) -> Result<(StatusCode, Vec<u8>), String> {
    let sep = resp.windows(4).position(|w| w == b"\r\n\r\n").ok_or(format!("{what}: malformed response ({} bytes, no header terminator)", resp.len()))?;
    let head = std::str::from_utf8(&resp[..sep]).map_err(|_| format!("{what}: non-utf8 headers"))?;
    let code = head.split_whitespace().nth(1).and_then(|s| s.parse::<u16>().ok()).ok_or(format!("{what}: bad status line"))?;
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return Err(format!("{what}: chunked response is not supported by the fork client"));
    }
    let status = StatusCode::from_u16(code).map_err(|_| format!("{what}: invalid status {code}"))?;
    Ok((status, resp[sep + 4..].to_vec()))
}

async fn connect(addr: &str) -> Result<TcpStream, String> {
    match tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr)).await {
        Ok(Ok(s)) => {
            let _ = s.set_nodelay(true);
            Ok(s)
        }
        Ok(Err(e)) => Err(format!("connect {addr}: {e}")),
        Err(_) => Err(format!("connect {addr}: no answer in {} s", CONNECT_TIMEOUT.as_secs())),
    }
}

/// POST the LAT1 header plus the scratch state file to the target's import
/// route, 8 MiB at a time (CONTRACT 1: no route buffers a whole 32k state).
async fn post_state(addr: &str, header: &[u8], path: &std::path::Path, payload_len: u64) -> Result<(StatusCode, Vec<u8>), String> {
    let mut stream = connect(addr).await?;
    let head = format!(
        "POST /v1/state/import HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/vnd.baro.state\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        header.len() as u64 + payload_len
    );
    stream.write_all(head.as_bytes()).await.map_err(|e| format!("write to {addr}: {e}"))?;
    stream.write_all(header).await.map_err(|e| format!("write to {addr}: {e}"))?;
    let mut f = tokio::fs::File::open(path).await.map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; CHUNK_LEN];
    loop {
        let n = f.read(&mut buf).await.map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        stream.write_all(&buf[..n]).await.map_err(|e| format!("write to {addr}: {e}"))?;
    }
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.map_err(|e| format!("read from {addr}: {e}"))?;
    parse_response(&resp, "target import")
}

pub(crate) async fn post_json(addr: &str, route: &str, body: &Value) -> Result<(StatusCode, Vec<u8>), String> {
    let mut stream = connect(addr).await?;
    let data = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    let mut req = format!(
        "POST {route} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        data.len()
    )
    .into_bytes();
    req.extend_from_slice(&data);
    stream.write_all(&req).await.map_err(|e| format!("write to {addr}: {e}"))?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.map_err(|e| format!("read from {addr}: {e}"))?;
    parse_response(&resp, "target fork")
}

pub(crate) fn relay(status: StatusCode, body: Vec<u8>) -> Response {
    (status, [(axum::http::header::CONTENT_TYPE, "application/json")], body).into_response()
}

/// `body` is the caller's `/v1/fork` request as received; it is forwarded to
/// the target with `target` removed and `prompt` replaced by this node's
/// token ids, so the target restores the prefix this node exported and
/// never re-tokenizes. A refusal by the target (the 409 identity path, a
/// 501, a 4xx) is relayed with the target's own status and body.
pub async fn fork_on_target(app: &Shared, target: &str, prompt: Vec<u32>, mut body: Value) -> Result<Response, ApiError> {
    let addr = parse_target(target)?;
    if prompt.len() < 2 {
        return Err(bad("a cross-node fork needs a prompt of at least 2 tokens (the state is exported at the prompt end minus one)"));
    }
    let t0 = Instant::now();
    let phase = |phase: &'static str| move |e: ApiError| match e {
        ApiError::Plain(code, msg) => ApiError::Phase(code, phase, msg),
        other => other,
    };
    let (path, header) = state::export_lat1_file(app, &state::ExportReq::for_tokens(prompt.clone())).await.map_err(phase("export"))?;
    let export_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let sent = post_state(&addr, &header.to_bytes(), &path, header.payload_len).await;
    let _ = std::fs::remove_file(&path);
    let (status, import_body) = sent.map_err(|e| ApiError::Phase(StatusCode::BAD_GATEWAY, "import", e))?;
    let import_s = t1.elapsed().as_secs_f64();
    if !status.is_success() {
        return Ok(relay(status, import_body));
    }
    let import: Value = serde_json::from_slice(&import_body).map_err(|e| gateway(format!("target import answered non-JSON: {e}")))?;

    if let Some(o) = body.as_object_mut() {
        o.remove("target");
        o.insert("prompt".into(), json!(prompt));
    }
    let t2 = Instant::now();
    let (status, fork_body) = post_json(&addr, "/v1/fork", &body).await.map_err(|e| ApiError::Phase(StatusCode::BAD_GATEWAY, "fork", e))?;
    let answer_s = t2.elapsed().as_secs_f64();
    if !status.is_success() {
        return Ok(relay(status, fork_body));
    }
    let mut answer: Value = serde_json::from_slice(&fork_body).map_err(|e| gateway(format!("target fork answered non-JSON: {e}")))?;
    if let Some(o) = answer.as_object_mut() {
        o.insert(
            "target".into(),
            json!({
                "node": addr,
                "pos": header.pos_hi,
                "prefix_hash": format!("{:016x}", header.prefix_hash),
                "format": if header.dtype == state::lat1::DTYPE_I8_BLOCK { "int8" } else { "f32" },
                "state_bytes": header.payload_len + state::lat1::HEADER_LEN as u64,
                "export_s": export_s,
                "import_s": import_s,
                "answer_s": answer_s,
                "import": import,
            }),
        );
    }
    Ok(Json(answer).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_target_accepts_an_address_and_strips_the_scheme() {
        assert_eq!(parse_target("10.99.7.2:8099").ok().as_deref(), Some("10.99.7.2:8099"));
        assert_eq!(parse_target("http://node-b:8099/").ok().as_deref(), Some("node-b:8099"));
    }

    #[test]
    fn parse_target_refuses_anything_that_is_not_host_port() {
        for t in ["", "node-b", "node-b:", ":8099", "node-b:99999", "http://node-b/path:1", "https://node-b:8099"] {
            assert!(parse_target(t).is_err(), "{t:?} should be refused");
        }
    }

    #[test]
    fn parse_response_splits_status_and_body() {
        let (s, b) = parse_response(b"HTTP/1.1 409 Conflict\r\ncontent-type: application/json\r\n\r\n{\"error\":\"state_identity\"}", "t").unwrap();
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(b, b"{\"error\":\"state_identity\"}");
    }

    #[test]
    fn parse_response_refuses_chunked_and_truncated() {
        assert!(parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", "t").is_err());
        assert!(parse_response(b"HTTP/1.1 200 OK\r\ncontent-le", "t").is_err());
    }

    /// One-connection mock node: reads a request with a Content-Length body,
    /// answers `status` with `reply`, and hands the request bytes back.
    async fn mock_node(status: &'static str, reply: &'static str) -> (String, tokio::task::JoinHandle<Vec<u8>>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let h = tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut got = Vec::new();
            let mut buf = vec![0u8; 1 << 16];
            loop {
                let n = s.read(&mut buf).await.unwrap();
                got.extend_from_slice(&buf[..n]);
                if let Some(sep) = got.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&got[..sep]).to_ascii_lowercase();
                    let len: usize = head.split("content-length: ").nth(1).unwrap().split("\r\n").next().unwrap().parse().unwrap();
                    if got.len() >= sep + 4 + len {
                        break;
                    }
                }
                assert!(n > 0, "peer closed before the body was complete");
            }
            let out = format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{reply}", reply.len());
            s.write_all(out.as_bytes()).await.unwrap();
            got
        });
        (addr, h)
    }

    #[tokio::test]
    async fn post_state_sends_header_then_file_bytes_with_the_exact_length() {
        let path = std::env::temp_dir().join(format!("baro-fork-target-{}.bin", std::process::id()));
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &payload).unwrap();
        let header = [7u8; 256];
        let (addr, h) = mock_node("200 OK", "{\"pos\":3}").await;
        let (status, body) = post_state(&addr, &header, &path, payload.len() as u64).await.unwrap();
        let got = h.await.unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"{\"pos\":3}");
        let sep = got.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&got[..sep]).to_string();
        assert!(head.starts_with("POST /v1/state/import HTTP/1.1"), "{head}");
        assert!(head.contains("Content-Type: application/vnd.baro.state"), "{head}");
        assert!(head.contains(&format!("Content-Length: {}", 256 + payload.len())), "{head}");
        assert_eq!(&got[sep + 4..sep + 4 + 256], &header[..]);
        assert_eq!(&got[sep + 4 + 256..], &payload[..]);
    }

    #[tokio::test]
    async fn a_refusal_by_the_target_keeps_its_status_and_body() {
        let path = std::env::temp_dir().join(format!("baro-fork-target-409-{}.bin", std::process::id()));
        std::fs::write(&path, b"x").unwrap();
        let reply = "{\"error\":\"state_identity\",\"field\":\"role_sha\"}";
        let (addr, h) = mock_node("409 Conflict", reply).await;
        let (status, body) = post_state(&addr, &[0u8; 256], &path, 1).await.unwrap();
        h.await.unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body, reply.as_bytes());
    }

    #[tokio::test]
    async fn an_unreachable_target_is_an_error_naming_the_address() {
        let e = connect("127.0.0.1:1").await.unwrap_err();
        assert!(e.contains("127.0.0.1:1"), "{e}");
    }
}
