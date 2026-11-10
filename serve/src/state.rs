//! P1 CONTRACT 3 (`docs/P1-STATE-API.md`): `GET /v1/state`, `POST
//! /v1/state/export`, `POST /v1/state/import`. A thin adapter over
//! `checkpoints.rs`'s existing engine-submission plumbing (`Registry`,
//! `Ckpt`, `Gen`/`check_and_submit`) plus a LAT1 wrap/unwrap, not a parallel
//! reimplementation: `create()` already runs the prefill and `state_save`
//! export needs, `fork()` already runs the `state_load` restore import
//! needs (room A, 2026-09-17 design review with codex).
//!
//! Build order: `GET /v1/state` (read-only, no GPU), then `POST
//! /v1/state/export` (`path` variant), both landed. This commit adds `POST
//! /v1/state/import`, also the `path` variant: the caller names an existing
//! raw `BAROST01`/`BAROST02` file (as `export`'s `path` form writes), the
//! response is JSON metadata. The LAT1-wrapped streaming body/response
//! CONTRACT 1 also describes, for both routes, is a following commit.
//!
//! Gaps found while building this, handled honestly rather than silently
//! (room A, 2026-09-17): CONTRACT 3's per-request `n = 0` prefill does not
//! exist (`check_and_submit` rejects `n == 0`); this uses `n = 1`, the same
//! working pattern `checkpoints::create` already ships. The per-request
//! `"format"` field cannot be honored: `serve/engine.mojo:86` reads
//! `BARO_STATE_INT8` once at server startup, so export 409s a mismatched
//! request instead of writing a silently wrong file. And `runtime_differs`
//! in import's response is always `null`, not `false`: a raw BAROST0x file
//! (`serve/engine.mojo`'s `save_state`) carries no `runtime` field at all
//! (magic, pos, CONV_SLOT, SSM_SLOT, kvn, salt, tokens -- that's the whole
//! 72-byte header plus the token array), only the LAT1 wrapper does, so
//! there is nothing to compare yet; `false` would claim a check that never
//! happened.

use super::*;
use crate::checkpoints::{sha256, Sha256};
use axum::body::{to_bytes, Body, Bytes};
use axum::http::{header::CONTENT_TYPE, HeaderValue};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_stream::wrappers::ReceiverStream;

struct HmacSha256 {
    inner: Sha256,
    outer: Sha256,
}

impl HmacSha256 {
    fn new(key: &[u8]) -> Self {
        let mut block = [0u8; 64];
        let key = if key.len() > 64 { sha256(key).to_vec() } else { key.to_vec() };
        block[..key.len()].copy_from_slice(&key);
        let mut inner = Sha256::new();
        let mut outer = Sha256::new();
        let mut pad = block;
        pad.iter_mut().for_each(|b| *b ^= 0x36);
        inner.update(&pad);
        pad.iter_mut().for_each(|b| *b ^= 0x36 ^ 0x5c);
        outer.update(&pad);
        Self { inner, outer }
    }

    fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    fn finalize(mut self) -> [u8; 32] {
        let inner = self.inner.finalize();
        self.outer.update(&inner);
        self.outer.finalize()
    }
}

/// CONTRACT 1: the LAT1 256-byte container header, byte-for-byte matching
/// `latentos/proto.mojo::LatentHeader` (upstream `~/AMDHQ/src/latentos`,
/// vendored read-only in this repo -- offsets copied from that file's own
/// doc comment independently, not derived from it at runtime, since Mojo
/// and Rust share no struct-layout mechanism). This is the HTTP layer's
/// own reader/writer: Mojo never touches LAT1, only the raw BAROST0x body
/// it wraps (`serve/engine.mojo`'s `save_state`/`load_state`).
pub mod lat1 {
    pub const MAGIC: u32 = 0x3154_414C; // 'LAT1'
    pub const VERSION: u16 = 1;
    pub const HEADER_LEN: usize = 256;
    pub const KIND_KV_PAGES: u8 = 1;
    pub const DTYPE_F32: u8 = 1;
    /// New for P1 (CONTRACT 1): not in `latentos/proto.mojo`'s DTYPE_F32/
    /// DTYPE_BF16 pair -- the HTTP layer's own extension for a BAROST02
    /// int8 body, which Mojo never sees as a "dtype" at all (only as the
    /// BAROST01/BAROST02 magic bytes).
    pub const DTYPE_I8_BLOCK: u8 = 3;

    #[derive(Debug, Clone)]
    pub struct Header {
        pub magic: u32,
        pub version: u16,
        pub kind: u8,
        pub dtype: u8,
        pub weights_uuid: [u8; 16],
        pub role_sha: [u8; 32],
        pub runtime: [u8; 32],
        pub sigma_id: [u8; 32],
        pub tokenizer_sha: [u8; 32],
        pub pos_hi: u32,
        pub ttl_s: u32,
        pub prefix_hash: u64,
        pub payload_len: u64,
        pub payload_sha: [u8; 32],
        pub hmac: [u8; 32],
    }

    impl Header {
        /// Every field this repo does not set (layer_lo/hi, pos_lo, rope_id,
        /// batch_class/m) stays zero, matching
        /// `LatentHeader.__init__`'s own defaults.
        pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
            let mut b = [0u8; HEADER_LEN];
            b[0..4].copy_from_slice(&self.magic.to_le_bytes());
            b[4..6].copy_from_slice(&self.version.to_le_bytes());
            b[6] = self.kind;
            b[7] = self.dtype;
            b[8..24].copy_from_slice(&self.weights_uuid);
            b[24..56].copy_from_slice(&self.role_sha);
            b[56..88].copy_from_slice(&self.runtime);
            b[88..120].copy_from_slice(&self.sigma_id);
            b[120..152].copy_from_slice(&self.tokenizer_sha);
            b[160..164].copy_from_slice(&self.pos_hi.to_le_bytes());
            b[172..176].copy_from_slice(&self.ttl_s.to_le_bytes());
            b[176..184].copy_from_slice(&self.prefix_hash.to_le_bytes());
            b[184..192].copy_from_slice(&self.payload_len.to_le_bytes());
            b[192..224].copy_from_slice(&self.payload_sha);
            b[224..256].copy_from_slice(&self.hmac);
            b
        }

        pub fn from_bytes(b: &[u8; HEADER_LEN]) -> Header {
            let mut weights_uuid = [0u8; 16];
            weights_uuid.copy_from_slice(&b[8..24]);
            let mut role_sha = [0u8; 32];
            role_sha.copy_from_slice(&b[24..56]);
            let mut runtime = [0u8; 32];
            runtime.copy_from_slice(&b[56..88]);
            let mut sigma_id = [0u8; 32];
            sigma_id.copy_from_slice(&b[88..120]);
            let mut tokenizer_sha = [0u8; 32];
            tokenizer_sha.copy_from_slice(&b[120..152]);
            let mut payload_sha = [0u8; 32];
            payload_sha.copy_from_slice(&b[192..224]);
            let mut hmac = [0u8; 32];
            hmac.copy_from_slice(&b[224..256]);
            Header {
                magic: u32::from_le_bytes(b[0..4].try_into().unwrap()),
                version: u16::from_le_bytes(b[4..6].try_into().unwrap()),
                kind: b[6],
                dtype: b[7],
                weights_uuid,
                role_sha,
                runtime,
                sigma_id,
                tokenizer_sha,
                pos_hi: u32::from_le_bytes(b[160..164].try_into().unwrap()),
                ttl_s: u32::from_le_bytes(b[172..176].try_into().unwrap()),
                prefix_hash: u64::from_le_bytes(b[176..184].try_into().unwrap()),
                payload_len: u64::from_le_bytes(b[184..192].try_into().unwrap()),
                payload_sha,
                hmac,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn to_bytes_from_bytes_roundtrips() {
            let h = Header {
                magic: MAGIC,
                version: VERSION,
                kind: KIND_KV_PAGES,
                dtype: DTYPE_F32,
                weights_uuid: [1u8; 16],
                role_sha: [2u8; 32],
                runtime: [3u8; 32],
                sigma_id: [0u8; 32],
                tokenizer_sha: [4u8; 32],
                pos_hi: 4096,
                ttl_s: 600,
                prefix_hash: 0xa3762c148f44e787,
                payload_len: 61079640,
                payload_sha: [5u8; 32],
                hmac: [6u8; 32],
            };
            let bytes = h.to_bytes();
            assert_eq!(bytes.len(), HEADER_LEN);
            let back = Header::from_bytes(&bytes);
            assert_eq!(back.magic, h.magic);
            assert_eq!(back.version, h.version);
            assert_eq!(back.kind, h.kind);
            assert_eq!(back.dtype, h.dtype);
            assert_eq!(back.weights_uuid, h.weights_uuid);
            assert_eq!(back.role_sha, h.role_sha);
            assert_eq!(back.runtime, h.runtime);
            assert_eq!(back.tokenizer_sha, h.tokenizer_sha);
            assert_eq!(back.pos_hi, h.pos_hi);
            assert_eq!(back.ttl_s, h.ttl_s);
            assert_eq!(back.prefix_hash, h.prefix_hash);
            assert_eq!(back.payload_len, h.payload_len);
            assert_eq!(back.payload_sha, h.payload_sha);
            assert_eq!(back.hmac, h.hmac);
        }

        #[test]
        fn magic_is_the_ascii_bytes_lat1_little_endian() {
            let h = Header {
                magic: MAGIC,
                version: VERSION,
                kind: KIND_KV_PAGES,
                dtype: DTYPE_F32,
                weights_uuid: [0u8; 16],
                role_sha: [0u8; 32],
                runtime: [0u8; 32],
                sigma_id: [0u8; 32],
                tokenizer_sha: [0u8; 32],
                pos_hi: 0,
                ttl_s: 0,
                prefix_hash: 0,
                payload_len: 0,
                payload_sha: [0u8; 32],
                hmac: [0u8; 32],
            };
            assert_eq!(&h.to_bytes()[0..4], b"LAT1");
        }
    }
}

/// CONTRACT 1: "Streams are written and read in 8 MiB chunks; no route
/// buffers a whole 32k state in the HTTP layer."
const CHUNK_LEN: usize = 8 * 1024 * 1024;

fn hex32_encode(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn state_hmac_key() -> Option<Vec<u8>> {
    std::env::var("BARO_STATE_HMAC_KEY").ok().filter(|key| !key.is_empty()).map(String::into_bytes)
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

async fn hmac_file(path: &std::path::Path, header: &lat1::Header, key: &[u8]) -> Result<[u8; 32], ApiError> {
    let mut hmac = HmacSha256::new(key);
    let mut bytes = header.to_bytes();
    bytes[224..].fill(0);
    hmac.update(&bytes);
    let mut f = tokio::fs::File::open(path).await.map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", path.display())))?;
    let mut buf = vec![0u8; CHUNK_LEN];
    loop {
        let n = f.read(&mut buf).await.map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", path.display())))?;
        if n == 0 { break; }
        hmac.update(&buf[..n]);
    }
    Ok(hmac.finalize())
}

/// The inverse of `checkpoints::Identity`'s own hex encoding: `app.identity.pack`
/// and `.tokenizer_sha` are always 64 lowercase hex chars (produced by the same
/// `hex(&sha256(...))` idiom throughout this crate), so this only ever returns
/// `None` for a value that was never one of ours to begin with.
fn hex32_decode(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn identity_mismatch(field: &str, ours: impl Into<String>, theirs: impl Into<String>) -> ApiError {
    ApiError::StateIdentity { field: field.into(), ours: ours.into(), theirs: theirs.into() }
}

/// A scratch state file that is removed when this guard drops (an early return, an error, a handler future
/// dropped by a client reset). `into_path` hands ownership to whoever streams or relays it next.
pub struct Scratch(Option<std::path::PathBuf>);

impl Scratch {
    pub fn new(path: std::path::PathBuf) -> Scratch {
        Scratch(Some(path))
    }

    pub fn path(&self) -> &std::path::Path {
        self.0.as_deref().unwrap_or(std::path::Path::new(""))
    }

    pub fn into_path(mut self) -> std::path::PathBuf {
        self.0.take().unwrap_or_default()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn scratch_path(app: &App, prefix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    app.ckpts.dir.join(format!("{prefix}-{}-{nanos}.baro", std::process::id()))
}

async fn hash_file(path: &std::path::Path) -> Result<[u8; 32], ApiError> {
    let mut f = tokio::fs::File::open(path).await.map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_LEN];
    loop {
        let n = f.read(&mut buf).await.map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

async fn stream_file_chunks(path: std::path::PathBuf, tx: mpsc::Sender<Result<Bytes, std::io::Error>>) {
    let result = async {
        let mut f = tokio::fs::File::open(&path).await?;
        let mut buf = vec![0u8; CHUNK_LEN];
        loop {
            let n = f.read(&mut buf).await?;
            if n == 0 {
                return Ok(());
            }
            if tx.send(Ok(Bytes::copy_from_slice(&buf[..n]))).await.is_err() {
                return Ok(()); // client went away; nothing left to report
            }
        }
    }
    .await;
    if let Err(e) = result {
        let _: Result<_, _> = tx.send(Err(e)).await;
    }
    let _ = std::fs::remove_file(&path);
}

/// CONTRACT 1: the unsalted `prefix_hash` field, `sha256(tokens[0:pos]` as
/// little-endian u32 bytes `)`, first 8 digest bytes read back as a
/// little-endian u64. Deliberately not `checkpoints::Registry::id_for`:
/// that hashes the WHOLE `tokens` slice (an off-by-one against `pos`, since
/// `pos` there is only a label in the id string) and formats a hex string
/// for the HTTP-facing checkpoint id, not this binary LAT1 header field.
pub fn exact_prefix_hash(tokens: &[u32], pos: usize) -> u64 {
    let mut bytes = Vec::with_capacity(pos * 4);
    for &t in &tokens[..pos] {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    let digest = sha256(&bytes);
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}


/// `GET /v1/state` (CONTRACT 3): no GPU work, reads straight from the
/// existing HTTP-facing `checkpoints::Registry` (room A design review: this
/// is "the chain" CONTRACT 3 means, not the Mojo engine's internal prefix
/// `Chain`, a separate, GPU-resident structure this Rust layer cannot see
/// into). `pinned`/`boundary` are always `false`: the Registry is a flat
/// TTL-and-cap cache with no priority-retention concept, unlike the Mojo
/// engine's own checkpoint chain (`serve/prefix.mojo`) -- reporting them as
/// `false` here is honest about that, not a stand-in for real values.
pub async fn list(State(app): State<Shared>) -> Json<Value> {
    let states: Vec<Value> = app
        .ckpts
        .live(now())
        .into_iter()
        .map(|c| {
            json!({
                "prefix_hash": format!("{:016x}", exact_prefix_hash(&c.tokens, c.pos)),
                "pos": c.pos,
                "bytes": c.bytes,
                "pinned": false,
                "boundary": false,
                "age_s": now().saturating_sub(c.created),
            })
        })
        .collect();
    Json(json!({
        "model": app.model,
        "weights_uuid": app.identity.weights_uuid_hex(),
        "portable": app.identity.portable,
        "kv": app.engine.limits.kv,
        "states": states,
    }))
}

/// State files default to BAROST02 (int8). Explicit `BARO_STATE_INT8=0`
/// selects BAROST01 (f32); no per-request override exists on the wire.
fn state_file_format() -> &'static str {
    if std::env::var("BARO_STATE_INT8").map(|v| v != "0").unwrap_or(true) { "int8" } else { "f32" }
}

/// CONTRACT 3: an engine running `BARO_KVQ=int8` refuses `state_save`/
/// `state_load` with its own wire error (`serve/engine.mojo:68,157`); map
/// that specific, known refusal to the documented 501 `kvq_state_open`
/// instead of `collect`'s generic 400 for an unrecognized engine error.
fn map_kvq_refusal(e: ApiError) -> ApiError {
    match &e {
        ApiError::Plain(_, msg) if msg.contains("quantized KV state is A2 step 3") => ApiError::Plain(
            StatusCode::NOT_IMPLEMENTED,
            "kvq_state_open: this engine's KV is quantized (BARO_KVQ), export/import need f32 KV (A2 step 3)".into(),
        ),
        _ => e,
    }
}

#[derive(Deserialize)]
struct ExportMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Deserialize)]
pub struct ExportReq {
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    messages: Option<Vec<ExportMessage>>,
    #[serde(default)]
    tokens: Option<Vec<u32>>,
    #[serde(default)]
    pos: Option<usize>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

/// `(prompt token ids, role-boundary positions)`; boundaries are only ever
/// non-empty for a `messages` request (`Text::role_boundaries`), matching
/// CONTRACT 3's default-`pos` rule.
fn export_prompt(app: &App, r: &ExportReq) -> Result<(Vec<u32>, Vec<u32>), ApiError> {
    if let Some(msgs) = &r.messages {
        let t = need_text(app)?;
        if msgs.is_empty() {
            return Err(bad("messages is empty"));
        }
        let chat_msgs: Vec<ChatMessage> = msgs
            .iter()
            .map(|m| Ok(ChatMessage { role: m.role.clone(), content: content_text(&m.content)?, tool_calls: None }))
            .collect::<Result<Vec<_>, ApiError>>()?;
        let rendered = t.apply_chat_template(&chat_msgs, None, None).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let prompt = t.encode(&rendered, true).map_err(bad)?;
        let boundaries = t.role_boundaries(&chat_msgs, None, None);
        return Ok((prompt, boundaries));
    }
    if let Some(tokens) = &r.tokens {
        if tokens.is_empty() {
            return Err(bad("tokens is empty"));
        }
        return Ok((tokens.clone(), vec![]));
    }
    if let Some(p) = &r.prompt {
        return Ok((prompt_ids(app, p)?, vec![]));
    }
    Err(bad("one of prompt, messages, or tokens is required"))
}

/// CONTRACT 3: "pos defaults to the last role boundary at or before the
/// prompt end, else the prompt end minus one."
fn default_pos(prompt_len: usize, boundaries: &[u32]) -> usize {
    boundaries
        .iter()
        .copied()
        .filter(|&b| (b as usize) <= prompt_len)
        .max()
        .map(|b| b as usize)
        .unwrap_or_else(|| prompt_len.saturating_sub(1))
}

/// `POST /v1/state/export`: dispatches on `path` (CONTRACT 3's request shape
/// is identical either way -- only the response differs, JSON metadata with
/// `path`, the raw LAT1 stream without it).
pub async fn export(State(app): State<Shared>, Json(r): Json<ExportReq>) -> Result<Response, ApiError> {
    match r.path.clone() {
        Some(path) => export_to_path(&app, &r, path).await.map(|v| Json(v).into_response()),
        None => export_stream(&app, &r).await,
    }
}

/// Common prefill: resolves the request's prompt/pos and the format 409,
/// then runs the `n=1, state_save` request `checkpoints::create` already
/// proves works, writing to `path`.
async fn export_prefill(app: &Shared, r: &ExportReq, path: &str) -> Result<(Vec<u32>, usize), ApiError> {
    let server_format = state_file_format();
    if let Some(requested) = &r.format {
        if requested != server_format {
            return Err(ApiError::Plain(
                StatusCode::CONFLICT,
                format!("this server writes state files as {server_format} (BARO_STATE_INT8 at startup), it cannot honor format={requested} per request"),
            ));
        }
    }
    let (prompt, boundaries) = export_prompt(app, r)?;
    if prompt.is_empty() {
        return Err(bad("prompt resolves to zero tokens"));
    }
    let pos = r.pos.unwrap_or_else(|| default_pos(prompt.len(), &boundaries));
    if pos == 0 || pos > prompt.len() {
        return Err(bad(format!("pos must be in 1..={} (the prompt length), got {pos}", prompt.len())));
    }
    let g = Gen {
        prompt: prompt.clone(),
        n: 1,
        spec: false,
        stream: false,
        stop: vec![],
        ckpt: vec![],
        state: (Some(path.to_string()), None),
        state_prefix: false,
        base_pos: 0,
        embeds: vec![],
        sample: protocol::SampleParams::default(),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let (_req_id, rx) = check_and_submit(app, &g)?;
    collect(app, rx).await.map_err(map_kvq_refusal)?;
    Ok((prompt, pos))
}

/// `path` variant: `{"path","bytes","pos","prefix_hash"}`, writing to the
/// caller's own path rather than the Registry's managed one -- this is a
/// one-off dump, not a tracked, TTL'd checkpoint.
async fn export_to_path(app: &Shared, r: &ExportReq, path: String) -> Result<Value, ApiError> {
    let (prompt, pos) = export_prefill(app, r, &path).await?;
    let bytes = std::fs::metadata(&path)
        .map(|m| m.len())
        .map_err(|e| ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine finished but wrote no state file at {path}: {e}")))?;
    Ok(json!({
        "path": path,
        "bytes": bytes,
        "pos": pos,
        "prefix_hash": format!("{:016x}", exact_prefix_hash(&prompt, pos)),
    }))
}

/// CONTRACT 1: the raw LAT1 stream. Writes to an internal scratch path
/// (never the caller's filesystem), hashes the resulting BAROST0x file in
/// one streamed pass to fill the header's `payload_sha`/`payload_len`
/// (the header, the stream's first 256 bytes, must carry the hash of bytes
/// that come after it -- so this is necessarily two passes over the file,
/// each holding only one `CHUNK_LEN` buffer at a time, never the whole
/// file), then streams header + file in a second pass. The file was just
/// written by the engine and is almost certainly still page-cache-hot, so
/// the second read is not a second disk trip in practice.
async fn export_stream(app: &Shared, r: &ExportReq) -> Result<Response, ApiError> {
    let (path, header) = export_lat1_file(app, r).await?;
    let header_bytes = header.to_bytes();
    let (tx, rx_ch) = mpsc::channel::<Result<Bytes, std::io::Error>>(2);
    if tx.send(Ok(Bytes::copy_from_slice(&header_bytes))).await.is_err() {
        let _ = std::fs::remove_file(&path);
        return Err(ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, "client closed the connection before the header was sent".into()));
    }
    tokio::spawn(stream_file_chunks(path, tx));
    let body = Body::from_stream(ReceiverStream::new(rx_ch));
    let mut resp = Response::new(body);
    resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/vnd.baro.state"));
    Ok(resp)
}

impl ExportReq {
    /// The export `/v1/fork`'s `target` form runs: the fork prompt's own
    /// token ids, `pos` left to `default_pos` (the prompt end minus one,
    /// which is also the only position `save_state` writes).
    pub fn for_tokens(tokens: Vec<u32>) -> ExportReq {
        ExportReq { prompt: None, messages: None, tokens: Some(tokens), pos: None, format: None, path: None }
    }
}

/// The prefill, the scratch BAROST0x file and its filled LAT1 header, shared
/// by the stream response above and `fork_target` (which sends the same
/// bytes to another node's import route). The caller owns the scratch file.
pub async fn export_lat1_file(app: &Shared, r: &ExportReq) -> Result<(std::path::PathBuf, lat1::Header), ApiError> {
    let server_format = state_file_format();
    let scratch = Scratch::new(scratch_path(app, "lat1-export"));
    let path = scratch.path().to_path_buf();
    std::fs::create_dir_all(&app.ckpts.dir).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", app.ckpts.dir.display())))?;
    let path_s = path.to_string_lossy().into_owned();
    let (prompt, pos) = match export_prefill(app, r, &path_s).await {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let payload_len = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => return Err(ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine finished but wrote no state file at {path_s}: {e}"))),
    };
    let payload_sha = hash_file(&path).await?;
    let dtype = if server_format == "int8" { lat1::DTYPE_I8_BLOCK } else { lat1::DTYPE_F32 };
    let role_sha = hex32_decode(&app.identity.pack).unwrap_or([0u8; 32]);
    let mut weights_uuid = [0u8; 16];
    weights_uuid.copy_from_slice(&role_sha[..16]);
    let tokenizer_sha = app.identity.tokenizer_sha.as_deref().and_then(hex32_decode).unwrap_or([0u8; 32]);
    let mut header = lat1::Header {
        magic: lat1::MAGIC,
        version: lat1::VERSION,
        kind: lat1::KIND_KV_PAGES,
        dtype,
        weights_uuid,
        role_sha,
        runtime: sha256(app.identity.runtime.as_bytes()),
        sigma_id: [0u8; 32],
        tokenizer_sha,
        pos_hi: pos as u32,
        ttl_s: 600,
        prefix_hash: exact_prefix_hash(&prompt, pos),
        payload_len,
        payload_sha,
        hmac: [0u8; 32],
    };
    if let Some(key) = state_hmac_key() {
        header.hmac = hmac_file(&path, &header, &key).await?;
    }
    Ok((scratch.into_path(), header))
}

/// `(pos, tokens[0:pos])` embedded in a `BAROST01`/`BAROST02`/`BAROST03` file (03 = an int8-KV engine's pool pages,
/// same header; refusing it meant an int8 engine could not import its own export, 2026-10-12)
/// (`serve/engine.mojo`'s `save_state`/`load_state`): magic(8) pos(i64 LE)
/// CONV_SLOT(i64 LE) SSM_SLOT(i64 LE) kvn(i64 LE) salt(32) tokens(pos*4,
/// i32 LE each) -- a 72-byte fixed header, then the token array; the salt
/// and payload past it are not read here, the engine re-checks the salt
/// itself during `load_state` and refuses a mismatch on its own.
fn read_state_header(path: &str) -> Result<(usize, Vec<u32>), ApiError> {
    let data = std::fs::read(path).map_err(|e| ApiError::Plain(StatusCode::BAD_REQUEST, format!("cannot read {path}: {e}")))?;
    if data.len() < 72 {
        return Err(bad(format!("{path} is too short to be a BAROST01/02/03 state file")));
    }
    if !matches!(&data[0..8], b"BAROST01" | b"BAROST02" | b"BAROST03") {
        return Err(bad(format!("{path} is not a BAROST01/02/03 state file")));
    }
    let pos = i64::from_le_bytes(data[8..16].try_into().unwrap());
    if pos < 0 {
        return Err(bad(format!("{path}: negative pos in header")));
    }
    let pos = pos as usize;
    let tok_start = 72;
    let tok_end = tok_start + pos * 4;
    if data.len() < tok_end {
        return Err(bad(format!(
            "{path} is truncated: expected {} token bytes after the header, has {}",
            tok_end - tok_start,
            data.len().saturating_sub(tok_start)
        )));
    }
    let tokens = data[tok_start..tok_end].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    Ok((pos, tokens))
}

#[derive(Deserialize)]
pub struct ImportReq {
    #[serde(default)]
    path: Option<String>,
}

/// `POST /v1/state/import`, `path` variant: `{"prefix_hash","pos",
/// "restore_ms","runtime_differs"}`. Reads the file's own embedded tokens
/// (`read_state_header`) to build the request `chain.lookup` needs to find
/// the checkpoint `state_load` just brought in (`serve/engine.mojo:852-867`:
/// load happens, THEN the request's own prompt is looked up against the
/// chain by salted hash).
///
/// `Chain::lookup` (`serve/prefix.mojo:337`) only accepts a checkpoint at
/// `pos <= n - 1`, n the request's prompt length: it always reserves the
/// prompt's last token as the live decode seed (confirmed against
/// `checkpoints::create`/`fork` above, which always submit `pos + 1`
/// tokens -- the checkpoint's own defining prompt plus a suffix). The file
/// embeds exactly `pos` tokens, so a request built from them alone always
/// has `n == pos` and can never clear that bar -- a live smoke against this
/// route 502'd every time before this fix (room A, 2026-09-17). The request
/// sent to the engine repeats the file's own last token once, giving
/// `n == pos + 1`; `exact_prefix_hash` below and the checkpoint's own
/// registered hash both still only read `tokens[0:pos]`, so the repeat
/// changes nothing that is checked, it only clears the reservation.
/// Dispatches on the request's `Content-Type`: `application/json` is the
/// `{"path":STR}` form, anything else is CONTRACT 1's raw LAT1 stream as the
/// request body. CONTRACT 3's request column genuinely differs in shape
/// between the two import forms (unlike export, where only the response
/// differs), so this takes the whole `Request` rather than a `Json<T>`
/// extractor.
pub async fn import(State(app): State<Shared>, req: AxumRequest) -> Result<Json<Value>, ApiError> {
    let is_json = req.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(|ct| ct.starts_with("application/json")).unwrap_or(false);
    if is_json {
        let bytes = to_bytes(req.into_body(), 1 << 16).await.map_err(|e| bad(format!("failed reading request body: {e}")))?;
        let r: ImportReq = serde_json::from_slice(&bytes).map_err(|e| bad(format!("invalid JSON body: {e}")))?;
        let Some(path) = r.path else {
            return Err(bad("path is required for the application/json form; send the LAT1 stream as the request body instead"));
        };
        return import_from_path(&app, &path, Value::Null).await;
    }
    import_stream(&app, req).await
}

/// `path` variant: `{"prefix_hash","pos","restore_ms","runtime_differs"}`.
/// Reads the file's own embedded tokens (`read_state_header`) to build the
/// request `chain.lookup` needs to find the checkpoint `state_load` just
/// brought in (`serve/engine.mojo:852-867`: load happens, THEN the
/// request's own prompt is looked up against the chain by salted hash).
///
/// `Chain::lookup` (`serve/prefix.mojo:337`) only accepts a checkpoint at
/// `pos <= n - 1`, n the request's prompt length: it always reserves the
/// prompt's last token as the live decode seed (confirmed against
/// `checkpoints::create`/`fork` above, which always submit `pos + 1`
/// tokens -- the checkpoint's own defining prompt plus a suffix). The file
/// embeds exactly `pos` tokens, so a request built from them alone always
/// has `n == pos` and can never clear that bar -- a live smoke against this
/// route 502'd every time before this fix (room A, 2026-09-17). The request
/// sent to the engine repeats the file's own last token once, giving
/// `n == pos + 1`; `exact_prefix_hash` below and the checkpoint's own
/// registered hash both still only read `tokens[0:pos]`, so the repeat
/// changes nothing that is checked, it only clears the reservation.
///
/// `runtime_differs` is the caller's to fill: the raw path form (no LAT1
/// header) has nothing to compare, so it is always `Null` there; the
/// streaming form below fills a real bool once it has read the header.
async fn import_from_path(app: &Shared, path: &str, runtime_differs: Value) -> Result<Json<Value>, ApiError> {
    let (pos, tokens) = read_state_header(path)?;
    if pos == 0 {
        return Err(bad(format!("{path}: pos is 0, nothing to import")));
    }
    let mut request_prompt = tokens.clone();
    request_prompt.push(tokens[pos - 1]);
    let g = Gen {
        prompt: request_prompt,
        n: 1,
        spec: false,
        stream: false,
        stop: vec![],
        ckpt: vec![],
        state: (None, Some(path.to_string())),
        state_prefix: false,
        base_pos: 0,
        embeds: vec![],
        sample: protocol::SampleParams::default(),
        schema: None,
        reasoning: None,
        embed: None,
    };
    let (_req_id, rx) = check_and_submit(app, &g)?;
    let (_acc, _text, stats) = collect(app, rx).await.map_err(map_kvq_refusal)?;
    let restore_s = stats.get("restore_s").and_then(Value::as_f64).unwrap_or(0.0);
    let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0) as usize;
    if cached < pos {
        return Err(ApiError::Plain(
            StatusCode::BAD_GATEWAY,
            format!("import ran but the chain lookup did not find the loaded state (cached={cached}, expected >={pos}): the file's salt or tokens likely do not match this pack"),
        ));
    }
    Ok(Json(json!({
        "prefix_hash": format!("{:016x}", exact_prefix_hash(&tokens, pos)),
        "pos": pos,
        "restore_ms": restore_s * 1000.0,
        "runtime_differs": runtime_differs,
    })))
}

/// CONTRACT 1's raw LAT1 stream as the request body, CONTRACT 2's ordered
/// identity check. Reads exactly 256 bytes for the header regardless of how
/// the underlying HTTP chunks land (a chunk boundary has no reason to align
/// with byte 256), then streams the remainder straight to a scratch file
/// while hashing it incrementally -- CONTRACT 1's 8 MiB-chunk, never-buffer-
/// the-whole-stream rule applies to the request side exactly as the response
/// side (`export_stream`).
async fn import_stream(app: &Shared, req: AxumRequest) -> Result<Json<Value>, ApiError> {
    let mut body = req.into_body().into_data_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(lat1::HEADER_LEN);
    while buf.len() < lat1::HEADER_LEN {
        match body.next().await {
            Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
            Some(Err(e)) => return Err(bad(format!("body stream error: {e}"))),
            None => return Err(bad(format!("stream ended before the {}-byte LAT1 header ({} bytes seen)", lat1::HEADER_LEN, buf.len()))),
        }
    }
    let header_bytes: [u8; lat1::HEADER_LEN] = buf[..lat1::HEADER_LEN].try_into().unwrap();
    let header = lat1::Header::from_bytes(&header_bytes);

    // CONTRACT 1: "kinds accepts only kv_pages and answers 400 for the
    // rest, naming P1's scope" -- a scope refusal, not an identity 409.
    if header.magic != lat1::MAGIC {
        return Err(identity_mismatch("magic", format!("{:08x}", lat1::MAGIC), format!("{:08x}", header.magic)));
    }
    if header.version != lat1::VERSION {
        return Err(identity_mismatch("version", lat1::VERSION.to_string(), header.version.to_string()));
    }
    if header.kind != lat1::KIND_KV_PAGES {
        return Err(bad(format!("kind {} is not kv_pages ({}); P1 does not carry any other kind", header.kind, lat1::KIND_KV_PAGES)));
    }
    if header.dtype != lat1::DTYPE_F32 && header.dtype != lat1::DTYPE_I8_BLOCK {
        return Err(bad(format!("dtype {} is neither f32 ({}) nor the int8-block dtype ({})", header.dtype, lat1::DTYPE_F32, lat1::DTYPE_I8_BLOCK)));
    }

    std::fs::create_dir_all(&app.ckpts.dir).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", app.ckpts.dir.display())))?;
    let scratch = Scratch::new(scratch_path(app, "lat1-import"));
    let path = scratch.path().to_path_buf();
    let mut file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => return Err(ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", path.display()))),
    };
    let mut hasher = Sha256::new();
    let first_payload = buf[lat1::HEADER_LEN..].to_vec();
    hasher.update(&first_payload);
    let mut received = first_payload.len() as u64;
    let write_err = |e: std::io::Error| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("writing scratch state file: {e}"));
    if let Err(e) = file.write_all(&first_payload).await {
        return Err(write_err(e));
    }
    loop {
        match body.next().await {
            Some(Ok(chunk)) => {
                hasher.update(&chunk);
                if let Err(e) = file.write_all(&chunk).await {
                    return Err(write_err(e));
                }
                received += chunk.len() as u64;
            }
            Some(Err(e)) => return Err(bad(format!("body stream error: {e}"))),
            None => break,
        }
    }
    if let Err(e) = file.flush().await {
        return Err(write_err(e));
    }
    drop(file);
    let payload_sha = hasher.finalize();

    // CONTRACT 2's ordered identity checks: payload_sha, role_sha,
    // tokenizer_sha, then pos against BARO_TMAX. Any miss is 409
    // state_identity and nothing is restored -- the scratch file is removed
    // before returning in every branch below, so `state_load` never sees it.
    if received != header.payload_len {
        return Err(identity_mismatch("payload_len", header.payload_len.to_string(), received.to_string()));
    }
    if payload_sha != header.payload_sha {
        return Err(identity_mismatch("payload_sha", hex32_encode(&header.payload_sha), hex32_encode(&payload_sha)));
    }
    let expected_hmac = match state_hmac_key() {
        Some(key) => hmac_file(&path, &header, &key).await?,
        None => [0u8; 32],
    };
    if !constant_time_eq(&header.hmac, &expected_hmac) {
        return Err(identity_mismatch("hmac", hex32_encode(&expected_hmac), hex32_encode(&header.hmac)));
    }
    let role_sha = hex32_decode(&app.identity.pack).unwrap_or([0u8; 32]);
    if header.role_sha != role_sha {
        return Err(identity_mismatch("role_sha", hex32_encode(&role_sha), hex32_encode(&header.role_sha)));
    }
    let tokenizer_sha = app.identity.tokenizer_sha.as_deref().and_then(hex32_decode).unwrap_or([0u8; 32]);
    if header.tokenizer_sha != tokenizer_sha {
        return Err(identity_mismatch("tokenizer_sha", hex32_encode(&tokenizer_sha), hex32_encode(&header.tokenizer_sha)));
    }
    let tmax = app.engine.limits.tmax;
    if header.pos_hi == 0 || header.pos_hi as u64 >= tmax as u64 {
        return Err(identity_mismatch("pos", format!("1..{tmax}"), header.pos_hi.to_string()));
    }
    // "A differing runtime is reported (runtime_differs:true), not
    // refused: the identity gate judges it" -- the only CONTRACT 2 field
    // that never blocks the restore.
    let runtime_differs = sha256(app.identity.runtime.as_bytes()) != header.runtime;

    let path_s = path.to_string_lossy().into_owned();
    // `scratch` removes the file when this function returns, after the engine has loaded it.
    import_from_path(app, &path_s, Value::Bool(runtime_differs)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex32_roundtrips_through_encode_and_decode() {
        let bytes = [0xa3u8, 0x76, 0x2c, 0x14, 0x8f, 0x44, 0xe7, 0x87, 0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 255];
        assert_eq!(hex32_decode(&hex32_encode(&bytes)), Some(bytes));
    }

    #[test]
    fn hex32_decode_rejects_the_wrong_length_or_non_hex() {
        assert_eq!(hex32_decode("short"), None);
        assert_eq!(hex32_decode(&"g".repeat(64)), None);
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231_case_1() {
        let mut h = HmacSha256::new(&[0x0b; 20]);
        h.update(b"Hi There");
        assert_eq!(hex32_encode(&h.finalize()), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    #[test]
    fn identity_mismatch_builds_the_contract_2_shape() {
        match identity_mismatch("role_sha", "aa", "bb") {
            ApiError::StateIdentity { field, ours, theirs } => {
                assert_eq!(field, "role_sha");
                assert_eq!(ours, "aa");
                assert_eq!(theirs, "bb");
            }
            _ => panic!("expected StateIdentity"),
        }
    }

    #[test]
    fn matches_a_hand_computed_vector() {
        // tokens[0:2] = [1, 2] as little-endian u32: 01 00 00 00 02 00 00 00.
        // Independently computed with Python (not just testing the algorithm
        // against itself):
        //   python3 -c "import hashlib,struct; \
        //     h=hashlib.sha256(bytes([1,0,0,0,2,0,0,0])).hexdigest(); \
        //     print(h[:16], struct.unpack('<Q', bytes.fromhex(h[:16]))[0])"
        //   -> 34fb5c825de7ca4a 5389374292907326260
        let tokens = [1u32, 2, 999999]; // pos=2 must ignore the trailing token
        assert_eq!(exact_prefix_hash(&tokens, 2), 5389374292907326260u64);
    }

    #[test]
    fn pos_excludes_the_token_at_pos() {
        // tokens[0:pos], not tokens[0:pos+1]: the token AT pos must not
        // change the hash (this is exactly the off-by-one id_for has).
        let a = exact_prefix_hash(&[10, 20, 30], 2);
        let b = exact_prefix_hash(&[10, 20, 999], 2);
        assert_eq!(a, b, "the hash must not depend on tokens[pos]");
        let c = exact_prefix_hash(&[10, 20, 30], 3);
        assert_ne!(a, c, "but it must depend on everything before pos");
    }

    #[test]
    fn empty_prefix_hashes_the_empty_byte_string() {
        assert_eq!(exact_prefix_hash(&[1, 2, 3], 0), exact_prefix_hash(&[], 0));
    }

    #[test]
    fn default_pos_picks_the_last_boundary_at_or_before_the_prompt_end() {
        assert_eq!(default_pos(10, &[3, 7, 12]), 7, "12 is past the prompt end, must be excluded");
        assert_eq!(default_pos(10, &[3, 7, 10]), 10, "a boundary AT the prompt end still counts");
    }

    #[test]
    fn default_pos_falls_back_to_prompt_end_minus_one_with_no_boundaries() {
        assert_eq!(default_pos(10, &[]), 9);
        assert_eq!(default_pos(0, &[]), 0, "saturating, never underflows");
    }

    #[test]
    fn map_kvq_refusal_rewrites_only_the_documented_engine_message() {
        let refusal = ApiError::Plain(
            StatusCode::BAD_REQUEST,
            "engine: BARO_STATE_SAVE: state files store f32 KV; this engine has BARO_KVQ=int8 (quantized KV state is A2 step 3)".into(),
        );
        match map_kvq_refusal(refusal) {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
                assert!(msg.contains("kvq_state_open"), "{msg}");
            }
            _ => panic!("expected ApiError::Plain"),
        }
        let unrelated = ApiError::Plain(StatusCode::BAD_REQUEST, "engine: prompt+n exceeds TMAX 128".into());
        match map_kvq_refusal(unrelated) {
            ApiError::Plain(code, msg) => {
                assert_eq!(code, StatusCode::BAD_REQUEST, "an unrelated engine error must pass through unchanged");
                assert!(msg.contains("TMAX"));
            }
            _ => panic!("expected ApiError::Plain"),
        }
    }

    fn ok<T>(r: Result<T, ApiError>) -> T {
        match r {
            Ok(v) => v,
            Err(_) => panic!("expected Ok"),
        }
    }

    /// A hand-built BAROST01 file, matching `serve/engine.mojo::save_state`
    /// byte for byte: magic, pos, CONV_SLOT, SSM_SLOT, kvn (i64 LE each),
    /// a 32-byte salt, then `pos` i32-LE tokens, then arbitrary payload
    /// bytes (never read by `read_state_header`).
    fn barost01_fixture(tokens: &[u32], extra_payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BAROST01");
        buf.extend_from_slice(&(tokens.len() as i64).to_le_bytes()); // pos
        buf.extend_from_slice(&100i64.to_le_bytes()); // CONV_SLOT, arbitrary
        buf.extend_from_slice(&200i64.to_le_bytes()); // SSM_SLOT, arbitrary
        buf.extend_from_slice(&50i64.to_le_bytes()); // kvn, arbitrary
        buf.extend_from_slice(&[0u8; 32]); // salt, not read by this function
        for &t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        buf.extend_from_slice(extra_payload);
        buf
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("baro-state-test-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn read_state_header_extracts_pos_and_tokens() {
        let path = write_temp("ok", &barost01_fixture(&[7, 8, 9], b"payload"));
        let (pos, tokens) = ok(read_state_header(path.to_str().unwrap()));
        assert_eq!(pos, 3);
        assert_eq!(tokens, vec![7, 8, 9]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_accepts_the_int8_magic_too() {
        let mut bytes = barost01_fixture(&[1], b"");
        bytes[0..8].copy_from_slice(b"BAROST02");
        let path = write_temp("int8", &bytes);
        let (pos, tokens) = ok(read_state_header(path.to_str().unwrap()));
        assert_eq!(pos, 1);
        assert_eq!(tokens, vec![1]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_rejects_a_bad_magic() {
        let mut bytes = barost01_fixture(&[1], b"");
        bytes[0..8].copy_from_slice(b"NOTASTAT");
        let path = write_temp("badmagic", &bytes);
        assert!(read_state_header(path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_state_header_rejects_a_truncated_token_array() {
        let mut bytes = barost01_fixture(&[1, 2, 3], b"");
        bytes.truncate(bytes.len() - 2); // cut into the last token's bytes
        let path = write_temp("truncated", &bytes);
        assert!(read_state_header(path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    /// A-M7: a client reset mid-upload drops the import handler's future; the scratch file (hundreds of MB in real
    /// life, outside the disk cap) was never removed.
    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_import_leaves_no_scratch_file() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.work/fix-front-test/scratch-drop");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut app = Arc::try_unwrap(crate::sse_tests::test_app()).ok().expect("fresh app");
        app.ckpts = checkpoints::Registry::new(dir.clone(), 1);
        let app: Shared = Arc::new(app);
        let mut h = lat1::Header::from_bytes(&[0u8; lat1::HEADER_LEN]);
        h.magic = lat1::MAGIC;
        h.version = lat1::VERSION;
        h.kind = lat1::KIND_KV_PAGES;
        h.dtype = lat1::DTYPE_F32;
        let mut first = h.to_bytes().to_vec();
        first.extend_from_slice(&[7u8; 10]);
        let body = Body::from_stream(tokio_stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(first))]).chain(tokio_stream::pending()));
        let req = AxumRequest::new(body);
        let r = tokio::time::timeout(std::time::Duration::from_millis(200), import_stream(&app, req)).await;
        assert!(r.is_err(), "the upload should still be pending");
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
        assert!(left.is_empty(), "scratch files left behind: {left:?}");
    }
}
