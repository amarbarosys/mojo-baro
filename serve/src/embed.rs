//! Standard embedder: `bge-small-en-v1.5` in a Mojo child process (`embed/serve_embed.mojo`), one load, CPU only.
//! `/v1/embeddings` and `/api/embeddings` route here when `model` is absent or names bge-small
//! (`embeddings.rs`); a chat model's name keeps the old last-hidden-state path.
//!
//! Wire (stdin/stdout lines, documented in `embed/serve_embed.mojo`): the child prints `ready`; a request line is
//! the batch's texts as hex of their UTF-8 bytes joined by a space (`-` is an empty text); the reply line is
//! `ok NTOK:HEX ...` (per text its token count and 384 little-endian f32, L2-normalised) or `err MESSAGE`.
//! The child starts on the first request (at boot under `--embed-only`) and is restarted after it dies.

use super::*;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

pub const MODEL: &str = "bge-small-en-v1.5";
pub const DIMS: usize = 384;

const START_TIMEOUT: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, PartialEq)]
pub enum Route {
    Bge,
    Chat,
    Unknown,
}

/// `model` absent or bge-small (any case, with or without the version) -> bge; the loaded chat model's name -> chat.
pub fn route(model: Option<&str>, chat_model: &str) -> Route {
    match model.map(str::trim) {
        None | Some("") => Route::Bge,
        Some(m) if m.eq_ignore_ascii_case(MODEL) || m.eq_ignore_ascii_case("bge-small") || m.eq_ignore_ascii_case("bge-small-en") => Route::Bge,
        Some(m) if m == chat_model => Route::Chat,
        Some(_) => Route::Unknown,
    }
}

struct Live {
    _child: Child,
    stdin: ChildStdin,
    out: BufReader<ChildStdout>,
}

pub struct EmbedSidecar {
    bin: String,
    model_dir: String,
    norm: String,
    max_tokens: u32,
    live: tokio::sync::Mutex<Option<Live>>,
}

pub fn encode_request(texts: &[String]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut line = String::new();
    for (i, t) in texts.iter().enumerate() {
        if i > 0 {
            line.push(' ');
        }
        if t.is_empty() {
            line.push('-');
        }
        for b in t.bytes() {
            line.push(H[(b >> 4) as usize] as char);
            line.push(H[(b & 15) as usize] as char);
        }
    }
    line
}

/// `ok NTOK:HEX ...` -> (token count, vector) per text.
pub fn parse_reply(line: &str, n: usize) -> Result<Vec<(usize, Vec<f32>)>, String> {
    let line = line.trim_end();
    if let Some(e) = line.strip_prefix("err ") {
        return Err(e.to_string());
    }
    let body = line.strip_prefix("ok ").ok_or_else(|| format!("embed process answered {:?}", &line[..line.len().min(80)]))?;
    let mut res = Vec::with_capacity(n);
    for item in body.split(' ') {
        let (ntok, hex) = item.split_once(':').ok_or("embed reply item without a colon")?;
        let ntok: usize = ntok.parse().map_err(|_| "embed reply token count is not a number")?;
        let hb = hex.as_bytes();
        if hb.len() != DIMS * 8 {
            return Err(format!("embed reply vector is {} hex digits, want {}", hb.len(), DIMS * 8));
        }
        let nib = |c: u8| (c as char).to_digit(16).map(|d| d as u8).ok_or("embed reply has a non-hex digit");
        let mut v = Vec::with_capacity(DIMS);
        for w in hb.chunks(8) {
            let mut b = [0u8; 4];
            for (k, p) in w.chunks(2).enumerate() {
                b[k] = nib(p[0])? << 4 | nib(p[1])?;
            }
            v.push(f32::from_le_bytes(b));
        }
        res.push((ntok, v));
    }
    if res.len() != n {
        return Err(format!("embed process returned {} vectors for {} texts", res.len(), n));
    }
    Ok(res)
}

impl EmbedSidecar {
    pub fn from_env() -> EmbedSidecar {
        let home = std::env::var("HOME").unwrap_or_default();
        EmbedSidecar {
            bin: std::env::var("BARO_EMBED_BIN").unwrap_or_else(|_| format!("{home}/.cache/baro/embed/serve_embed")),
            model_dir: std::env::var("BARO_EMBED_MODEL").unwrap_or_else(|_| format!("{home}/Models/bge-small-en-v1.5")),
            norm: std::env::var("BARO_EMBED_NORM").unwrap_or_else(|_| format!("{home}/.cache/baro/embed/norm.tsv")),
            max_tokens: std::env::var("BARO_EMBED_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(512),
            live: tokio::sync::Mutex::new(None),
        }
    }

    async fn spawn(&self) -> Result<Live, String> {
        let mut child = Command::new(&self.bin)
            .arg(&self.model_dir)
            .arg(&self.norm)
            .arg(self.max_tokens.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("embed process {}: {e} (build it with embed/build.sh)", self.bin))?;
        let stdin = child.stdin.take().ok_or("embed process has no stdin")?;
        let mut out = BufReader::new(child.stdout.take().ok_or("embed process has no stdout")?);
        let mut first = String::new();
        match tokio::time::timeout(START_TIMEOUT, out.read_line(&mut first)).await {
            Ok(Ok(n)) if n > 0 && first.trim() == "ready" => {}
            Ok(Ok(_)) => return Err(format!("embed process did not say ready (got {:?}); model {} norm {}", first.trim(), self.model_dir, self.norm)),
            Ok(Err(e)) => return Err(format!("embed process: {e}")),
            Err(_) => return Err("embed process did not become ready in 60 s".into()),
        }
        Ok(Live { _child: child, stdin, out })
    }

    /// Start the child now (`--embed-only` boot) so a missing binary or model fails at launch.
    pub async fn start(&self) -> Result<(), String> {
        let mut g = self.live.lock().await;
        if g.is_none() {
            *g = Some(self.spawn().await?);
        }
        Ok(())
    }

    /// One batch, serialised with other batches; the child parallelises the rows of a batch.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<(usize, Vec<f32>)>, String> {
        let mut g = self.live.lock().await;
        if g.is_none() {
            *g = Some(self.spawn().await?);
        }
        let live = g.as_mut().expect("just started");
        let mut line = encode_request(texts);
        line.push('\n');
        let io = async {
            live.stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
            live.stdin.flush().await.map_err(|e| e.to_string())?;
            let mut reply = String::new();
            let n = live.out.read_line(&mut reply).await.map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("embed process closed its output".to_string());
            }
            Ok(reply)
        };
        match tokio::time::timeout(REQUEST_TIMEOUT, io).await {
            Ok(Ok(reply)) => parse_reply(&reply, texts.len()),
            Ok(Err(e)) => {
                *g = None;
                Err(e)
            }
            Err(_) => {
                *g = None;
                Err("embed process timed out after 120 s".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply_for(vals: &[f32]) -> String {
        vals.iter().flat_map(|f| f.to_le_bytes()).map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn route_picks_bge_for_absent_and_named_models() {
        assert_eq!(route(None, "qwen"), Route::Bge);
        assert_eq!(route(Some(""), "qwen"), Route::Bge);
        assert_eq!(route(Some("bge-small-en-v1.5"), "qwen"), Route::Bge);
        assert_eq!(route(Some("BGE-Small-EN-v1.5"), "qwen"), Route::Bge);
        assert_eq!(route(Some("bge-small"), "qwen"), Route::Bge);
    }

    #[test]
    fn route_keeps_the_chat_model_on_the_old_path_and_refuses_others() {
        assert_eq!(route(Some("qwen"), "qwen"), Route::Chat);
        assert_eq!(route(Some("text-embedding-3-small"), "qwen"), Route::Unknown);
    }

    #[test]
    fn request_line_is_hex_with_dash_for_empty() {
        assert_eq!(encode_request(&["hi".into(), "".into(), "\u{e9}".into()]), "6869 - c3a9");
        assert_eq!(encode_request(&["".into()]), "-");
    }

    #[test]
    fn reply_parses_counts_and_vectors() {
        let v: Vec<f32> = (0..DIMS).map(|i| i as f32 * 0.5 - 3.0).collect();
        let line = format!("ok 7:{} 2:{}\n", reply_for(&v), reply_for(&v));
        let got = parse_reply(&line, 2).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].0, got[1].0), (7, 2));
        assert_eq!(got[0].1, v);
    }

    #[test]
    fn reply_refuses_errors_short_vectors_and_wrong_counts() {
        assert_eq!(parse_reply("err bad hex digit\n", 1).unwrap_err(), "bad hex digit");
        assert!(parse_reply("ok 3:abcd", 1).is_err());
        assert!(parse_reply("garbage", 1).is_err());
        let v = vec![0.0f32; DIMS];
        assert!(parse_reply(&format!("ok 3:{}", reply_for(&v)), 2).is_err());
    }
}
