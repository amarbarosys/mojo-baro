//! Vision input through llama.cpp's projector (docs/vision.md): an `image_url`
//! content part becomes `<|vision_start|><|image_pad|>...<|vision_end|>` in the
//! rendered text plus one `.bemb` embeddings file for the engine, produced by
//! `tools/mtmd-embed` (libmtmd, CPU by default) and cached by content hash.
//!
//! Configuration, all env: `BARO_MTMD_EMBED` (the sidecar binary), `BARO_MMPROJ`
//! (projector gguf), `BARO_TEXT_GGUF` (the text gguf, opened vocab-only),
//! `BARO_IMAGE_MAX_TOKENS` (default 576), `BARO_VISION_DIR` (cache, default
//! `.work/vision/cache`). Unset => image parts are refused with a 400.
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

use crate::{bad, ApiError};

pub struct Vision {
    embed_bin: String,
    mmproj: String,
    text_gguf: String,
    max_tokens: u32,
    dir: PathBuf,
}

impl Vision {
    pub fn from_env() -> Option<Vision> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Some(Vision {
            embed_bin: get("BARO_MTMD_EMBED")?,
            mmproj: get("BARO_MMPROJ")?,
            text_gguf: get("BARO_TEXT_GGUF")?,
            max_tokens: get("BARO_IMAGE_MAX_TOKENS").and_then(|v| v.parse().ok()).unwrap_or(576),
            dir: PathBuf::from(get("BARO_VISION_DIR").unwrap_or_else(|| ".work/vision/cache".into())),
        })
    }

    /// Image bytes -> (absolute `.bemb` path, image token count). Cached on
    /// the content hash and the token cap; the sidecar runs once per new image.
    pub fn encode(&self, bytes: &[u8], ext: &str) -> Result<(String, u32), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("vision dir {}: {e}", self.dir.display()))?;
        let stem = self.dir.join(format!("{:016x}-{}", fnv1a64(bytes), self.max_tokens));
        let bemb = stem.with_extension("bemb");
        let meta = stem.with_extension("json");
        let abs = |p: &PathBuf| std::fs::canonicalize(p).map(|a| a.to_string_lossy().into_owned());
        if bemb.is_file() {
            if let Ok(m) = std::fs::read_to_string(&meta) {
                if let Some(n) = serde_json::from_str::<Value>(&m).ok().and_then(|v| v["n_tokens"].as_u64()) {
                    return Ok((abs(&bemb).map_err(|e| e.to_string())?, n as u32));
                }
            }
        }
        let img = stem.with_extension(ext);
        std::fs::write(&img, bytes).map_err(|e| format!("write {}: {e}", img.display()))?;
        let out = Command::new(&self.embed_bin)
            .args([&self.mmproj, &self.text_gguf])
            .arg(&img)
            .arg(&bemb)
            .args(["--max-tokens", &self.max_tokens.to_string()])
            .output()
            .map_err(|e| format!("run {}: {e}", self.embed_bin))?;
        if !out.status.success() {
            return Err(format!("mtmd-embed failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        let line = String::from_utf8_lossy(&out.stdout);
        let v: Value = serde_json::from_str(line.trim()).map_err(|e| format!("mtmd-embed output {line:?}: {e}"))?;
        let n = v["n_tokens"].as_u64().ok_or("mtmd-embed output lacks n_tokens")? as u32;
        std::fs::write(&meta, line.trim()).map_err(|e| format!("write {}: {e}", meta.display()))?;
        eprintln!("vision: encoded {} -> {} tokens ({})", img.display(), n, line.trim());
        Ok((abs(&bemb).map_err(|e| e.to_string())?, n))
    }
}

/// `data:image/<ext>;base64,<...>`, `file:///abs/path` or an absolute path.
pub fn decode_image_url(url: &str) -> Result<(Vec<u8>, String), String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (head, body) = rest.split_once(',').ok_or("data URL without a comma")?;
        if !head.ends_with(";base64") {
            return Err("only base64 data URLs are supported".into());
        }
        let mime = head.trim_end_matches(";base64");
        let ext = mime.strip_prefix("image/").unwrap_or("png").to_string();
        return Ok((base64_decode(body)?, ext));
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    if !path.starts_with('/') {
        return Err("image_url must be a data: URL or an absolute file path".into());
    }
    let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("png").to_string();
    std::fs::read(path).map(|b| (b, ext)).map_err(|e| format!("read {path}: {e}"))
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err(format!("invalid base64 byte {c}")),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn fnv1a64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// A message's `content`: a string, or OpenAI-style parts. Text parts are
/// concatenated; each `image_url` part is encoded through the sidecar and
/// rendered as the model's vision span, its `.bemb` path pushed to `embeds`
/// in prompt order (the engine pairs runs and files in that order).
pub async fn content_with_images(v: &Value, embeds: &mut Vec<String>) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | None if p.get("text").is_some() => {
                        out.push_str(p["text"].as_str().unwrap_or(""));
                    }
                    Some("image_url") => {
                        let url = p["image_url"]
                            .as_str()
                            .or_else(|| p["image_url"]["url"].as_str())
                            .ok_or_else(|| bad("image_url part without a url"))?;
                        let vis = Vision::from_env().ok_or_else(|| {
                            bad("vision is not configured: set BARO_MTMD_EMBED, BARO_MMPROJ and BARO_TEXT_GGUF")
                        })?;
                        let (bytes, ext) = decode_image_url(url).map_err(bad)?;
                        let (path, n) = tokio::task::spawn_blocking(move || vis.encode(&bytes, &ext))
                            .await
                            .map_err(|e| bad(format!("vision task: {e}")))?
                            .map_err(bad)?;
                        embeds.push(path);
                        out.push_str("<|vision_start|>");
                        for _ in 0..n {
                            out.push_str("<|image_pad|>");
                        }
                        out.push_str("<|vision_end|>");
                    }
                    Some(other) => return Err(bad(format!("unsupported content part type {other:?}"))),
                    None => return Err(bad("content part without type or text")),
                }
            }
            Ok(out)
        }
        _ => Err(bad("message content must be a string or an array of parts")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert!(base64_decode("a*").is_err());
    }

    #[test]
    fn data_url_ext() {
        let (b, ext) = decode_image_url("data:image/png;base64,aGk=").unwrap();
        assert_eq!((b.as_slice(), ext.as_str()), (&b"hi"[..], "png"));
        assert!(decode_image_url("relative.png").is_err());
    }
}
