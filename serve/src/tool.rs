//! Tools the Baro phone app borrows from the workstation. `POST /tool/crawl {"url"}` fetches the page with crawl4ai
//! (`BARO_CRAWL4AI`, default `http://127.0.0.1:11235`; bearer token from `BARO_CRAWL4AI_TOKEN` or `CRAWL4AI_API_TOKEN`)
//! and answers `{"text": markdown, "source": "crawl4ai"}`. The app asks here first and falls back to its own request.

use super::*;
use std::time::Duration;

/// Longest text sent back; the app cuts it further to the model's budget.
const MAX_TEXT: usize = 64 * 1024;

fn endpoint() -> String {
    std::env::var("BARO_CRAWL4AI").unwrap_or_else(|_| "http://127.0.0.1:11235".into()).trim_end_matches('/').to_string()
}

/// Only web URLs: crawl4ai would also open `file:` and other schemes on this machine.
fn web_url(req: &Value) -> Option<&str> {
    req.get("url")?.as_str().filter(|u| u.starts_with("http://") || u.starts_with("https://"))
}

/// [text] cut to at most [MAX_TEXT] bytes on a character boundary.
fn capped(text: &str) -> &str {
    let mut end = text.len().min(MAX_TEXT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn gateway(msg: String) -> ApiError {
    ApiError::Plain(StatusCode::BAD_GATEWAY, msg)
}

pub async fn crawl(Json(req): Json<Value>) -> Result<Json<Value>, ApiError> {
    let url = web_url(&req).ok_or_else(|| bad("\"url\" must be an http(s) URL"))?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build().map_err(|e| gateway(format!("http client: {e}")))?;
    let mut call = client.post(format!("{}/md", endpoint())).json(&json!({"url": url, "f": "fit"}));
    if let Some(token) = std::env::var("BARO_CRAWL4AI_TOKEN").or_else(|_| std::env::var("CRAWL4AI_API_TOKEN")).ok().filter(|t| !t.is_empty()) {
        call = call.bearer_auth(token);
    }
    let r = call.send().await.map_err(|e| gateway(format!("crawl4ai: {e}")))?;
    if !r.status().is_success() {
        return Err(gateway(format!("crawl4ai answered {}", r.status())));
    }
    let v: Value = r.json().await.map_err(|e| gateway(format!("crawl4ai sent no JSON: {e}")))?;
    let text = v.get("markdown").and_then(Value::as_str).unwrap_or("");
    if text.trim().is_empty() {
        return Err(gateway("crawl4ai returned no text".into()));
    }
    Ok(Json(json!({"text": capped(text), "source": "crawl4ai"})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_http_urls_pass() {
        assert_eq!(web_url(&json!({"url": "https://example.com/a"})), Some("https://example.com/a"));
        assert_eq!(web_url(&json!({"url": "file:///etc/passwd"})), None);
        assert_eq!(web_url(&json!({"url": 5})), None);
        assert_eq!(web_url(&json!({})), None);
    }

    #[test]
    fn the_cut_keeps_whole_characters() {
        let s = "é".repeat(MAX_TEXT); // 2 bytes each
        let c = capped(&s);
        assert!(c.len() <= MAX_TEXT && c.chars().all(|ch| ch == 'é'));
        assert_eq!(capped("short"), "short");
    }
}
