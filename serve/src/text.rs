//! Text in, text out: the pack's `tokenizer.json` (HF `tokenizers` format,
//! written by tools/gguf-tokenizer.py) plus the optional
//! `tokenizer-meta.json` (bos/eos ids, chat template). Absent tokenizer =>
//! the server still runs, token-id endpoints only.

use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

/// Python `json.dumps` layout (`", "` and `": "`, insertion order, raw UTF-8): what HF and llama.cpp
/// templates get from `tojson`. minijinja's own filter is compact and HTML-escapes `<`, `>`, `&`, `'`.
struct PyJson;

impl serde_json::ser::Formatter for PyJson {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
    /// Shortest round-trip digits like Python's `repr`, exponent spelled `e+22` / `e-07`.
    fn write_f64<W: ?Sized + std::io::Write>(&mut self, w: &mut W, value: f64) -> std::io::Result<()> {
        let s = format!("{value:?}");
        match s.split_once('e') {
            Some((m, e)) => {
                let (sign, d) = e.strip_prefix('-').map_or(("+", e), |d| ("-", d));
                write!(w, "{m}e{sign}{d:0>2}")
            }
            None => w.write_all(s.as_bytes()),
        }
    }
}

fn py_tojson(v: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyJson);
    v.serialize(&mut ser)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string()))?;
    let s = String::from_utf8(buf).map_err(|e| minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string()))?;
    Ok(minijinja::Value::from_safe_string(s))
}

#[derive(Debug, Default, Deserialize)]
struct Meta {
    #[serde(default)]
    chat_template: Option<String>,
    #[serde(default, alias = "bos_id")]
    bos_token_id: Option<u32>,
    #[serde(default, alias = "eos_id")]
    eos_token_id: Option<u32>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    add_bos: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// A5: assistant tool calls, already in the shape the chat template
    /// expects (`arguments` as a parsed object, not an OpenAI-style JSON
    /// string) -- `main.rs` converts on the way in.
    pub tool_calls: Option<Value>,
}

pub struct Text {
    tok: Tokenizer,
    meta: Meta,
    /// Token ids that end a generation (eos, plus Qwen's <|im_end|> /
    /// <|endoftext|> when the vocabulary has them).
    pub stop_ids: Vec<u32>,
}

impl Text {
    pub fn load(tokenizer_json: &Path, template_file: Option<&Path>) -> Result<Text, String> {
        let tok = Tokenizer::from_file(tokenizer_json).map_err(|e| format!("{}: {e}", tokenizer_json.display()))?;
        let meta_path = tokenizer_json.with_file_name("tokenizer-meta.json");
        let mut meta: Meta = match std::fs::read_to_string(&meta_path) {
            Ok(s) => serde_json::from_str(&s).map_err(|e| format!("{}: {e}", meta_path.display()))?,
            Err(_) => Meta::default(),
        };
        if let Some(path) = template_file {
            meta.chat_template = Some(
                std::fs::read_to_string(path)
                    .map_err(|e| format!("chat template {}: {e}", path.display()))?,
            );
        }
        if let Some(template) = &meta.chat_template {
            Self::validate_template(template, template_file.unwrap_or(meta_path.as_path()))?;
        }
        let mut stop_ids = Vec::new();
        if let Some(e) = meta.eos_token_id {
            stop_ids.push(e);
        }
        for s in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tok.token_to_id(s) {
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }
        Ok(Text { tok, meta, stop_ids })
    }

    fn validate_template(template: &str, source: &Path) -> Result<(), String> {
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", template)
            .map(|_| ())
            .map_err(|e| format!("chat template {}: {e}", source.display()))
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        let enc = self.tok.encode(text, add_special).map_err(|e| e.to_string())?;
        let mut ids = enc.get_ids().to_vec();
        if add_special && self.meta.add_bos == Some(true) {
            if let Some(bos) = self.meta.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tok.decode(ids, false).map_err(|e| e.to_string())
    }

    /// Render the chat template (the pack's, else ChatML which is what Qwen
    /// ships) with the generation prompt appended. `tools`: the OpenAI
    /// `tools` array, passed to the template verbatim (A5, `tools on
    /// /v1/chat/completions` -- the template decides how to render them,
    /// same as vLLM and llama.cpp). `template_kwargs`: extra top-level
    /// template variables (A5 `chat_template_kwargs`, e.g. Qwen/Qwythos's
    /// `enable_thinking`), merged into the render context so the caller
    /// never needs to know the template's own variable names.
    /// True when the chat template writes tool calls in the Qwen3.5 XML wire
    /// (`<function=NAME>` / `<parameter=P>`), false for the JSON wire.
    pub fn tool_wire_xml(&self) -> bool {
        self.meta.chat_template.as_deref().is_some_and(|t| t.contains("<function="))
    }

    /// The wire `parse_tool_calls` reads and the tool-name regexes describe: `<function=` XML, or JSON with a
    /// `"name"` key inside `<tool_call>`. False for any other wire (MiniCPM5 `<function name="...">`, Spark-X2.5
    /// `<tool_call>NAME<arg_key>`): forcing one of ours onto it would corrupt the call (2026-11-07 cast survey).
    pub fn tool_wire_known(&self) -> bool {
        self.meta.chat_template.as_deref().is_some_and(|t| t.contains("<function=") || (t.contains("<tool_call>") && t.contains("\"name")))
    }

    pub fn apply_chat_template(
        &self, messages: &[ChatMessage], tools: Option<&Value>, template_kwargs: Option<&Value>,
    ) -> Result<String, String> {
        self.render(messages, true, tools, template_kwargs)
    }

    fn render(
        &self, messages: &[ChatMessage], add_generation_prompt: bool, tools: Option<&Value>, template_kwargs: Option<&Value>,
    ) -> Result<String, String> {
        let Some(tpl) = &self.meta.chat_template else {
            let mut s = String::new();
            for m in messages {
                s.push_str("<|im_start|>");
                s.push_str(&m.role);
                s.push('\n');
                s.push_str(&m.content);
                s.push_str("<|im_end|>\n");
            }
            if add_generation_prompt {
                s.push_str("<|im_start|>assistant\n");
            }
            return Ok(s);
        };
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", py_tojson);
        env.add_template("chat", tpl).map_err(|e| format!("chat_template: {e}"))?;
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut v = serde_json::json!({"role": m.role, "content": m.content});
                if let Some(tc) = &m.tool_calls {
                    v["tool_calls"] = tc.clone();
                }
                v
            })
            .collect();
        let mut ctx = serde_json::json!({
            "messages": msgs,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.meta.bos_token.clone().unwrap_or_default(),
            "eos_token": self.meta.eos_token.clone().unwrap_or_default(),
        });
        if let Some(t) = tools {
            ctx["tools"] = t.clone();
        }
        if let Some(Value::Object(kwargs)) = template_kwargs {
            for (k, v) in kwargs {
                ctx[k] = v.clone();
            }
        }
        env.get_template("chat")
            .and_then(|t| t.render(minijinja::Value::from_serialize(&ctx)))
            .map_err(|e| format!("chat_template: {e}"))
    }

    /// M1b role-boundary checkpoint hints: the token length after each
    /// message when the conversation up to that message is rendered with no
    /// generation prompt. A later turn's full prompt starts with exactly
    /// these same bytes (the history before it does not change), so these
    /// positions double as restore points -- and the engine's prefix hash
    /// (`serve/prefix.mojo`) catches a mismatch rather than trusting this
    /// blindly, so a template this trick does not fit just wastes a hint,
    /// never corrupts a response.
    ///
    /// A prefix that does not render or tokenize on its own is skipped, not
    /// propagated: real templates validate the *whole* conversation (e.g.
    /// Qwen agent templates `raise_exception` a system-only prefix with "no
    /// user query found"), so an early k can legitimately fail here even
    /// though the full render (this function's caller renders separately)
    /// succeeds. Hints are an optimization; a request must never 500 over
    /// one being unavailable. A prefix that panics the renderer counts as
    /// one that does not render: minijinja 2.24 indexes `vec[0]` when a
    /// template reverses an empty list (`messages[1:][::-1]` on a
    /// system-only prefix, RegesCore persona templates).
    ///
    /// One more hint: where the last message's content starts, found by
    /// rendering the conversation again with that content emptied and taking
    /// the common token prefix. Requests that differ only in the last
    /// message (the same persona or system prompt, a new task each time)
    /// then share a checkpoint right before the text that differs, instead of
    /// the last 1024-grid point below it (regescore Blender A/B: 4096 and 5120
    /// restored of a ~5100 / ~5500-token shared prefix).
    pub fn role_boundaries(&self, messages: &[ChatMessage], tools: Option<&Value>, kwargs: Option<&Value>) -> Vec<u32> {
        let render_ids = |msgs: &[ChatMessage], gen: bool| -> Option<Vec<u32>> {
            let render = std::panic::AssertUnwindSafe(|| self.render(msgs, gen, tools, kwargs));
            let Ok(Ok(rendered)) = std::panic::catch_unwind(render) else { return None };
            self.encode(&rendered, true).ok()
        };
        let ids_of = |msgs: &[ChatMessage]| render_ids(msgs, false);
        let mut out = Vec::with_capacity(messages.len() + 1);
        let mut full = None;
        for k in 1..=messages.len() {
            let Some(ids) = ids_of(&messages[..k]) else { continue };
            out.push(ids.len() as u32);
            if k == messages.len() {
                full = Some(ids);
            }
        }
        if let Some(full) = full {
            let mut open = messages.to_vec();
            if let Some(m) = open.last_mut() {
                m.content.clear();
            }
            if let Some(ids) = ids_of(&open) {
                let start = common_prefix(&ids, &full);
                if start > 0 && start < full.len() {
                    out.push(start as u32);
                }
            }
        }
        // Where the next turn diverges from this prompt: the conversation with a reply appended
        // renders the assistant turn as history, this prompt ends in the generation prompt
        // (RegesCore: `<think></think>` only in the latter, 4 ids before the end).
        let mut next = messages.to_vec();
        next.push(ChatMessage { role: "assistant".into(), content: "x".into(), tool_calls: None });
        if let (Some(a), Some(b)) = (render_ids(&next, false), render_ids(messages, true)) {
            let at = common_prefix(&a, &b);
            if at > 0 && at < b.len() {
                out.push(at as u32);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }
}

/// Length of the longest common prefix of two token sequences.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Incremental detokenizer for streaming: emits only the new, complete text,
/// holding back a trailing replacement character from a split multi-byte
/// sequence. It decodes a short window of recent ids, not the whole prefix (that
/// was O(n^2) per stream: 10 s of decode at 16k tokens). The window is trimmed to
/// `KEEP` ids only when the tail's own decode is a suffix of the window's decode,
/// so a decoder that is not concatenative (a leading-space strip, say) just keeps
/// the whole window and stays correct.
pub struct Detok {
    win: Vec<u32>,
    /// Bytes of `decode(win)` already emitted.
    emitted: usize,
}

const KEEP: usize = 8;

impl Detok {
    pub fn new() -> Detok {
        Detok {
            win: Vec::new(),
            emitted: 0,
        }
    }

    pub fn push(&mut self, text: &Text, id: u32) -> String {
        self.win.push(id);
        let Ok(full) = text.decode(&self.win) else {
            return String::new();
        };
        let stable = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
        if stable.len() <= self.emitted || !stable.is_char_boundary(self.emitted) {
            return String::new();
        }
        let delta = stable[self.emitted..].to_string();
        self.emitted = stable.len();
        if stable.len() == full.len() && self.win.len() >= 2 * KEEP {
            let from = self.win.len() - KEEP;
            if let Ok(tail) = text.decode(&self.win[from..]) {
                if full.ends_with(&tail) {
                    self.win.drain(..from);
                    self.emitted = tail.len();
                }
            }
        }
        delta
    }

}

impl Default for Detok {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback_shape() {
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta::default(),
            stop_ids: vec![],
        };
        let s = t
            .apply_chat_template(
                &[
                    ChatMessage { role: "system".into(), content: "be brief".into(), tool_calls: None },
                    ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None },
                ],
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn role_boundary_renders_are_prefixes_of_the_full_render() {
        // The BPE-with-no-vocab tokenizer here can't produce meaningful
        // token lengths (every encode is empty), so this exercises the
        // string-level invariant role_boundaries relies on -- that each
        // prefix render is a byte-exact prefix of the full render -- not
        // the token-length numbers themselves (covered with a real
        // tokenizer by tools/test_server.sh's M1b case).
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta::default(),
            stop_ids: vec![],
        };
        let msgs = [
            ChatMessage { role: "system".into(), content: "be brief".into(), tool_calls: None },
            ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None },
        ];
        let full = t.render(&msgs, false, None, None).unwrap();
        for k in 1..=msgs.len() {
            let prefix = t.render(&msgs[..k], false, None, None).unwrap();
            assert!(full.starts_with(&prefix), "message {k} render is not a prefix: {prefix:?} vs {full:?}");
        }
        assert!(t.role_boundaries(&msgs, None, None).len() <= msgs.len());
    }

    #[test]
    fn role_boundaries_skips_a_prefix_that_fails_to_render() {
        // messages[1] is out of bounds for a 1-message prefix -- a stand-in
        // for a real chat template's own validation rejecting a shorter
        // prefix (Qwen agent templates `raise_exception` a system-only
        // prefix with "no user query found"). k=1 must be skipped, not
        // turn the whole call into an error.
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some("{% for m in messages %}<{{ m.role }}>{% endfor %}{{ messages[1].role }}".into()),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let msgs = [
            ChatMessage { role: "system".into(), content: "be brief".into(), tool_calls: None },
            ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None },
        ];
        // Doesn't panic or return an Err; k=1 (renders fine, but as an
        // empty-BPE encode -> filtered by the same empty-encode behaviour
        // as any other prefix) contributes nothing, k=2 does not error.
        let _ = t.role_boundaries(&msgs, None, None);
        assert!(t.render(&msgs, false, None, None).is_ok(), "the full render (k=2, what a real request sends) must succeed");
        assert!(t.render(&msgs[..1], false, None, None).is_err(), "k=1 must actually fail to render, or this test proves nothing");
    }

    #[test]
    fn role_boundaries_survives_a_renderer_panic() {
        // RegesCore persona templates reverse `messages[1:]`; on a
        // system-only prefix that list is empty and minijinja 2.24 panics
        // instead of rendering nothing.
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some("{% for m in messages[1:][::-1] %}<{{ m.role }}>{% endfor %}".into()),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let msgs = [
            ChatMessage { role: "system".into(), content: "be brief".into(), tool_calls: None },
            ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None },
        ];
        let _ = t.role_boundaries(&msgs, None, None);
        assert!(t.render(&msgs, false, None, None).is_ok());
    }

    #[test]
    fn jinja_template_renders_with_pycompat() {
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some(
                    "{% for m in messages %}<{{ m.role }}>{{ m.content.strip() }}</>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}".into(),
                ),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let s = t
            .apply_chat_template(
                &[ChatMessage { role: "user".into(), content: "  hi  ".into(), tool_calls: None }],
                None,
                None,
            )
            .unwrap();
        assert_eq!(s, "<user>hi</><assistant>");
    }

    #[test]
    fn custom_template_validation_rejects_invalid_jinja() {
        let err = Text::validate_template("{% if", Path::new("custom.jinja")).unwrap_err();
        assert!(err.contains("custom.jinja"));
        assert!(err.contains("chat template"));
    }

    #[test]
    fn custom_template_file_overrides_metadata_template() {
        let dir = std::env::temp_dir().join(format!("baro-template-test-{}", std::process::id()));
        let tokenizer_path = dir.join("tokenizer.json");
        let template_path = dir.join("custom.jinja");
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        std::fs::write(&tokenizer_path, tokenizer.to_string(false).unwrap()).unwrap();
        std::fs::write(dir.join("tokenizer-meta.json"), r#"{"chat_template":"metadata"}"#).unwrap();
        std::fs::write(&template_path, "custom").unwrap();

        let text = Text::load(&tokenizer_path, Some(&template_path)).unwrap();
        assert_eq!(text.render(&[], false, None, None).unwrap(), "custom");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn chat_template_kwargs_reach_the_render_context() {
        // A5: chat_template_kwargs is a bag of extra top-level template
        // variables, merged in verbatim -- this template reads one back
        // that isn't part of the fixed ctx (messages/add_generation_prompt/
        // bos_token/eos_token), the same shape Qwen/Qwythos's real template
        // reads `enable_thinking` from.
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some("{% if enable_thinking is defined and enable_thinking is false %}off{% else %}on{% endif %}".into()),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let msgs = [ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None }];
        let without = t.render(&msgs, false, None, None).unwrap();
        let with_false = t.render(&msgs, false, None, Some(&serde_json::json!({"enable_thinking": false}))).unwrap();
        assert_eq!(without, "on");
        assert_eq!(with_false, "off");
        assert_ne!(without, with_false, "chat_template_kwargs must actually change the render");
    }

    #[test]
    fn tools_array_reaches_the_template() {
        let t = Text {
            tok: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            meta: Meta {
                chat_template: Some("{% if tools %}{{ tools|length }} tools{% else %}no tools{% endif %}".into()),
                ..Meta::default()
            },
            stop_ids: vec![],
        };
        let msgs = [ChatMessage { role: "user".into(), content: "hi".into(), tool_calls: None }];
        let without = t.render(&msgs, false, None, None).unwrap();
        let tools = serde_json::json!([{"type": "function", "function": {"name": "get_weather"}}]);
        let with = t.render(&msgs, false, Some(&tools), None).unwrap();
        assert_eq!(without, "no tools");
        assert_eq!(with, "1 tools");
    }

    /// Expected strings are Python `json.dumps(x, ensure_ascii=False)` output (HF's `tojson`), not llama.cpp's,
    /// which prints floats with 6 digits (`123456789.0` -> `1.23457e+08`, `1.0` -> `1`).
    #[test]
    fn tojson_is_python_json_dumps() {
        let v = serde_json::json!({"z": "é→😀", "a": ["<b>&'\"\\/\t\n\u{1}\u{7f}", 1, 2.5, null, true, {}, []], "é": {"k": -3}});
        let got = py_tojson(minijinja::Value::from_serialize(&v)).unwrap().to_string();
        let want = format!(
            "{}{}{}",
            r#"{"z": "é→😀", "a": ["<b>&'\"\\/\t\n\u0001"#, "\u{7f}", r#"", 1, 2.5, null, true, {}, []], "é": {"k": -3}}"#
        );
        assert_eq!(got, want);
        let f: Value = serde_json::from_str("[1.0, 0.1, -0.0, 100.0, 123456789.0, 3.14159265358979, 1e22, 1e-07, 1e16, 1e15, 0.0001, 0.00001, 1.5e300, -2.5e-10]").unwrap();
        let got = py_tojson(minijinja::Value::from_serialize(&f)).unwrap().to_string();
        assert_eq!(
            got,
            "[1.0, 0.1, -0.0, 100.0, 123456789.0, 3.14159265358979, 1e+22, 1e-07, 1e+16, 1000000000000000.0, 0.0001, 1e-05, 1.5e+300, -2.5e-10]"
        );
    }

    /// Pack tokenizer for the streaming-detokenizer tests. Panics when the pack is absent, like
    /// `wire_test::load_text`: a skipped test would print "ok" and prove nothing.
    fn pack_text() -> Text {
        let pack = std::env::var("BARO_WIRE_PACK").map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.work/ailab/pack-kq"));
        Text::load(&pack.join("tokenizer.json"), None).unwrap_or_else(|e| panic!("BARO_WIRE_PACK {}: {e}", pack.display()))
    }

    /// `n_ids` tokens of mixed text: prose, code, CJK, emoji (multi-byte ids split across tokens), accents.
    fn corpus_ids(t: &Text, n_ids: usize) -> Vec<u32> {
        let parts = [
            "The quick brown fox jumps over the lazy dog. ",
            "fn main() {\n    let x = vec![1, 2, 3];\n    println!(\"{x:?}\");\n}\n",
            "\u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}\u{3002}\u{8fd9}\u{662f}\u{4e00}\u{4e2a}\u{6d4b}\u{8bd5}\u{3002}",
            " \u{1f600}\u{1f389}\u{1f44d}\u{1f3fd} caf\u{e9} na\u{ef}ve \u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629}\n\n",
            "   indented\ttext  with   runs   of   spaces. ",
            "\u{1f600}\u{1f601}\u{1f602}\u{1f923}\u{1f60a}\u{1f60d}\u{1f929}\u{1f973}\u{1f9d1}\u{200d}\u{1f4bb}\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467} ",
        ];
        let mut ids = Vec::new();
        let mut k = 0;
        while ids.len() < n_ids {
            ids.extend(t.encode(parts[k % parts.len()], false).unwrap());
            k += 1;
        }
        ids.truncate(n_ids);
        ids
    }

    /// The whole-prefix algorithm `Detok` replaced, kept as the reference.
    fn detok_reference(t: &Text, ids: &[u32]) -> Vec<String> {
        let mut emitted = 0;
        (1..=ids.len())
            .map(|k| {
                let Ok(full) = t.decode(&ids[..k]) else { return String::new() };
                let stable = full.strip_suffix('\u{FFFD}').unwrap_or(&full);
                if stable.len() <= emitted || !stable.is_char_boundary(emitted) {
                    return String::new();
                }
                let d = stable[emitted..].to_string();
                emitted = stable.len();
                d
            })
            .collect()
    }

    #[test]
    fn detok_matches_the_whole_prefix_reference_step_by_step() {
        let t = pack_text();
        let ids = corpus_ids(&t, 3000);
        let want = detok_reference(&t, &ids);
        let mut d = Detok::new();
        let got: Vec<String> = ids.iter().map(|&id| d.push(&t, id)).collect();
        assert_eq!(got, want);
        assert!(got.iter().any(|s| s.contains('\u{1f600}')), "corpus must exercise a multi-byte token split");
    }

    /// `BARO_DETOK_N` tokens through one `Detok` (a long stream); prints ms and token count.
    #[test]
    fn detok_stream_cost() {
        let t = pack_text();
        let n: usize = std::env::var("BARO_DETOK_N").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let ids = corpus_ids(&t, n);
        let t0 = std::time::Instant::now();
        let mut d = Detok::new();
        let bytes: usize = ids.iter().map(|&id| d.push(&t, id).len()).sum();
        if n > 256 {
            eprintln!("detok {n} tokens: {} ms, {bytes} bytes out", t0.elapsed().as_millis());
        }
    }
}
