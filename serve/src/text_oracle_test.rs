//! front-text oracle (docs/lanes/2026-11-09-mojo-front.md, item front-text). `cargo test --release text_oracle`
//! writes `exchange/receipts/front/text-oracle.jsonl`: what the Rust `Text` and `Detok` produce, row by row,
//! for the Mojo `serve/front/text.mojo` to equal byte for byte (`serve/front/test_text.mojo`).
//!
//! Row kinds: `header` (one per pack: the files used, with sha256, and `stop_ids`), `p0` (a wire_test body through
//! the chat template), `gate` (a bench/mtp-prompts prompt, chat-rendered), `raw` (a text through `encode` only),
//! `chat` (an edge conversation), `chatml` (no template in the pack: the ChatML fallback).
//! Packs: `kq` is `BARO_WIRE_PACK` (default .work/ailab/pack-kq) as is; `bos` and `nochat` are the same
//! tokenizer.json with a changed sidecar, written under .work/front-text/packs (`add_bos` true, so `add_special`
//! changes ids; no `chat_template`). The Mojo test refuses a file whose sha256 differs from the header.
use super::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn sh(cmd: &str) -> String {
    let o = std::process::Command::new("sh").arg("-c").arg(cmd).output().unwrap();
    assert!(o.status.success(), "{cmd}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().split_whitespace().next().unwrap_or("").to_string()
}

fn sha_file(p: &Path) -> String {
    sh(&format!("sha256sum '{}'", p.display()))
}

/// First 96 MiB of the gguf: the whole of what `serve/tokenizer.mojo` reads (HEADER_MAX).
fn sha_gguf_header(p: &Path) -> String {
    sh(&format!("head -c {} '{}' | sha256sum", 96u64 << 20, p.display()))
}

fn p0_bodies() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for dir in ["states-run", "states-run-tp"] {
        let d = root().join("bench/ailab").join(dir);
        let man = std::fs::read_to_string(d.join("MANIFEST.tsv")).unwrap();
        for line in man.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            out.push((format!("{dir}/{}", f[0]), std::fs::read(d.join(format!("{}.json", f[0]))).unwrap()));
        }
    }
    assert_eq!(out.len(), 20 * 3 + 104 * 2);
    out
}

fn to_msgs(v: &[Value]) -> Vec<ChatMessage> {
    v.iter()
        .map(|m| ChatMessage {
            role: m["role"].as_str().unwrap().to_string(),
            content: m["content"].as_str().unwrap().to_string(),
            tool_calls: m.get("tool_calls").cloned(),
        })
        .collect()
}

/// (name, directory to load, the directory as the fixture shows it). The real pack is its canonical path (outside
/// the repo); the derived ones are relative to the repo root, where the Mojo test runs, so the fixture is the
/// same bytes in every checkout.
fn pack_dirs(kq: &Path) -> Vec<(&'static str, PathBuf, String)> {
    let kq = &kq.canonicalize().unwrap();
    let base = root().join(".work/front-text/packs");
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(kq.join("tokenizer-meta.json")).unwrap()).unwrap();
    let mut out = vec![("kq", kq.to_path_buf(), kq.display().to_string())];
    for name in ["bos", "nochat"] {
        let d = base.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let link = d.join("tokenizer.json");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(kq.join("tokenizer.json"), &link).unwrap();
        let mut m = meta.clone();
        if name == "bos" {
            m["add_bos"] = json!(true);
            m["bos_token_id"] = json!(248044);
            m["bos_token"] = json!("<|endoftext|>");
        } else {
            m.as_object_mut().unwrap().remove("chat_template");
        }
        std::fs::write(d.join("tokenizer-meta.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
        out.push((name, d, format!(".work/front-text/packs/{name}")));
    }
    out
}

/// A row of `encode` + `decode` + `Detok` for one text. `full_detok`: the per-id deltas go in the row; else only
/// whether their concatenation equals `decode`.
fn text_row(t: &Text, pack: &str, kind: &str, id: &str, text: &str, full_detok: bool, extra: Value) -> Value {
    let ids = t.encode(text, true).unwrap();
    let plain = t.encode(text, false).unwrap();
    let dec = t.decode(&ids).unwrap();
    let mut d = Detok::new();
    let deltas: Vec<String> = ids.iter().map(|&i| d.push(t, i)).collect();
    let mut row = extra;
    row["pack"] = json!(pack);
    row["kind"] = json!(kind);
    row["id"] = json!(id);
    row["prompt"] = json!(text);
    row["ids"] = json!(ids);
    row["ids_plain"] = if plain == ids { Value::Null } else { json!(plain) };
    row["decode_is_prompt"] = json!(dec == text);
    if dec != text {
        row["decode"] = json!(dec);
    }
    if full_detok {
        row["detok"] = json!(deltas);
    }
    row["detok_concat_is_decode"] = json!(deltas.concat() == dec);
    row
}

fn chat_extra(t: &Text, msgs: &[ChatMessage], tools: Option<&Value>, kwargs: Option<&Value>) -> Value {
    json!({"role_boundaries": t.role_boundaries(msgs, tools, kwargs)})
}

/// Float boundary values as a client would send them. `ten22` is the integer 10**22 in the request text; serde_json
/// reads an integer past u64 as f64, so Rust (and the Mojo `Json`, which does the same) see the float 1e22.
fn float_args() -> Value {
    let ten22: Value = serde_json::from_str("10000000000000000000000").unwrap();
    json!({"p01": 1e22, "p02": 1e16, "p03": 1e15, "p04": 1e-4, "p05": 1e-5, "p06": 1e-7, "p07": 1.5e300, "p08": 1.2345678901234568e17,
        "p09": 0.1, "p10": 100.0, "p11": 1.0, "p12": -0.0, "p13": 123456789.125, "p14": 2.5, "p15": -1e-10, "p16": ten22})
}

fn edge_chats() -> Vec<(&'static str, Value)> {
    let tools = json!([
        {"type": "function", "function": {"name": "calc", "description": "a < b & c > d 'q' \"dq\" é 日本 \u{1f600}",
            "parameters": {"type": "object", "properties": {
                "x": {"type": "number", "default": 0.1}, "big": {"type": "number", "default": 1e22},
                "small": {"type": "number", "default": 1e-7}, "whole": {"type": "number", "default": 100.0},
                "neg": {"type": "number", "default": -0.0}, "n": {"type": "integer", "default": 3},
                "list": {"type": "array", "items": {"type": "string"}, "default": ["a", "b"]}}}}}
    ]);
    let calls = json!([{"function": {"name": "calc", "arguments":
        {"a": 1, "b": 2.5, "c": "x<y&z", "d": [1, 2], "e": {"f": null}, "g": true, "h": 1e22, "i": "é日本"}}}]);
    vec![
        ("plain", json!({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]})),
        ("tools-floats", json!({"messages": [{"role": "user", "content": "add"}], "tools": tools, "kwargs": {"enable_thinking": false}})),
        ("tool-history", json!({"messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "{\"r\": 3.5}"}, {"role": "user", "content": "again"}],
            "tools": tools, "kwargs": {"enable_thinking": true}})),
        ("system-only", json!({"messages": [{"role": "system", "content": "only system"}]})),
        ("unicode", json!({"messages": [{"role": "user", "content": "héllo \u{1f600}\n\n日本語 e\u{301}\t tab"}]})),
        ("kwargs-override", json!({"messages": [{"role": "user", "content": "x"}], "kwargs": {"add_generation_prompt": false, "enable_thinking": true}})),
        ("assistant-last", json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})),
        ("float-args", json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": float_args()}}]},
            {"role": "tool", "content": "ok"}, {"role": "user", "content": "again"}]})),
        ("tojson-control", json!({"messages": [{"role": "user", "content": "go"}], "kwargs": {"enable_thinking": false},
            "tools": [{"type": "function", "function": {"name": "c", "description": "bs\u{8} ff\u{c} soh\u{1} del\u{7f} nel\u{85} ls\u{2028} tab\t nl\n cr\r q\" bs\\ slash/ amp& lt<",
                "parameters": {"type": "object", "properties": {}}}}]})),
        ("empty-user", json!({"messages": [{"role": "user", "content": ""}]})),
    ]
}

fn raw_edges() -> Vec<String> {
    let mut v: Vec<String> = [
        "", " ", "\n\n", "a\u{1f600}b", "\u{1f600}", "日本語", "e\u{301}", "\u{1d54f}\u{1d550}", "\u{10ffff}", "tab\there",
        "<|im_start|>user\nhi<|im_end|>\n", "<think>\n\n</think>", "<tool_call>\n<function=f>\n</function>\n</tool_call>",
        "<|endoftext|>", "  leading and trailing  ", "12345678901234567890", "a\r\nb",
    ].iter().map(|s| s.to_string()).collect();
    v.push("x".repeat(130));
    v
}

#[test]
fn text_oracle() {
    let kq = std::env::var("BARO_WIRE_PACK").map(PathBuf::from).unwrap_or_else(|_| root().join(".work/ailab/pack-kq"));
    let out_path = root().join("exchange/receipts/front/text-oracle.jsonl");
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    let mut put = |v: Value| {
        out.write_all(serde_json::to_string(&v).unwrap().as_bytes()).unwrap();
        out.write_all(b"\n").unwrap();
    };
    let gate: Vec<(String, String)> = {
        let mut names: Vec<_> = std::fs::read_dir(root().join("bench/mtp-prompts")).unwrap()
            .map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
        names.sort();
        assert_eq!(names.len(), 20, "the 20 gate prompts");
        names.iter().map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(p).unwrap())).collect()
    };
    let mut split_seen = false;
    for (pack, dir, shown) in pack_dirs(&kq) {
        let tj = dir.join("tokenizer.json");
        let meta = dir.join("tokenizer-meta.json");
        let t = Text::load(&tj, None).unwrap_or_else(|e| panic!("{}: {e}", tj.display()));
        let mv: Value = serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
        let gguf = PathBuf::from(mv["source_gguf"].as_str().unwrap());
        put(json!({"kind": "header", "pack": pack, "tokenizer_json": format!("{shown}/tokenizer.json"),
            "tokenizer_sha256": sha_file(&tj), "meta": format!("{shown}/tokenizer-meta.json"), "meta_sha256": sha_file(&meta),
            "gguf": gguf.display().to_string(), "gguf_size": std::fs::metadata(&gguf).unwrap().len(),
            "gguf_header_sha256": sha_gguf_header(&gguf), "gguf_sha256": sha_file(&gguf),
            "stop_ids": t.stop_ids, "tool_xml": t.tool_wire_xml()}));
        for (i, s) in raw_edges().iter().enumerate() {
            let r = text_row(&t, pack, "raw", &format!("edge{i:02}"), s, true, json!({}));
            let d: Vec<&str> = r["detok"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
            split_seen |= d.windows(2).any(|w| w[0].is_empty() && w[1].chars().any(|c| c.len_utf8() == 4));
            put(r);
        }
        for (name, text) in &gate {
            put(text_row(&t, pack, "raw", name, text, true, json!({})));
        }
        for (name, req) in edge_chats() {
            let msgs = to_msgs(req["messages"].as_array().unwrap());
            let tools = req.get("tools");
            let kwargs = req.get("kwargs");
            let kind = if pack == "nochat" { "chatml" } else { "chat" };
            let extra = chat_extra(&t, &msgs, tools, kwargs);
            match t.apply_chat_template(&msgs, tools, kwargs) {
                Ok(p) => {
                    let mut r = text_row(&t, pack, kind, name, &p, true, extra);
                    r["req"] = json!({"messages": req["messages"], "tools": tools.cloned().unwrap_or(Value::Null), "kwargs": kwargs.cloned().unwrap_or(Value::Null)});
                    put(r);
                }
                Err(e) => put(json!({"pack": pack, "kind": kind, "id": name, "error": e, "role_boundaries": extra["role_boundaries"],
                    "req": {"messages": req["messages"], "tools": tools.cloned().unwrap_or(Value::Null), "kwargs": kwargs.cloned().unwrap_or(Value::Null)}})),
            }
        }
        if pack != "kq" {
            continue;
        }
        for (name, text) in &gate {
            let msgs = vec![ChatMessage { role: "user".into(), content: text.clone(), tool_calls: None }];
            let p = t.apply_chat_template(&msgs, None, None).unwrap();
            put(text_row(&t, pack, "gate", name, &p, true, chat_extra(&t, &msgs, None, None)));
        }
        for (id, raw) in p0_bodies() {
            let v: Value = serde_json::from_slice(&raw).unwrap();
            let msgs: Vec<ChatMessage> = v["messages"].as_array().unwrap().iter().map(|m| ChatMessage {
                role: m["role"].as_str().unwrap().to_string(),
                content: m["content"].as_str().unwrap().to_string(),
                tool_calls: m.get("tool_calls").and_then(Value::as_array).map(|calls| {
                    // the same conversion as main.rs `tool_calls_for_template`
                    let c: Vec<ToolCallIn> = serde_json::from_value(Value::Array(calls.clone())).unwrap();
                    tool_calls_for_template(&c)
                }),
            }).collect();
            let tools = v.get("tools");
            let kwargs = v.get("chat_template_kwargs");
            let p = t.apply_chat_template(&msgs, tools, kwargs).unwrap_or_else(|e| panic!("{id}: {e}"));
            put(text_row(&t, pack, "p0", &id, &p, false, chat_extra(&t, &msgs, tools, kwargs)));
        }
    }
    assert!(split_seen, "corpus must exercise a 4-byte codepoint split across tokens");
    drop(put);
    out.flush().unwrap();
}
