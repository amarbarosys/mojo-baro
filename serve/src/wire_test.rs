//! P2 wire compat (docs/plan-ailab-server.md): the request bodies street-sim's `ask_openai` posts (bench/ailab/states-run
//! and states-run-tp, recorded by bench/ailab/capture.py) go through the server's own request parser and chat template,
//! and the response message the server builds has the fields `ask_openai` reads. The runner re-asks WITHOUT tools on
//! any HTTPError, so a body that fails to parse (4xx) would silently degrade a body's whole ask: these tests are the
//! guard. `BARO_WIRE_PACK` (default .work/ailab/pack-kq) names the pack whose tokenizer.json and template render the bodies.
use super::*;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// (id, round, bytes) of every body listed in the MANIFEST.tsv of both sets; the list and the files must agree.
fn bodies() -> Vec<(String, usize, Vec<u8>)> {
    let mut out = Vec::new();
    for dir in ["states-run", "states-run-tp"] {
        let d = root().join("bench/ailab").join(dir);
        let man = std::fs::read_to_string(d.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
        for line in man.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 4, "{dir}: bad MANIFEST row {line:?}");
            let raw = std::fs::read(d.join(format!("{}.json", f[0]))).unwrap_or_else(|e| panic!("{dir}/{}: {e}", f[0]));
            assert_eq!(raw.len(), f[3].parse::<usize>().unwrap(), "{dir}/{}: size differs from MANIFEST", f[0]);
            out.push((format!("{dir}/{}", f[0]), f[1].parse().unwrap(), raw));
        }
        let files = std::fs::read_dir(&d).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".json")).count();
        assert_eq!(files, man.lines().count(), "{dir}: files and MANIFEST rows differ");
    }
    assert_eq!(out.len(), 20 * 3 + 104 * 2, "the 20x3 + 104x2 bodies");
    out
}

fn tool_names(tools: &Value) -> Vec<String> {
    tools.as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn every_p0_body_is_a_valid_chat_request() {
    for (id, round, raw) in bodies() {
        let v: Value = serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("{id}: not json: {e}"));
        let r: ChatReq = serde_json::from_value(v).unwrap_or_else(|e| panic!("{id}: the server rejects the body: {e}"));
        assert_eq!(r.model.as_deref(), Some("local"), "{id}");
        assert!(r.tool_choice.is_none() && !r.stream && !r.checkpoint && r.response_format.is_none(), "{id}: unexpected fields");
        assert_eq!(r.chat_template_kwargs, Some(json!({"enable_thinking": false})), "{id}");
        assert_eq!(r.max_tokens, Some(200), "{id}: the runner's own cap");
        let tools = r.tools.as_ref().unwrap_or_else(|| panic!("{id}: no tools"));
        let names = tool_names(tools);
        assert!(names.contains(&"state".to_string()) && names.contains(&"learn".to_string()), "{id}: tool set {names:?}");
        for t in tools.as_array().unwrap() {
            assert!(t["function"]["parameters"].get("required").is_none(), "{id}: a tool carries `required`");
        }
        assert_eq!(r.messages.len(), 2 + 2 * (round - 1), "{id}: message count for round {round}");
        assert_eq!((r.messages[0].role.as_str(), r.messages[1].role.as_str()), ("system", "user"), "{id}");
        for pair in r.messages[2..].chunks(2) {
            let (a, t) = (&pair[0], &pair[1]);
            assert_eq!((a.role.as_str(), t.role.as_str()), ("assistant", "tool"), "{id}: history order");
            let calls = a.tool_calls.as_deref().unwrap_or_else(|| panic!("{id}: assistant turn without tool_calls"));
            let tpl = tool_calls_for_template(calls);
            for c in tpl.as_array().unwrap() {
                assert!(names.contains(&c["function"]["name"].as_str().unwrap().to_string()), "{id}: history calls an unknown tool");
                assert!(c["function"]["arguments"].is_object(), "{id}: history arguments did not parse back to an object");
            }
            assert!(t.tool_call_id.is_some() && t.content.is_string(), "{id}: tool message shape");
        }
    }
}

fn load_text() -> Text {
    let pack = std::env::var("BARO_WIRE_PACK").map(PathBuf::from).unwrap_or_else(|_| root().join(".work/ailab/pack-kq"));
    Text::load(&pack.join("tokenizer.json"), None).unwrap_or_else(|e| panic!("BARO_WIRE_PACK {}: {e}", pack.display()))
}

#[test]
fn every_p0_body_renders_through_the_chat_template() {
    let t = load_text();
    let mut longest = 0;
    for (id, _, raw) in bodies() {
        let r: ChatReq = serde_json::from_slice(&raw).unwrap();
        let msgs: Vec<ChatMessage> = r
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.as_str().unwrap_or_else(|| panic!("{id}: non-string content")).to_string(),
                tool_calls: m.tool_calls.as_deref().map(tool_calls_for_template),
            })
            .collect();
        let (tools, force) = resolve_tool_choice(r.tools.clone(), r.tool_choice.as_ref(), t.tool_wire_xml()).ok().unwrap();
        assert!(tools.is_some() && force.is_empty(), "{id}: tools dropped or a call forced without tool_choice");
        let rendered = t.apply_chat_template(&msgs, tools.as_ref(), r.chat_template_kwargs.as_ref()).unwrap_or_else(|e| panic!("{id}: template: {e}"));
        for n in tool_names(tools.as_ref().unwrap()) {
            assert!(rendered.contains(&n), "{id}: tool {n} missing from the prompt");
        }
        for m in &msgs {
            let head: String = m.content.chars().take(40).collect();
            assert!(rendered.contains(&head), "{id}: a {} message is missing from the prompt", m.role);
        }
        let ids = t.encode(&rendered, true).unwrap_or_else(|e| panic!("{id}: encode: {e}"));
        assert!(ids.len() < 16384 - 200, "{id}: prompt of {} tokens", ids.len());
        longest = longest.max(ids.len());
    }
    eprintln!("wire: longest rendered P0 prompt {longest} tokens");
}

#[test]
fn a_generated_call_becomes_the_message_ask_openai_reads() {
    let (_, _, raw) = bodies().into_iter().next().unwrap();
    let v: Value = serde_json::from_slice(&raw).unwrap();
    for tool in v["tools"].as_array().unwrap() {
        let f = &tool["function"];
        let name = f["name"].as_str().unwrap();
        let mut gen = format!("<tool_call>\n<function={name}>\n");
        for (p, spec) in f["parameters"]["properties"].as_object().unwrap() {
            let val = if spec["type"] == "number" { "2.5" } else { "hello" };
            gen += &format!("<parameter={p}>\n{val}\n</parameter>\n");
        }
        gen += "</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(&gen);
        let mut message = json!({"role": "assistant", "content": content});
        message["tool_calls"] = tool_calls_response_json(&calls);
        // what ask_openai does with it
        assert!(message["content"].is_string(), "{name}: content is not a string");
        let c = &message["tool_calls"][0];
        assert!(c["id"].as_str().is_some_and(|s| !s.is_empty()), "{name}: no id");
        assert_eq!(c["function"]["name"], json!(name));
        let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or_else(|| panic!("{name}: arguments is not a JSON string"))).unwrap();
        assert!(args.is_object(), "{name}: arguments is not an object");
        for (p, spec) in f["parameters"]["properties"].as_object().unwrap() {
            if spec["type"] == "number" {
                assert!(args[p].is_number(), "{name}.{p}: a number came back as {}", args[p]);
            } else {
                assert_eq!(args[p], json!("hello"), "{name}.{p}");
            }
        }
    }
    let (content, calls) = parse_tool_calls("<tool_call>\n<function=state>\n</function>\n</tool_call>\n<tool_call>\n<function=actions>\n</function>\n</tool_call>");
    assert_eq!((content.as_str(), calls.len()), ("", 2));
    let (text, none) = parse_tool_calls("I will wait here.");
    assert_eq!((text.as_str(), none.len()), ("I will wait here.", 0));
}
