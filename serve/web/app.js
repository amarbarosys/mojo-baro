"use strict";

const LS_CONV = "baro.conversation";
const LS_SETTINGS = "baro.settings.v3";

const DEFAULT_SETTINGS = {
  serverUrl: location.origin,
  temperature: 0.7,
  maxTokens: 512,
  topP: 1, topK: 0, minP: 0, seed: null, frequencyPenalty: 0, presencePenalty: 0, stopSequences: "", spec: false, thinking: true, hidden: false, logprobs: false, jsonSchema: "",
  systemPrompt: "",
};

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode / quota */
  }
}

let settings = { ...DEFAULT_SETTINGS, ...loadJSON(LS_SETTINGS, {}) };
let conversation = loadJSON(LS_CONV, []);
let pendingAbort = null;
let pendingResponseId = null;

const messagesEl = document.getElementById("messages");
const emptyHint = document.getElementById("empty-hint");
const composer = document.getElementById("composer");
const promptEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-button");
const micBtn = document.getElementById("mic");
const settingsBtn = document.getElementById("settings-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const settingsBackdrop = document.getElementById("settings-backdrop");
const settingsClose = document.getElementById("settings-close");
const settingsClear = document.getElementById("settings-clear");
const serverUrlEl = document.getElementById("server-address");
const temperatureEl = document.getElementById("temperature");
const temperatureVal = document.getElementById("temperature-val");
const maxTokensEl = document.getElementById("max-tokens");
const systemPromptEl = document.getElementById("system-prompt");

const modelSelect = document.getElementById("model-select");
const modelLoadBtn = document.getElementById("model-load-btn");
const modelStopBtn = document.getElementById("model-stop-btn");
const statusPill = document.getElementById("status-pill");
const telemetryBadge = document.getElementById("telemetry-badge");

const field = (id) => document.getElementById(id);

/* Lightweight Markdown & Thinking Parser */
function parseMarkdown(text) {
  if (!text) return "";
  
  // 1. Separate <think> reasoning blocks if present
  let thinkHtml = "";
  let mainText = text;
  
  const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/i);
  if (thinkMatch) {
    const thinkContent = escapeHtml(thinkMatch[1].trim());
    const isComplete = text.includes("</think>");
    const openAttr = isComplete ? "" : " open";
    thinkHtml = `<details class="reasoning-box"${openAttr}><summary>Thinking Process</summary><div class="reasoning-content">${thinkContent}</div></details>`;
    mainText = text.replace(/<think>[\s\S]*?(?:<\/think>|$)/i, "").trim();
  }
  
  // 2. Code blocks ```lang ... ```
  let html = escapeHtml(mainText);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><button class="copy-btn" onclick="copyCode(this)">Copy</button><code>${code}</code></pre>`;
  });
  
  // 3. Inline code `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  
  // 4. Bold **text**
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // 5. Line breaks
  html = html.replace(/\n/g, '<br>');
  
  return thinkHtml + html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

window.copyCode = function(btn) {
  const code = btn.nextElementSibling ? btn.nextElementSibling.innerText : "";
  if (!code) return;
  navigator.clipboard.writeText(code).then(() => {
    const orig = btn.innerText;
    btn.innerText = "Copied!";
    setTimeout(() => { btn.innerText = orig; }, 1500);
  });
};

function renderMessages() {
  messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
  emptyHint.hidden = conversation.length > 0;
  for (const m of conversation) {
    messagesEl.appendChild(bubbleFor(m.role, m.content));
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function bubbleFor(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "assistant") {
    el.dataset.role = "assistant-message";
    el.innerHTML = parseMarkdown(text);
  } else {
    el.textContent = text;
  }
  return el;
}

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(busy) {
  sendBtn.disabled = false;
  sendBtn.classList.toggle("stop", busy);
  sendBtn.textContent = busy ? "■" : "➤";
  promptEl.disabled = busy;
}

/* Model Management & Status Polling */
async function fetchModels() {
  try {
    const r = await fetch(new URL("/api/models", settings.serverUrl));
    if (!r.ok) return;
    const models = await r.json();
    modelSelect.innerHTML = "";
    if (models.length === 0) {
      modelSelect.innerHTML = `<option value="">No models found</option>`;
      return;
    }
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.path;
      opt.textContent = `${m.name} (${m.size_gb} GB)`;
      modelSelect.appendChild(opt);
    }
  } catch {
    modelSelect.innerHTML = `<option value="">Model service unavailable</option>`;
  }
}

async function checkStatus() {
  try {
    const r = await fetch(new URL("/api/status", settings.serverUrl));
    if (!r.ok) throw new Error("Offline");
    const st = await r.json();
    if (st.status === "running") {
      statusPill.className = "status-pill ready";
      statusPill.title = `Loaded: ${st.model} (PID ${st.pid})`;
      modelStopBtn.hidden = false;
    } else {
      statusPill.className = "status-pill offline";
      statusPill.title = "Offline / No Model Loaded";
      modelStopBtn.hidden = true;
    }
  } catch {
    statusPill.className = "status-pill offline";
    statusPill.title = "Backend Service Offline";
    modelStopBtn.hidden = true;
  }
}

modelLoadBtn.addEventListener("click", async () => {
  const selectedModel = modelSelect.value;
  if (!selectedModel) return;

  statusPill.className = "status-pill loading";
  statusPill.title = "Loading model...";
  modelLoadBtn.disabled = true;

  try {
    const r = await fetch(new URL("/api/load", settings.serverUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel })
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || "Failed to load model");
    
    // Poll until backend service port responds
    let retries = 0;
    while (retries < 60) {
      await new Promise((res) => setTimeout(res, 1000));
      try {
        const sr = await fetch(new URL("/api/status", settings.serverUrl));
        const st = await sr.json();
        if (st.status === "running") break;
      } catch {}
      retries++;
    }
  } catch (err) {
    alert(`Load Error: ${err.message}`);
  } finally {
    modelLoadBtn.disabled = false;
    checkStatus();
  }
});

modelStopBtn.addEventListener("click", async () => {
  try {
    await fetch(new URL("/api/stop", settings.serverUrl), { method: "POST" });
  } catch {}
  checkStatus();
});

function buildMessages(userText) {
  const msgs = [];
  if (settings.systemPrompt) msgs.push({ role: "system", content: settings.systemPrompt });
  for (const m of conversation) msgs.push({ role: m.role, content: m.content });
  msgs.push({ role: "user", content: userText });
  return msgs;
}

async function sendMessage(userText) {
  conversation.push({ role: "user", content: userText });
  saveJSON(LS_CONV, conversation);
  renderMessages();

  const assistantBubble = bubbleFor("assistant", "");
  emptyHint.hidden = true;
  messagesEl.appendChild(assistantBubble);
  scrollToEnd();

  const controller = new AbortController();
  pendingAbort = controller;
  pendingResponseId = null;
  setBusy(true);

  let text = "";
  let tokenCount = 0;
  let startTime = performance.now();
  let ttft = 0;

  telemetryBadge.textContent = "0 tok/s | TTFT: --";

  try {
    const resp = await fetch(new URL("/v1/chat/completions", settings.serverUrl), {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: buildMessages(userText),
        stream: true,
        temperature: settings.temperature,
        max_tokens: Number(settings.maxTokens) || 512,
        top_p: Number(settings.topP), top_k: Number(settings.topK), min_p: Number(settings.minP),
        ...(settings.seed === null || settings.seed === "" ? {} : { seed: Number(settings.seed) }),
        frequency_penalty: Number(settings.frequencyPenalty), presence_penalty: Number(settings.presencePenalty),
        ...(settings.stopSequences ? { stop: settings.stopSequences.split(",").map((s) => s.trim()).filter(Boolean) } : {}),
        spec: Boolean(settings.spec), hidden: Boolean(settings.hidden),
        ...(settings.logprobs ? { logprobs: true } : {}),
        chat_template_kwargs: { enable_thinking: Boolean(settings.thinking) },
        ...(settings.jsonSchema ? { response_format: { type: "json_schema", json_schema: { name: "response", schema: JSON.parse(settings.jsonSchema) } } } : {}),
      }),
    });

    if (!resp.ok || !resp.body) {
      const detail = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${detail.slice(0, 300)}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      if (tokenCount === 0) {
        ttft = Math.round(performance.now() - startTime);
      }

      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop() ?? "";

      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;

        let obj;
        try {
          obj = JSON.parse(payload);
        } catch {
          continue;
        }

        if (obj.id) pendingResponseId = obj.id;
        const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
        if (delta && typeof delta.content === "string") {
          text += delta.content;
          tokenCount++;
          
          const elapsedSec = (performance.now() - startTime) / 1000;
          const tokSec = (tokenCount / Math.max(elapsedSec, 0.001)).toFixed(1);
          telemetryBadge.textContent = `${tokSec} tok/s | TTFT: ${ttft}ms`;

          assistantBubble.innerHTML = parseMarkdown(text);
          scrollToEnd();
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      assistantBubble.innerHTML = parseMarkdown(text || `*(error: ${err.message})*`);
    }
  } finally {
    pendingAbort = null;
    setBusy(false);
  }

  conversation.push({ role: "assistant", content: text });
  saveJSON(LS_CONV, conversation);
}

async function stopGeneration() {
  if (!pendingAbort) return;
  pendingAbort.abort();
  if (pendingResponseId) {
    try {
      await fetch(new URL("/v1/cancel", settings.serverUrl), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id: pendingResponseId }),
      });
    } catch {
      /* best effort */
    }
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (pendingAbort) {
    stopGeneration();
    return;
  }
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";
  autoGrow();
  sendMessage(text);
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function autoGrow() {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 140)}px`;
}
promptEl.addEventListener("input", autoGrow);

newChatBtn.addEventListener("click", () => {
  if (pendingAbort) stopGeneration();
  conversation = [];
  saveJSON(LS_CONV, conversation);
  renderMessages();
  telemetryBadge.textContent = "0 tok/s";
});

function openSettings() {
  serverUrlEl.value = settings.serverUrl;
  temperatureEl.value = settings.temperature;
  temperatureVal.textContent = settings.temperature;
  maxTokensEl.value = settings.maxTokens;
  field("top-p").value = settings.topP; field("top-k").value = settings.topK; field("min-p").value = settings.minP;
  field("seed").value = settings.seed ?? ""; field("stop-sequences").value = settings.stopSequences;
  field("frequency-penalty").value = settings.frequencyPenalty; field("presence-penalty").value = settings.presencePenalty;
  field("spec").checked = settings.spec; field("thinking").checked = settings.thinking; field("hidden").checked = settings.hidden;
  field("logprobs").checked = settings.logprobs; field("json-schema").value = settings.jsonSchema;
  systemPromptEl.value = settings.systemPrompt;
  settingsBackdrop.hidden = false;
}

function closeSettingsAndSave() {
  settings.serverUrl = serverUrlEl.value.trim() || DEFAULT_SETTINGS.serverUrl;
  settings.temperature = Number(temperatureEl.value);
  settings.maxTokens = Number(maxTokensEl.value) || 512;
  settings.topP = Number(field("top-p").value); settings.topK = Number(field("top-k").value); settings.minP = Number(field("min-p").value);
  settings.seed = field("seed").value === "" ? null : Number(field("seed").value); settings.stopSequences = field("stop-sequences").value;
  settings.frequencyPenalty = Number(field("frequency-penalty").value); settings.presencePenalty = Number(field("presence-penalty").value);
  settings.spec = field("spec").checked; settings.thinking = field("thinking").checked; settings.hidden = field("hidden").checked;
  settings.logprobs = field("logprobs").checked; settings.jsonSchema = field("json-schema").value.trim();
  settings.systemPrompt = systemPromptEl.value;
  saveJSON(LS_SETTINGS, settings);
  settingsBackdrop.hidden = true;
  fetchModels();
  checkStatus();
}

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettingsAndSave);
settingsBackdrop.addEventListener("click", (e) => {
  if (e.target === settingsBackdrop) closeSettingsAndSave();
});
temperatureEl.addEventListener("input", () => {
  temperatureVal.textContent = temperatureEl.value;
});
settingsClear.addEventListener("click", () => {
  conversation = [];
  saveJSON(LS_CONV, conversation);
  renderMessages();
  settingsBackdrop.hidden = true;
  telemetryBadge.textContent = "0 tok/s";
});

// Voice input (P3a): PCM WAV -> POST /v1/audio/transcriptions.
// Hidden outside a secure context, since getUserMedia requires one.
if (window.isSecureContext && navigator.mediaDevices && (window.AudioContext || window.webkitAudioContext)) {
  micBtn.hidden = false;
  let recording = null;

  function wavBlob(samples, rate) {
    const b = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(b);
    const text = (at, s) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
    text(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
    text(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    text(36, "data"); v.setUint32(40, samples.length * 2, true);
    samples.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
    return new Blob([b], { type: "audio/wav" });
  }

  micBtn.addEventListener("click", async () => {
    if (recording) {
      recording.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const Context = window.AudioContext || window.webkitAudioContext;
      const context = new Context();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const sink = context.createGain();
      const chunks = [];
      sink.gain.value = 0;
      processor.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      source.connect(processor); processor.connect(sink); sink.connect(context.destination);
      recording = { stop() {
        const rate = context.sampleRate;
        source.disconnect(); processor.disconnect(); sink.disconnect(); stream.getTracks().forEach((t) => t.stop());
        context.close(); recording = null; micBtn.classList.remove("recording");
        const samples = new Float32Array(chunks.reduce((n, x) => n + x.length, 0));
        chunks.reduce((at, x) => (samples.set(x, at), at + x.length), 0);
        const form = new FormData();
        form.append("file", wavBlob(samples, rate), "voice.wav");
        form.append("response_format", "text");
        void (async () => { try {
          const r = await fetch(new URL("/v1/audio/transcriptions", settings.serverUrl), {
            method: "POST",
            body: form,
          });
          const text = await r.text();
          if (r.ok && text) {
            promptEl.value = (promptEl.value ? `${promptEl.value} ` : "") + text.trim();
            promptEl.dispatchEvent(new Event("input"));
            promptEl.focus();
          }
        } catch {
          /* transcription unavailable; the typed path still works */
        }})();
      }};
      micBtn.classList.add("recording");
    } catch {
      /* mic permission denied or unavailable */
    }
  });
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/web/sw.js", { scope: "/" }).catch(() => {});
}

// Initial Setup
renderMessages();
fetchModels();
checkStatus();
setInterval(checkStatus, 5000);
