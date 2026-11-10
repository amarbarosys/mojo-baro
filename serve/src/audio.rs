//! P3a speech-in sidecar: `POST /v1/audio/transcriptions` proxied to a
//! `whisper-server` child, started on demand and stopped after idle.
//! whisper-server binds its port only after its model is loaded
//! (`examples/server/server.cpp`: `whisper_init_from_file_with_params`
//! before `listen_after_bind`), so a successful TCP connect is a valid
//! readiness check. The child is driven as a plain HTTP peer on loopback
//! (`POST /inference`, multipart in, JSON out), not over stdio like the LLM
//! engine in `engine.rs`.

use super::*;
use axum::extract::Multipart;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

static TTS_SEQ: AtomicU64 = AtomicU64::new(1);

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.into())
}

fn env_home(key: &str, rel: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| format!("{}/{}", std::env::var("HOME").unwrap_or_default(), rel))
}

fn env_num<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

pub struct AudioSidecar {
    bin: String,
    model: String,
    language: String,
    beam: u32,
    threads: u32,
    port: u16,
    idle_secs: u64,
    /// P3a preflight (`docs/PLATFORM-PLAN.md`): one clip through the
    /// endpoint on CPU with a small model, before the real GPU gate. When
    /// set, the sidecar is not wrapped in `gpu-wait` at all -- it is not a
    /// GPU workload.
    no_gpu: bool,
    /// Absolute path, not a bare name: a `gpu-wait run` job does not put
    /// `~/.local/bin` on the child's `PATH` (found live in the P3a timed
    /// gate, `bench/p3a-gate.sh` line 54's own nested call hit the same
    /// thing), so a bare `Command::new("gpu-wait")` fails inside one.
    gpu_wait_bin: String,
    tts_bin: String,
    tts_voice: String,
    tts_out_dir: String,
    tts_timeout_secs: u64,
    child: tokio::sync::Mutex<Option<Child>>,
    last_used: AtomicU64,
    /// Held across a whole start (spawn plus the wait for the port): a concurrent caller waits here and then finds
    /// the sidecar ready, instead of seeing a spawned child and connecting to a port that is not open yet.
    startup: tokio::sync::Mutex<()>,
    /// Calls in flight: the reaper never kills the sidecar under one (a call longer than `idle_secs` was killed).
    busy: std::sync::atomic::AtomicUsize,
    /// One reaper task at a time (each fresh spawn used to start another one).
    reaper_on: std::sync::atomic::AtomicBool,
}

/// A call's hold on the sidecar: counts it busy, and re-stamps `last_used` when the call ends.
pub struct Busy {
    app: Shared,
}

impl Busy {
    fn hold(app: &Shared) -> Busy {
        app.audio.busy.fetch_add(1, Ordering::SeqCst);
        app.audio.last_used.store(now(), Ordering::SeqCst);
        Busy { app: app.clone() }
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        self.app.audio.busy.fetch_sub(1, Ordering::SeqCst);
        self.app.audio.last_used.store(now(), Ordering::SeqCst);
    }
}

impl AudioSidecar {
    pub fn from_env() -> AudioSidecar {
        AudioSidecar {
            bin: env_home("BARO_WHISPER_BIN", "Models/whisper.cpp/build/bin/whisper-server"),
            model: env_home("BARO_WHISPER_MODEL", "Models/whisper/ggml-large-v3-turbo-q5_0.bin"),
            language: env_str("BARO_WHISPER_LANGUAGE", "en"),
            beam: env_num("BARO_WHISPER_BEAM", 5u32),
            threads: env_num("BARO_WHISPER_THREADS", 8u32),
            port: env_num("BARO_WHISPER_PORT", 8090u16),
            idle_secs: env_num("BARO_WHISPER_IDLE_SECS", 300u64),
            no_gpu: env_str("BARO_WHISPER_NO_GPU", "0") == "1",
            gpu_wait_bin: env_home("BARO_GPU_WAIT", ".local/bin/gpu-wait"),
            tts_bin: env_home("BARO_TTS_BIN", "Projects/ai-models/tts-local/scripts/tts-piper"),
            tts_voice: env_str("BARO_TTS_VOICE", "en_US-lessac-medium"),
            tts_out_dir: env_str("BARO_TTS_OUT_DIR", ".work/baro-tts"),
            tts_timeout_secs: env_num("BARO_TTS_TIMEOUT_SECS", 120u64),
            child: tokio::sync::Mutex::new(None),
            last_used: AtomicU64::new(0),
            startup: tokio::sync::Mutex::new(()),
            busy: std::sync::atomic::AtomicUsize::new(0),
            reaper_on: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Kill the sidecar if one is running. Explicit, not left to
    /// `kill_on_drop`: a background reaper task (`reap_when_idle`) holds its
    /// own `Shared` clone for as long as the sidecar has been used even
    /// once, so `App`'s refcount never reaches zero on its own at shutdown,
    /// and `kill_on_drop` never fires. Found live: a plain `kill <pid>`
    /// (SIGTERM) on `baro-serve` terminated it via the kernel's default
    /// disposition -- no handler installed, no unwind, no `Drop` -- and left
    /// whisper-server orphaned under systemd, holding ~24.5 GB of VRAM after
    /// the P3a timed gate's own cleanup trap believed it had killed it.
    pub async fn shutdown(&self) {
        let child = self.child.lock().await.take();
        let Some(mut child) = child else { return };
        let _ = child.kill().await;
    }
}

/// Start the sidecar if it is not already up (or has exited), wait for its
/// port to accept connections, and arm the idle reaper. A resident, not a
/// one-shot job: spawned through `gpu-wait run --shared`, the daemon's own
/// pattern for a long-lived service (`gpuwaitingroom` README, "Resident
/// service"). `gpu-wait run` drops the shell env, so a job's `PATH` does not
/// carry `~/.local/bin` -- `a.gpu_wait_bin` is always an absolute path,
/// never a bare name (found live: `bench/p3a-gate.sh`'s own nested snapshot
/// call hit the same PATH gap). When `GPU_WAITING_ROOM_JOB` is already set,
/// this process is itself running inside a job, so it spawns the sidecar
/// bare instead of calling `gpu-wait run` again -- nesting would queue the
/// sidecar behind the job admitting it.
async fn ensure_running(app: &Shared) -> Result<Busy, String> {
    let a = &app.audio;
    let _startup = a.startup.lock().await;
    {
        let mut guard = a.child.lock().await;
        if let Some(child) = guard.as_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                return Ok(Busy::hold(app));
            }
            *guard = None; // exited: fall through and respawn
        }
        let mut whisper_args = vec![
            a.bin.clone(),
            "-m".into(),
            a.model.clone(),
            "-l".into(),
            a.language.clone(),
            "-bs".into(),
            a.beam.to_string(),
            "-t".into(),
            a.threads.to_string(),
            // Matches the reference `whisper-cli -nt` in bench/p3a-gate.sh:
            // enabling timestamps changes whisper.cpp's segment text join,
            // dropping punctuation at segment boundaries (P3a CPU preflight,
            // isolated by diffing a direct call with and without this flag).
            "-nt".into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            a.port.to_string(),
        ];
        if a.no_gpu {
            whisper_args.push("-ng".into());
        }
        let job_id = std::env::var("GPU_WAITING_ROOM_JOB").ok();
        let mut cmd = if a.no_gpu {
            eprintln!("audio: whisper sidecar running bare (--no-gpu, not a GPU workload)");
            let mut c = Command::new(&whisper_args[0]);
            c.args(&whisper_args[1..]);
            c
        } else if let Some(job) = &job_id {
            eprintln!("audio: whisper sidecar running bare, already admitted (GPU_WAITING_ROOM_JOB={job})");
            let mut c = Command::new(&whisper_args[0]);
            c.args(&whisper_args[1..]);
            c
        } else {
            eprintln!("audio: whisper sidecar spawned via {} run --shared", a.gpu_wait_bin);
            let mut gpu_wait_args = vec!["run".to_string(), "--shared".into(), "--vram".into(), "4".into(), "--".into()];
            gpu_wait_args.extend(whisper_args);
            let mut c = Command::new(&a.gpu_wait_bin);
            c.args(&gpu_wait_args);
            c
        };
        // PR_SET_PDEATHSIG covers what a SIGTERM handler in main() cannot:
        // a SIGKILL or an OOM kill on baro-serve gives it no chance to run
        // any handler at all, but the kernel still delivers SIGTERM to this
        // child directly the moment its parent dies, by any means.
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn whisper sidecar: {e}"))?;
        *guard = Some(child);
    }
    a.last_used.store(now(), Ordering::SeqCst);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        if TcpStream::connect(("127.0.0.1", a.port)).await.is_ok() {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            // Not listening: do not leave a live child behind for the next caller to mistake for a ready sidecar.
            if let Some(mut c) = a.child.lock().await.take() {
                let _ = c.kill().await;
            }
            return Err("whisper sidecar did not open its port in time".into());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    if a.reaper_on.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
        tokio::spawn(reap_when_idle(app.clone()));
    }
    Ok(Busy::hold(app))
}

async fn reap_when_idle(app: Shared) {
    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;
        let idle = now().saturating_sub(app.audio.last_used.load(Ordering::SeqCst));
        if idle < app.audio.idle_secs || app.audio.busy.load(Ordering::SeqCst) > 0 {
            continue;
        }
        let mut guard = app.audio.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
            eprintln!("audio: whisper sidecar stopped after {idle}s idle");
        }
        app.audio.reaper_on.store(false, Ordering::SeqCst);
        return;
    }
}

fn multipart_field(boundary: &str, name: &str, value: &str) -> Vec<u8> {
    format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").into_bytes()
}

fn multipart_file(boundary: &str, name: &str, filename: &str, content_type: &str, data: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
    )
    .into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    out
}

fn parse_http_response(resp: &[u8]) -> Result<(StatusCode, String, Vec<u8>), String> {
    let sep = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("whisper sidecar: malformed response (no header terminator)")?;
    let head = std::str::from_utf8(&resp[..sep]).map_err(|_| "whisper sidecar: non-utf8 headers")?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or("whisper sidecar: empty response")?;
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("whisper sidecar: bad status line")?;
    let status = StatusCode::from_u16(code).map_err(|_| "whisper sidecar: invalid status code")?;
    let content_type = lines
        .find_map(|l| l.to_ascii_lowercase().starts_with("content-type:").then(|| l[13..].trim().to_string()))
        .unwrap_or_else(|| "application/octet-stream".into());
    Ok((status, content_type, resp[sep + 4..].to_vec()))
}

/// One multipart request to whisper-server's `/inference`, `Connection:
/// close` so reading to EOF is a valid way to collect the whole response
/// (a local loopback peer, not a keep-alive pool worth building here).
async fn forward(app: &Shared, boundary: &str, body: Vec<u8>) -> Result<(StatusCode, String, Vec<u8>), String> {
    let port = app.audio.port;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.map_err(|e| format!("connect whisper sidecar: {e}"))?;
    let mut head = format!(
        "POST /inference HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    head.extend_from_slice(&body);
    stream.write_all(&head).await.map_err(|e| format!("write to whisper sidecar: {e}"))?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.map_err(|e| format!("read from whisper sidecar: {e}"))?;
    parse_http_response(&resp)
}

pub async fn transcriptions(State(app): State<Shared>, mut multipart: Multipart) -> Result<Response, ApiError> {
    let mut file: Option<(String, String, Vec<u8>)> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut temperature: Option<String> = None;
    loop {
        let field = multipart.next_field().await.map_err(|e| bad(format!("multipart: {e}")))?;
        let Some(field) = field else { break };
        match field.name().unwrap_or("") {
            "file" => {
                let filename = field.file_name().unwrap_or("audio.wav").to_string();
                let content_type = field.content_type().unwrap_or("audio/wav").to_string();
                let data = field.bytes().await.map_err(|e| bad(format!("reading file field: {e}")))?.to_vec();
                file = Some((filename, content_type, data));
            }
            "language" => language = field.text().await.ok(),
            "response_format" => response_format = field.text().await.ok(),
            "temperature" => temperature = field.text().await.ok(),
            _ => {}
        }
    }
    let (filename, content_type, data) = file.ok_or_else(|| bad("multipart field \"file\" is required"))?;

    let boundary = format!("bAr0{}", now());
    let mut body = Vec::new();
    if let Some(l) = &language {
        body.extend(multipart_field(&boundary, "language", l));
    }
    if let Some(f) = &response_format {
        body.extend(multipart_field(&boundary, "response_format", f));
    }
    if let Some(t) = &temperature {
        body.extend(multipart_field(&boundary, "temperature", t));
    }
    body.extend(multipart_file(&boundary, "file", &filename, &content_type, &data));
    body.extend(format!("--{boundary}--\r\n").into_bytes());

    let _busy = ensure_running(&app).await.map_err(|e| ApiError::Plain(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let (status, resp_content_type, resp_body) =
        forward(&app, &boundary, body).await.map_err(|e| ApiError::Plain(StatusCode::BAD_GATEWAY, e))?;

    Ok((status, [(axum::http::header::CONTENT_TYPE, resp_content_type)], resp_body).into_response())
}

fn default_tts_voice() -> String {
    "en_US-lessac-medium".into()
}

fn default_tts_format() -> String {
    "wav".into()
}

fn default_tts_speed() -> f32 {
    1.0
}

#[derive(Deserialize)]
pub struct SpeechReq {
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    input: String,
    #[serde(default = "default_tts_voice")]
    voice: String,
    #[serde(default = "default_tts_format")]
    response_format: String,
    #[serde(default = "default_tts_speed")]
    speed: f32,
}

pub async fn speech(State(app): State<Shared>, Json(r): Json<SpeechReq>) -> Result<Response, ApiError> {
    if r.input.trim().is_empty() {
        return Err(bad("input is empty"));
    }
    if r.response_format != "wav" {
        return Err(bad("only response_format=wav is supported"));
    }
    if (r.speed - 1.0).abs() > f32::EPSILON {
        return Err(bad("only speed=1 is supported"));
    }
    let voice = if r.voice.is_empty() { app.audio.tts_voice.clone() } else { r.voice };
    tokio::fs::create_dir_all(&app.audio.tts_out_dir).await
        .map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("create TTS output directory: {e}")))?;
    let seq = TTS_SEQ.fetch_add(1, Ordering::Relaxed);
    let output = std::path::Path::new(&app.audio.tts_out_dir)
        .join(format!("speech-{}-{seq}.wav", std::process::id()));
    let result = tokio::time::timeout(
        Duration::from_secs(app.audio.tts_timeout_secs),
        Command::new(&app.audio.tts_bin).arg(&r.input).arg(&voice).arg(&output).output(),
    )
    .await
    .map_err(|_| ApiError::Plain(StatusCode::GATEWAY_TIMEOUT, "TTS timed out".into()))?
    .map_err(|e| ApiError::Plain(StatusCode::SERVICE_UNAVAILABLE, format!("start TTS runner: {e}")))?;
    if !result.status.success() {
        let _ = tokio::fs::remove_file(&output).await;
        return Err(ApiError::Plain(StatusCode::BAD_GATEWAY, format!("TTS runner failed: {}", String::from_utf8_lossy(&result.stderr))));
    }
    let body = tokio::fs::read(&output).await
        .map_err(|e| ApiError::Plain(StatusCode::BAD_GATEWAY, format!("read TTS output: {e}")))?;
    let _ = tokio::fs::remove_file(&output).await;
    if body.is_empty() {
        return Err(ApiError::Plain(StatusCode::BAD_GATEWAY, "TTS runner returned an empty WAV".into()));
    }
    Ok((StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "audio/wav")], body).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    fn stub_app(port: u16) -> Shared {
        let mut a = AudioSidecar::from_env();
        a.bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stub_whisper.sh").display().to_string();
        a.no_gpu = true;
        a.port = port;
        let mut app = Arc::try_unwrap(crate::sse_tests::test_app()).ok().expect("fresh app");
        app.audio = a;
        Arc::new(app)
    }

    /// A-M9: a second caller that arrived while the first was still waiting for the port found a spawned child, returned
    /// "ready" at once and its call hit a closed port (a 502 on the first two audio calls).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_first_calls_both_find_the_port_open() {
        let port = free_port();
        let app = stub_app(port);
        let probe = |app: Shared| async move {
            let busy = ensure_running(&app).await?;
            TcpStream::connect(("127.0.0.1", port)).await.map_err(|e| format!("port closed after ensure_running: {e}"))?;
            Ok::<Busy, String>(busy)
        };
        let (r1, r2) = tokio::join!(probe(app.clone()), probe(app.clone()));
        let (b1, b2) = (r1.expect("first call"), r2.expect("second call"));
        assert_eq!(app.audio.busy.load(Ordering::SeqCst), 2, "both calls count as busy");
        drop(b1);
        drop(b2);
        assert_eq!(app.audio.busy.load(Ordering::SeqCst), 0);
        assert!(app.audio.reaper_on.load(Ordering::SeqCst), "one reaper runs");
        app.audio.shutdown().await;
    }

    /// A-M9: the reaper never kills the sidecar under a call that is still running.
    #[tokio::test(flavor = "current_thread")]
    async fn a_busy_call_holds_the_idle_clock() {
        let app = stub_app(free_port());
        app.audio.last_used.store(1, Ordering::SeqCst);
        let b = Busy::hold(&app);
        assert!(app.audio.last_used.load(Ordering::SeqCst) > 1, "a call stamps last_used when it starts");
        app.audio.last_used.store(1, Ordering::SeqCst);
        drop(b);
        assert!(app.audio.last_used.load(Ordering::SeqCst) > 1, "and when it ends, so a long call is not idle");
    }
}
