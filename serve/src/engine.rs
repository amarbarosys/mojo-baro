//! The engine child process: one `serve/engine.mojo` running with
//! `BARO_SERVE=1`, driven over its stdin/stdout by a single worker task.
//! The engine advances one request at a time; the worker admits later lines
//! while routing each response event by request id.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{mpsc, watch};

use crate::protocol::{cancel_line, parse_line, DoneStats, EngineMsg, Request, SampleParams};

/// What a request's consumer sees, in order: zero or more tokens, then
/// exactly one `Done` or `Error`.
#[derive(Debug, Clone)]
pub enum Event {
    Tok { tok: u32, logprob: Option<f64>, top_logprobs: Vec<(u32, f64)> },
    /// P0a-e: the requested embedding vector, sent once, before `Done`.
    Embed(Vec<f32>),
    Hidden(Vec<f32>),
    LogitsTopK(Vec<(u32, f64)>),
    /// Latent fork/resume response with the created/resumed latent_id.
    #[allow(dead_code)]
    Latent { latent_id: String, tokens: Option<u32> },
    /// T3b: the engine's answer to an IPC op; terminal, like `Done`.
    Ipc(serde_json::Value),
    Done(Box<DoneStats>),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub tmax: u32,
    pub mrows: u32,
    pub kmax: u32,
    pub spec_k: u32,
    pub pack: String,
    pub kv: String,
    /// From the ready line when the engine sends them (wire seam).
    pub tier: Option<String>,
    pub seqs: Option<u32>,
}

struct Job {
    req: Request,
    out: mpsc::UnboundedSender<Event>,
}

/// The one phase word of an engine child. Written only through `StateCell::set`, read by `submit`, `/health` and `/api/ps`.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    Starting,
    Idle,
    /// Requests on the wire right now.
    Busy(u32),
    /// The child is gone; the reason carries its exit status or the pipe error.
    Dead(String),
    /// A graceful stop is closing stdin and waiting for the child.
    Stopping,
}

impl EngineState {
    pub fn word(&self) -> &'static str {
        match self {
            EngineState::Starting => "starting",
            EngineState::Idle => "idle",
            EngineState::Busy(_) => "busy",
            EngineState::Dead(_) => "dead",
            EngineState::Stopping => "stopping",
        }
    }

    pub fn is_live(&self) -> bool {
        matches!(self, EngineState::Idle | EngineState::Busy(_))
    }

    /// The reason, for a dead engine.
    pub fn reason(&self) -> Option<&str> {
        if let EngineState::Dead(r) = self { Some(r) } else { None }
    }
}

/// The state plus the single transition function: logs `engine <label>: OLD -> NEW (why)` for every change
/// except a flip inside the live states (Idle/Busy(n) change on every request).
pub struct StateCell {
    label: String,
    v: std::sync::Mutex<EngineState>,
}

impl StateCell {
    fn new(label: String, v: EngineState) -> StateCell {
        StateCell { label, v: std::sync::Mutex::new(v) }
    }

    pub fn get(&self) -> EngineState {
        self.v.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set(&self, new: EngineState, why: &str) {
        let mut g = self.v.lock().unwrap_or_else(|e| e.into_inner());
        if *g == new {
            return;
        }
        // Dead and Stopping are final for this child: nothing moves it back to a live state.
        if matches!(*g, EngineState::Dead(_)) || (*g == EngineState::Stopping && new.is_live()) {
            return;
        }
        if !(g.is_live() && new.is_live()) {
            eprintln!("engine {}: {} -> {} ({why})", self.label, g.word(), new.word());
        }
        *g = new;
    }
}

pub struct Engine {
    tx: mpsc::Sender<Job>,
    queued: Arc<AtomicUsize>,
    state: Arc<StateCell>,
    /// The request id the engine is actively decoding, if any; set/cleared
    /// by the worker around `run_one`. `cancel` reads it to decide whether a
    /// cancel would land on anything, and the worker alone still owns
    /// `stdin` -- no sharing needed to write a cancel line, since the worker
    /// writes it itself once `cancel_tx` wakes its select loop.
    current: Arc<tokio::sync::Mutex<Option<u64>>>,
    cancel_tx: mpsc::UnboundedSender<u64>,
    pub limits: Limits,
    /// `shutdown` flips it; the worker (which owns the child and stdin) closes stdin and reaps the child.
    stop: watch::Sender<bool>,
    worker: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// How long the engine may take to print its ready line (pack load).
const READY_TIMEOUT: Duration = Duration::from_secs(600);
/// Requests waiting beyond this many are refused with 503.
const QUEUE_CAP: usize = 64;

impl Engine {
    /// Spawn the engine, wait for its ready line, start the worker.
    /// The child inherits this process's environment (BARO_* knobs pass
    /// through); `BARO_SERVE=1` and `BARO_PACK` are set here. `extra_env`
    /// applies on top (P4: `ROCR_VISIBLE_DEVICES`/`HSA_OVERRIDE_GFX_VERSION`
    /// for a device pin explicit in baro-serve's own args, not only in
    /// whatever launched it -- the same list for every engine this call
    /// spawns, since `EnginePool` uses it pool-wide, one device per
    /// process), each entry a `(KEY, VALUE)` pair; an empty value unsets
    /// the key instead of setting it (`igpu-env` needs `HIP_VISIBLE_DEVICES`
    /// unset, not set to empty).
    pub async fn spawn(engine: &Path, pack: &Path, extra_env: &[(String, String)]) -> Result<Engine, String> {
        let mut cmd = tokio::process::Command::new(engine);
        cmd.env("BARO_SERVE", "1").env("BARO_PACK", pack);
        if std::env::var_os("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT").is_none() {
            cmd.env("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "2");
        }
        for (k, v) in extra_env {
            if v.is_empty() {
                cmd.env_remove(k);
            } else {
                cmd.env(k, v);
            }
        }
        // The engine dies with baro-serve, even a SIGKILL of it while the engine is stuck mid-decode and never
        // reads the stdin EOF: an orphaned engine holds the card's VRAM (bench/ailab/serve-selftest.sh kill-stuck).
        // PDEATHSIG fires when the spawning THREAD exits, not the process: spawn only from main's block_on, never spawn_blocking.
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: only async-signal-safe calls (prctl, getppid) run between fork and exec.
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("baro-serve exited during engine spawn"));
                }
                Ok(())
            });
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", engine.display()))?;
        // P4: the GPU work happens in THIS child, not in baro-serve itself --
        // a preflight matching rocm-smi PIDs against baro-serve's own pid
        // checks the wrong process. Log the child's pid so a gate script can
        // grep it instead.
        let label = child.id().map(|p| format!("pid {p}")).unwrap_or_else(|| "pid unknown".into());
        eprintln!("engine child pid: {} ({})", child.id().map(|p| p.to_string()).unwrap_or_else(|| "unknown".into()), engine.display());
        let stdin = child.stdin.take().ok_or("engine stdin not piped")?;
        let stdout = child.stdout.take().ok_or("engine stdout not piped")?;
        let mut lines = BufReader::new(stdout).lines();

        let limits = tokio::time::timeout(READY_TIMEOUT, async {
            loop {
                match lines.next_line().await {
                    Ok(Some(l)) => match parse_line(&l) {
                        EngineMsg::Ready(ready) => {
                            return Ok(Limits {
                                tmax: ready.tmax,
                                mrows: ready.mrows,
                                kmax: ready.kmax,
                                spec_k: ready.spec_k,
                                pack: ready.pack,
                                kv: ready.kv,
                                tier: ready.tier,
                                seqs: ready.seqs,
                            })
                        }
                        EngineMsg::Log(s) => eprintln!("engine: {s}"),
                        other => eprintln!("engine: unexpected before ready: {other:?}"),
                    },
                    Ok(None) => return Err("engine exited before its ready line".to_string()),
                    Err(e) => return Err(format!("engine stdout: {e}")),
                }
            }
        })
        .await
        .map_err(|_| "engine did not become ready in time".to_string())??;

        let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAP);
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel::<u64>();
        let (stop, stop_rx) = watch::channel(false);
        let queued = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(StateCell::new(label, EngineState::Starting));
        state.set(EngineState::Idle, "ready line");
        let current = Arc::new(tokio::sync::Mutex::new(None));
        let handle = tokio::spawn(worker(rx, stdin, lines.into_inner(), child, queued.clone(), state.clone(), cancel_rx, current.clone(), stop_rx));
        Ok(Engine { tx, queued, state, current, cancel_tx, limits, stop, worker: tokio::sync::Mutex::new(Some(handle)) })
    }

    pub fn alive(&self) -> bool {
        self.state.get().is_live()
    }

    pub fn state(&self) -> EngineState {
        self.state.get()
    }

    /// Requests waiting or running.
    pub fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    /// Queue one request under the given id (assigned by the caller --
    /// `EnginePool` owns a single id counter shared by every engine in the
    /// pool, so ids stay unique across engines). Returns a receiver of its
    /// events; dropping it without cancelling discards the rest of that
    /// request's output, the engine still runs it to completion.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &self, id: u64, prompt: Vec<u32>, n: u32, spec: bool, stop: Vec<Vec<u32>>, ckpt: Vec<u32>, sample: SampleParams,
        schema: Option<serde_json::Value>, reasoning: Option<bool>,
        state: (Option<String>, Option<String>), state_prefix: bool, embed: Option<bool>, embeds: Vec<String>,
    ) -> Result<mpsc::UnboundedReceiver<Event>, String> {
        if !self.alive() {
            return Err("engine process has exited".into());
        }
        let (out, rx) = mpsc::unbounded_channel();
        let job = Job {
            req: Request { id, prompt, n, spec, stop, ckpt, state_save: state.0, state_load: state.1, state_prefix, sample, schema, reasoning, embed, embeds, latent_fork: None, latent_resume: None, ipc: Default::default() },
            out,
        };
        self.queued.fetch_add(1, Ordering::SeqCst);
        match self.tx.try_send(job) {
            Ok(()) => Ok(rx),
            Err(_) => {
                self.queued.fetch_sub(1, Ordering::SeqCst);
                Err(format!("queue full ({QUEUE_CAP} requests waiting)"))
            }
        }
    }

    /// T3b: a prepared request line (IPC ops, the IPC-ingest fork); `req.id` is set here.
    pub fn submit_req(&self, id: u64, mut req: Request) -> Result<mpsc::UnboundedReceiver<Event>, String> {
        if !self.alive() {
            return Err("engine process has exited".into());
        }
        req.kind().map_err(str::to_string)?;
        req.id = id;
        let (out, rx) = mpsc::unbounded_channel();
        self.queued.fetch_add(1, Ordering::SeqCst);
        match self.tx.try_send(Job { req, out }) {
            Ok(()) => Ok(rx),
            Err(_) => {
                self.queued.fetch_sub(1, Ordering::SeqCst);
                Err(format!("queue full ({QUEUE_CAP} requests waiting)"))
            }
        }
    }

    /// Cancel `id` if the engine is actively decoding it right now. Returns
    /// `false` for a request that has not started (still queued), already
    /// finished, or does not exist -- cancelling a queued-but-not-yet-running
    /// request is not supported (brief CHAT/C1: not in this step).
    pub async fn cancel(&self, id: u64) -> bool {
        if *self.current.lock().await != Some(id) {
            return false;
        }
        self.cancel_tx.send(id).is_ok()
    }

    /// Close the engine's stdin (it exits at EOF) and reap it. The worker owns both: it finishes the requests on the
    /// wire, closes stdin, waits up to 30 s for the child, then kills it.
    pub async fn shutdown(&self) {
        let handle = self.worker.lock().await.take();
        let Some(mut handle) = handle else { return };
        self.state.set(EngineState::Stopping, "shutdown");
        let _ = self.stop.send(true);
        if tokio::time::timeout(Duration::from_secs(40), &mut handle).await.is_err() {
            eprintln!("engine worker did not stop in time; killing the child");
            handle.abort();
        }
    }
}

/// `BARO_POOL` engine processes (default 1 = today's single engine), each
/// with its own pack load and its own request queue. `submit` routes to the
/// engine with the fewest requests waiting or running (ties -> lowest
/// index), so at `BARO_POOL=1` this is exactly today's single-queue
/// behaviour; at `BARO_POOL=2` a second request starts on the other engine
/// instead of waiting behind the first. One id counter, owned by the pool
/// and never by an `Engine`, so ids stay unique across engines (`cancel`
/// and the API's `cmpl-<id>`/`chatcmpl-<id>` depend on that).
pub struct EnginePool {
    engines: Vec<Engine>,
    next_id: AtomicU64,
    pub limits: Limits,
}

impl EnginePool {
    /// No LLM engine (`--audio-only`, P3a): every text/decode route's existing
    /// `submit` error path answers 503 with no engine process spawned, no
    /// pack load, no VRAM held. `tmax` is `u32::MAX` so `check_and_submit`'s
    /// exceed-context check never intercepts the request first; `submit`
    /// itself is the thing that refuses.
    pub fn empty() -> EnginePool {
        EnginePool {
            engines: Vec::new(),
            next_id: AtomicU64::new(1),
            limits: Limits { tmax: u32::MAX, mrows: 0, kmax: 0, spec_k: 0, pack: "none (--audio-only)".into(), kv: "f32".into(), tier: None, seqs: None },
        }
    }

    pub async fn spawn(engine: &Path, pack: &Path, pool_size: usize, extra_env: &[(String, String)]) -> Result<EnginePool, String> {
        let mut engines = Vec::with_capacity(pool_size.max(1));
        for _ in 0..pool_size.max(1) {
            engines.push(Engine::spawn(engine, pack, extra_env).await?);
        }
        let limits = engines[0].limits.clone();
        Ok(EnginePool {
            engines,
            next_id: AtomicU64::new(1),
            limits,
        })
    }

    pub fn pool_size(&self) -> usize {
        self.engines.len()
    }

    pub fn alive(&self) -> bool {
        self.engines.iter().any(Engine::alive)
    }

    /// The pool's phase for `/health`: `none` (no engine, `--audio-only`), else the first live word
    /// (idle before busy), else the first non-live one.
    pub fn state_word(&self) -> &'static str {
        let st: Vec<EngineState> = self.engines.iter().map(Engine::state).collect();
        if st.is_empty() {
            return "none";
        }
        if st.iter().any(|s| *s == EngineState::Idle) {
            return "idle";
        }
        st.iter().find(|s| s.is_live()).or_else(|| st.first()).map(EngineState::word).unwrap_or("none")
    }

    /// One `{state, depth, reason}` per engine.
    pub fn engine_states(&self) -> Vec<serde_json::Value> {
        self.engines
            .iter()
            .map(|e| {
                let s = e.state();
                serde_json::json!({"state": s.word(), "depth": e.queue_depth(), "reason": s.reason()})
            })
            .collect()
    }

    /// Requests waiting or running, summed over every engine in the pool.
    pub fn queue_depth(&self) -> usize {
        self.engines.iter().map(Engine::queue_depth).sum()
    }

    /// Per-engine breakdown of `queue_depth`: lets the C4 gate tell "two
    /// engines each running one request" apart from "one engine running
    /// both while the other sits idle," which the summed `queue_depth`
    /// alone cannot.
    pub fn queue_depths(&self) -> Vec<usize> {
        self.engines.iter().map(Engine::queue_depth).collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &self, prompt: Vec<u32>, n: u32, spec: bool, stop: Vec<Vec<u32>>, ckpt: Vec<u32>, sample: SampleParams,
        schema: Option<serde_json::Value>, reasoning: Option<bool>,
        state: (Option<String>, Option<String>), state_prefix: bool, embed: Option<bool>, embeds: Vec<String>,
    ) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        if self.engines.is_empty() {
            return Err("no engine loaded (--audio-only)".into());
        }
        // A dead engine drains to depth 0 and would win the pick: only live engines compete (ties -> lowest index).
        let mut pick: Option<(usize, usize)> = None;
        for (i, e) in self.engines.iter().enumerate().filter(|(_, e)| e.alive()) {
            let q = e.queue_depth();
            if pick.is_none_or(|(_, best_q)| q < best_q) {
                pick = Some((i, q));
            }
        }
        let Some((best, _)) = pick else { return Err("engine process has exited".into()) };
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let rx = self.engines[best].submit(id, prompt, n, spec, stop, ckpt, sample, schema, reasoning, state, state_prefix, embed, embeds)?;
        Ok((id, rx))
    }

    /// T3b: IPC state lives in one engine process, so IPC lines always go to engine 0.
    pub fn submit_req(&self, req: Request) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        if self.engines.is_empty() {
            return Err("no engine loaded (--audio-only)".into());
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        Ok((id, self.engines[0].submit_req(id, req)?))
    }

    /// Tries every engine (a request's id does not say which one it landed
    /// on); at most one will have it as its `current` decode.
    pub async fn cancel(&self, id: u64) -> bool {
        for e in &self.engines {
            if e.cancel(id).await {
                return true;
            }
        }
        false
    }

    pub async fn shutdown(&self) {
        for e in &self.engines {
            e.shutdown().await;
        }
    }
}

async fn worker(
    mut rx: mpsc::Receiver<Job>,
    mut stdin: ChildStdin,
    mut reader: BufReader<ChildStdout>,
    mut child: Child,
    queued: Arc<AtomicUsize>,
    state: Arc<StateCell>,
    mut cancel_rx: mpsc::UnboundedReceiver<u64>,
    current: Arc<tokio::sync::Mutex<Option<u64>>>,
    mut stop_rx: watch::Receiver<bool>,
) {
    let mut active = HashMap::<u64, mpsc::UnboundedSender<Event>>::new();
    let mut order = VecDeque::<u64>::new();
    let mut line_buf = String::with_capacity(128);
    let mut dead: Option<String> = None;
    loop {
        // Idle: wait for a job, a stop, or the child's own death (an engine that dies while idle is seen now,
        // with its exit status, not at the next write).
        let job = tokio::select! {
            biased;
            _ = stop_rx.changed() => break,
            status = child.wait() => {
                dead = Some(format!("engine exited while idle ({})", status.map_or_else(|e| format!("wait failed: {e}"), |s| s.to_string())));
                break;
            }
            j = rx.recv() => match j { Some(j) => j, None => break },
        };
        // A request whose consumer is already gone never reaches the engine.
        if job.out.is_closed() {
            queued.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        active.clear();
        order.clear();
        let mut cancelled = std::collections::HashSet::<u64>::new();
        let mut rx_open = true;
        let mut error = None;
        let first_id = job.req.id;
        let first_line = job.req.line();
        let first_out = job.out;
        active.insert(first_id, first_out.clone());
        order.push_back(first_id);
        *current.lock().await = Some(first_id);
        state.set(EngineState::Busy(1), "request on the wire");
        if let Err(e) = write_line(&mut stdin, first_line).await {
            active.remove(&first_id);
            order.pop_front();
            let _ = first_out.send(Event::Error(e.clone()));
            queued.fetch_sub(1, Ordering::SeqCst);
            error = Some(e);
        }

        while error.is_none() && !active.is_empty() {
            line_buf.clear();
            // Set when a send to a request's consumer fails: the receiver was dropped (client gone).
            let mut dropped: Option<u64> = None;
            tokio::select! {
                maybe_job = rx.recv(), if rx_open => match maybe_job {
                    Some(job) => {
                        let id = job.req.id;
                        let line = job.req.line();
                        let out = job.out;
                        active.insert(id, out.clone());
                        order.push_back(id);
                        if let Some(running) = *current.lock().await {
                            eprintln!("engine: wire admitted request {id} while request {running} is in flight");
                        }
                        state.set(EngineState::Busy(active.len() as u32), "request admitted");
                        if let Err(e) = write_line(&mut stdin, line).await {
                            active.remove(&id);
                            order.retain(|queued_id| *queued_id != id);
                            let _ = out.send(Event::Error(e.clone()));
                            queued.fetch_sub(1, Ordering::SeqCst);
                            error = Some(e);
                        }
                    }
                    None => rx_open = false,
                },
                read_res = reader.read_line(&mut line_buf) => match read_res {
                    Ok(n) if n > 0 => match parse_line(&line_buf) {
                        EngineMsg::Tok { id, tok, logprob, top_logprobs } => {
                            if let Some(out) = active.get(&id) {
                                if out.send(Event::Tok { tok, logprob, top_logprobs }).is_err() {
                                    dropped = Some(id);
                                }
                            } else {
                                eprintln!("engine: line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Embed { id, vector } => {
                            if let Some(out) = active.get(&id) {
                                let _ = out.send(Event::Embed(vector));
                            } else {
                                eprintln!("engine: line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Hidden { id, vector } => {
                            if let Some(out) = active.get(&id) {
                                let _ = out.send(Event::Hidden(vector));
                            } else {
                                eprintln!("engine: line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::LogitsTopK { id, values } => {
                            if let Some(out) = active.get(&id) {
                                let _ = out.send(Event::LogitsTopK(values));
                            } else {
                                eprintln!("engine: line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Done { id, stats } => {
                            if let Some(out) = active.remove(&id) {
                                let _ = out.send(Event::Done(stats));
                                queued.fetch_sub(1, Ordering::SeqCst);
                                order.retain(|queued_id| *queued_id != id);
                                *current.lock().await = order.front().copied();
                            } else {
                                eprintln!("engine: terminal line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Ipc { id, value } => {
                            if let Some(out) = active.remove(&id) {
                                let _ = out.send(Event::Ipc(*value));
                                queued.fetch_sub(1, Ordering::SeqCst);
                                order.retain(|queued_id| *queued_id != id);
                                *current.lock().await = order.front().copied();
                            } else {
                                eprintln!("engine: ipc line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Error { id, error: message } => {
                            if let Some(out) = active.remove(&id) {
                                let _ = out.send(Event::Error(message));
                                queued.fetch_sub(1, Ordering::SeqCst);
                                order.retain(|queued_id| *queued_id != id);
                                *current.lock().await = order.front().copied();
                            } else {
                                eprintln!("engine: error line for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::LatentResponse(lat) => {
                            if let Some(out) = active.get(&lat.id) {
                                let _ = out.send(Event::Latent { latent_id: lat.latent_id, tokens: lat.tokens });
                            } else {
                                eprintln!("engine: latent response for unknown request ignored: {}", line_buf.trim_end());
                            }
                        }
                        EngineMsg::Ready { .. } => eprintln!("engine: unexpected ready line after startup"),
                        EngineMsg::Log(s) => eprintln!("engine: {s}"),
                    },
                    Ok(_) => error = Some("engine exited mid-request".into()),
                    Err(e) => error = Some(format!("engine stdout: {e}")),
                },
                Some(cancel_id) = cancel_rx.recv() => {
                    if *current.lock().await == Some(cancel_id) {
                        if let Err(e) = write_line(&mut stdin, cancel_line(cancel_id)).await {
                            error = Some(e);
                        }
                    }
                }
            }
            // The consumer is gone: stop the decode now instead of running the request to its end.
            if let Some(id) = dropped {
                if error.is_none() && cancelled.insert(id) {
                    eprintln!("engine: client dropped request {id}; cancelling");
                    if let Err(e) = write_line(&mut stdin, cancel_line(id)).await {
                        error = Some(e);
                    }
                }
            }
        }

        *current.lock().await = None;
        if let Some(e) = error {
            for (_, out) in active.drain() {
                let _ = out.send(Event::Error(e.clone()));
                queued.fetch_sub(1, Ordering::SeqCst);
            }
            eprintln!("engine: worker stopped: {e}");
            // The exit status is the reason a user can act on: read it (the child is usually already gone).
            let status = match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
                Ok(Ok(s)) => s.to_string(),
                Ok(Err(we)) => format!("wait failed: {we}"),
                Err(_) => "still running".into(),
            };
            dead = Some(format!("{e} ({status})"));
            break;
        }
        state.set(EngineState::Idle, "request finished");
    }
    if let Some(reason) = dead.take() {
        state.set(EngineState::Dead(reason.clone()), &reason);
    } else {
        state.set(EngineState::Stopping, "stop requested");
    }
    // Refuse whatever is still queued, then close stdin so the engine sees EOF.
    rx.close();
    while let Some(job) = rx.recv().await {
        queued.fetch_sub(1, Ordering::SeqCst);
        let _ = job.out.send(Event::Error("engine process has exited".into()));
    }
    let _ = stdin.shutdown().await;
    drop(stdin);
    if state.get() == EngineState::Stopping {
        match tokio::time::timeout(Duration::from_secs(30), child.wait()).await {
            Ok(st) => state.set(EngineState::Dead(format!("stopped ({})", st.map_or_else(|e| format!("wait failed: {e}"), |s| s.to_string()))), "child exited after stdin EOF"),
            Err(_) => {
                eprintln!("engine did not exit after stdin EOF; killing");
                let _ = child.kill().await;
                state.set(EngineState::Dead("killed after 30 s without exiting".into()), "no exit after stdin EOF");
            }
        }
    }
}

async fn write_line(stdin: &mut ChildStdin, line: String) -> Result<(), String> {
    stdin.write_all(line.as_bytes()).await.map_err(|e| format!("write to engine stdin: {e}"))?;
    stdin.flush().await.map_err(|e| format!("flush engine stdin: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Engine` with no child process: the test owns the queue's receiver.
    fn fake(alive: bool, queued: usize) -> (Engine, mpsc::Receiver<Job>) {
        let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAP);
        let (cancel_tx, _) = mpsc::unbounded_channel::<u64>();
        let limits = Limits { tmax: 1, mrows: 1, kmax: 1, spec_k: 0, pack: "t".into(), kv: "f32".into(), tier: None, seqs: None };
        let st = if alive { EngineState::Idle } else { EngineState::Dead("test".into()) };
        let e = Engine {
            tx,
            queued: Arc::new(AtomicUsize::new(queued)),
            state: Arc::new(StateCell::new("test".into(), st)),
            current: Arc::new(tokio::sync::Mutex::new(None)),
            cancel_tx,
            limits: limits.clone(),
            stop: watch::channel(false).0,
            worker: tokio::sync::Mutex::new(None),
        };
        (e, rx)
    }

    /// A real `Engine` over `tests/stub_engine.sh` in the given mode; the stub's event log is `.work/fix-front-test/<tag>.log`.
    async fn stub(mode: &str, tag: &str) -> (Engine, std::path::PathBuf) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let log = root.join("../.work/fix-front-test").join(format!("{tag}.log"));
        std::fs::create_dir_all(log.parent().unwrap()).unwrap();
        let _ = std::fs::remove_file(&log);
        let env = vec![("STUB_MODE".to_string(), mode.to_string()), ("STUB_LOG".to_string(), log.display().to_string())];
        let e = Engine::spawn(&root.join("tests/stub_engine.sh"), Path::new("none"), &env).await.expect("stub engine");
        (e, log)
    }

    async fn until(what: &str, mut f: impl FnMut() -> bool) {
        for _ in 0..100 {
            if f() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for: {what}");
    }

    fn pool(engines: Vec<Engine>) -> EnginePool {
        let limits = engines[0].limits.clone();
        EnginePool { engines, next_id: AtomicU64::new(1), limits }
    }

    fn submit(p: &EnginePool) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        p.submit(vec![1], 1, false, vec![], vec![], SampleParams::default(), None, None, (None, None), false, None, vec![])
    }

    /// A-M1: a dead engine drains to queue depth 0, so it used to win the least-queued pick and every
    /// other request answered 503 while the live engine sat busy.
    #[test]
    fn submit_skips_a_dead_engine() {
        let (dead, _rx0) = fake(false, 0);
        let (live, mut rx1) = fake(true, 3);
        let p = pool(vec![dead, live]);
        assert!(submit(&p).is_ok(), "dead engine 0 must not be preferred");
        assert!(rx1.try_recv().is_ok(), "the job must land on the live engine");
    }

    #[test]
    fn submit_with_every_engine_dead_names_the_exit() {
        let (a, _ra) = fake(false, 0);
        let (b, _rb) = fake(false, 0);
        let p = pool(vec![a, b]);
        assert_eq!(submit(&p).unwrap_err(), "engine process has exited");
    }

    /// A-M2: a client that drops its receiver (closed tab) must stop the decode. Before, the engine ran the
    /// request to its end and every later request queued behind it.
    #[tokio::test(flavor = "current_thread")]
    async fn a_dropped_receiver_cancels_the_running_request() {
        let (e, log) = stub("slow", "a-m2-drop").await;
        let id = 7;
        let mut rx = e.submit(id, vec![1], 400, false, vec![], vec![], SampleParams::default(), None, None, (None, None), false, None, vec![]).unwrap();
        for _ in 0..2 {
            assert!(matches!(rx.recv().await, Some(Event::Tok { .. })));
        }
        drop(rx);
        until("the engine saw the cancel line", || std::fs::read_to_string(&log).unwrap_or_default().contains("cancel 7")).await;
        until("the queue to drain", || e.queue_depth() == 0).await;
        assert!(e.alive());
    }

    /// A-M1: an engine that dies while idle used to stay `alive` until the next write failed; its exit status
    /// was never read. Now the state turns Dead with the reason, and `submit` refuses at once.
    #[tokio::test(flavor = "current_thread")]
    async fn an_engine_that_dies_idle_is_seen_with_its_exit_status() {
        let (e, _log) = stub("die_idle", "a-m1-die-idle").await;
        assert_eq!(e.state(), EngineState::Idle);
        until("the dead state", || !e.alive()).await;
        let st = e.state();
        assert_eq!(st.word(), "dead");
        assert!(st.reason().unwrap_or("").contains("exit status: 3"), "{st:?}");
        let r = e.submit(1, vec![1], 1, false, vec![], vec![], SampleParams::default(), None, None, (None, None), false, None, vec![]);
        assert_eq!(r.unwrap_err(), "engine process has exited");
    }

    /// A-M1: nothing closed the engine's stdin, so a graceful stop waited the full 30 s and then killed.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_closes_stdin_and_the_child_exits_at_once() {
        let (e, log) = stub("ok", "a-m1-shutdown").await;
        let t0 = std::time::Instant::now();
        tokio::time::timeout(Duration::from_secs(5), e.shutdown()).await.expect("shutdown took over 5 s");
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(std::fs::read_to_string(&log).unwrap_or_default().contains("stdin-eof"));
        assert_eq!(e.state().word(), "dead");
        assert!(e.state().reason().unwrap_or("").starts_with("stopped"), "{:?}", e.state());
    }

    /// A-M5: a malformed line is refused before it is queued.
    #[test]
    fn submit_req_refuses_a_line_with_two_ops() {
        let (e, mut rx) = fake(true, 0);
        let mut req = Request::default();
        req.ipc.ipc_stats = Some(true);
        req.ipc.ipc_release = Some(1);
        assert!(e.submit_req(1, req).unwrap_err().starts_with("malformed request"));
        assert!(rx.try_recv().is_err() && e.queue_depth() == 0);
    }
}
