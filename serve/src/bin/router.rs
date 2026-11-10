//! CPU-only `baro-router` skeleton.
//!
//! The router owns placement and discovery; `baro-serve` remains the engine
//! process. Placement uses resident-prefix locality first, rendezvous hashing
//! second, and pending-count rank as the final fallback.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post};
use axum::{Json, Router};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpStream;

const BARO_SERVICE: &str = "_baro-node._tcp";
const PAIR_SERVICE: &str = "_nvpair-node._tcp";
// PAIR uses a fixed SRV port for the node record; the actual node-info port is
// carried by the ni TXT entry.  Consumers must not mistake the SRV port for
// the HTTP endpoint.
const PAIR_SRV_PORT: u16 = 14318;
// Proxied request bodies are chat/completions JSON, not state streams (P1's
// own 8 MiB-chunked stream lives on baro-serve directly, never through this
// path) -- 32 MiB is generous headroom, not a real expected size.
const MAX_PROXY_BODY: usize = 32 * 1024 * 1024;

/// CONTRACT 1's unsalted prefix hash, dependency-free SHA-256. A separate
/// copy from `serve/src/checkpoints.rs`'s: `router` is a different binary
/// target under `src/bin/`, sharing no module tree with `main.rs` without a
/// `lib.rs` this crate does not have -- `checkpoints.rs`'s own comment
/// already accepts one dependency-free sha256 per binary over adding a crate
/// dependency, this is the same call for the second binary. Must compute the
/// identical value `state.rs::exact_prefix_hash` does: CONTRACT 4 (P1-STATE-API.md)
/// only works if the router's hash and the engine's own `GET /v1/state`
/// `prefix_hash` are the same number.
mod prefix_hash {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    fn sha256(data: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
        let mut msg = data.to_vec();
        let bit_len = (data.len() as u64).wrapping_mul(8);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in msg.chunks(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ (!e & g);
                let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
                *x = x.wrapping_add(y);
            }
        }
        let mut out = [0u8; 32];
        for (i, v) in h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }

    /// `sha256(tokens[0:pos]` as little-endian u32 bytes `)`, first 8 digest
    /// bytes read back as a little-endian u64.
    pub fn exact_prefix_hash(tokens: &[u32], pos: usize) -> u64 {
        let mut bytes = Vec::with_capacity(pos * 4);
        for &t in &tokens[..pos.min(tokens.len())] {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        let digest = sha256(&bytes);
        u64::from_le_bytes(digest[..8].try_into().unwrap())
    }

    pub fn rendezvous_score(prefix: u64, engine_id: &str) -> u64 {
        let mut bytes = prefix.to_le_bytes().to_vec();
        bytes.extend_from_slice(engine_id.as_bytes());
        u64::from_le_bytes(sha256(&bytes)[..8].try_into().unwrap())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn matches_state_rs_own_test_vector() {
            // Same vector serve/src/state.rs::tests::matches_a_hand_computed_vector
            // uses, independently verified there against Python's hashlib --
            // this MUST agree, not just be internally consistent, since CONTRACT 4
            // compares this value against the engine's own GET /v1/state number.
            let tokens = [1u32, 2, 999999];
            assert_eq!(exact_prefix_hash(&tokens, 2), 5389374292907326260u64);
        }
    }
}

#[derive(Clone)]
struct Shared(Arc<RouterState>);

struct RouterState {
    node_id: String,
    http_port: u16,
    discovery: DiscoveryAdvertisement,
    mdns: Option<MdnsRuntime>,
    engines: EngineRegistry,
    workloads: WorkloadCatalog,
    next_request: AtomicU64,
    /// Pooled upstream connections (router shape idea 8): one `reqwest::Client`
    /// reused for every proxied request and every `probe_loop` enrichment
    /// fetch, not one per request.
    http_client: reqwest::Client,
    prefix_locks: PrefixLocks,
}

#[derive(Clone)]
struct PrefixLocks(Arc<tokio::sync::Mutex<HashMap<u64, Arc<PrefixLock>>>>);

struct PrefixLock {
    done: std::sync::atomic::AtomicBool,
    notify: tokio::sync::Notify,
}

struct PrefixLease {
    key: u64,
    entry: Arc<PrefixLock>,
    locks: PrefixLocks,
    owner: bool,
}

impl PrefixLocks {
    async fn acquire(&self, key: u64) -> PrefixLease {
        let mut locks = self.0.lock().await;
        if let Some(entry) = locks.get(&key) {
            return PrefixLease { key, entry: entry.clone(), locks: self.clone(), owner: false };
        }
        let entry = Arc::new(PrefixLock { done: std::sync::atomic::AtomicBool::new(false), notify: tokio::sync::Notify::new() });
        locks.insert(key, entry.clone());
        PrefixLease { key, entry, locks: self.clone(), owner: true }
    }
}

/// How long a waiter holds for the owner of its cold prefix before it goes on without the lock
/// (`ROUTER_PREFIX_WAIT_MS`, default 60 s): an upstream that accepts and never answers must not hold every waiter.
fn prefix_wait_timeout() -> std::time::Duration {
    let ms = std::env::var("ROUTER_PREFIX_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(60_000);
    std::time::Duration::from_millis(ms)
}

impl PrefixLease {
    /// Waits for the owner to finish; returns false when `prefix_wait_timeout` ran out first (the caller proceeds).
    async fn wait(&self) -> bool {
        let released = async {
            while !self.entry.done.load(Ordering::Acquire) {
                let notified = self.entry.notify.notified();
                if self.entry.done.load(Ordering::Acquire) {
                    break;
                }
                notified.await;
            }
        };
        tokio::time::timeout(prefix_wait_timeout(), released).await.is_ok()
    }
}

impl Drop for PrefixLease {
    fn drop(&mut self) {
        if !self.owner {
            return;
        }
        let key = self.key;
        let entry = self.entry.clone();
        let locks = self.locks.clone();
        tokio::spawn(async move {
            entry.done.store(true, Ordering::Release);
            entry.notify.notify_waiters();
            let mut map = locks.0.lock().await;
            if map.get(&key).is_some_and(|current| Arc::ptr_eq(current, &entry)) {
                map.remove(&key);
            }
        });
    }
}

#[derive(Debug, Clone, Serialize)]
struct DiscoveryAdvertisement {
    instance: String,
    service: String,
    pair_service: String,
    port: u16,
    txt: BTreeMap<String, String>,
}

impl DiscoveryAdvertisement {
    fn new(node_id: &str, port: u16) -> Self {
        let mut txt = BTreeMap::new();
        txt.insert("v".into(), "1".into());
        txt.insert("ni".into(), port.to_string());
        // PAIR keys its directory by the UUID in the consolidated node record.
        txt.insert("uuid".into(), node_id.into());
        txt.insert("node_id".into(), node_id.into());
        Self {
            instance: format!("baro-{node_id}"),
            service: BARO_SERVICE.into(),
            pair_service: PAIR_SERVICE.into(),
            port,
            txt,
        }
    }

    /// The service records the mDNS adapter must advertise and browse.
    fn browse_services(&self) -> [&'static str; 2] {
        [BARO_SERVICE, PAIR_SERVICE]
    }
}

struct MdnsRuntime {
    daemon: ServiceDaemon,
    // Every record `register`ed below, by full name: gate 3's own failure
    // (room A, 2026-09-17 -- PAIR's discovery:get-nodes returned a stale
    // UUID from an earlier test run instead of the current one) traced to
    // this: mdns-sd's PTR/TXT records carry a 75-minute TTL
    // (`DNS_OTHER_TTL` in `service_info.rs`) and neither `ServiceDaemon` nor
    // this struct sent a goodbye (TTL=0) on exit, only `Drop`ping the
    // channel handles -- which `mdns-sd` does not treat as an unregister at
    // all. Every peer that ever saw the record, PAIR's scanner included,
    // keeps believing it for up to 75 minutes after the process is gone.
    // Compounded by three router processes orphaned by earlier ad-hoc runs
    // that were simply never killed (`ps` receipt, gate-3 diagnosis,
    // exchange/lane-P0B-report.md): still alive, still re-announcing,
    // still winning the race against the current run's fresh record.
    fullnames: Vec<String>,
}

impl MdnsRuntime {
    /// Sends an explicit goodbye for every record this process registered,
    /// then stops the daemon. Must run before the process exits (wired
    /// through `axum::serve(..).with_graceful_shutdown` on SIGINT/SIGTERM)
    /// so a killed router leaves nothing for the next run, or a live peer's
    /// cache, to trip over.
    fn unregister_all(&self) {
        for fullname in &self.fullnames {
            match self.daemon.unregister(fullname) {
                Ok(recv) => match recv.recv_timeout(Duration::from_secs(2)) {
                    Ok(status) => eprintln!("mDNS unregister {fullname}: {status:?}"),
                    Err(e) => eprintln!("mDNS unregister {fullname}: no ack ({e})"),
                },
                Err(e) => eprintln!("mDNS unregister {fullname}: {e}"),
            }
        }
        if let Err(e) = self.daemon.shutdown() {
            eprintln!("mDNS daemon shutdown: {e}");
        }
    }
}

fn start_mdns(advertisement: &DiscoveryAdvertisement, node_id: &str, advertise_ip: &str) -> Result<MdnsRuntime, String> {
    let daemon = ServiceDaemon::new().map_err(|e| format!("mDNS daemon: {e}"))?;
    let host = std::env::var("BARO_ROUTER_MDNS_HOST")
        .unwrap_or_else(|_| format!("baro-{node_id}.local."));
    let mut properties = HashMap::new();
    for (key, value) in &advertisement.txt {
        properties.insert(key.clone(), value.clone());
    }
    let mut fullnames = Vec::new();
    for service in advertisement.browse_services() {
        let service_type = format!("{service}.local.");
        let srv_port = if service == PAIR_SERVICE { PAIR_SRV_PORT } else { advertisement.port };
        let info = ServiceInfo::new(
            &service_type,
            &advertisement.instance,
            &host,
            advertise_ip,
            srv_port,
            properties.clone(),
        )
        .map_err(|e| format!("mDNS service {service}: {e}"))?;
        fullnames.push(info.get_fullname().to_string());
        daemon.register(info).map_err(|e| format!("mDNS register {service}: {e}"))?;
    }
    let browse_type = format!("{}.local.", advertisement.pair_service);
    let receiver = daemon.browse(&browse_type).map_err(|e| format!("mDNS browse: {e}"))?;
    std::thread::Builder::new()
        .name("baro-router-mdns".into())
        .spawn(move || {
            while let Ok(event) = receiver.recv() {
                if let ServiceEvent::ServiceResolved(service) = event {
                    eprintln!("mDNS discovered {}:{}", service.get_hostname(), service.get_port());
                }
            }
        })
        .map_err(|e| format!("mDNS browse thread: {e}"))?;
    Ok(MdnsRuntime { daemon, fullnames })
}

#[derive(Debug, Clone, Serialize)]
struct EngineInfo {
    id: String,
    address: String,
    /// `scheme://host:port` every upstream request is built on.
    #[serde(skip)]
    url: String,
    /// Upstream bearer key; the client's own `Authorization` never leaves the router.
    #[serde(skip)]
    key: Option<String>,
    port: u16,
    running: bool,
    healthy: bool,
    adopted: bool,
    /// The one word for the three bools above plus the hysteresis (the bools stay for compatibility).
    state: Health,
    pending: usize,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_timeout_ms: Option<u64>,
    /// CONTRACT 4 (P1-STATE-API.md): this engine's `GET /v1/state`
    /// `states[].prefix_hash` values, refreshed by `probe_loop`. The
    /// locality term compares an incoming request's own computed hash
    /// against this list, never the reverse.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    resident_prefix_hashes: Vec<String>,
    /// CONTRACT 4: "/v1/node-info carries, per engine, the GET /v1/state
    /// object verbatim." The raw response body, not re-derived fields --
    /// absent (not an empty object) for an engine that has never answered
    /// GET /v1/state (not adopted yet, or not a baro-serve engine at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    resident_state: Option<Value>,
}

struct EngineRecord {
    info: EngineInfo,
    health: HealthHysteresis,
}

#[derive(Clone)]
struct EngineRegistry(Arc<RwLock<Vec<EngineRecord>>>);

impl EngineRegistry {
    fn from_env() -> Self {
        let records = std::env::var("BARO_ROUTER_ENGINES")
            .unwrap_or_default()
            .split(',')
            .filter_map(parse_engine)
            .map(|info| EngineRecord { info, health: HealthHysteresis::default() })
            .collect();
        Self(Arc::new(RwLock::new(records)))
    }

    /// Adds an engine, replacing any engine with the same id.
    fn add(&self, info: EngineInfo) {
        let mut engines = self.0.write().expect("engine registry poisoned");
        engines.retain(|e| e.info.id != info.id);
        engines.push(EngineRecord { info, health: HealthHysteresis::default() });
    }

    fn remove(&self, id: &str) -> bool {
        let mut engines = self.0.write().expect("engine registry poisoned");
        let before = engines.len();
        engines.retain(|e| e.info.id != id);
        engines.len() != before
    }

    /// `engine/model` names one engine: (engine id, upstream model name).
    fn pin(&self, model: &str) -> Option<(String, String)> {
        let (id, rest) = model.split_once('/')?;
        let engines = self.0.read().expect("engine registry poisoned");
        engines.iter().any(|e| e.info.id == id).then(|| (id.to_string(), rest.to_string()))
    }

    fn ids(&self) -> Vec<String> {
        self.0.read().expect("engine registry poisoned").iter().map(|e| e.info.id.clone()).collect()
    }

    fn snapshot(&self) -> Vec<EngineInfo> {
        self.0.read().expect("engine registry poisoned").iter().map(|e| e.info.clone()).collect()
    }

    /// `prefer`: an engine id the caller wants tie-broken in its favor and
    /// its `pending` treated as one lower (CONTRACT 4: "rank improved by the
    /// equivalent of one pending job") -- `None` for plain rank, gates 1-3's
    /// behavior, unaffected by any of this. `exclude`: ids already tried and
    /// failed to connect this request (proxy's retry loop), skipped outright
    /// rather than merely deprioritized, since a failed connect already
    /// demoted the engine via `set_probe`.
    fn choose(&self, prefer: Option<&str>, exclude: &[String]) -> Option<EngineChoice> {
        self.0
            .read()
            .expect("engine registry poisoned")
            .iter()
            .filter(|e| e.info.running && e.info.healthy && !exclude.iter().any(|x| x == &e.info.id))
            .min_by_key(|e| {
                let preferred = prefer == Some(e.info.id.as_str());
                let key_pending = if preferred { e.info.pending.saturating_sub(1) } else { e.info.pending };
                (key_pending, !preferred, e.info.id.clone())
            })
            .map(|e| {
                let preferred = prefer == Some(e.info.id.as_str());
                EngineChoice { id: e.info.id.clone(), url: e.info.url.clone(), key: e.info.key.clone(), placement: if preferred { "locality" } else { "rank" }.into() }
            })
    }

    fn choose_consistent(&self, prefix: u64, exclude: &[String]) -> Option<EngineChoice> {
        self.0
            .read()
            .expect("engine registry poisoned")
            .iter()
            .filter(|e| e.info.running && e.info.healthy && !exclude.iter().any(|x| x == &e.info.id))
            .max_by_key(|e| (prefix_hash::rendezvous_score(prefix, &e.info.id), std::cmp::Reverse(e.info.id.clone())))
            .map(|e| EngineChoice { id: e.info.id.clone(), url: e.info.url.clone(), key: e.info.key.clone(), placement: "hash".into() })
    }

    fn len(&self) -> usize {
        self.0.read().expect("engine registry poisoned").len()
    }

    fn set_probe(&self, id: &str, ok: bool) {
        let mut engines = self.0.write().expect("engine registry poisoned");
        if let Some(engine) = engines.iter_mut().find(|e| e.info.id == id) {
            let healthy = engine.health.observe(ok);
            engine.info.running = ok;
            engine.info.adopted = ok;
            engine.info.healthy = healthy;
            engine.info.state = engine.health.state();
        }
    }

    /// Why `choose` found nothing: an engine added in the last 5 s has not been probed yet, which is not "down".
    fn none_reason(&self) -> String {
        let unknown = self.0.read().expect("engine registry poisoned").iter().filter(|e| e.info.state == Health::Unknown).count();
        if unknown > 0 {
            format!("no healthy adopted engine ({unknown} engine(s) not probed yet, retry in a few seconds)")
        } else {
            "no healthy adopted engine".into()
        }
    }

    /// `/health` status from the engine states: ok (all up, or none registered), degraded (some up), starting
    /// (none up, some unprobed), down.
    fn status_word(&self) -> &'static str {
        let engines = self.0.read().expect("engine registry poisoned");
        let total = engines.len();
        let up = engines.iter().filter(|e| e.info.running && e.info.healthy).count();
        let unknown = engines.iter().filter(|e| e.info.state == Health::Unknown).count();
        if total == 0 || up == total {
            "ok"
        } else if up > 0 {
            "degraded"
        } else if unknown > 0 {
            "starting"
        } else {
            "down"
        }
    }

    fn set_resident(&self, id: &str, hashes: Vec<String>, raw: Option<Value>) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.resident_prefix_hashes = hashes;
            engine.info.resident_state = raw;
        }
    }

    /// Engines that do not list `model`. Empty when no engine lists it, so an
    /// unknown name keeps plain placement instead of failing.
    fn without_model(&self, model: &str) -> Vec<String> {
        let engines = self.0.read().expect("engine registry poisoned");
        if !engines.iter().any(|e| e.info.models.iter().any(|m| m == model)) {
            return Vec::new();
        }
        engines.iter().filter(|e| !e.info.models.iter().any(|m| m == model)).map(|e| e.info.id.clone()).collect()
    }

    /// (model, engine) for every healthy engine, first engine wins a duplicate id.
    fn all_models(&self) -> Vec<(String, String)> {
        let mut seen = std::collections::HashSet::new();
        self.0
            .read()
            .expect("engine registry poisoned")
            .iter()
            .filter(|e| e.info.running && e.info.healthy)
            .flat_map(|e| e.info.models.iter().map(move |m| (m.clone(), e.info.id.clone())))
            .filter(|(m, _)| seen.insert(m.clone()))
            .collect()
    }

    fn set_models(&self, id: &str, models: Vec<String>) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.models = models;
        }
    }

    fn bump_pending(&self, id: &str) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.pending += 1;
        }
    }

    fn drop_pending(&self, id: &str) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.pending = engine.info.pending.saturating_sub(1);
        }
    }
}

/// Model ids from an OpenAI `GET /v1/models` (`data[].id`) or an Ollama
/// `GET /api/tags` (`models[].name`) body.
fn parse_models(body: &Value) -> Vec<String> {
    let list = |key: &str, field: &str| -> Vec<String> {
        body.get(key)
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(|m| m.get(field).and_then(Value::as_str).map(str::to_string)).collect())
            .unwrap_or_default()
    };
    let openai = list("data", "id");
    if openai.is_empty() { list("models", "name") } else { openai }
}

async fn fetch_models(http_client: &reqwest::Client, url: &str, key: Option<&str>) -> Vec<String> {
    for path in ["/v1/models", "/api/tags"] {
        let mut req = http_client.get(format!("{url}{path}"));
        if let Some(key) = key {
            req = req.bearer_auth(key);
        }
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(body) = resp.json::<Value>().await {
                    let models = parse_models(&body);
                    if !models.is_empty() {
                        return models;
                    }
                }
            }
        }
    }
    Vec::new()
}

fn requested_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body).ok()?.get("model")?.as_str().map(str::to_string)
}

async fn openai_models(State(Shared(state)): State<Shared>) -> Json<Value> {
    let data: Vec<Value> = state.engines.all_models().into_iter().map(|(id, engine)| json!({"id": id, "object": "model", "owned_by": engine})).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn ollama_tags(State(Shared(state)): State<Shared>) -> Json<Value> {
    let models: Vec<Value> = state.engines.all_models().into_iter().map(|(id, _)| json!({"name": id, "model": id})).collect();
    Json(json!({"models": models}))
}

/// `id=URL[@ENV_NAME]`: the key, when named, is read from that env var.
fn parse_engine(raw: &str) -> Option<EngineInfo> {
    let (id, endpoint) = raw.trim().split_once('=')?;
    let (endpoint, key) = match endpoint.rsplit_once('@') {
        Some((url, var)) if !var.is_empty() && var.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_') => {
            (url, Some(std::env::var(var).ok()?))
        }
        _ => (endpoint, None),
    };
    let mut info = engine_info(id, endpoint)?;
    info.key = key;
    Some(info)
}

fn engine_info(id: &str, endpoint: &str) -> Option<EngineInfo> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    let (scheme, host) = match endpoint.split_once("://") {
        Some((scheme @ ("http" | "https"), host)) => (scheme, host),
        Some(_) => return None,
        None => ("http", endpoint),
    };
    let (address, port) = match host.rsplit_once(':') {
        Some((_, port)) => (host.to_string(), port.parse().ok()?),
        None if scheme == "https" => (format!("{host}:443"), 443),
        None => return None,
    };
    Some(EngineInfo {
        id: id.trim().to_string(),
        url: format!("{scheme}://{address}"),
        key: None,
        address,
        port,
        running: false,
        healthy: false,
        adopted: false,
        state: Health::Unknown,
        pending: 0,
        models: Vec::new(),
        prefill_timeout_ms: Some(20_000),
        resident_prefix_hashes: Vec::new(),
        resident_state: None,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct HealthHysteresis {
    healthy: bool,
    failures: u8,
    passes: u8,
}

/// unknown: never probed; up; flapping: moving toward up or down (one pass so far, or failing but not yet 3);
/// down: not healthy and failing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Health {
    Unknown,
    Up,
    Flapping,
    Down,
}

impl HealthHysteresis {
    fn state(&self) -> Health {
        match (self.healthy, self.passes, self.failures) {
            (false, 0, 0) => Health::Unknown,
            (true, _, 0) => Health::Up,
            (true, _, _) => Health::Flapping,
            (false, p, _) if p > 0 => Health::Flapping,
            _ => Health::Down,
        }
    }

    fn observe(&mut self, ok: bool) -> bool {
        if ok {
            self.failures = 0;
            self.passes = self.passes.saturating_add(1);
            if self.passes >= 2 {
                self.healthy = true;
            }
        } else {
            self.passes = 0;
            self.failures = self.failures.saturating_add(1);
            if self.failures >= 3 {
                self.healthy = false;
            }
        }
        self.healthy
    }
}

/// One request's phase at the router, written only through `WorkloadCatalog::set`. The serialized words are the
/// old `connected`-less set (`done`, `failed_connect`) plus `connecting`, `streaming` and `stream_broke`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Connecting,
    /// Response headers are in hand and the body is flowing: the engine still counts as busy.
    Streaming,
    Done,
    FailedConnect,
    /// The body ended with an error, or the client went away before the end.
    StreamBroke,
}

impl Phase {
    fn word(self) -> &'static str {
        match self {
            Phase::Connecting => "connecting",
            Phase::Streaming => "streaming",
            Phase::Done => "done",
            Phase::FailedConnect => "failed_connect",
            Phase::StreamBroke => "stream_broke",
        }
    }
}

/// Rows kept in the catalog; the oldest go first (it used to grow forever).
const CATALOG_CAP: usize = 1024;

#[derive(Debug, Clone, Serialize)]
struct Workload {
    request_id: String,
    method: String,
    path: String,
    state: Phase,
    node_id: String,
    engine: Option<String>,
    /// "rank" (plain, gates 1-3) or "locality" (CONTRACT 4: the catalog row
    /// names the locality term as the reason the request landed here).
    placement: String,
    started_ms: u128,
}

#[derive(Clone)]
struct WorkloadCatalog(Arc<Mutex<Vec<Workload>>>);

impl WorkloadCatalog {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    /// One row per request id: the first call adds it, later ones move it to its new phase.
    fn set(&self, mut workload: Workload, phase: Phase) {
        workload.state = phase;
        let mut rows = self.0.lock().expect("workload catalog poisoned");
        match rows.iter_mut().rev().find(|w| w.request_id == workload.request_id) {
            Some(row) => row.state = phase,
            None => {
                if rows.len() >= CATALOG_CAP {
                    rows.remove(0);
                }
                rows.push(workload);
            }
        }
    }

    fn snapshot(&self) -> Vec<Workload> {
        self.0.lock().expect("workload catalog poisoned").clone()
    }
}

#[derive(Debug, Clone)]
struct FilteredRequest {
    method: Method,
    path: String,
    request_id: String,
}

#[derive(Debug, Clone)]
struct EngineChoice {
    id: String,
    url: String,
    key: Option<String>,
    /// CONTRACT 4: this engine was picked because it holds the request's
    /// resident prefix, not because it was plain-rank best.
    placement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendState {
    NotConnected,
    /// `reqwest::RequestBuilder::send()` is atomic (connect, write the
    /// request, await the response headers) -- this router never observes a
    /// TCP-connected-but-no-response-bytes window the way a raw socket proxy
    /// would, so no real call site constructs this variant. Kept (not
    /// deleted) because `retry_allowed`'s policy for it is still the
    /// documented, tested rule for whichever future path gets that
    /// granularity (a streamed request body, HTTP/2 multiplexing).
    #[allow(dead_code)]
    ConnectedNoBytes,
    /// Never constructed since A-M11 (the body guard carries the phase); kept for `retry_allowed`'s documented rule.
    #[allow(dead_code)]
    FirstByteSent,
}

impl SendState {
    fn retry_allowed(self, cancel_ack: bool, cancel_timed_out: bool) -> bool {
        match self {
            Self::NotConnected => true,
            Self::ConnectedNoBytes => cancel_ack || cancel_timed_out,
            Self::FirstByteSent => false,
        }
    }
}

fn request_filter(request: &Request, request_id: String) -> Result<FilteredRequest, String> {
    let path = request.uri().path().to_string();
    if path == "/v1/node-info" || path == "/v1/workloads" || path == "/health" {
        return Err("control endpoint is not proxyable".into());
    }
    Ok(FilteredRequest { method: request.method().clone(), path, request_id })
}

fn choose_engine(registry: &EngineRegistry, affinity: Option<&PrefixAffinity>, exclude: &[String]) -> Result<EngineChoice, String> {
    if let Some(affinity) = affinity {
        if let Some(engine) = affinity.resident.as_deref() {
            return registry.choose(Some(engine), exclude).ok_or_else(|| registry.none_reason());
        }
        if let Some(choice) = registry.choose_consistent(affinity.hash, exclude) {
            return Ok(choice);
        }
    }
    registry.choose(None, exclude).ok_or_else(|| registry.none_reason())
}

fn fail_to_connect(error: String) -> Response {
    error_while_proxy(format!("upstream connection failed: {error}"))
}

fn response_filter(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

fn error_while_proxy(error: String) -> Response {
    (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "upstream", "message": error}}))).into_response()
}

/// Headers that name properties of THIS hop (router<->engine or
/// router<->client), never forwarded across a proxy: RFC 9110 section 7.6.1
/// plus `content-length`, which the streamed body invalidates on the way
/// out (axum/hyper recompute framing for the new body).
fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection" | "keep-alive" | "proxy-authenticate" | "proxy-authorization" | "te" | "trailers" | "transfer-encoding" | "upgrade" | "content-length" | "host"
    )
}

/// Streams the upstream engine's response straight through: SSE token-by-
/// token output (`stream:true` chat completions) must reach the downstream
/// client as it arrives, never buffered whole first. `reqwest` and `axum`
/// both build on the same `http` crate types (status, header names/values),
/// so nothing here needs converting, only re-filtering hop-by-hop headers.
fn stream_response(resp: reqwest::Response, guard: StreamGuard) -> Response {
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = Body::from_stream(Tracked { inner: Box::pin(resp.bytes_stream()), guard });
    let mut builder = Response::builder().status(status);
    for (name, value) in headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder.body(body).unwrap_or_else(|error| error_while_proxy(format!("building proxied response: {error}")))
}

/// Ends a proxied request when its body stream is dropped (end, error or client gone): the engine's pending count
/// drops then, not when the headers arrive (a long generation counted as idle), and the catalog row gets its last phase.
struct StreamGuard {
    engines: EngineRegistry,
    workloads: WorkloadCatalog,
    engine_id: String,
    workload: Workload,
    phase: Phase,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.engines.drop_pending(&self.engine_id);
        log(&self.workloads, self.workload.clone(), self.phase);
    }
}

struct Tracked<S> {
    inner: S,
    guard: StreamGuard,
}

impl<S, T, E> futures_core::Stream for Tracked<S>
where
    S: futures_core::Stream<Item = Result<T, E>> + Unpin,
{
    type Item = Result<T, E>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let polled = std::pin::Pin::new(&mut self.inner).poll_next(cx);
        match &polled {
            std::task::Poll::Ready(None) => self.guard.phase = Phase::Done,
            std::task::Poll::Ready(Some(Err(_))) => self.guard.phase = Phase::StreamBroke,
            _ => {}
        }
        polled
    }
}

/// CONTRACT 4 (P1-STATE-API.md): "a request whose unsalted prefix_hash at
/// any role boundary matches a resident state on engine E gets E's rank
/// improved." Scope limited to `/v1/completions` and `/api/generate`'s raw
/// `prompt` string, which is already the rendered text -- a `messages`-
/// shaped chat request would need this process to apply the chat template
/// itself to get a "rendered prompt" at all, which only the engine's own
/// tokenizer+template pipeline can do; such a request gets no locality
/// preference and falls back to plain rank, exactly gates 1-3's behavior.
/// Tokenizes against whichever engine plain rank would pick right now (all
/// engines share one pack's tokenizer by construction), so this adds one
/// HTTP round trip before the real placement decision, never GPU work.
struct PrefixAffinity {
    hash: u64,
    resident: Option<String>,
}

async fn locality_preference(state: &RouterState, path: &str, body: &[u8]) -> Option<PrefixAffinity> {
    if path != "/v1/completions" && path != "/api/generate" {
        return None;
    }
    let prompt = serde_json::from_slice::<Value>(body).ok()?.get("prompt")?.as_str()?.to_string();
    if prompt.is_empty() {
        return None;
    }
    let reference = state.engines.choose(None, &[])?;
    let tokenize_url = format!("{}/tokenize", reference.url);
    let resp = state.http_client.post(&tokenize_url).json(&json!({"content": prompt, "add_special": true})).send().await.ok()?;
    let parsed: Value = resp.json().await.ok()?;
    let tokens: Vec<u32> = parsed.get("tokens")?.as_array()?.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect();
    if tokens.is_empty() {
        return None;
    }
    // "At any role boundary" (CONTRACT 4): a raw prompt has none, so the one
    // real candidate is the same position every checkpoint in this codebase
    // defaults to for a boundary-less prompt -- len - 1, reserving the last
    // token as the live decode seed (checkpoints::create's own c.pos, and
    // state.rs::default_pos's fallback; the same reservation P1's
    // Chain::lookup enforces, room A 2026-09-17). len itself is checked too,
    // in case a resident state was saved at an explicit full-length pos.
    let candidates = [tokens.len().saturating_sub(1), tokens.len()];
    let engines = state.engines.snapshot();
    let mut fallback = None;
    for pos in candidates {
        if pos == 0 {
            continue;
        }
        let hash = format!("{:016x}", prefix_hash::exact_prefix_hash(&tokens, pos));
        let hash_value = prefix_hash::exact_prefix_hash(&tokens, pos);
        fallback = Some(PrefixAffinity { hash: hash_value, resident: None });
        if let Some(engine) = engines.iter().find(|e| e.resident_prefix_hashes.contains(&hash)) {
            return Some(PrefixAffinity { hash: hash_value, resident: Some(engine.id.clone()) });
        }
    }
    fallback
}

fn log(catalog: &WorkloadCatalog, workload: Workload, phase: Phase) {
    eprintln!("router request={} method={} path={} state={} engine={:?}", workload.request_id, workload.method, workload.path, phase.word(), workload.engine);
    catalog.set(workload, phase);
}

async fn health(State(Shared(state)): State<Shared>) -> Json<Value> {
    Json(json!({"status": state.engines.status_word(), "node_id": state.node_id, "engines": state.engines.snapshot().len()}))
}

async fn node_info(State(Shared(state)): State<Shared>) -> Json<Value> {
    let engines = state.engines.snapshot();
    Json(json!({
        "schema": 1,
        "node_id": state.node_id,
        "hostUuid": state.node_id,
        "GPUs": [],
        "telemetryValid": false,
        "msSince": 0,
        "http_port": state.http_port,
        "engines": engines,
        "pending": engines.iter().map(|e| e.pending).sum::<usize>(),
        "discovery": &state.discovery,
        // CONTRACT 4: each engine entry above carries its own resident_state
        // (GET /v1/state verbatim) and resident_prefix_hashes; no separate
        // top-level "state_locality" key exists (gate 3 asserts its absence).
    }))
}

#[derive(Deserialize)]
struct EngineSpec {
    id: String,
    url: String,
    key: Option<String>,
}

/// `BARO_ROUTER_ADMIN_TOKEN`, when set, guards engine registration.
fn admin_ok(headers: &HeaderMap) -> bool {
    match std::env::var("BARO_ROUTER_ADMIN_TOKEN") {
        Ok(token) => headers.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {token}")),
        Err(_) => true,
    }
}

async fn add_engine(State(Shared(state)): State<Shared>, headers: HeaderMap, Json(spec): Json<EngineSpec>) -> Response {
    if !admin_ok(&headers) {
        return response_filter(StatusCode::UNAUTHORIZED, json!({"error": "admin token required"}));
    }
    if spec.id.is_empty() || spec.id.contains('/') {
        return response_filter(StatusCode::BAD_REQUEST, json!({"error": "id must be non-empty and contain no '/'"}));
    }
    let Some(mut info) = engine_info(&spec.id, &spec.url) else {
        return response_filter(StatusCode::BAD_REQUEST, json!({"error": "url must be http(s)://host[:port]"}));
    };
    info.key = spec.key.filter(|k| !k.is_empty());
    state.engines.add(info);
    response_filter(StatusCode::OK, json!({"id": spec.id}))
}

async fn remove_engine(State(Shared(state)): State<Shared>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if !admin_ok(&headers) {
        return response_filter(StatusCode::UNAUTHORIZED, json!({"error": "admin token required"}));
    }
    let status = if state.engines.remove(&id) { StatusCode::OK } else { StatusCode::NOT_FOUND };
    response_filter(status, json!({"id": id}))
}

async fn workloads(State(Shared(state)): State<Shared>) -> Json<Vec<Workload>> {
    Json(state.workloads.snapshot())
}

/// Named phases (router shape idea 1): `request_filter` and `choose_engine`
/// happen once per attempt below; `connected`/`fail_to_connect` collapse into
/// the single `req.send()` match (reqwest owns TCP connect and the HTTP
/// exchange together); `response_filter` is every early control-endpoint or
/// no-engine reply; `error_while_proxy` and `log` close every path.
///
/// Retry policy (idea 2, "failover by what was sent, not only by what came
/// back"): a `send()` failure before any response arrived (`is_connect()`/
/// `is_timeout()`, `SendState::NotConnected` in this loop's terms) demotes
/// the engine immediately (`set_probe(false)`, not waiting for
/// `probe_loop`'s next cycle) and retries the next-ranked engine, once per
/// remaining engine. A failure AFTER `send()` returns a response is a
/// mid-stream break on a request already flowing to the client -- CONTRACT 3
/// (P0b) gate 2's "requests in flight on it fail loudly": no retry, the
/// broken stream reaches the client as a truncated/errored response, never a
/// silent hang or a duplicate generation.
async fn proxy(State(Shared(state)): State<Shared>, request: Request) -> Response {
    let request_id = format!("router-{}", state.next_request.fetch_add(1, Ordering::Relaxed));
    let filtered = match request_filter(&request, request_id.clone()) {
        Ok(value) => value,
        Err(error) => return response_filter(StatusCode::NOT_FOUND, json!({"error": error})),
    };
    let (parts, body) = request.into_parts();
    let body_bytes = match to_bytes(body, MAX_PROXY_BODY).await {
        Ok(value) => value,
        Err(error) => return error_while_proxy(format!("reading request body: {error}")),
    };
    let path_and_query = parts.uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| filtered.path.clone());
    let affinity = locality_preference(&state, &filtered.path, &body_bytes).await;
    let _prefix_lease = match affinity.as_ref().filter(|affinity| affinity.resident.is_none()) {
        Some(affinity) => {
            let lease = state.prefix_locks.acquire(affinity.hash).await;
            if !lease.owner && !lease.wait().await {
                eprintln!("router: prefix {:x}: owner did not answer within {:?}; proceeding without the lock", affinity.hash, prefix_wait_timeout());
            }
            Some(lease)
        }
        None => None,
    };

    let send_state = SendState::NotConnected;
    let mut tried: Vec<String> = Vec::new();
    let pinned = requested_model(&body_bytes).and_then(|m| state.engines.pin(&m));
    let (body_bytes, model_excluded) = match pinned {
        Some((id, model)) => {
            let mut body: Value = serde_json::from_slice(&body_bytes).expect("requested_model parsed it");
            body["model"] = Value::String(model);
            (axum::body::Bytes::from(body.to_string()), state.engines.ids().into_iter().filter(|e| *e != id).collect())
        }
        None => {
            let excluded = requested_model(&body_bytes).map(|m| state.engines.without_model(&m)).unwrap_or_default();
            (body_bytes, excluded)
        }
    };
    loop {
        let exclude: Vec<String> = tried.iter().chain(&model_excluded).cloned().collect();
        let choice = match choose_engine(&state.engines, affinity.as_ref(), &exclude) {
            Ok(value) => value,
            Err(error) => return response_filter(StatusCode::SERVICE_UNAVAILABLE, json!({"error": error})),
        };
        state.engines.bump_pending(&choice.id);
        let workload = Workload {
            request_id: filtered.request_id.clone(),
            method: filtered.method.to_string(),
            path: filtered.path.clone(),
            state: Phase::Connecting,
            node_id: state.node_id.clone(),
            engine: Some(choice.id.clone()),
            placement: choice.placement.clone(),
            started_ms: now_ms(),
        };

        let url = format!("{}{}", choice.url, path_and_query);
        let mut req = state.http_client.request(parts.method.clone(), &url).body(body_bytes.clone());
        for (name, value) in parts.headers.iter() {
            if is_hop_by_hop(name) || name == "authorization" {
                continue;
            }
            req = req.header(name.clone(), value.clone());
        }
        if let Some(key) = &choice.key {
            req = req.bearer_auth(key);
        }

        state.workloads.set(workload.clone(), Phase::Connecting);
        match req.send().await {
            Ok(resp) => {
                // Response headers are in hand: committed (`retry_allowed` is false from here on). The row moves to
                // streaming and the engine stays counted busy until the body ends (StreamGuard).
                state.workloads.set(workload.clone(), Phase::Streaming);
                let guard = StreamGuard {
                    engines: state.engines.clone(),
                    workloads: state.workloads.clone(),
                    engine_id: choice.id.clone(),
                    workload,
                    phase: Phase::StreamBroke,
                };
                return stream_response(resp, guard);
            }
            Err(error) => {
                state.engines.drop_pending(&choice.id);
                let never_connected = error.is_connect() || error.is_timeout();
                if never_connected {
                    state.engines.set_probe(&choice.id, false);
                }
                log(&state.workloads, workload, Phase::FailedConnect);
                let more_engines_left = tried.len() + 1 < state.engines.len();
                if never_connected && send_state.retry_allowed(false, false) && more_engines_left {
                    tried.push(choice.id);
                    continue;
                }
                return fail_to_connect(error.to_string());
            }
        }
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}

fn parse_args() -> Result<(String, String, u16), String> {
    let mut host = "127.0.0.1".to_string();
    let mut node_id = std::env::var("BARO_ROUTER_NODE_ID").unwrap_or_else(|_| "local".into());
    let mut port = 8090u16;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--host" => host = value()?,
            "--node-id" => node_id = value()?,
            "--port" => port = value()?.parse().map_err(|e| format!("--port: {e}"))?,
            "-h" | "--help" => {
                println!("usage: baro-router [--host HOST] [--node-id ID] [--port PORT]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok((host, node_id, port))
}

/// CONTRACT 4: refreshes each engine's resident-state cache from its own
/// `GET /v1/state` every cycle, alongside the existing health probe. A
/// non-baro-serve engine (Ollama, LM Studio, adopted by port probe) or one
/// that is merely unhealthy has no such route; that failure clears the
/// cache (never leaves a stale hash believed resident) rather than erroring
/// the probe loop itself.
/// Is this engine able to take work? Its own `/health` decides: 2xx is yes, 503 (loading, warming, engine dead) is no.
/// An upstream with no `/health` route (Ollama, LM Studio: 404 or 405) is judged by the open port, as before. A port
/// that accepts a connection proves nothing about a baro-serve: it binds before the pack loads and stays open after
/// its engine died.
async fn probe_ok(http_client: &reqwest::Client, address: &str, url: &str) -> bool {
    if TcpStream::connect(address).await.is_err() {
        return false;
    }
    match tokio::time::timeout(Duration::from_secs(3), http_client.get(format!("{url}/health")).send()).await {
        Ok(Ok(resp)) => resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::METHOD_NOT_ALLOWED,
        _ => false,
    }
}

async fn probe_loop(registry: EngineRegistry, http_client: reqwest::Client) {
    loop {
        for engine in registry.snapshot() {
            let ok = probe_ok(&http_client, &engine.address, &engine.url).await;
            registry.set_probe(&engine.id, ok);
            if !ok {
                registry.set_resident(&engine.id, Vec::new(), None);
                registry.set_models(&engine.id, Vec::new());
                continue;
            }
            registry.set_models(&engine.id, fetch_models(&http_client, &engine.url, engine.key.as_deref()).await);
            let state_url = format!("{}/v1/state", engine.url);
            let resident = match http_client.get(&state_url).send().await {
                Ok(resp) if resp.status().is_success() => resp.json::<Value>().await.ok(),
                _ => None,
            };
            match resident {
                Some(value) => {
                    let hashes = value
                        .get("states")
                        .and_then(Value::as_array)
                        .map(|states| states.iter().filter_map(|s| s.get("prefix_hash").and_then(Value::as_str).map(str::to_string)).collect())
                        .unwrap_or_default();
                    registry.set_resident(&engine.id, hashes, Some(value));
                }
                None => registry.set_resident(&engine.id, Vec::new(), None),
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[tokio::main]
async fn main() {
    let (host, node_id, port) = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("baro-router: {error}");
            std::process::exit(2);
        }
    };
    let engines = EngineRegistry::from_env();
    let discovery = DiscoveryAdvertisement::new(&node_id, port);
    let advertise_ip = std::env::var("BARO_ROUTER_ADVERTISE_IP").unwrap_or_else(|_| "127.0.0.1".into());
    eprintln!("discovery advertise={} browse={:?} txt={:?}", discovery.service, discovery.browse_services(), discovery.txt);
    let mdns = match start_mdns(&discovery, &node_id, &advertise_ip) {
        Ok(runtime) => Some(runtime),
        Err(error) => {
            eprintln!("baro-router: mDNS unavailable: {error}");
            None
        }
    };
    let http_client = reqwest::Client::new();
    let state = Arc::new(RouterState {
        node_id,
        http_port: port,
        discovery,
        mdns,
        engines: engines.clone(),
        workloads: WorkloadCatalog::new(),
        next_request: AtomicU64::new(1),
        http_client: http_client.clone(),
        prefix_locks: PrefixLocks(Arc::new(tokio::sync::Mutex::new(HashMap::new()))),
    });
    let state_for_shutdown = state.clone();
    tokio::spawn(probe_loop(engines, http_client));
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/node-info", get(node_info))
        .route("/v1/workloads", get(workloads))
        .route("/v1/models", get(openai_models))
        .route("/v1/engines", post(add_engine))
        .route("/v1/engines/{id}", delete(remove_engine))
        .route("/api/tags", get(ollama_tags))
        .route("/v1/{*path}", any(proxy))
        .route("/api/{*path}", any(proxy))
        .with_state(Shared(state));
    let address: SocketAddr = match format!("{host}:{port}").parse() {
        Ok(address) => address,
        Err(error) => {
            eprintln!("baro-router: invalid listen address: {error}");
            std::process::exit(2);
        }
    };
    let listener = match tokio::net::TcpListener::bind(address).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("baro-router: bind {address}: {error}");
            std::process::exit(1);
        }
    };
    println!("listening on http://{address}");
    if let Err(error) = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(state_for_shutdown)).await {
        eprintln!("baro-router: {error}");
    }
}

/// Waits for SIGINT or SIGTERM, then sends mDNS goodbyes before the caller
/// lets the process exit. `mdns-sd`'s own `Drop` (there isn't one) or a bare
/// process exit does not do this -- see `MdnsRuntime`'s own doc comment for
/// the gate-3 failure this traces to.
async fn shutdown_signal(state: Arc<RouterState>) {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok() };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(error) => eprintln!("baro-router: SIGTERM handler: {error}"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    if state.mdns.is_some() {
        // unregister_all blocks on a flume recv_timeout per record; off the
        // async executor thread so a slow ack never stalls other tasks.
        let state = state.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(runtime) = state.mdns.as_ref() {
                runtime.unregister_all();
            }
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A baro-serve binds its port before the pack loads and keeps it open after its engine died, so an open port
    /// kept a node in rotation that answered every request with 503. The probe reads `/health`.
    #[tokio::test]
    async fn the_probe_reads_health_not_the_open_port() {
        async fn up(code: u16) -> (String, String) {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = l.local_addr().unwrap().to_string();
            let app = Router::new().route("/health", get(move || async move { StatusCode::from_u16(code).unwrap() }));
            tokio::spawn(async move {
                let _ = axum::serve(l, app).await;
            });
            (addr.clone(), format!("http://{addr}"))
        }
        let c = reqwest::Client::new();
        let (a, u) = up(200).await;
        assert!(probe_ok(&c, &a, &u).await);
        let (a, u) = up(503).await;
        assert!(!probe_ok(&c, &a, &u).await, "a loading or dead node is not healthy");
        let (a, u) = up(404).await;
        assert!(probe_ok(&c, &a, &u).await, "an upstream with no /health route is judged by its open port");
        assert!(!probe_ok(&c, "127.0.0.1:1", "http://127.0.0.1:1").await, "a closed port is down");
    }

    #[test]
    fn health_requires_three_failures_and_two_passes() {
        let mut state = HealthHysteresis::default();
        assert!(!state.observe(true));
        assert!(state.observe(true));
        assert!(state.observe(false));
        assert!(state.observe(false));
        assert!(!state.observe(false));
    }

    /// A-M10: three bools and a hysteresis counter said "not healthy" for a new engine, a failing one and a
    /// recovering one alike; /health answered ok whatever the engines did.
    /// A-M11: `pending` dropped when the response HEADERS arrived, so a long generation counted as idle; the
    /// catalog pushed a row per event and grew forever.
    #[tokio::test]
    async fn pending_lasts_until_the_body_ends_and_the_catalog_has_one_row_per_request() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001")]);
        let workloads = WorkloadCatalog::new();
        let row = |id: &str| Workload { request_id: id.into(), method: "POST".into(), path: "/v1/x".into(), state: Phase::Connecting, node_id: "n".into(),
                                        engine: Some("a".into()), placement: "rank".into(), started_ms: 0 };
        let pending = || registry.snapshot()[0].pending;
        registry.bump_pending("a");
        workloads.set(row("r1"), Phase::Connecting);
        workloads.set(row("r1"), Phase::Streaming);
        let guard = StreamGuard { engines: registry.clone(), workloads: workloads.clone(), engine_id: "a".into(), workload: row("r1"), phase: Phase::StreamBroke };
        let body = tokio_stream::iter(vec![Ok::<_, String>(1u8), Ok(2)]);
        let mut tracked = Tracked { inner: body, guard };
        assert_eq!((pending(), workloads.snapshot().len(), workloads.snapshot()[0].state), (1, 1, Phase::Streaming), "headers in hand: still busy");
        while tokio_stream::StreamExt::next(&mut tracked).await.is_some() {}
        drop(tracked);
        assert_eq!((pending(), workloads.snapshot()[0].state), (0, Phase::Done));
        // The client goes away mid-body: the row says so.
        registry.bump_pending("a");
        workloads.set(row("r2"), Phase::Streaming);
        let guard = StreamGuard { engines: registry.clone(), workloads: workloads.clone(), engine_id: "a".into(), workload: row("r2"), phase: Phase::StreamBroke };
        let mut tracked = Tracked { inner: tokio_stream::iter(vec![Ok::<_, String>(1u8), Ok(2)]), guard };
        let _ = tokio_stream::StreamExt::next(&mut tracked).await;
        drop(tracked);
        let rows = workloads.snapshot();
        assert_eq!((pending(), rows.len(), rows[1].state), (0, 2, Phase::StreamBroke));
        for i in 0..(CATALOG_CAP + 5) {
            workloads.set(row(&format!("x{i}")), Phase::Done);
        }
        assert_eq!(workloads.snapshot().len(), CATALOG_CAP);
    }

    #[test]
    fn the_health_state_has_one_word() {
        let mut h = HealthHysteresis::default();
        assert_eq!(h.state(), Health::Unknown);
        h.observe(true);
        assert_eq!(h.state(), Health::Flapping, "one pass is moving toward up");
        h.observe(true);
        assert_eq!(h.state(), Health::Up);
        h.observe(false);
        assert_eq!(h.state(), Health::Flapping, "one failure of an up engine");
        h.observe(false);
        h.observe(false);
        assert_eq!(h.state(), Health::Down);
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        assert_eq!(registry.status_word(), "ok");
        registry.set_probe("a", false);
        registry.set_probe("a", false);
        registry.set_probe("a", false);
        assert_eq!(registry.status_word(), "degraded");
        let fresh = EngineRegistry(Arc::new(RwLock::new(vec![])));
        fresh.add(parse_engine("n=http://127.0.0.1:9009").unwrap());
        assert_eq!(fresh.status_word(), "starting");
        assert!(fresh.none_reason().contains("not probed yet"));
    }

    #[test]
    fn sent_aware_retry_never_retries_after_first_byte() {
        assert!(SendState::NotConnected.retry_allowed(false, false));
        assert!(!SendState::ConnectedNoBytes.retry_allowed(false, false));
        assert!(SendState::ConnectedNoBytes.retry_allowed(true, false));
        assert!(SendState::ConnectedNoBytes.retry_allowed(false, true));
        assert!(!SendState::FirstByteSent.retry_allowed(true, true));
    }

    fn healthy_registry(specs: &[(&str, &str)]) -> EngineRegistry {
        let records = specs
            .iter()
            .map(|(id, endpoint)| EngineRecord { info: parse_engine(&format!("{id}={endpoint}")).unwrap(), health: HealthHysteresis::default() })
            .collect();
        let registry = EngineRegistry(Arc::new(RwLock::new(records)));
        for (id, _) in specs {
            registry.set_probe(id, true);
            registry.set_probe(id, true); // 2 passes -> healthy (HealthHysteresis's own threshold)
        }
        registry
    }

    #[test]
    fn choose_plain_rank_picks_the_lowest_pending() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        registry.bump_pending("b");
        let choice = registry.choose(None, &[]).unwrap();
        assert_eq!(choice.id, "a");
        assert_eq!(choice.placement, "rank");
    }

    #[test]
    fn choose_consistent_is_stable_and_marks_hash_placement() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        let first = registry.choose_consistent(42, &[]).unwrap();
        let second = registry.choose_consistent(42, &[]).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(first.placement, "hash");
    }

    #[tokio::test]
    async fn cold_prefix_lock_releases_waiters() {
        let locks = PrefixLocks(Arc::new(tokio::sync::Mutex::new(HashMap::new())));
        let owner = locks.acquire(42).await;
        let waiter = locks.acquire(42).await;
        assert!(owner.owner);
        assert!(!waiter.owner);
        drop(owner);
        waiter.wait().await;
    }

    /// A-M12: an owner whose upstream never answers held every waiter forever; the wait has a deadline now.
    #[tokio::test]
    async fn a_waiter_does_not_wait_forever_for_a_stuck_owner() {
        std::env::set_var("ROUTER_PREFIX_WAIT_MS", "100");
        let locks = PrefixLocks(Arc::new(tokio::sync::Mutex::new(HashMap::new())));
        let _owner = locks.acquire(7).await;
        let waiter = locks.acquire(7).await;
        let r = tokio::time::timeout(std::time::Duration::from_millis(1500), waiter.wait()).await;
        assert!(r.is_ok(), "the waiter is still blocked on a stuck owner");
    }

    #[test]
    fn choose_locality_bonus_breaks_a_tie_in_the_preferred_engines_favor() {
        // CONTRACT 4: "rank improved by the equivalent of one pending job."
        // b (1 pending) ties a (0 pending) once b's bonus applies (1 - 1 = 0);
        // the tie must resolve to the preferred engine, not silently fall
        // through to plain rank or an arbitrary id ordering.
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        registry.bump_pending("b");
        let choice = registry.choose(Some("b"), &[]).unwrap();
        assert_eq!(choice.id, "b");
        assert_eq!(choice.placement, "locality");
    }

    #[test]
    fn choose_locality_bonus_never_overrides_a_genuinely_worse_engine() {
        // b is 3 requests behind, not 1: the +1 bonus must not manufacture a
        // false preference past what CONTRACT 4 actually promises.
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        registry.bump_pending("b");
        registry.bump_pending("b");
        registry.bump_pending("b");
        let choice = registry.choose(Some("b"), &[]).unwrap();
        assert_eq!(choice.id, "a");
        assert_eq!(choice.placement, "rank");
    }

    #[test]
    fn choose_excludes_listed_ids() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        let choice = registry.choose(None, &["a".to_string()]).unwrap();
        assert_eq!(choice.id, "b");
    }

    #[test]
    fn choose_returns_none_when_every_engine_is_excluded_or_unhealthy() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001")]);
        assert!(registry.choose(None, &["a".to_string()]).is_none());
    }

    #[test]
    fn is_hop_by_hop_strips_connection_framing_not_ordinary_headers() {
        assert!(is_hop_by_hop(&HeaderName::from_static("connection")));
        assert!(is_hop_by_hop(&HeaderName::from_static("content-length")));
        assert!(is_hop_by_hop(&HeaderName::from_static("host")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("content-type")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("authorization")));
    }

    #[test]
    fn requested_model_reads_body_field() {
        assert_eq!(requested_model(br#"{"model":"m1","messages":[]}"#).as_deref(), Some("m1"));
        assert_eq!(requested_model(b"not json"), None);
        assert_eq!(requested_model(br#"{"prompt":"x"}"#), None);
    }

    #[test]
    fn model_routing_excludes_non_owners_and_falls_back_on_unknown() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        registry.set_models("a", vec!["ma".into()]);
        registry.set_models("b", vec!["mb".into(), "shared".into()]);
        assert_eq!(choose_engine(&registry, None, &registry.without_model("mb")).unwrap().id, "b");
        assert_eq!(choose_engine(&registry, None, &registry.without_model("ma")).unwrap().id, "a");
        assert!(registry.without_model("unknown").is_empty());
    }

    #[test]
    fn all_models_merges_and_dedupes() {
        let registry = healthy_registry(&[("a", "http://127.0.0.1:9001"), ("b", "http://127.0.0.1:9002")]);
        registry.set_models("a", vec!["m".into(), "x".into()]);
        registry.set_models("b", vec!["m".into(), "y".into()]);
        let got = registry.all_models();
        assert_eq!(got, vec![("m".into(), "a".into()), ("x".into(), "a".into()), ("y".into(), "b".into())]);
    }

    #[test]
    fn models_parse_openai_and_ollama_shapes() {
        let openai = serde_json::json!({"object": "list", "data": [{"id": "a.gguf"}, {"id": "b"}]});
        assert_eq!(parse_models(&openai), vec!["a.gguf", "b"]);
        let ollama = serde_json::json!({"models": [{"name": "llama3:8b", "model": "llama3:8b"}]});
        assert_eq!(parse_models(&ollama), vec!["llama3:8b"]);
        assert!(parse_models(&serde_json::json!({"error": {"code": 404}})).is_empty());
    }

    #[test]
    fn engine_parser_keeps_port_probe_address() {
        let info = parse_engine("ollama=http://127.0.0.1:11434/").unwrap();
        assert_eq!(info.address, "127.0.0.1:11434");
        assert_eq!(info.port, 11434);
        assert!(!info.adopted);
        assert_eq!(info.url, "http://127.0.0.1:11434");
    }

    #[test]
    fn engine_parser_keeps_https() {
        let info = parse_engine("remote=https://api.example.com/").unwrap();
        assert_eq!(info.url, "https://api.example.com:443");
        assert_eq!(info.address, "api.example.com:443");
        let bare = parse_engine("lan=10.0.0.5:8090").unwrap();
        assert_eq!(bare.url, "http://10.0.0.5:8090");
        assert!(parse_engine("bad=ftp://h:1").is_none());
    }

    #[test]
    fn engine_parser_reads_key_from_env() {
        std::env::set_var("ROUTER_TEST_KEY_1", "sk-test");
        let info = parse_engine("oai=https://api.example.com@ROUTER_TEST_KEY_1").unwrap();
        assert_eq!(info.key.as_deref(), Some("sk-test"));
        assert_eq!(info.url, "https://api.example.com:443");
        assert!(parse_engine("oai=https://api.example.com@ROUTER_TEST_KEY_UNSET").is_none());
    }

    #[test]
    fn registry_pins_engine_prefixed_models() {
        let registry = EngineRegistry(Arc::new(RwLock::new(Vec::new())));
        registry.add(engine_info("u1-home", "127.0.0.1:9000").unwrap());
        assert_eq!(registry.pin("u1-home/qwen3:8b"), Some(("u1-home".into(), "qwen3:8b".into())));
        assert_eq!(registry.pin("library/qwen3"), None);
        registry.add(engine_info("u1-home", "127.0.0.1:9001").unwrap());
        assert_eq!(registry.ids(), vec!["u1-home"]);
        assert!(registry.remove("u1-home"));
        assert!(!registry.remove("u1-home"));
    }

    #[test]
    fn discovery_answers_both_service_shapes() {
        let advertisement = DiscoveryAdvertisement::new("test", 8090);
        assert_eq!(advertisement.browse_services(), [BARO_SERVICE, PAIR_SERVICE]);
        assert_eq!(advertisement.txt.get("ni"), Some(&"8090".to_string()));
    }
}
