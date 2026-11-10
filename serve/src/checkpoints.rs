//! Checkpoint API (LatentOS `docs/design/latent-os/10-checkpoint-api.md`,
//! sec 8): the engine's BAROST01 state file as a first-class object.
//!
//! `POST /v1/checkpoints` runs a prompt once and asks the engine to write its
//! full state (KV + conv + SSM + tokens) to a file under `BARO_CKPT_DIR`;
//! `POST /v1/checkpoints/{id}/fork` runs branches that load that file before
//! the prefix lookup, so only the suffix is prefilled. The payload never
//! crosses HTTP: the object is metadata, the file is the payload.
//!
//! Identity: `pack` is `<pack>/identity.json`'s `pack_sha256` (P1-STATE-API.md
//! CONTRACT 2, coordinator amendment 0c68cb4) when that file is present and
//! its recorded `pack_bytes`/`pack_mtime_ns` still match `pack.bin` on disk;
//! otherwise the hex sha256 of the pack PATH STRING, as before (`load_state`
//! refuses a mismatch here with "saved from a different pack"). `runtime` is
//! the engine repo's git HEAD; `tokenizer_sha` is sha256 of the tokenizer
//! file, the same algorithm `tools/engine-pack.py --identity` uses for
//! `identity.json`'s `tokenizer_sha256`, computed independently here (one
//! algorithm, not one shared computation) rather than trusted from the file.
//! `weights_uuid` comes from `identity.json`'s GGUF UUID when present, or the
//! validated pack digest as the stable pack UUID fallback.

use super::*;
use axum::extract::Path as UrlPath;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

// ---- sha256, dependency-free -----------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
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

/// Incremental SHA-256: `update` any number of times, `finalize` once. P1
/// (CONTRACT 1) hashes multi-GB state streams in 8 MiB chunks as they cross
/// the wire; a one-shot `sha256(&[u8])` would need the whole stream in one
/// `Vec` first, exactly what that contract forbids.
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    len: u64,
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 { h: H0, buf: [0u8; 64], buf_len: 0, len: 0 }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                compress(&mut self.h, &self.buf);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            compress(&mut self.h, block.try_into().unwrap());
            data = rest;
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            compress(&mut self.h, &self.buf);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut self.h, &self.buf);
        let mut out = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Sha256 {
        Sha256::new()
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

pub fn hex(b: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(b.len() * 2);
    for &x in b {
        out.push(LUT[(x >> 4) as usize] as char);
        out.push(LUT[(x & 0x0f) as usize] as char);
    }
    out
}

// ---- identity ---------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub pack: String,
    pub runtime: String,
    pub tokenizer_sha: Option<String>,
    pub weights_uuid: Option<String>,
    /// True when `pack` came from a validated `identity.json` (CONTRACT 2's
    /// `pack_sha256`), false when it fell back to a hash of the pack PATH
    /// string. `GET /v1/state` reports the false case as `"portable":false`.
    pub portable: bool,
}

/// `<pack>/identity.json`'s shape (`tools/engine-pack.py`); only the fields
/// this reader needs. `tokenizer_sha256` and `source`/`general_uuid` are not
/// read here: this struct exists to validate and extract `pack_sha256`.
#[derive(Debug, Clone, Deserialize)]
struct PackIdentityFile {
    pack_sha256: String,
    pack_bytes: u64,
    pack_mtime_ns: u64,
    #[serde(default)]
    general_uuid: Option<String>,
}

/// `pack_sha256` from `<pack>/identity.json`, only when the file parses and
/// its `pack_bytes`/`pack_mtime_ns` still match `<pack>/pack.bin` right now
/// (a stat, not a re-hash of a file that can be many GB). Any miss returns
/// `None`, never an error: the caller's own path-hash fallback is not a
/// defect, it is CONTRACT 2's documented "identity.json absent" path.
fn read_pack_identity(pack: &Path) -> Option<PackIdentityFile> {
    let text = std::fs::read_to_string(pack.join("identity.json")).ok()?;
    let parsed: PackIdentityFile = serde_json::from_str(&text).ok()?;
    // A sha256 hex digest is always exactly 64 ASCII hex chars; reject
    // anything else here rather than let a hand-edited or truncated file
    // panic later at `Identity::weights_uuid_hex`'s `&self.pack[..32]`.
    if parsed.pack_sha256.len() != 64 || !parsed.pack_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let meta = std::fs::metadata(pack.join("pack.bin")).ok()?;
    let mtime_ns = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos() as u64;
    if meta.len() != parsed.pack_bytes || mtime_ns != parsed.pack_mtime_ns {
        return None;
    }
    Some(parsed)
}

#[cfg(test)]
fn read_pack_sha256(pack: &Path) -> Option<String> {
    read_pack_identity(pack).map(|identity| identity.pack_sha256)
}

fn normalize_uuid(value: &str) -> Option<String> {
    let compact: String = value.chars().filter(|c| *c != '-').collect();
    (compact.len() == 32 && compact.bytes().all(|b| b.is_ascii_hexdigit())).then(|| compact.to_ascii_lowercase())
}

impl Identity {
    pub fn compute(pack: &Path, tokenizer: &Path) -> Identity {
        // The engine salts its checkpoint hashes with sha256 of the pack
        // path STRING it was given (serve/prefix.mojo `Chain.__init__`), so
        // the same spelling is hashed here -- unchanged fallback for a pack
        // with no valid identity.json (CONTRACT 2).
        let pack_s = pack.to_string_lossy().into_owned();
        let runtime = std::env::var("BARO_RUNTIME").ok().unwrap_or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".into())
        });
        let tokenizer_sha = std::fs::read(tokenizer).ok().map(|b| hex(&sha256(&b)));
        let recorded = read_pack_identity(pack);
        let recorded_pack_sha256 = recorded.as_ref().map(|identity| identity.pack_sha256.clone());
        let portable = recorded_pack_sha256.is_some();
        let pack_field = recorded_pack_sha256.clone().unwrap_or_else(|| hex(&sha256(pack_s.as_bytes())));
        let weights_uuid = recorded
            .as_ref()
            .and_then(|identity| identity.general_uuid.as_deref().and_then(normalize_uuid))
            .or_else(|| portable.then(|| pack_field[..32].to_string()));
        Identity { pack: pack_field, runtime, tokenizer_sha, weights_uuid, portable }
    }

    /// CONTRACT 2 LAT1 mapping: `weights_uuid` (16 B) is `pack_sha256`'s
    /// first 16 bytes, as hex -- only meaningful when `portable` (a
    /// path-hash fallback is not a content identity, so it must not be
    /// dressed up as one).
    pub fn weights_uuid_hex(&self) -> Option<&str> {
        self.weights_uuid.as_deref()
    }

    fn json(&self) -> Value {
        json!({"pack": self.pack, "runtime": self.runtime, "tokenizer_sha": self.tokenizer_sha, "weights_uuid": self.weights_uuid})
    }

    /// First field a client-supplied identity disagrees on, if any. Absent
    /// or null client fields are not compared.
    pub fn first_mismatch(&self, claimed: &Value) -> Option<String> {
        for (name, ours) in [
            ("pack", Some(self.pack.as_str())),
            ("runtime", Some(self.runtime.as_str())),
            ("tokenizer_sha", self.tokenizer_sha.as_deref()),
            ("weights_uuid", self.weights_uuid.as_deref()),
        ] {
            if let Some(theirs) = claimed.get(name).and_then(Value::as_str) {
                if ours != Some(theirs) {
                    return Some(name.to_string());
                }
            }
        }
        None
    }
}

// ---- registry ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Ckpt {
    pub bytes: u64,
    pub created: u64,
    pub ttl_s: u64,
    pub expires_at: u64,
    pub pos: usize,
    pub id: String,
    pub path: PathBuf,
    pub tokens: Vec<u32>,
    pub identity: Identity,
}

impl Ckpt {
    pub fn json(&self) -> Value {
        json!({"id": self.id, "object": "checkpoint", "pos": self.pos, "bytes": self.bytes,
               "created": self.created, "ttl_s": self.ttl_s, "expires_at": self.expires_at,
               "identity": self.identity.json()})
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskCkpt {
    pub bytes: u64,
    pub created: u64,
    pub ttl_s: u64,
    pub expires_at: u64,
    pub pos: usize,
    pub id: String,
    pub tokens: Vec<u32>,
    pub identity: Identity,
}

impl From<&Ckpt> for DiskCkpt {
    fn from(c: &Ckpt) -> Self {
        Self { id: c.id.clone(), pos: c.pos, tokens: c.tokens.clone(), bytes: c.bytes, created: c.created,
               ttl_s: c.ttl_s, expires_at: c.expires_at, identity: c.identity.clone() }
    }
}

#[derive(Debug)]
pub enum InsertError {
    ApiCap(usize),
    DiskCap { used: u64, needed: u64, cap: u64 },
    Io(String),
}

pub struct Registry {
    pub dir: PathBuf,
    pub cap: usize,
    /// Optional total bytes allowed for `.baro` state files. Set with
    /// `BARO_CKPT_DISK_BYTES`; unset or zero means unlimited. A create that
    /// would exceed it is rejected with HTTP 507 and leaves no state file.
    pub disk_cap: Option<u64>,
    items: Mutex<HashMap<String, Ckpt>>,
    /// Checkpoints a request is using right now (id -> holders): `sweep` leaves them and their files alone.
    leases: Arc<Mutex<HashMap<String, usize>>>,
}

/// A request's hold on a checkpoint: while it lives the sweep neither forgets the entry nor deletes its file, so the
/// engine finds the state file it was told to load. Dropped when the request ends.
pub struct Lease {
    id: String,
    leases: Arc<Mutex<HashMap<String, usize>>>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut m = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = m.get_mut(&self.id) {
            *n -= 1;
            if *n == 0 {
                m.remove(&self.id);
            }
        }
    }
}

/// Where a request's restored state came from, reported as `state_source` so a caller reads which branch ran instead of
/// inferring it from `restored` and `checkpoint_from` (which was set for two of the three).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateSource {
    /// No checkpoint: the whole prompt was prefilled.
    None,
    /// The caller named it (`checkpoint_id`, `/v1/turn`, checkpoint fork).
    Explicit,
    /// The route created it for this request (`checkpoint: true`).
    Created,
    /// The longest cached prefix of the rendered prompt, picked by the server.
    Auto,
}

impl StateSource {
    pub fn word(self) -> &'static str {
        match self {
            StateSource::None => "none",
            StateSource::Explicit => "explicit",
            StateSource::Created => "created",
            StateSource::Auto => "auto",
        }
    }
}

pub const DEFAULT_TTL_S: u64 = 300;

impl Registry {
    pub fn from_env() -> Registry {
        let dir = PathBuf::from(std::env::var("BARO_CKPT_DIR").unwrap_or_else(|_| ".work/checkpoints".into()));
        let cap = std::env::var("BARO_CKPT_API").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let disk_cap = std::env::var("BARO_CKPT_DISK_BYTES").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0);
        Registry { dir, cap, disk_cap, items: Mutex::new(HashMap::new()), leases: Default::default() }
    }

    #[cfg(test)]
    pub fn new(dir: PathBuf, cap: usize) -> Registry {
        Registry { dir, cap, disk_cap: None, items: Mutex::new(HashMap::new()), leases: Default::default() }
    }

    /// `lat:state@pos<pos>:<first 16 hex of sha256(tokens as little-endian u32)>`.
    pub fn id_for(tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        let pos = tokens.len().saturating_sub(1);
        format!("lat:state@pos{pos}:{}", &hex(&sha256(&bytes))[..16])
    }

    pub fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.baro", id.replace([':', '@'], "_")))
    }

    pub fn sidecar_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", id.replace([':', '@'], "_")))
    }

    fn hydrate(&self, items: &mut HashMap<String, Ckpt>, now: u64) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("json") { continue }
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let Ok(meta) = serde_json::from_slice::<DiskCkpt>(&bytes) else { continue };
            if meta.expires_at <= now {
                if self.is_leased(&meta.id) {
                    continue;
                }
                let _ = std::fs::remove_file(self.path_for(&meta.id));
                let _ = std::fs::remove_file(path);
                continue;
            }
            let state_path = self.path_for(&meta.id);
            let Ok(file_meta) = std::fs::metadata(&state_path) else {
                // A sidecar with no state file is an orphan (an engine error, or a file removed by hand): nothing can
                // load it, and it used to pile up in BARO_CKPT_DIR for good.
                eprintln!("checkpoints: removing orphan sidecar {} (no state file)", path.display());
                let _ = std::fs::remove_file(path);
                continue;
            };
            if file_meta.len() != meta.bytes { continue }
            items.entry(meta.id.clone()).or_insert(Ckpt {
                id: meta.id,
                pos: meta.pos,
                tokens: meta.tokens,
                path: state_path,
                bytes: meta.bytes,
                created: meta.created,
                ttl_s: meta.ttl_s,
                expires_at: meta.expires_at,
                identity: meta.identity,
            });
        }
    }

    /// Drop expired entries (and their files); returns the live count.
    fn sweep(&self, items: &mut HashMap<String, Ckpt>, now: u64) -> usize {
        let gone: Vec<String> = items.values().filter(|c| c.expires_at <= now && !self.is_leased(&c.id)).map(|c| c.id.clone()).collect();
        for id in gone {
            if let Some(c) = items.remove(&id) {
                let _ = std::fs::remove_file(&c.path);
                let _ = std::fs::remove_file(self.sidecar_for(&id));
            }
        }
        items.len()
    }

    fn is_leased(&self, id: &str) -> bool {
        self.leases.lock().unwrap_or_else(|e| e.into_inner()).contains_key(id)
    }

    /// `get` plus a lease taken under the same lock, so no sweep can run between the lookup and the hold. The
    /// caller keeps the `Lease` until the engine request that reads the file is done.
    pub fn lease(&self, id: &str, now: u64) -> Option<(Ckpt, Lease)> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now);
        self.sweep(&mut items, now);
        let c = items.get(id).cloned()?;
        *self.leases.lock().unwrap_or_else(|e| e.into_inner()).entry(id.to_string()).or_insert(0) += 1;
        Some((c, Lease { id: id.to_string(), leases: self.leases.clone() }))
    }

    pub fn live(&self, now: u64) -> Vec<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now);
        self.sweep(&mut items, now);
        let mut v: Vec<Ckpt> = items.values().cloned().collect();
        v.sort_by_key(|c| c.created);
        v
    }

    pub fn get(&self, id: &str, now: u64) -> Option<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now);
        self.sweep(&mut items, now);
        items.get(id).cloned()
    }

    /// Finds the cached checkpoint with the longest prefix matching `tokens`.
    pub fn find_longest_prefix(&self, tokens: &[u32], now: u64) -> Option<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now);
        self.sweep(&mut items, now);
        let mut best: Option<Ckpt> = None;
        for c in items.values() {
            if tokens.len() >= c.tokens.len() && tokens[..c.tokens.len()] == c.tokens[..]
                && best.as_ref().is_none_or(|b| c.tokens.len() > b.tokens.len()) {
                    best = Some(c.clone());
                }
        }
        best
    }

    /// Err(live count) when the cap is reached and `id` is not already held.
    pub fn insert(&self, c: Ckpt, now: u64) -> Result<(), InsertError> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now);
        let n = self.sweep(&mut items, now);
        if n >= self.cap && !items.contains_key(&c.id) {
            return Err(InsertError::ApiCap(n));
        }
        let used: u64 = items.values().map(|x| x.bytes).sum();
        if let Some(cap) = self.disk_cap {
            let old = items.get(&c.id).map(|x| x.bytes).unwrap_or(0);
            let projected = used.saturating_sub(old).saturating_add(c.bytes);
            if projected > cap {
                return Err(InsertError::DiskCap { used: used.saturating_sub(old), needed: c.bytes, cap });
            }
        }
        let sidecar = self.sidecar_for(&c.id);
        std::fs::create_dir_all(&self.dir).map_err(|e| InsertError::Io(format!("{}: {e}", self.dir.display())))?;
        let data = serde_json::to_vec_pretty(&DiskCkpt::from(&c)).map_err(|e| InsertError::Io(e.to_string()))?;
        std::fs::write(&sidecar, data).map_err(|e| InsertError::Io(format!("{}: {e}", sidecar.display())))?;
        items.insert(c.id.clone(), c);
        Ok(())
    }

    pub fn disk_usage(&self, now: u64) -> u64 {
        self.live(now).iter().map(|c| c.bytes).sum()
    }

    pub fn remove(&self, id: &str) -> Option<Ckpt> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        self.hydrate(&mut items, now());
        let c = items.remove(id);
        if let Some(c) = &c {
            let _ = std::fs::remove_file(&c.path);
            let _ = std::fs::remove_file(self.sidecar_for(id));
        }
        c
    }
}

// ---- routes -----------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateReq {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    ttl_s: Option<u64>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

pub fn insert_error(err: InsertError, cap: usize) -> ApiError {
    match err {
        InsertError::ApiCap(n) => ApiError::Plain(StatusCode::INSUFFICIENT_STORAGE, format!("checkpoint cap {cap} reached ({n} live)")),
        InsertError::DiskCap { used, needed, cap } => ApiError::Plain(StatusCode::INSUFFICIENT_STORAGE, format!("checkpoint disk budget {cap} bytes exceeded (used {used}, need {needed})")),
        InsertError::Io(msg) => ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("checkpoint metadata: {msg}")),
    }
}

pub fn not_found(id: &str) -> ApiError {
    ApiError::Plain(StatusCode::NOT_FOUND, format!("no live checkpoint {id}"))
}

/// The engine refuses a state file saved from another pack; that is an
/// identity mismatch on `pack`, not an engine fault.
fn map_engine_err(e: ApiError) -> ApiError {
    match &e {
        ApiError::Plain(_, msg) if msg.contains("saved from a different pack") => ApiError::Mismatch("pack".into()),
        _ => e,
    }
}

/// An engine error for a request that loads checkpoint `c`, while its state file is gone from disk: the checkpoint
/// expired or was removed before the engine read it. That is 410, not a 400 the caller cannot tell from a malformed request.
pub fn expired_if_gone(e: ApiError, c: &Ckpt) -> ApiError {
    match &e {
        ApiError::Plain(StatusCode::BAD_REQUEST | StatusCode::BAD_GATEWAY, _) if !c.path.exists() => {
            ApiError::Plain(StatusCode::GONE, format!("checkpoint {} expired or was removed before the engine loaded it", c.id))
        }
        _ => e,
    }
}

pub async fn create(State(app): State<Shared>, Json(r): Json<CreateReq>) -> Result<Response, ApiError> {
    let tokens = prompt_ids(&app, &r.prompt)?;
    let id = Registry::id_for(&tokens);
    if let Some(c) = app.ckpts.get(&id, now()) {
        return Ok(Json(json!({"created": false, "checkpoint": c.json()})).into_response());
    }
    let spec = spec_default(&app, r.spec);
    let c = create_from_tokens_full(&app, tokens, r.ttl_s, spec, r.max_tokens, r.stop, r.sampler.to_sample_params(None), r.model).await?;
    Ok(Json(json!({"created": true, "checkpoint": c.0, "completion": c.1})).into_response())
}

/// The checkpoint of `tokens` (created if absent), for the chat route's
/// `checkpoint: true` (QS batch item 8): one 1-token greedy prefill that
/// writes the state file, then the caller's own request forks it hot.
pub async fn create_from_tokens(app: &Shared, tokens: Vec<u32>, ttl_s: Option<u64>, spec: bool) -> Result<Ckpt, ApiError> {
    let id = Registry::id_for(&tokens);
    if let Some(c) = app.ckpts.get(&id, now()) {
        return Ok(c);
    }
    let (_, _, c) = create_from_tokens_full(app, tokens, ttl_s, spec, Some(1), None, crate::protocol::SampleParams::default(), None).await?;
    Ok(c)
}

#[allow(clippy::too_many_arguments)]
async fn create_from_tokens_full(
    app: &Shared, tokens: Vec<u32>, ttl_s: Option<u64>, spec: bool, max_tokens: Option<u32>, stop: Option<StopParam>,
    sample: crate::protocol::SampleParams, model: Option<String>,
) -> Result<(Value, Value, Ckpt), ApiError> {
    if tokens.len() < 2 {
        return Err(bad("prompt must hold at least 2 tokens (the checkpoint is taken at len-1)"));
    }
    let ttl_s = ttl_s.unwrap_or(DEFAULT_TTL_S).max(1);
    let id = Registry::id_for(&tokens);
    let t_now = now();
    let live = app.ckpts.live(t_now).len();
    if live >= app.ckpts.cap {
        return Err(ApiError::Plain(StatusCode::INSUFFICIENT_STORAGE, format!("checkpoint cap {} reached ({live} live)", app.ckpts.cap)));
    }
    std::fs::create_dir_all(&app.ckpts.dir).map_err(|e| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", app.ckpts.dir.display())))?;
    let path = app.ckpts.path_for(&id);
    let abs = std::fs::canonicalize(&app.ckpts.dir).unwrap_or_else(|_| app.ckpts.dir.clone()).join(path.file_name().unwrap_or_default());
    let model = model.unwrap_or_else(|| app.model.clone());
    let g = Gen {
        prompt: tokens.clone(),
        n: max_tokens.unwrap_or(1),
        spec,
        stream: false,
        stop: compute_stop(app, stop),
        ckpt: vec![],
        state: (Some(abs.to_string_lossy().into_owned()), None),
        state_prefix: false,
        base_pos: 0,
        embeds: vec![],
        sample,
        schema: None,
        reasoning: None,
        embed: None,
    };
    let n_prompt = g.prompt.len();
    let (req_id, rx) = check_and_submit(app, &g)?;
    let (acc, text_out, stats) = collect(app, rx).await?;
    let bytes = std::fs::metadata(&abs).map(|m| m.len()).map_err(|e| {
        ApiError::Plain(StatusCode::BAD_GATEWAY, format!("engine finished but wrote no state file at {}: {e}", abs.display()))
    })?;
    let c = Ckpt { id: id.clone(), pos: tokens.len() - 1, tokens, path: abs, bytes, created: t_now, ttl_s, expires_at: t_now + ttl_s,
                   identity: app.identity.clone() };
    let cj = c.json();
    if let Err(err) = app.ckpts.insert(c, t_now) {
        let _ = std::fs::remove_file(app.ckpts.path_for(&id));
        let _ = std::fs::remove_file(app.ckpts.sidecar_for(&id));
        return Err(insert_error(err, app.ckpts.cap));
    }
    let c = app.ckpts.get(&id, t_now).ok_or_else(|| ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, "checkpoint vanished after insert".into()))?;
    let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
    let completion = json!({"id": format!("cmpl-{req_id}"), "model": model, "text": text_out, "tokens": acc.tokens,
                            "finish_reason": reason, "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats});
    Ok((cj, completion, c))
}

pub async fn list(State(app): State<Shared>) -> Json<Value> {
    let items: Vec<Value> = app.ckpts.live(now()).iter().map(Ckpt::json).collect();
    Json(json!({"object": "list", "data": items, "cap": app.ckpts.cap, "disk_cap": app.ckpts.disk_cap, "disk_bytes": app.ckpts.disk_usage(now()), "dir": app.ckpts.dir}))
}

pub async fn get_one(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Result<Json<Value>, ApiError> {
    let c = app.ckpts.get(&id, now()).ok_or_else(|| not_found(&id))?;
    Ok(Json(c.json()))
}

pub async fn delete(State(app): State<Shared>, UrlPath(id): UrlPath<String>) -> Json<Value> {
    Json(json!({"id": id, "deleted": app.ckpts.remove(&id).is_some()}))
}

#[derive(Deserialize)]
pub struct CkptForkReq {
    #[serde(default)]
    model: Option<String>,
    branches: Vec<CkptForkBranch>,
    /// Optional identity the client believes the checkpoint has; any field
    /// that disagrees with this server's is refused with 409 before the
    /// engine is touched.
    #[serde(default)]
    identity: Option<Value>,
}

#[derive(Deserialize)]
pub struct CkptForkBranch {
    /// Text or token ids appended after the checkpointed prompt.
    #[serde(default)]
    prompt_suffix: Option<Value>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    spec: Option<bool>,
    #[serde(default)]
    stop: Option<StopParam>,
    #[serde(flatten)]
    sampler: SamplerFields,
}

pub async fn fork(State(app): State<Shared>, UrlPath(id): UrlPath<String>, Json(r): Json<CkptForkReq>) -> Result<Response, ApiError> {
    if r.branches.is_empty() {
        return Err(bad("branches must be non-empty"));
    }
    // Held until the last branch is done: a later request's sweep must not delete the file between two branches.
    let (c, _lease) = app.ckpts.lease(&id, now()).ok_or_else(|| not_found(&id))?;
    if c.identity.pack != app.identity.pack {
        return Err(ApiError::Mismatch("pack".into()));
    }
    if let Some(field) = r.identity.as_ref().and_then(|v| app.identity.first_mismatch(v)) {
        return Err(ApiError::Mismatch(field));
    }
    let model = r.model.unwrap_or_else(|| app.model.clone());
    let mut branches = Vec::with_capacity(r.branches.len());
    for (i, b) in r.branches.into_iter().enumerate() {
        // P2b: the engine line carries the suffix only (`state_prefix`); the
        // engine prepends the checkpoint's own token list from the state file
        // header, so the fork request is O(suffix) instead of O(P) (front gap
        // 15 ms at 8k, 55 ms at 32k before, state-timing 2026-09-24).
        // The state file holds tokens[0..pos] (pos = len(P) - 1: the last
        // prompt token is the one whose forward produced the checkpointed
        // step), so the line carries the rest of P plus the suffix.
        let mut prompt: Vec<u32> = c.tokens[c.pos..].to_vec();
        if let Some(sfx) = &b.prompt_suffix {
            let extra = match sfx {
                // A suffix continues the prompt, so no BOS / special prefix.
                Value::String(s) => need_text(&app)?.encode(s, false).map_err(bad)?,
                other => prompt_ids(&app, other)?,
            };
            prompt.extend(extra);
        }
        let g = Gen {
            prompt,
            n: b.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            spec: spec_default(&app, b.spec),
            stream: false,
            stop: compute_stop(&app, b.stop),
            ckpt: vec![],
            state: (None, Some(c.path.to_string_lossy().into_owned())),
            state_prefix: true,
            base_pos: c.pos,
            embeds: vec![],
            sample: b.sampler.to_sample_params(None),
            schema: None,
            reasoning: None,
            embed: None,
        };
        let n_prompt = c.pos + g.prompt.len();
        let (req_id, rx) = check_and_submit(&app, &g)?;
        let (acc, text_out, stats) = collect(&app, rx).await.map_err(map_engine_err).map_err(|e| expired_if_gone(e, &c))?;
        let cached = stats.get("cached").and_then(Value::as_u64).unwrap_or(0) as usize;
        let (tier, tier_source) = app.metrics.record_fork(&stats);
        let reason = acc.finish_reason(stats.get("finish").and_then(Value::as_str));
        branches.push(json!({"tier": tier, "tier_source": tier_source, "state_source": StateSource::Explicit.word(),
            "index": i, "id": format!("fork-{req_id}"), "text": text_out, "tokens": acc.tokens,
            "finish_reason": reason, "restored": cached >= c.pos, "cached_tokens": cached,
            "usage": usage_json(n_prompt, acc.tokens.len(), &stats), "timings": stats,
        }));
    }
    Ok(Json(json!({"object": "checkpoint.fork", "model": model, "checkpoint": id, "pos": c.pos, "branches": branches})).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha256_streamed_in_arbitrary_chunks_matches_the_one_shot_hash() {
        // Sized past several 64-byte compression blocks and one padding
        // boundary, so both the block loop and the finalize-only tail path
        // in `Sha256::update`/`finalize` run: the P1 import loop feeds
        // 8 MiB chunks that never land on a 64-byte boundary either.
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let whole = sha256(&data);
        for chunk_size in [1usize, 3, 64, 65, 8191, 8192] {
            let mut h = Sha256::new();
            for chunk in data.chunks(chunk_size) {
                h.update(chunk);
            }
            assert_eq!(h.finalize(), whole, "chunk_size={chunk_size}");
        }
    }

    fn write_pack_fixture(dir: &Path, pack_bytes: &[u8]) -> u64 {
        std::fs::write(dir.join("pack.bin"), pack_bytes).unwrap();
        let meta = std::fs::metadata(dir.join("pack.bin")).unwrap();
        meta.modified().unwrap().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }

    // sha256(b"hello world"), verified with Python's hashlib, not guessed
    // (the same lesson as state.rs's fabricated-then-corrected test vector):
    //   python3 -c "import hashlib; print(hashlib.sha256(b'hello world').hexdigest())"
    const HELLO_WORLD_SHA256: &str = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    #[test]
    fn read_pack_sha256_returns_the_recorded_hash_when_stat_matches() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"{HELLO_WORLD_SHA256}","pack_bytes":11,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), Some(HELLO_WORLD_SHA256.to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_is_none_on_a_size_mismatch() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-size-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"{HELLO_WORLD_SHA256}","pack_bytes":999,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_is_none_when_identity_json_is_absent() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-absent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_pack_fixture(&dir, b"hello world");
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_pack_sha256_rejects_a_malformed_hash_instead_of_panicking_later() {
        let dir = std::env::temp_dir().join(format!("baro-identity-test-malformed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mtime_ns = write_pack_fixture(&dir, b"hello world");
        // Too short: weights_uuid_hex's &pack[..32] would panic on this if
        // it were ever accepted.
        std::fs::write(
            dir.join("identity.json"),
            format!(r#"{{"pack_sha256":"short","pack_bytes":11,"pack_mtime_ns":{mtime_ns}}}"#),
        )
        .unwrap();
        assert_eq!(read_pack_sha256(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn weights_uuid_hex_is_none_when_not_portable() {
        let me = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: None, weights_uuid: None, portable: false };
        assert_eq!(me.weights_uuid_hex(), None);
    }

    #[test]
    fn weights_uuid_hex_is_the_first_32_hex_chars_when_portable() {
        let first_half = "a".repeat(32);
        let pack = first_half.clone() + &"b".repeat(32);
        let me = Identity { pack, runtime: "r".into(), tokenizer_sha: None, weights_uuid: Some(first_half.clone()), portable: true };
        assert_eq!(me.weights_uuid_hex(), Some(first_half.as_str()));
    }

    #[test]
    fn id_is_stable_and_position_bound() {
        let a = Registry::id_for(&[1, 2, 3]);
        assert_eq!(a, Registry::id_for(&[1, 2, 3]));
        assert!(a.starts_with("lat:state@pos2:"));
        assert_ne!(a, Registry::id_for(&[1, 2, 4]));
    }

    #[test]
    fn registry_expires_and_caps() {
        let dir = std::env::temp_dir().join(format!("baro-ckpt-test-{}", std::process::id()));
        let reg = Registry::new(dir.clone(), 1);
        let ident = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: None, weights_uuid: None, portable: false };
        let mk = |id: &str, exp: u64| Ckpt { id: id.into(), pos: 1, tokens: vec![1, 2], path: reg.path_for(id), bytes: 0, created: 0, ttl_s: 1, expires_at: exp, identity: ident.clone() };
        assert!(reg.insert(mk("a", 10), 5).is_ok());
        assert!(matches!(reg.insert(mk("b", 10), 5), Err(InsertError::ApiCap(1))));
        assert!(reg.get("a", 5).is_some());
        assert!(reg.get("a", 10).is_none(), "expired at its expires_at");
        assert!(reg.insert(mk("b", 20), 11).is_ok(), "cap frees after expiry");
        assert_eq!(reg.live(11).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn lease_dir(tag: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.work/fix-front-test").join(tag);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn ckpt_with_file(reg: &Registry, tokens: Vec<u32>, expires_at: u64) -> Ckpt {
        let id = Registry::id_for(&tokens);
        let path = reg.path_for(&id);
        std::fs::write(&path, b"state").unwrap();
        let identity = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: None, weights_uuid: None, portable: false };
        Ckpt { id, pos: tokens.len() - 1, tokens, path, bytes: 5, created: 1, ttl_s: 60, expires_at, identity }
    }

    /// A-M6: a request read `c.path` from the registry and a later request's sweep deleted the file before the engine
    /// loaded it (the engine error came back as a 400). A lease holds the entry and its file until the request ends.
    #[test]
    fn a_lease_keeps_the_state_file_through_a_sweep() {
        let dir = lease_dir("a-m6-lease");
        let reg = Registry::new(dir.clone(), 4);
        let c = ckpt_with_file(&reg, vec![1, 2, 3], 10);
        let id = c.id.clone();
        reg.insert(c, 5).ok().unwrap();
        let (held, lease) = reg.lease(&id, 5).expect("live at 5");
        assert!(reg.live(100).len() == 1 && reg.get(&id, 100).is_some(), "a leased entry survives the sweep");
        assert!(held.path.exists(), "the file is still there for the engine");
        drop(lease);
        assert!(reg.get(&id, 100).is_none(), "once the lease is dropped the sweep takes it");
        assert!(!held.path.exists(), "and removes the file");
    }

    /// A-M14: the response says which branch of the ladder ran.
    #[test]
    fn state_source_words_are_the_documented_four() {
        let words: Vec<&str> = [StateSource::None, StateSource::Explicit, StateSource::Created, StateSource::Auto].iter().map(|s| s.word()).collect();
        assert_eq!(words, ["none", "explicit", "created", "auto"]);
    }

    #[test]
    fn a_sidecar_with_no_state_file_is_removed() {
        let dir = lease_dir("a-m6-orphan");
        let reg = Registry::new(dir.clone(), 4);
        let c = ckpt_with_file(&reg, vec![4, 5, 6], 1000);
        let sidecar = reg.sidecar_for(&c.id);
        std::fs::write(&sidecar, serde_json::to_vec(&DiskCkpt::from(&c)).unwrap()).unwrap();
        std::fs::remove_file(&c.path).unwrap();
        assert!(reg.live(5).is_empty());
        assert!(!sidecar.exists(), "the orphan sidecar must not pile up");
    }

    #[test]
    fn an_engine_error_for_a_vanished_checkpoint_file_is_a_410() {
        let dir = lease_dir("a-m6-gone");
        let reg = Registry::new(dir.clone(), 4);
        let c = ckpt_with_file(&reg, vec![7, 8, 9], 1000);
        let plain = |code| ApiError::Plain(code, "engine: cannot open".into());
        assert!(matches!(expired_if_gone(plain(StatusCode::BAD_REQUEST), &c), ApiError::Plain(StatusCode::BAD_REQUEST, _)), "file present: unchanged");
        std::fs::remove_file(&c.path).unwrap();
        assert!(matches!(expired_if_gone(plain(StatusCode::BAD_REQUEST), &c), ApiError::Plain(StatusCode::GONE, _)));
        assert!(matches!(expired_if_gone(plain(StatusCode::TOO_MANY_REQUESTS), &c), ApiError::Plain(StatusCode::TOO_MANY_REQUESTS, _)), "other statuses are not rewritten");
    }

    #[test]
    fn registry_hydrates_sidecar_after_restart_and_keeps_identity() {
        let dir = std::env::temp_dir().join(format!("baro-ckpt-restart-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let reg = Registry::new(dir.clone(), 1);
        let identity = Identity { pack: "pack-a".into(), runtime: "r".into(), tokenizer_sha: None, weights_uuid: Some("a".repeat(32)), portable: true };
        let id = Registry::id_for(&[1, 2]);
        let path = reg.path_for(&id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, b"state").unwrap();
        let c = Ckpt { id: id.clone(), pos: 1, tokens: vec![1, 2], path, bytes: 5, created: 1, ttl_s: 60, expires_at: 61, identity: identity.clone() };
        assert!(reg.insert(c, 2).is_ok());
        let restarted = Registry::new(dir.clone(), 1);
        let loaded = restarted.get(&id, 2).unwrap();
        assert_eq!(loaded.identity.pack, identity.pack);
        assert_eq!(loaded.bytes, 5);
        assert!(restarted.remove(&id).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identity_mismatch_names_the_first_field() {
        let me = Identity { pack: "p".into(), runtime: "r".into(), tokenizer_sha: Some("t".into()), weights_uuid: None, portable: false };
        assert_eq!(me.first_mismatch(&json!({"pack": "p", "runtime": "x"})), Some("runtime".into()));
        assert_eq!(me.first_mismatch(&json!({"tokenizer_sha": null})), None);
        assert_eq!(me.first_mismatch(&json!({})), None);
    }
}
