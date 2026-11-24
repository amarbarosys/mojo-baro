//! The Baro phone app's Agent-Brain folder, synced with the workstation (`BARO_BRAIN_DIR`, default `~/Brain/baro-agent`).
//! `GET /brain/pull` answers the whole folder as a zip; `POST /brain/push` takes the phone's merged folder as a zip.
//! The phone merges first (it knows what changed since the last sync); the push is the server's safety net: a file
//! whose own mtime is newer than the pushed one is kept, and the pushed text lands beside it as `<file>.conflict-<date>`.
//! Zip entries are stored, not deflated (the app writes them that way), and carry the mtime in the 0x5455 extra field.

use super::*;
use axum::body::Bytes;
use std::fs;
use std::path::Path;

const MAX_FILES: usize = 500;
const MAX_FILE: usize = 1 << 20;
const MAX_TOTAL: usize = 16 << 20;

fn dir() -> PathBuf {
    match std::env::var("BARO_BRAIN_DIR") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join("Brain/baro-agent"),
    }
}

#[derive(Debug)]
struct Entry {
    path: String,
    data: Vec<u8>,
    /// Seconds since the epoch; 0 when the zip did not say.
    mtime: u64,
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

/// A relative path with no empty, `.` or `..` part and no backslash: it cannot leave the folder.
fn safe(path: &str) -> bool {
    !path.is_empty() && !path.starts_with('/') && !path.contains('\\') && path.split('/').all(|c| !c.is_empty() && c != "." && c != "..")
}

fn walk(root: &Path, rel: &str, out: &mut Vec<Entry>) -> std::io::Result<()> {
    for e in fs::read_dir(root.join(rel))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        let path = if rel.is_empty() { name } else { format!("{rel}/{name}") };
        let meta = e.metadata()?;
        if meta.file_type().is_symlink() {
            continue;
        } else if meta.is_dir() {
            walk(root, &path, out)?;
        } else if meta.len() as usize <= MAX_FILE && out.len() < MAX_FILES {
            let mtime = meta.modified().ok().and_then(|m| m.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            out.push(Entry { data: fs::read(e.path())?, path, mtime });
        }
    }
    Ok(())
}

fn u16le(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn u32le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn write_zip(entries: &[Entry]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for e in entries {
        let extra = {
            let mut x = vec![0x55, 0x54, 5, 0, 1];
            u32le(&mut x, e.mtime as u32);
            x
        };
        let crc = crc32(&e.data);
        let offset = out.len() as u32;
        let name = e.path.as_bytes();
        for (v, central_dir) in [(&mut out, false), (&mut central, true)] {
            u32le(v, if central_dir { 0x0201_4b50 } else { 0x0403_4b50 });
            if central_dir {
                u16le(v, 20); // made by
            }
            u16le(v, 20); // needed
            u16le(v, 0x0800); // UTF-8 names
            u16le(v, 0); // stored
            u16le(v, 0);
            u16le(v, 0x21); // DOS time and date: the extra field has the real time
            u32le(v, crc);
            u32le(v, e.data.len() as u32);
            u32le(v, e.data.len() as u32);
            u16le(v, name.len() as u16);
            u16le(v, extra.len() as u16);
            if central_dir {
                v.extend_from_slice(&[0; 2 + 2 + 2 + 4]); // comment length, disk, internal and external attributes
                u32le(v, offset);
            }
            v.extend_from_slice(name);
            v.extend_from_slice(&extra);
            if !central_dir {
                v.extend_from_slice(&e.data);
            }
        }
    }
    let cd_at = out.len() as u32;
    out.extend_from_slice(&central);
    u32le(&mut out, 0x0605_4b50);
    out.extend_from_slice(&[0; 4]);
    u16le(&mut out, entries.len() as u16);
    u16le(&mut out, entries.len() as u16);
    u32le(&mut out, central.len() as u32);
    u32le(&mut out, cd_at);
    u16le(&mut out, 0);
    out
}

fn read_zip(z: &[u8]) -> Result<Vec<Entry>, String> {
    let le16 = |at: usize| z.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize);
    let le32 = |at: usize| z.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let mut out = Vec::new();
    let mut at = 0;
    while le32(at) == Some(0x0403_4b50) {
        let (flags, method, csize, size) = (le16(at + 6).ok_or("short zip")?, le16(at + 8).ok_or("short zip")?, le32(at + 18).ok_or("short zip")?, le32(at + 22).ok_or("short zip")?);
        if method != 0 || flags & 8 != 0 || csize != size {
            return Err("zip entries must be stored (no compression, no data descriptor)".into());
        }
        let (n, x) = (le16(at + 26).ok_or("short zip")?, le16(at + 28).ok_or("short zip")?);
        let name = z.get(at + 30..at + 30 + n).ok_or("short zip")?;
        let extra = z.get(at + 30 + n..at + 30 + n + x).ok_or("short zip")?;
        let data = z.get(at + 30 + n + x..at + 30 + n + x + size).ok_or("short zip")?;
        let mut mtime = 0;
        let mut i = 0;
        while i + 4 <= extra.len() {
            let len = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
            if extra[i] == 0x55 && extra[i + 1] == 0x54 && len >= 5 && extra[i + 4] & 1 != 0 && i + 9 <= extra.len() {
                mtime = u32::from_le_bytes([extra[i + 5], extra[i + 6], extra[i + 7], extra[i + 8]]) as u64;
            }
            i += 4 + len;
        }
        let path = String::from_utf8(name.to_vec()).map_err(|_| "a file name is not UTF-8")?;
        if !path.ends_with('/') {
            out.push(Entry { path, data: data.to_vec(), mtime });
        }
        at += 30 + n + x + size;
    }
    Ok(out)
}

/// `YYYY-MM-DD` (UTC) of [secs]; the days-to-date step is Howard Hinnant's civil_from_days.
fn date(secs: u64) -> String {
    let z = (secs / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn set_mtime(path: &Path, secs: u64) {
    if secs > 0 {
        if let Ok(f) = fs::File::options().write(true).open(path) {
            let _ = f.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(secs));
        }
    }
}

/// Puts [entries] into [root]: a new or identical file is written or skipped; a changed file is overwritten only when the
/// pushed mtime is not older than its own, else the pushed text is kept as `<file>.conflict-<today>`. Returns (written, conflicts, unchanged).
fn merge(root: &Path, entries: Vec<Entry>, today: &str) -> std::io::Result<(usize, usize, usize)> {
    let (mut written, mut conflicts, mut same) = (0, 0, 0);
    for e in entries {
        let p = root.join(&e.path);
        let own = fs::read(&p).ok();
        if own.as_deref() == Some(&e.data[..]) {
            same += 1;
            continue;
        }
        let own_mtime = fs::metadata(&p).ok().and_then(|m| m.modified().ok()).and_then(|m| m.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let target = if own.is_some() && own_mtime > e.mtime {
            conflicts += 1;
            root.join(format!("{}.conflict-{today}", e.path))
        } else {
            written += 1;
            p
        };
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, &e.data)?;
        set_mtime(&target, e.mtime);
    }
    Ok((written, conflicts, same))
}

fn io_err(e: std::io::Error) -> ApiError {
    ApiError::Plain(StatusCode::INTERNAL_SERVER_ERROR, format!("brain folder: {e}"))
}

pub async fn pull() -> Result<Response, ApiError> {
    let root = dir();
    let mut files = Vec::new();
    if root.is_dir() {
        walk(&root, "", &mut files).map_err(io_err)?;
    }
    Ok(([("content-type", "application/zip")], write_zip(&files)).into_response())
}

pub async fn push(body: Bytes) -> Result<Json<Value>, ApiError> {
    let entries = read_zip(&body).map_err(bad)?;
    if entries.len() > MAX_FILES || entries.iter().map(|e| e.data.len()).sum::<usize>() > MAX_TOTAL || entries.iter().any(|e| e.data.len() > MAX_FILE) {
        return Err(bad("the folder is too big to sync (500 files, 1 MB each, 16 MB in all)"));
    }
    if let Some(e) = entries.iter().find(|e| !safe(&e.path)) {
        return Err(bad(format!("unsafe path in the zip: {}", e.path)));
    }
    let today = date(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    let (written, conflicts, unchanged) = merge(&dir(), entries, &today).map_err(io_err)?;
    Ok(Json(json!({"written": written, "conflicts": conflicts, "unchanged": unchanged})))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(path: &str, data: &str, mtime: u64) -> Entry {
        Entry { path: path.into(), data: data.as_bytes().to_vec(), mtime }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("baro-brain-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_zip_round_trips_with_names_data_and_mtime() {
        let z = write_zip(&[e("AGENTS.md", "héllo", 1_700_000_000), e("skills/a/SKILL.md", "", 5)]);
        let back = read_zip(&z).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!((back[0].path.as_str(), back[0].data.as_slice(), back[0].mtime), ("AGENTS.md", "héllo".as_bytes(), 1_700_000_000));
        assert_eq!(back[1].path, "skills/a/SKILL.md");
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn a_compressed_zip_is_refused() {
        let mut z = write_zip(&[e("a.md", "x", 1)]);
        z[8] = 8; // method: deflate
        assert!(read_zip(&z).unwrap_err().contains("stored"));
    }

    #[test]
    fn only_paths_inside_the_folder_pass() {
        assert!(safe("notes/a.md") && safe("AGENTS.md.conflict-2026-11-21"));
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "a//b", "a\\b", "./a"] {
            assert!(!safe(bad), "{bad}");
        }
    }

    #[test]
    fn dates_are_utc_civil_dates() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(1_795_219_200), "2026-11-21");
        assert_eq!(date(951_782_400), "2000-02-29");
    }

    #[test]
    fn merge_writes_new_skips_same_and_keeps_a_newer_own_file() {
        let root = scratch("merge");
        fs::write(root.join("own.md"), "mine").unwrap(); // mtime: now
        let r = merge(&root, vec![e("new/a.md", "A", 100), e("own.md", "mine", 1), e("own.md", "pushed", 1)], "2026-11-21").unwrap();
        assert_eq!(r, (1, 1, 1));
        assert_eq!(fs::read_to_string(root.join("new/a.md")).unwrap(), "A");
        assert_eq!(fs::read_to_string(root.join("own.md")).unwrap(), "mine");
        assert_eq!(fs::read_to_string(root.join("own.md.conflict-2026-11-21")).unwrap(), "pushed");
        // a pushed file as new as ours wins
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 10;
        assert_eq!(merge(&root, vec![e("own.md", "newer", now)], "2026-11-21").unwrap(), (1, 0, 0));
        assert_eq!(fs::read_to_string(root.join("own.md")).unwrap(), "newer");
        let _ = fs::remove_dir_all(root);
    }
}
