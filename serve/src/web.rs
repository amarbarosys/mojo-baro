//! P6-pwa: serves `serve/web/` (the PWA client) embedded at compile time.
//! `GET /` returns the app shell; `GET /web/*path` returns everything else
//! by exact match against the embedded table. No build step, no CDN: the
//! bytes shipped here are exactly the files in `serve/web/`.

use axum::extract::Path;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

struct Asset {
    path: &'static str,
    bytes: &'static [u8],
    content_type: &'static str,
}

const INDEX_HTML: &[u8] = include_bytes!("../web/index.html");

const ASSETS: &[Asset] = &[
    Asset { path: "index.html", bytes: INDEX_HTML, content_type: "text/html; charset=utf-8" },
    Asset {
        path: "app.js",
        bytes: include_bytes!("../web/app.js"),
        content_type: "text/javascript; charset=utf-8",
    },
    Asset {
        path: "app.css",
        bytes: include_bytes!("../web/app.css"),
        content_type: "text/css; charset=utf-8",
    },
    Asset {
        path: "manifest.webmanifest",
        bytes: include_bytes!("../web/manifest.webmanifest"),
        content_type: "application/manifest+json",
    },
    Asset {
        path: "sw.js",
        bytes: include_bytes!("../web/sw.js"),
        content_type: "text/javascript; charset=utf-8",
    },
    Asset {
        path: "icons/icon.svg",
        bytes: include_bytes!("../web/icons/icon.svg"),
        content_type: "image/svg+xml",
    },
];

pub async fn index() -> Response {
    serve(INDEX_HTML, "text/html; charset=utf-8", false)
}

pub async fn asset(Path(path): Path<String>) -> Response {
    match ASSETS.iter().find(|a| a.path == path) {
        Some(a) => serve(a.bytes, a.content_type, a.path == "sw.js"),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn serve(bytes: &'static [u8], content_type: &'static str, service_worker: bool) -> Response {
    let mut res =
        ([(header::CONTENT_TYPE, HeaderValue::from_static(content_type))], bytes).into_response();
    if service_worker {
        // sw.js lives under /web/ but registers with scope "/"; this header
        // is what Service Workers allow to grant a script control outside
        // its own directory (not part of the Fetch spec's normal
        // same-directory scope rule).
        res.headers_mut().insert("service-worker-allowed", HeaderValue::from_static("/"));
    }
    res
}
