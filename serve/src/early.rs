//! A-M16: the port is bound before the engine loads, and answers `/health` with 503 and the phase (`starting`, `warming`)
//! until the real app takes the socket over. Before this a client saw `connection refused` for minutes, which reads as
//! "wrong port". The real server still prints its `listening on` line only when it can serve (scripts wait for it).
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;

pub struct Early {
    phase: Arc<Mutex<&'static str>>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

fn router(phase: Arc<Mutex<&'static str>>) -> Router {
    let p = phase.clone();
    let word = move || *p.lock().unwrap_or_else(|e| e.into_inner());
    let w = word.clone();
    Router::new()
        // 503, not 200: the gate scripts poll `curl -sf /health` and treat a 200 as "ready". The body still names the phase.
        .route("/health", get(move || async move { (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": w()}))) }))
        .fallback(move || async move {
            let phase = word();
            let body = json!({"error": {"message": format!("baro-serve is {phase}; retry when /health says ok"), "type": phase, "code": 503}});
            (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
        })
}

impl Early {
    /// Serve the early router on `listener` (a clone of the socket the real server will own).
    pub fn start(listener: std::net::TcpListener) -> Early {
        let phase = Arc::new(Mutex::new("starting"));
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let app = router(phase.clone());
        let task = tokio::spawn(async move {
            let Ok(l) = tokio::net::TcpListener::from_std(listener) else { return };
            let _ = axum::serve(l, app).with_graceful_shutdown(async move { let _ = stopped.await; }).await;
        });
        Early { phase, stop: Some(stop), task }
    }

    pub fn set(&self, phase: &'static str) {
        *self.phase.lock().unwrap_or_else(|e| e.into_inner()) = phase;
    }

    /// The real server is ready: stop answering (waits briefly for open connections to close).
    pub async fn stop(mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(3), &mut self.task).await;
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::IntoFuture;

    async fn get_json(addr: &str, path: &str) -> (u16, serde_json::Value) {
        let r = reqwest::get(format!("http://{addr}{path}")).await.expect("the early server must answer");
        (r.status().as_u16(), r.json().await.unwrap_or(serde_json::Value::Null))
    }

    /// A-M16: the port is open during the load and says what the server is doing; the real app then takes the socket.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_port_answers_with_a_phase_before_the_app_exists() {
        let std_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        std_l.set_nonblocking(true).unwrap();
        let addr = std_l.local_addr().unwrap().to_string();
        let early = Early::start(std_l.try_clone().unwrap());
        let (code, v) = get_json(&addr, "/health").await;
        assert_eq!((code, v["status"].as_str()), (503, Some("starting")), "a poll for a 200 must not see a server that is still loading");
        early.set("warming");
        assert_eq!(get_json(&addr, "/health").await.1["status"], "warming");
        let (code, v) = get_json(&addr, "/v1/chat/completions").await;
        assert_eq!((code, v["error"]["type"].as_str()), (503, Some("warming")));
        early.stop().await;
        let app = Router::new().route("/health", get(|| async { Json(json!({"status": "ok"})) }));
        tokio::spawn(axum::serve(tokio::net::TcpListener::from_std(std_l).unwrap(), app).into_future());
        let (code, v) = get_json(&addr, "/health").await;
        assert_eq!((code, v["status"].as_str()), (200, Some("ok")), "the real server must take the socket over");
    }
}
