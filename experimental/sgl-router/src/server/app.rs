// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{outcome_from_status, RequestLogContext};
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use std::sync::{Arc, OnceLock};
use tower_http::catch_panic::CatchPanicLayer;

/// Infra endpoints whose *successful* polls are logged at DEBUG rather than
/// INFO.
fn is_infra_path(path: &str) -> bool {
    matches!(path, "/healthz" | "/readyz" | "/metrics")
}

/// Collapse an HTTP method to a bounded allow-list for use as a metric label.
/// `http::Method` accepts any RFC-7230 extension token.
/// the raw method on `requests_total` / `responses_total` would be
/// caller-controlled unbounded-cardinality input.
fn normalize_method(method: &axum::http::Method) -> &'static str {
    use axum::http::Method;
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::PATCH => "PATCH",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::TRACE => "TRACE",
        Method::CONNECT => "CONNECT",
        _ => "other",
    }
}

/// Router pod identity stamped on every access-log line.
static POD_ID: OnceLock<String> = OnceLock::new();

fn pod_id() -> &'static str {
    POD_ID.get_or_init(|| {
        std::env::var("POD_NAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".to_string())
    })
}

/// Outermost middleware: the single access-log and edge-counter site. Their difference = received-but-not-answered, invisible to post-dispatch `worker_requests_total`. `route` is the matched template (not the raw URI) and `method` a known-verb allow-list, so neither label's cardinality is caller-controlled. Also the only place that sees every HTTP exchange on every route, so it is where `inflight_http` is taken — the count the termination drain reports on. The guard rides the response body rather than being dropped here: a streaming completion has barely started when this function returns. Dispatched requests carry a [`RequestLogContext`] naming the worker, model and outcome; everything else logs those empty.
async fn access_log_and_record(
    State(ctx): State<Arc<AppContext>>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let method_label = normalize_method(&method);
    let path = req.uri().path().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_owned();
    let start = std::time::Instant::now();

    ctx.metrics.record_ingress(&route, method_label);
    let inflight = ctx.inflight_http.enter();
    let resp = next.run(req).await;
    let status = resp.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    ctx.metrics
        .record_response(&route, method_label, status.as_u16());

    // A healthy probe is noise; a failing one is an incident signal.
    if is_infra_path(&path) && status.is_success() {
        tracing::debug!(
            method = %method,
            path = %path,
            status = status.as_u16(),
            latency_ms,
            "http_request",
        );
    } else {
        // Per-worker fields are present only when a handler dispatched the request and attached them.
        let log_ctx = resp.extensions().get::<RequestLogContext>();
        tracing::info!(
            pod_id = %pod_id(),
            request_id = %request_id,
            method = %method,
            path = %path,
            status = status.as_u16(),
            outcome = log_ctx
                .map(|c| c.outcome)
                .unwrap_or_else(|| outcome_from_status(status.as_u16()))
                .as_str(),
            worker = log_ctx.map(|c| c.worker_url.as_str()).unwrap_or(""),
            engine_rid = log_ctx
                .and_then(|c| c.engine_rid.as_deref())
                .unwrap_or(""),
            model = log_ctx.map(|c| c.model_id.as_str()).unwrap_or(""),
            stream = log_ctx.is_some_and(|c| c.streaming),
            latency_ms,
            "http_request",
        );
    }
    resp.map(|body| crate::server::inflight::track_body(body, inflight))
}

/// Middleware: log multiple PAYLOAD_TOO_LARGE responses with the request
/// method and full URI, at WARN, so an operator investigating "client X gets 413s"
/// has a server-side breadcrumb.
async fn log_413(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let resp = next.run(req).await;
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        tracing::warn!(
            %method,
            %uri,
            "request rejected with 413 PAYLOAD_TOO_LARGE (body exceeded route limit)",
        );
    }
    resp
}

pub fn build_router(ctx: Arc<AppContext>) -> Router {
    let router = Router::new()
        .route("/healthz", get(crate::server::routes::health::healthz))
        .route("/readyz", get(crate::server::routes::health::readyz))
        .route("/metrics", get(crate::server::routes::metrics::metrics))
        .route(
            "/v1/models",
            get(crate::server::routes::models::list_models),
        )
        .route(
            "/v1/tokenize",
            post(crate::server::routes::tokenize::tokenize),
        )
        .route(
            "/v1/detokenize",
            post(crate::server::routes::tokenize::detokenize),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat::chat_completions)
                .layer(DefaultBodyLimit::max(MAX_CHAT_BODY_BYTES))
                .layer(middleware::from_fn(log_413)),
        )
        .route(
            "/flush_cache",
            post(crate::server::routes::cache::flush_cache),
        );
    // A route that panics on purpose, so the panic-handling layers below are exercised as `build_router` composes them.
    #[cfg(test)]
    let router = router.route(
        "/__test_panic",
        get(|| async {
            panic!("handler exploded");
            #[allow(unreachable_code)]
            StatusCode::OK
        }),
    );
    router
        .layer(CatchPanicLayer::custom(
            |_: Box<dyn std::any::Any + Send>| {
                ApiError::Internal(anyhow::anyhow!("handler panicked")).into_response()
            },
        ))
        // After routing, so MatchedPath is set for every route.
        .layer(middleware::from_fn_with_state(
            ctx.clone(),
            access_log_and_record,
        ))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::RequestOutcome;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;
    use tracing_subscriber::fmt::MakeWriter;

    /// Capture the `tracing` output of one test into a buffer.
    #[derive(Clone)]
    struct VecWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for VecWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for VecWriter {
        type Writer = VecWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Install a permissive global subscriber once per test binary. `tracing`
    /// caches each callsite's "interest" the first time it is hit.
    fn prime_tracing_callsites() {
        static PRIMED: OnceLock<()> = OnceLock::new();
        PRIMED.get_or_init(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::fmt()
                    .with_max_level(tracing::Level::TRACE)
                    .with_writer(std::io::sink)
                    .finish(),
            );
        });
    }

    /// Install a buffer-backed subscriber for the current thread. The returned
    /// guard must stay alive for the duration of the capture.
    fn capture_logs() -> (Arc<Mutex<Vec<u8>>>, tracing::subscriber::DefaultGuard) {
        prime_tracing_callsites();
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(VecWriter(buf.clone()))
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        (buf, guard)
    }

    fn captured(buf: &Arc<Mutex<Vec<u8>>>) -> String {
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    /// Metric labels must stay bounded: standard verbs pass through.
    #[test]
    fn normalize_method_collapses_unknown_verbs() {
        use axum::http::Method;
        assert_eq!(normalize_method(&Method::GET), "GET");
        assert_eq!(normalize_method(&Method::POST), "POST");
        let exotic = Method::from_bytes(b"BREW").unwrap();
        assert_eq!(
            normalize_method(&exotic),
            "other",
            "an unknown verb must collapse to `other`, not mint a new label series",
        );
    }

    /// An unrouted path must collapse to `route="unmatched"` and never put the raw URI in a metric label.
    #[tokio::test]
    async fn unmatched_route_does_not_leak_the_raw_uri_into_metrics() {
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/not/a/real/route-9f3c")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"sgl_router_requests_total{route="unmatched",method="GET"} 1"#),
            "an unrouted request must be counted under route=\"unmatched\": {m}",
        );
        assert!(
            !m.contains("route-9f3c"),
            "the raw URI must never reach a metric label: {m}",
        );
    }

    /// A handler panic must become a that the `access_log_and_record` middleware still observes. hyper catches a handler panic.
    #[tokio::test]
    async fn handler_panic_becomes_a_counted_500_with_the_router_error_envelope() {
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/__test_panic")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(
            res.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "a handler panic must surface as 500, not a dropped connection",
        );
        assert_eq!(
            res.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("internal_error"),
            "a panic-500 must carry the same error envelope as every other \
             router-originated error",
        );
        let m = ctx.metrics.render();
        assert!(
            m.contains(
                r#"sgl_router_responses_total{route="/__test_panic",method="GET",status_code="500"} 1"#
            ),
            "the middleware must observe and count the panic-500; got:\n{m}",
        );
    }

    /// A request rejected BEFORE any handler runs — here an unrouted path.
    #[tokio::test]
    async fn request_that_never_reaches_a_handler_is_still_logged() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/nope")
            .header("x-request-id", "rid-unrouted")
            .body(Body::empty())
            .unwrap();
        let res = build_router(ctx).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        let logs = captured(&buf);
        assert!(
            logs.contains("http_request") && logs.contains("rid-unrouted"),
            "every request must be logged by the middleware; captured:\n{logs}",
        );
        assert!(
            logs.contains("status=404") && logs.contains("outcome=\"client_error\""),
            "the access log must carry the final status and its outcome; captured:\n{logs}",
        );
    }

    /// A routed request carries the worker and model it was dispatched to, via the [`RequestLogContext`] its handler attaches to the response —.
    #[tokio::test]
    async fn routed_response_context_names_the_worker_in_the_access_log() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let app = Router::new()
            .route(
                "/routed",
                get(|| async {
                    let mut resp = StatusCode::OK.into_response();
                    resp.extensions_mut().insert(RequestLogContext {
                        worker_url: "http://worker-a:30000".into(),
                        model_id: "tiny".into(),
                        streaming: false,
                        outcome: RequestOutcome::Cancelled,
                        engine_rid: Some("1f0c2b7a4e9d4f3ab6c5d8e7f0a1b2c3".into()),
                    });
                    resp
                }),
            )
            .layer(middleware::from_fn_with_state(
                Arc::clone(&ctx),
                access_log_and_record,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/routed")
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let logs = captured(&buf);
        assert!(
            logs.contains("worker=\"http://worker-a:30000\"") && logs.contains("model=\"tiny\""),
            "a routed request must be logged with its worker and model; captured:\n{logs}",
        );
        assert!(
            logs.contains("engine_rid=\"1f0c2b7a4e9d4f3ab6c5d8e7f0a1b2c3\"")
                && logs.contains("request_id="),
            "the minted rid must be logged beside the caller's request id; captured:\n{logs}",
        );
        // The handler's outcome must win over the status-derived fallback —
        // otherwise the log and `worker_requests_total` can disagree about a
        // request the handler classified itself.
        assert!(
            logs.contains("outcome=\"cancelled\"") && !logs.contains("outcome=\"success\""),
            "the handler's own outcome must win over the status fallback; captured:\n{logs}",
        );
    }

    /// A SUCCEEDING infra probe (`/healthz`, `/readyz`, `/metrics`) is polled every few seconds by the kubelet and Prometheus.
    #[tokio::test]
    async fn successful_infra_probe_is_counted_but_not_logged_at_info() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert!(
            !captured(&buf).contains("/healthz"),
            "a healthy probe must not reach the INFO access log; captured:\n{}",
            captured(&buf),
        );

        // Positive control: a non-infra request through the same capture must
        // produce a line, proving the absence above is the filter.
        let control = Request::builder()
            .method("GET")
            .uri("/nope")
            .body(Body::empty())
            .unwrap();
        let _ = build_router(Arc::clone(&ctx))
            .oneshot(control)
            .await
            .unwrap();
        assert!(
            captured(&buf).contains("http_request"),
            "log capture is broken — the negative assertion above proves nothing; captured:\n{}",
            captured(&buf),
        );

        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"sgl_router_requests_total{route="/healthz",method="GET"} 1"#)
                && m.contains(
                    r#"sgl_router_responses_total{route="/healthz",method="GET",status_code="200"} 1"#
                ),
            "infra probes must still be counted at the edge: {m}",
        );
    }

    /// A FAILING infra probe is the opposite of noise.
    #[tokio::test]
    async fn failing_readiness_probe_is_logged_at_info() {
        let (buf, _guard) = capture_logs();
        let ctx = Arc::new(AppContext::stub());
        assert!(!ctx.is_ready(), "stub context must start unready");

        let req = Request::builder()
            .method("GET")
            .uri("/readyz")
            .body(Body::empty())
            .unwrap();
        let res = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

        let logs = captured(&buf);
        assert!(
            logs.contains("http_request")
                && logs.contains("/readyz")
                && logs.contains("status=503"),
            "a failing readiness probe must reach the INFO access log; captured:\n{logs}",
        );
    }

    /// The `method` label must be bounded where it is USED, not merely where it is computed.
    #[tokio::test]
    async fn unknown_verb_is_collapsed_in_the_metric_labels() {
        let ctx = Arc::new(AppContext::stub());
        let req = Request::builder()
            .method(axum::http::Method::from_bytes(b"BREW").unwrap())
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        let _ = build_router(Arc::clone(&ctx)).oneshot(req).await.unwrap();

        let m = ctx.metrics.render();
        assert!(
            m.contains(r#"method="other""#),
            "an unknown verb must be counted under method=\"other\": {m}",
        );
        assert!(
            !m.contains("BREW"),
            "the raw verb must never reach a metric label: {m}",
        );
    }
}
