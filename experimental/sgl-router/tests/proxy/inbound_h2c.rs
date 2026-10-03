// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Inbound cleartext-HTTP/2 (h2c) listener tests for the router's own server.

use std::sync::Arc;
use std::time::Duration;

use sgl_router::config::PolicyKind;
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;

use axum::http::Version;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn build_ctx() -> Arc<AppContext> {
    // Reuse the shared fixture rather than restating the whole `Config`: this test is about the listener, not the routing policy.
    let mut cfg = crate::common::cache_aware_fixture::config();
    cfg.model.policy = PolicyKind::RoundRobin;
    cfg.model.cache_aware = None;

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

/// Spawn the real router (`build_router`) behind `axum::serve` on an ephemeral
/// port — the exact serve path used in `main.rs`.
/// unconditionally, so no worker is needed. Returns the base URL; the accept
/// loop is dropped when the test runtime shuts down.
async fn spawn_router() -> String {
    let ctx = build_ctx();
    ctx.mark_ready();
    let app = build_router(ctx);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn inbound_accepts_h2c_prior_knowledge() {
    let base = spawn_router().await;

    // `http2_prior_knowledge()` sends the HTTP/2 connection preface directly
    // over cleartext (no ALPN, no h1 upgrade).
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .unwrap();

    let resp = client
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("h2c prior-knowledge client must reach the router listener");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.version(),
        Version::HTTP_2,
        "listener must negotiate HTTP/2 for an h2c prior-knowledge client",
    );
}

#[tokio::test]
async fn inbound_still_accepts_http1() {
    let base = spawn_router().await;

    // The default reqwest client speaks HTTP/1.1 over cleartext.
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("HTTP/1.1 client must still reach the router listener");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.version(),
        Version::HTTP_11,
        "an HTTP/1.1 client must keep negotiating HTTP/1.1 on the same port",
    );
}
