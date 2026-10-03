// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--override-sampling-params` end to end: from the CLI flag an operator writes in a manifest.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{Cli, Config};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "tiny";

const OVERRIDES: &str = r#"{"temperature": 1, "top_p": 0.95, "top_k": 1000,
                            "frequency_penalty": 0, "presence_penalty": 0, "n": 1}"#;

/// Build the config the way a deployment does — through `Cli`, so what these
/// tests pin is the flag spelling in a manifest.
/// could drift from what the parser produces.
fn config(flags: &[&str]) -> Config {
    let mut argv = vec![
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
    ];
    argv.extend_from_slice(flags);
    <Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .expect("flags must parse")
}

fn build_ctx(url: String, flags: &[&str]) -> Arc<AppContext> {
    let cfg = config(flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, body: Value) -> StatusCode {
    send_raw(ctx, serde_json::to_vec(&body).unwrap()).await.0
}

/// Send a body verbatim, so a test can express what `serde_json::Value`
/// cannot — a repeated key. Returns the status and the router's own error
/// code header.
async fn send_raw(ctx: Arc<AppContext>, body: Vec<u8>) -> (StatusCode, Option<String>) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = build_router(ctx).oneshot(req).await.unwrap();
    let code = resp
        .headers()
        .get("x-router-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    (resp.status(), code)
}

fn captured(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).expect("captured body is valid JSON"))
}

/// The values an operator configures replace the engine's own defaults.
#[tokio::test]
async fn configured_values_reach_the_engine_when_the_request_omits_them() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let status = send(
        ctx,
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(1)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));
    assert_eq!(body.get("top_k"), Some(&json!(1000)));
    assert_eq!(body.get("frequency_penalty"), Some(&json!(0)));
    assert_eq!(body.get("presence_penalty"), Some(&json!(0)));
    assert_eq!(body.get("n"), Some(&json!(1)));
}

/// Under the default `reject` mode a conflicting request is a that never reaches a worker — the contract costs no engine round-trip.
#[tokio::test]
async fn reject_mode_400s_a_conflicting_request_without_touching_the_engine() {
    // Covers the whole rejection contract: status, wire error code, and that the engine is never reached.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
    });
    let (status, code) = send_raw(ctx, serde_json::to_vec(&body).unwrap()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // Its own error code, so an operator rolling `reject` across a fleet can alert.
    assert_eq!(code.as_deref(), Some("sampling_contract_violation"));
    assert!(
        captured(&mock).is_none(),
        "a rejected request must not reach the engine"
    );
}

/// A repeated sampling key must not become a router-side.
#[tokio::test]
async fn duplicate_sampling_key_is_judged_on_its_last_value() {
    // Last value agrees with the contract -> served.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let (status, _) = send_raw(
        ctx,
        br#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],
             "temperature":0.7,"temperature":1}"#
            .to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        captured(&mock).and_then(|b| b.get("temperature").cloned()),
        Some(json!(1)),
        "the engine must see the last value"
    );

    // Last value disagrees -> rejected, even though the first one matched.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let (status, _) = send_raw(
        ctx,
        br#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],
             "temperature":1,"temperature":0.7}"#
            .to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// `temperature`, `top_p`, `top_k`, `min_p` and `repetition_penalty` are the parameters the engine resolves.
#[tokio::test]
async fn engine_defaulted_parameters_reach_the_engine() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        mock.url.clone(),
        &[
            "--override-sampling-params",
            r#"{"min_p": 0.05, "repetition_penalty": 1.1}"#,
        ],
    );
    let status = send(
        ctx,
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("engine must receive a body");
    assert_eq!(body.get("min_p"), Some(&json!(0.05)));
    assert_eq!(body.get("repetition_penalty"), Some(&json!(1.1)));
}

/// The same request under `allow` is forwarded with the client's value intact.
#[tokio::test]
async fn allow_mode_forwards_the_client_value_and_fills_the_rest() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        mock.url.clone(),
        &[
            "--override-sampling-params",
            OVERRIDES,
            "--sampling-param-conflict",
            "allow",
        ],
    );
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.7)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));
    assert_eq!(body.get("top_k"), Some(&json!(1000)));
    assert_eq!(body.get("n"), Some(&json!(1)));
}

/// A band accepts anything inside it, 400s outside, and injects nothing: temperature tunable in [, ].
#[tokio::test]
async fn a_temperature_band_admits_in_range_values_and_injects_nothing() {
    let flags = [
        "--override-sampling-params",
        r#"{"temperature": {"min": 0, "max": 1}, "top_p": 0.95}"#,
    ];

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.6,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.6)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));

    // Omitted: the band names no value, so the engine's own default applies.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), None);

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(captured(&mock).is_none());
}

/// With the flag unset the router touches nothing.
#[tokio::test]
async fn unset_flag_forwards_the_body_untouched() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &[]);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.7)));
    assert_eq!(body.get("top_p"), None);
    assert_eq!(body.get("top_k"), None);
    assert_eq!(body.get("n"), None);
}
