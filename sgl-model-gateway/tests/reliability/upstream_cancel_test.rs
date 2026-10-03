//! Upstream request cancellation tests Verifies that when a client disconnects mid-stream.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        clear_fail_status_code, clear_slow_stream_chunks, clear_stream_error_after_chunks,
        get_stream_tracking_state, reset_stream_tracker, set_fail_status_code,
        set_slow_stream_chunks, set_stream_error_after_chunks, wait_for_stream_finish,
        StreamTrackingState, MOCK_STREAM_BUFFER,
    },
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

/// Read up to `max_chunks` data frames from a streaming response body.
async fn read_n_chunks(body: &mut Body, max_chunks: usize) -> usize {
    let mut chunks_read = 0;
    while chunks_read < max_chunks {
        match body.frame().await {
            Some(Ok(frame)) if frame.is_data() => {
                chunks_read += 1;
            }
            Some(Ok(_)) => continue,
            _ => break,
        }
    }
    chunks_read
}

/// Read up to `max_chunks` data frames, returning the count and accumulated
/// bytes so callers can parse the SSE payload (e.g. to capture a `response.id`
/// before dropping the body).
async fn read_n_chunks_with_bytes(body: &mut Body, max_chunks: usize) -> (usize, Vec<u8>) {
    let mut chunks_read = 0;
    let mut buf: Vec<u8> = Vec::new();
    while chunks_read < max_chunks {
        match body.frame().await {
            Some(Ok(frame)) if frame.is_data() => {
                if let Ok(data) = frame.into_data() {
                    buf.extend_from_slice(&data);
                }
                chunks_read += 1;
            }
            Some(Ok(_)) => continue,
            _ => break,
        }
    }
    (chunks_read, buf)
}

/// Extract the `id` field from the first `response.created` SSE event in `buf`.
/// Returns `None` if the event hasn't arrived yet (caller should read more).
fn extract_response_id_from_sse(buf: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(buf).ok()?;
    let data_line = s
        .lines()
        .find(|l| l.starts_with("data:") && l.contains("\"response.created\""))?;
    let json_str = data_line.trim_start_matches("data:").trim();
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    value
        .get("response")
        .and_then(|r| r.get("id"))
        .and_then(|id| id.as_str())
        .map(|s| s.to_string())
}

/// Safety timeout for `wait_for_stream_finish` — the worker notifies the instant its producer task exits.
const STREAM_FINISH_TIMEOUT: Duration = Duration::from_secs(3);

async fn assert_cancelled_before_completion(port: u16) -> StreamTrackingState {
    let state = wait_for_stream_finish(port, STREAM_FINISH_TIMEOUT)
        .await
        .unwrap_or_else(|| {
            panic!(
                "Stream tracking state should exist for worker port {}",
                port
            )
        });

    assert!(
        !state.completed,
        "Stream should NOT have completed - gateway should have cancelled it. \
         Chunks sent: {}, total: {}",
        state.chunks_sent, state.total_chunks
    );
    assert!(
        state.chunks_sent < state.total_chunks,
        "Worker should have sent fewer chunks than total ({} < {}). \
         Stream was not cancelled in time.",
        state.chunks_sent,
        state.total_chunks
    );
    state
}

#[cfg(test)]
mod upstream_cancel_tests {
    use super::*;

    /// Test that the gateway cancels the upstream stream when the client disconnects before consuming all chunks.
    #[tokio::test]
    async fn test_streaming_cancel_on_client_disconnect() {
        let worker_port = 20250;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4250);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me a long story"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(
            chunks_read > 0,
            "Should have read at least one chunk before disconnecting"
        );

        // Snapshot the worker counter the moment we drop, then wait for the
        // producer task to fire its exit notifier.
        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + MOCK_STREAM_BUFFER,
            "Chunks_sent grew by more than the channel buffer ({} -> {}); \
             gateway likely did not propagate cancel.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a fully consumed stream is NOT cancelled prematurely — the worker sends all chunks.
    #[tokio::test]
    async fn test_streaming_completes_when_client_consumes_all() {
        let worker_port = 20251;
        let total_chunks: usize = 5;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4251);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Short response"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let _full_body = resp.into_body().collect().await.unwrap().to_bytes();

        let state = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT)
            .await
            .expect("Stream tracking state should exist");
        assert!(
            state.completed,
            "Stream should have completed when client consumed all chunks. \
             Chunks sent: {}, total: {}",
            state.chunks_sent, state.total_chunks
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a non-streaming request is not affected by cancel logic.
    #[tokio::test]
    async fn test_non_streaming_request_unaffected() {
        let config = TestRouterConfig::round_robin(4252);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20252)]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(
            body.get("object").and_then(|v| v.as_str()),
            Some("chat.completion"),
            "Non-streaming response should be a complete chat.completion object"
        );

        ctx.shutdown().await;
    }

    /// Cancel before the worker emits any chunk.
    #[tokio::test]
    async fn test_streaming_cancel_before_first_chunk() {
        let worker_port = 20253;
        let total_chunks: usize = 10;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4253);
        let ctx = AppTestContext::new_with_config(
            // 200ms per-chunk delay; the first chunk takes the full 200ms because the worker sleeps before emitting.
            config,
            vec![TestWorkerConfig::slow(worker_port, 200)],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drop immediately, before pulling any frame.
        drop(resp.into_body());

        let state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            state.chunks_sent <= 4,
            "Worker should have sent very few chunks (≤ buffer capacity), \
             saw {} of {}",
            state.chunks_sent,
            state.total_chunks
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Disconnecting *after* the stream completes naturally must be a no-op — no panic, no spurious "cancel" log.
    #[tokio::test]
    async fn test_streaming_cancel_after_done_is_noop() {
        let worker_port = 20254;
        let total_chunks: usize = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4254);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 5)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain entire body, then drop after a short pause.
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(!body.is_empty());
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(body);

        let state = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT)
            .await
            .expect("tracking state");
        assert!(state.completed, "Stream should have completed cleanly");
        assert_eq!(state.chunks_sent, state.total_chunks);

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Cancelling one client request must not affect a concurrent request that's hitting a *different* upstream.
    #[tokio::test]
    async fn test_cancel_one_request_does_not_affect_concurrent() {
        let worker_a = 20255;
        let worker_b = 20256;
        let total_a: usize = 20;
        let total_b: usize = 5;

        reset_stream_tracker(worker_a);
        reset_stream_tracker(worker_b);
        set_slow_stream_chunks(worker_a, total_a);
        set_slow_stream_chunks(worker_b, total_b);

        let config = TestRouterConfig::round_robin(4255);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::slow(worker_a, 50),
                TestWorkerConfig::slow(worker_b, 10),
            ],
        )
        .await;

        // Parallel requests. Round-robin should send them to different workers.
        let app = ctx.create_app().await;
        let app2 = app.clone();

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Long"}],
            "stream": true
        });
        let body_str = Arc::new(serde_json::to_string(&payload).unwrap());

        let body_str_a = Arc::clone(&body_str);
        let h_cancel = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from((*body_str_a).clone()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let mut body = resp.into_body();
            let _ = read_n_chunks(&mut body, 2).await;
            drop(body);
        });

        // Small stagger so round-robin index advances.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let body_str_b = Arc::clone(&body_str);
        let h_consume = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from((*body_str_b).clone()))
                .unwrap();
            let resp = app2.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let _ = resp.into_body().collect().await.unwrap();
        });

        h_cancel.await.unwrap();
        h_consume.await.unwrap();

        // Worker that should have completed normally.
        let state_b = wait_for_stream_finish(worker_b, STREAM_FINISH_TIMEOUT)
            .await
            .expect("worker_b tracking state");
        // Worker that should have been cancelled.
        let state_a = wait_for_stream_finish(worker_a, STREAM_FINISH_TIMEOUT)
            .await
            .expect("worker_a tracking state");

        // We don't know which worker got which request because round-robin
        // is shared across the run, so accept either ordering.
        let (cancelled, completed) = if state_a.completed {
            (state_b, state_a)
        } else {
            (state_a, state_b)
        };
        assert!(
            !cancelled.completed,
            "Cancelled stream should not have completed (sent {}/{})",
            cancelled.chunks_sent, cancelled.total_chunks
        );
        assert!(
            completed.completed,
            "Concurrent stream should have completed (sent {}/{})",
            completed.chunks_sent, completed.total_chunks
        );

        clear_slow_stream_chunks(worker_a);
        clear_slow_stream_chunks(worker_b);
        ctx.shutdown().await;
    }

    /// Mid-stream worker error must surface as `Stream error: …` to the client, must NOT be silently swallowed as cancel.
    #[tokio::test]
    async fn test_streaming_worker_error_propagates_not_cancel() {
        let worker_port = 20257;
        let total_chunks: usize = 10;
        let error_after: usize = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);

        let config = TestRouterConfig::round_robin(4256);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Collecting the body should yield a transport-level error AFTER the first few chunks. axum surfaces the upstream error.
        let mut body = resp.into_body();
        let mut combined = Vec::<u8>::new();
        let mut saw_transport_err = false;
        loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        combined.extend_from_slice(&data);
                    }
                }
                Some(Err(_)) => {
                    saw_transport_err = true;
                    break;
                }
                None => break,
            }
        }
        let combined_str = String::from_utf8_lossy(&combined);
        assert!(
            saw_transport_err
                || combined_str.contains("Stream error")
                || combined_str.contains("simulated upstream worker crash"),
            "Client should observe a stream error event or transport error; got: {}",
            combined_str
        );

        // Worker should have sent some chunks but not all, and the stream
        // should NOT be marked completed (we crashed before [DONE]).
        let state = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT)
            .await
            .expect("tracking state");
        assert!(!state.completed, "Errored stream should not be completed");
        assert!(
            state.chunks_sent <= error_after,
            "Worker reported {} chunks_sent, expected ≤ {}",
            state.chunks_sent,
            error_after
        );

        clear_stream_error_after_chunks(worker_port);
        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// PD-disagg streaming cancel: client disconnects mid-decode-stream.
    #[tokio::test]
    async fn test_pd_streaming_cancel_on_client_disconnect() {
        let prefill_port = 20258;
        let decode_port = 20259;
        let total_chunks: usize = 20;

        reset_stream_tracker(decode_port);
        set_slow_stream_chunks(decode_port, total_chunks);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4257)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                // Use a slow decode worker (50ms per chunk).
                let mut w = TestWorkerConfig::decode(decode_port);
                w.response_delay_ms = 50;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "PD streaming test",
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(
            chunks_read > 0,
            "Should have read at least one chunk before disconnecting"
        );

        let snapshot = get_stream_tracking_state(decode_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(decode_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + MOCK_STREAM_BUFFER,
            "Decode chunks_sent grew by more than the buffer ({} -> {}); \
             gateway likely did not propagate cancel through PD path.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(decode_port);
        ctx.shutdown().await;
    }

    /// /v1/responses with no persistence (`store: false`, no conversation): client disconnect must propagate.
    #[tokio::test]
    async fn test_responses_streaming_cancel_no_persistence() {
        let worker_port = 20260;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4258)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        // First events are response.created/response.in_progress, then chunks start.
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(chunks_read > 0);

        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + MOCK_STREAM_BUFFER,
            "/responses chunks_sent grew by more than the buffer ({} -> {}); \
             gateway likely did not propagate cancel through /responses path.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// /v1/responses with persistence (`store: true`): the upstream is intentionally NOT cancelled on client disconnect.
    #[tokio::test]
    async fn test_responses_streaming_persistence_drains_after_disconnect() {
        let worker_port = 20261;
        let total_chunks: usize = 6;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4259)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 20)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        // Read enough frames to capture the response.created event so we can later look up the stored response.
        let (_, captured) = read_n_chunks_with_bytes(&mut body, 3).await;
        let response_id =
            extract_response_id_from_sse(&captured).expect("response.created event with id");
        drop(body);

        // With persistence the gateway keeps reading; worker should
        // eventually mark the stream completed despite client gone.
        let state = wait_for_stream_finish(worker_port, Duration::from_secs(5))
            .await
            .expect("tracking state");
        assert!(
            state.completed,
            "Persistence path should drain upstream to completion despite \
             client disconnect (sent {}/{})",
            state.chunks_sent, state.total_chunks
        );
        assert_eq!(state.chunks_sent, state.total_chunks);

        // Draining is necessary but not sufficient — also verify the gateway called persist_conversation_items and the response landed.
        let storage = ctx.app_context.response_storage.clone();
        let stored = {
            use data_connector::ResponseId;
            let id = ResponseId::from(response_id.clone());
            let mut found = None;
            for _ in 0..20 {
                if let Ok(Some(r)) = storage.get_response(&id).await {
                    found = Some(r);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            found.unwrap_or_else(|| {
                panic!(
                    "Response {} should have been persisted after client \
                     disconnect on store=true /responses stream",
                    response_id
                )
            })
        };
        assert_eq!(stored.id.0, response_id);

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Dual of `_drains_after_disconnect`: with `store=true` and a mid-stream upstream error.
    #[tokio::test]
    async fn test_responses_simple_streaming_error_skips_persistence() {
        let worker_port = 20266;
        let total_chunks: usize = 10;
        let error_after: usize = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);
        let _guard = StreamInjectionGuard(worker_port);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4266)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 20)])
                .await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (_, f_pre) = breaker_counts(&worker);
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the body so the response.created event lands in `captured` and the upstream error is observed.
        let mut body = resp.into_body();
        let mut captured: Vec<u8> = Vec::new();
        while let Some(Ok(frame)) = body.frame().await {
            if frame.is_data() {
                if let Ok(data) = frame.into_data() {
                    captured.extend_from_slice(&data);
                }
            }
        }
        drop(body);
        let response_id =
            extract_response_id_from_sse(&captured).expect("response.created event with id");

        // Wait for the gateway-side producer task to exit so persistence (or its skip) and breaker tick have run.
        let _ = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT).await;

        // Co-assert that the breaker tick fired.
        let (_, f_post) = breaker_counts(&worker);
        assert!(
            f_post > f_pre,
            "/responses simple: mid-stream upstream error must record a \
             breaker failure. failures {}→{}",
            f_pre,
            f_post
        );

        let storage = ctx.app_context.response_storage.clone();
        use data_connector::ResponseId;
        let id = ResponseId::from(response_id.clone());
        if let Ok(Some(_)) = storage.get_response(&id).await {
            panic!(
                "Response {} should NOT have been persisted after \
                 mid-stream upstream error on store=true /responses",
                response_id
            );
        }

        ctx.shutdown().await;
    }

    /// OpenAI-mode (non-/responses) chat-completions cancel: this exercises the `OpenAIRouter` impl.
    #[tokio::test]
    async fn test_openai_router_streaming_cancel() {
        let worker_port = 20262;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4260)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(chunks_read > 0);

        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + MOCK_STREAM_BUFFER,
            "OpenAI-mode chunks_sent grew by more than the buffer ({} -> {})",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Tool-interception streaming cancel: when the client disconnects mid second-turn.
    #[tokio::test]
    async fn test_tool_interception_streaming_cancel_on_client_disconnect() {
        use smg::routers::{RouterFactory, RouterTrait};

        use crate::common::{
            mock_mcp_server::MockMCPServer,
            mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
        };

        let worker_port = 20263;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let mut mcp = MockMCPServer::start().await.expect("start mcp");
        let mcp_yaml = format!(
            "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
            mcp.url()
        );
        let dir = tempfile::tempdir().expect("tmpdir");
        let cfg_path = dir.path().join("mcp.yaml");
        std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");

        let mut worker = MockWorker::new(MockWorkerConfig {
            port: worker_port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 50,
            fail_rate: 0.0,
        });
        let worker_url = worker.start().await.expect("start worker");
        // Allow the mock worker's HTTP listener to bind before the router probes its health.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let router_cfg = RouterConfig::builder()
            .openai_mode(vec![worker_url])
            .random_policy()
            .host("127.0.0.1")
            .port(4263)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx = crate::common::create_test_context_with_mcp_config(
            router_cfg,
            cfg_path.to_str().unwrap(),
        )
        .await;
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&ctx).await.expect("router"));
        let app = crate::common::test_app::create_test_app_with_context(router, ctx);

        let payload = json!({
            "model": "mock-model",
            "input": "search something",
            "stream": true,
            "store": false,
            "tools": [{
                "type": "mcp",
                "server_label": "mock",
                "server_url": mcp.url()
            }]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 8).await;

        // Wait until the slow second-turn upstream request has started producing chunks — only the slow-stream branch.
        let mut waited_ms: u64 = 0;
        loop {
            if let Some(s) = get_stream_tracking_state(worker_port) {
                if s.chunks_sent > 0 {
                    break;
                }
            }
            if waited_ms >= 5000 {
                panic!(
                    "Second-turn upstream stream never started producing chunks \
                     within 5s — tool-interception path did not reach select!"
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            waited_ms += 50;
        }

        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + MOCK_STREAM_BUFFER,
            "Tool-interception second-turn chunks_sent grew by more than \
             the buffer ({} -> {}) — cancel did not propagate to upstream",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        worker.stop().await;
        mcp.stop().await;
    }

    /// Tool-interception path: client disconnects WHILE the gateway is still waiting for the upstream's response headers.
    #[tokio::test]
    async fn test_tool_interception_cancel_during_send() {
        use smg::routers::{RouterFactory, RouterTrait};

        use crate::common::{
            mock_mcp_server::MockMCPServer,
            mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
        };

        let worker_port = 20264;
        // slow_stream is configured so that IF the mock ever gets past the pre-response delay.
        let total_chunks: usize = 5;
        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let mut mcp = MockMCPServer::start().await.expect("start mcp");
        let mcp_yaml = format!(
            "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
            mcp.url()
        );
        let dir = tempfile::tempdir().expect("tmpdir");
        let cfg_path = dir.path().join("mcp.yaml");
        std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");

        let mut worker = MockWorker::new(MockWorkerConfig {
            port: worker_port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            // 1500ms pre-response delay: long enough.
            response_delay_ms: 1500,
            fail_rate: 0.0,
        });
        let worker_url = worker.start().await.expect("start worker");
        tokio::time::sleep(Duration::from_millis(200)).await;

        let router_cfg = RouterConfig::builder()
            .openai_mode(vec![worker_url])
            .random_policy()
            .host("127.0.0.1")
            .port(4264)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx = crate::common::create_test_context_with_mcp_config(
            router_cfg,
            cfg_path.to_str().unwrap(),
        )
        .await;
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&ctx).await.expect("router"));
        let app = crate::common::test_app::create_test_app_with_context(router, ctx);

        let payload = json!({
            "model": "mock-model",
            "input": "search something",
            "stream": true,
            "store": false,
            "tools": [{
                "type": "mcp",
                "server_label": "mock",
                "server_url": mcp.url()
            }]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drop immediately, while the mock is still inside its 1500ms pre-response sleep.
        let body = resp.into_body();
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(body);

        // Give the gateway and mock plenty of time to process the cancel and finish their respective sleeps.
        tokio::time::sleep(Duration::from_millis(2500)).await;

        assert!(
            get_stream_tracking_state(worker_port).is_none(),
            "Mock worker initialised the slow-stream tracker, which means \
             its handler completed the pre-response sleep — i.e. the gateway \
             waited for upstream headers instead of cancelling send().await \
             on client disconnect. Tracker state: {:?}",
            get_stream_tracking_state(worker_port)
        );

        clear_slow_stream_chunks(worker_port);
        worker.stop().await;
        mcp.stop().await;
    }

    /// Tool-interception path with `store=true`: when the second-turn upstream errors mid-stream.
    #[tokio::test]
    async fn test_tool_interception_streaming_error_skips_persistence() {
        use data_connector::ResponseId;
        use smg::routers::{RouterFactory, RouterTrait};

        use crate::common::{
            mock_mcp_server::MockMCPServer,
            mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
        };

        let worker_port = 20267;
        let total_chunks: usize = 10;
        let error_after: usize = 2;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);
        let _guard = StreamInjectionGuard(worker_port);

        let mut mcp = MockMCPServer::start().await.expect("start mcp");
        let mcp_yaml = format!(
            "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
            mcp.url()
        );
        let dir = tempfile::tempdir().expect("tmpdir");
        let cfg_path = dir.path().join("mcp.yaml");
        std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");

        let mut worker = MockWorker::new(MockWorkerConfig {
            port: worker_port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 20,
            fail_rate: 0.0,
        });
        let worker_url = worker.start().await.expect("start worker");
        tokio::time::sleep(Duration::from_millis(200)).await;

        let router_cfg = RouterConfig::builder()
            .openai_mode(vec![worker_url.clone()])
            .random_policy()
            .host("127.0.0.1")
            .port(4267)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx = crate::common::create_test_context_with_mcp_config(
            router_cfg,
            cfg_path.to_str().unwrap(),
        )
        .await;
        let storage = ctx.response_storage.clone();
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&ctx).await.expect("router"));
        let pinned_worker = ctx
            .worker_registry
            .get_by_url(&worker_url)
            .expect("worker should be registered after router create");
        let (_, f_pre) = breaker_counts(&pinned_worker);
        let app = crate::common::test_app::create_test_app_with_context(router, ctx);

        let payload = json!({
            "model": "mock-model",
            "input": "search something",
            "stream": true,
            "store": true,
            "tools": [{
                "type": "mcp",
                "server_label": "mock",
                "server_url": mcp.url()
            }]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain fully so the second-turn upstream error is observed.
        let mut body = resp.into_body();
        let mut captured: Vec<u8> = Vec::new();
        while let Some(Ok(frame)) = body.frame().await {
            if frame.is_data() {
                if let Ok(data) = frame.into_data() {
                    captured.extend_from_slice(&data);
                }
            }
        }
        drop(body);
        let response_id =
            extract_response_id_from_sse(&captured).expect("response.created event with id");

        // Wait for the second-turn producer to exit.
        let state = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT)
            .await
            .expect("second-turn producer to exit within timeout");

        assert!(
            state.chunks_sent >= error_after && !state.completed,
            "Second-turn must have entered slow_stream and errored after \
             {} chunks (got chunks_sent={}, completed={})",
            error_after,
            state.chunks_sent,
            state.completed,
        );

        // Co-assert the breaker tick fired so a "skip-persistence + record nothing" regression can't silently pass this.
        let (_, f_post) = breaker_counts(&pinned_worker);
        assert!(
            f_post > f_pre,
            "tool-interception: second-turn upstream error must record a \
             breaker failure. failures {}→{}",
            f_pre,
            f_post
        );

        let id = ResponseId::from(response_id.clone());
        if let Ok(Some(_)) = storage.get_response(&id).await {
            panic!(
                "Response {} should NOT have been persisted after \
                 second-turn upstream error on store=true tool-interception path",
                response_id
            );
        }

        worker.stop().await;
        mcp.stop().await;
    }

    /// After enough consecutive mid-stream upstream errors, the `BreakerTrackedStream` drop should record failures often enough.
    #[tokio::test]
    async fn test_streaming_errors_trip_circuit_breaker() {
        use smg::config::CircuitBreakerConfig;

        let worker_port = 20265;
        let total_chunks: usize = 10;
        let error_after: usize = 1;
        let failure_threshold = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);

        let config = TestRouterConfig::round_robin_with_circuit_breaker(
            4265,
            CircuitBreakerConfig {
                failure_threshold,
                success_threshold: 2,
                timeout_duration_secs: 30,
                window_duration_secs: 60,
            },
        );

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let app = ctx.create_app().await;

        // Drain each request fully so the BreakerTrackedStream sees
        // `Some(Err(...))` and tags the terminal state as Errored before
        // Drop fires `record_failure`.
        for _ in 0..failure_threshold {
            let payload = json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true
            });
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            let mut body = resp.into_body();
            while body.frame().await.is_some() {}
        }

        let worker = ctx
            .app_context
            .worker_registry
            .get_by_url(&worker_url)
            .expect("worker should be registered");
        let breaker = worker.circuit_breaker();
        assert!(
            !matches!(breaker.state(), smg::core::CircuitState::Closed),
            "Circuit breaker should NOT be Closed after {} streaming errors; \
             state = {:?}, consecutive_failures = {}, total_failures = {}, \
             total_successes = {}",
            failure_threshold,
            breaker.state(),
            breaker.consecutive_failures(),
            breaker.total_failures(),
            breaker.total_successes(),
        );

        clear_slow_stream_chunks(worker_port);
        clear_stream_error_after_chunks(worker_port);
        ctx.shutdown().await;
    }

    // -------- Breaker accounting tests -------- For single-upstream-call
    // streaming paths (http chat, OpenAI chat, PD generate, /responses
    // simple).

    /// Returns `(total_successes, total_failures)` for the given worker.
    fn breaker_counts(worker: &Arc<dyn smg::core::Worker>) -> (u64, u64) {
        let breaker = worker.circuit_breaker();
        (breaker.total_successes(), breaker.total_failures())
    }

    /// Capture the worker for a given URL once at test start. See
    /// `breaker_counts` for why repeated `get_by_url` lookups are unsafe.
    fn pin_worker(ctx: &AppTestContext, worker_url: &str) -> Arc<dyn smg::core::Worker> {
        ctx.app_context
            .worker_registry
            .get_by_url(worker_url)
            .expect("worker should be registered")
    }

    /// RAII cleanup for per-port stream injection state.
    struct StreamInjectionGuard(u16);
    impl Drop for StreamInjectionGuard {
        fn drop(&mut self) {
            clear_stream_error_after_chunks(self.0);
            clear_slow_stream_chunks(self.0);
        }
    }

    /// http chat: client disconnect mid-stream must NOT tick the breaker.
    #[tokio::test]
    async fn test_disconnect_does_not_move_breaker_http_chat() {
        let worker_port = 20270;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4270);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);

        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "long"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 3).await;
        drop(body);

        // Wait until the upstream producer exits so the body Drop has run and any breaker tick has landed.
        let _ = assert_cancelled_before_completion(worker_port).await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            (s_post - s_pre, f_post - f_pre),
            (0, 0),
            "http chat: client disconnect must not move breaker. \
             successes {}→{}, failures {}→{}",
            s_pre,
            s_post,
            f_pre,
            f_post
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// OpenAIRouter chat: client disconnect mid-stream must NOT tick the breaker.
    #[tokio::test]
    async fn test_disconnect_does_not_move_breaker_openai_chat() {
        let worker_port = 20271;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4271)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);

        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "long"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 3).await;
        drop(body);

        let _ = assert_cancelled_before_completion(worker_port).await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            (s_post - s_pre, f_post - f_pre),
            (0, 0),
            "openai chat: client disconnect must not move breaker. \
             successes {}→{}, failures {}→{}",
            s_pre,
            s_post,
            f_pre,
            f_post
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// PD streaming client disconnect: - decode breaker must show zero delta.
    #[tokio::test]
    async fn test_disconnect_does_not_move_breaker_pd_decode() {
        let prefill_port = 20272;
        let decode_port = 20273;
        let total_chunks: usize = 20;

        reset_stream_tracker(decode_port);
        set_slow_stream_chunks(decode_port, total_chunks);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4272)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                let mut w = TestWorkerConfig::decode(decode_port);
                w.response_delay_ms = 50;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode_worker = pin_worker(&ctx, &decode_url);
        let prefill_worker = pin_worker(&ctx, &prefill_url);

        let (s_pre_decode, f_pre_decode) = breaker_counts(&decode_worker);
        let (s_pre_prefill, f_pre_prefill) = breaker_counts(&prefill_worker);

        let payload = json!({
            "text": "PD streaming breaker test",
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 3).await;
        drop(body);

        let _ = assert_cancelled_before_completion(decode_port).await;

        let (s_post_decode, f_post_decode) = breaker_counts(&decode_worker);
        assert_eq!(
            (s_post_decode - s_pre_decode, f_post_decode - f_pre_decode),
            (0, 0),
            "PD decode: client disconnect must not move breaker. \
             successes {}→{}, failures {}→{}",
            s_pre_decode,
            s_post_decode,
            f_pre_decode,
            f_post_decode
        );

        let (s_post_prefill, f_post_prefill) = breaker_counts(&prefill_worker);
        assert_eq!(
            (
                s_post_prefill - s_pre_prefill,
                f_post_prefill - f_pre_prefill
            ),
            (1, 0),
            "PD prefill: 2xx must record exactly one success regardless of \
             client decode-stream disconnect. successes {}→{}, failures {}→{}",
            s_pre_prefill,
            s_post_prefill,
            f_pre_prefill,
            f_post_prefill
        );

        clear_slow_stream_chunks(decode_port);
        ctx.shutdown().await;
    }

    /// /v1/responses simple (no-persist): client disconnect must not move the worker's circuit breaker.
    #[tokio::test]
    async fn test_disconnect_does_not_move_breaker_responses_simple() {
        let worker_port = 20274;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4274)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);

        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 3).await;
        drop(body);

        let _ = assert_cancelled_before_completion(worker_port).await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            (s_post - s_pre, f_post - f_pre),
            (0, 0),
            "/responses simple: client disconnect must not move breaker. \
             successes {}→{}, failures {}→{}",
            s_pre,
            s_post,
            f_pre,
            f_post
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// /v1/responses simple, mid-stream error: the spawned forwarder in `handle_simple_streaming_passthrough` must record a failure.
    #[tokio::test]
    async fn test_responses_simple_mid_stream_error_records_failure() {
        let worker_port = 20275;
        let total_chunks: usize = 10;
        let error_after: usize = 2;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4275)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);

        let (_, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the body so we observe the mid-stream error.
        let mut body = resp.into_body();
        while body.frame().await.is_some() {}
        drop(body);

        // Wait for the producer task to exit so any breaker tick is observable.
        let _ = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT).await;

        let (_, f_post) = breaker_counts(&worker);
        assert!(
            f_post > f_pre,
            "/responses simple: mid-stream upstream error must record \
             at least one failure on the breaker. failures {}→{}",
            f_pre,
            f_post
        );

        clear_stream_error_after_chunks(worker_port);
        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// OpenAIRouter chat, mid-stream upstream error: breaker MUST record at least one failure.
    #[tokio::test]
    async fn test_openai_chat_mid_stream_error_records_failure() {
        let worker_port = 20276;
        let total_chunks: usize = 10;
        let error_after: usize = 2;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4276)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);

        let (_, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });

        reset_stream_tracker(worker_port);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let _ = resp.status();
        let mut body = resp.into_body();
        while body.frame().await.is_some() {}
        drop(body);
        let _ = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT).await;

        let (_s_post, f_post) = breaker_counts(&worker);
        assert!(
            f_post > f_pre,
            "openai chat: mid-stream upstream error must record at \
             least one failure on the breaker. failures {}→{}",
            f_pre,
            f_post
        );

        clear_stream_error_after_chunks(worker_port);
        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    // -------- 5xx-streaming and happy-path success coverage --------

    /// http::Router streaming 5xx must record `record_failure`, not success.
    #[tokio::test]
    async fn test_http_chat_streaming_5xx_records_failure() {
        let worker_port = 20290;
        let config = TestRouterConfig::round_robin(4290);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(worker_port, 1.0)],
        )
        .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            s_post - s_pre,
            0,
            "http chat streaming 5xx must not record success"
        );
        assert!(
            f_post > f_pre,
            "http chat streaming 5xx must record at least one failure. \
             failures {}→{}",
            f_pre,
            f_post
        );

        ctx.shutdown().await;
    }

    /// OpenAIRouter streaming 5xx must record `record_failure`, not success.
    #[tokio::test]
    async fn test_openai_chat_streaming_5xx_records_failure() {
        let worker_port = 20291;
        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4291)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(worker_port, 1.0)],
        )
        .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            s_post - s_pre,
            0,
            "openai chat streaming 5xx must not record success"
        );
        assert!(
            f_post > f_pre,
            "openai chat streaming 5xx must record at least one failure. \
             failures {}→{}",
            f_pre,
            f_post
        );

        ctx.shutdown().await;
    }

    /// PD decode 5xx on streaming request: decode breaker records failure, not success.
    #[tokio::test]
    async fn test_pd_decode_streaming_5xx_records_failure() {
        let prefill_port = 20292;
        let decode_port = 20293;
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4292)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                let mut w = TestWorkerConfig::decode(decode_port);
                w.fail_rate = 1.0;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode = pin_worker(&ctx, &decode_url);
        let prefill = pin_worker(&ctx, &prefill_url);
        let (s_pre, f_pre) = breaker_counts(&decode);
        let (_s_pre_prefill, f_pre_prefill) = breaker_counts(&prefill);

        let payload = json!({ "text": "x", "stream": true });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        let (s_post, f_post) = breaker_counts(&decode);
        assert_eq!(
            s_post - s_pre,
            0,
            "PD decode streaming 5xx must not record success"
        );
        assert!(
            f_post > f_pre,
            "PD decode streaming 5xx must record at least one failure. \
             failures {}→{}",
            f_pre,
            f_post
        );

        // Healthy prefill must not be penalised when only decode returns 5xx.
        let (_s_post_prefill, f_post_prefill) = breaker_counts(&prefill);
        assert_eq!(
            f_post_prefill - f_pre_prefill,
            0,
            "PD streaming: healthy prefill must not be penalised when only \
             decode returns 5xx. prefill failures {}→{}",
            f_pre_prefill,
            f_post_prefill
        );

        ctx.shutdown().await;
    }

    /// PD non-streaming, decode 4xx: the decode breaker MUST NOT record a failure.
    #[tokio::test]
    async fn test_pd_decode_non_streaming_4xx_does_not_penalise_breaker() {
        let prefill_port = 20313;
        let decode_port = 20314;

        set_fail_status_code(decode_port, 400);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4313)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                let mut w = TestWorkerConfig::decode(decode_port);
                w.fail_rate = 1.0;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode = pin_worker(&ctx, &decode_url);
        let prefill = pin_worker(&ctx, &prefill_url);
        let (_s_pre_decode, f_pre_decode) = breaker_counts(&decode);
        let (_s_pre_prefill, f_pre_prefill) = breaker_counts(&prefill);

        // Non-streaming /generate request.
        let payload = json!({ "text": "x" });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        // Legacy semantics.
        let (_s_post_decode, f_post_decode) = breaker_counts(&decode);
        assert_eq!(
            f_post_decode - f_pre_decode,
            0,
            "PD decode 4xx is a client fault, not a worker fault — the \
             decode breaker must not record a failure. failures {}→{}",
            f_pre_decode,
            f_post_decode,
        );

        // Prefill stayed healthy and must also not be penalised by a client-caused decode 4xx.
        let (_s_post_prefill, f_post_prefill) = breaker_counts(&prefill);
        assert_eq!(
            f_post_prefill - f_pre_prefill,
            0,
            "PD prefill must not be penalised by a decode 4xx. \
             failures {}→{}",
            f_pre_prefill,
            f_post_prefill,
        );

        clear_fail_status_code(decode_port);
        ctx.shutdown().await;
    }

    /// /v1/responses simple streaming 5xx must record `record_failure`, not success.
    #[tokio::test]
    async fn test_responses_simple_streaming_5xx_records_failure() {
        let worker_port = 20294;
        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4294)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(worker_port, 1.0)],
        )
        .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "input": "x",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            s_post - s_pre,
            0,
            "/responses simple 5xx must not record success"
        );
        assert!(
            f_post > f_pre,
            "/responses simple 5xx must record at least one failure. \
             failures {}→{}",
            f_pre,
            f_post
        );

        ctx.shutdown().await;
    }

    /// /v1/responses simple, clean stream: must record exactly one success and no failures.
    #[tokio::test]
    async fn test_responses_simple_clean_stream_records_one_success() {
        let worker_port = 20295;
        let total_chunks: usize = 4;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4295)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 5)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (s_pre, f_pre) = breaker_counts(&worker);

        let payload = json!({
            "model": "mock-model",
            "input": "x",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = resp.into_body().collect().await;
        let _ = wait_for_stream_finish(worker_port, STREAM_FINISH_TIMEOUT).await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            (s_post - s_pre, f_post - f_pre),
            (1, 0),
            "/responses simple clean stream must record exactly 1 success \
             and 0 failures. successes {}→{}, failures {}→{}",
            s_pre,
            s_post,
            f_pre,
            f_post
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// PD generate, clean stream: decode breaker must record exactly one success, prefill exactly one success.
    #[tokio::test]
    async fn test_pd_clean_stream_records_one_success() {
        let prefill_port = 20296;
        let decode_port = 20297;
        let total_chunks: usize = 4;

        reset_stream_tracker(decode_port);
        set_slow_stream_chunks(decode_port, total_chunks);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4296)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                let mut w = TestWorkerConfig::decode(decode_port);
                w.response_delay_ms = 5;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode = pin_worker(&ctx, &decode_url);
        let prefill = pin_worker(&ctx, &prefill_url);
        let (s_pre_decode, f_pre_decode) = breaker_counts(&decode);
        let (s_pre_prefill, f_pre_prefill) = breaker_counts(&prefill);

        let payload = json!({ "text": "x", "stream": true });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = resp.into_body().collect().await;
        let _ = wait_for_stream_finish(decode_port, STREAM_FINISH_TIMEOUT).await;

        let (s_post_decode, f_post_decode) = breaker_counts(&decode);
        assert_eq!(
            (s_post_decode - s_pre_decode, f_post_decode - f_pre_decode),
            (1, 0),
            "PD decode clean stream must record exactly 1 success and 0 failures. \
             successes {}→{}, failures {}→{}",
            s_pre_decode,
            s_post_decode,
            f_pre_decode,
            f_post_decode
        );
        let (s_post_prefill, f_post_prefill) = breaker_counts(&prefill);
        assert_eq!(
            (
                s_post_prefill - s_pre_prefill,
                f_post_prefill - f_pre_prefill
            ),
            (1, 0),
            "PD prefill clean stream must record exactly 1 success and 0 failures. \
             successes {}→{}, failures {}→{}",
            s_pre_prefill,
            s_post_prefill,
            f_pre_prefill,
            f_post_prefill
        );

        clear_slow_stream_chunks(decode_port);
        ctx.shutdown().await;
    }

    /// PD generate, prefill 5xx (decode never reached): prefill breaker records failure, decode breaker untouched.
    #[tokio::test]
    async fn test_pd_prefill_5xx_records_failure() {
        let prefill_port = 20298;
        let decode_port = 20299;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4298)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                {
                    let mut p = TestWorkerConfig::prefill(prefill_port);
                    p.fail_rate = 1.0;
                    p
                },
                TestWorkerConfig::decode(decode_port),
            ],
        )
        .await;
        let app = ctx.create_app().await;
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill = pin_worker(&ctx, &prefill_url);
        let decode = pin_worker(&ctx, &decode_url);
        let (s_pre_p, f_pre_p) = breaker_counts(&prefill);
        let (s_pre_d, f_pre_d) = breaker_counts(&decode);

        let payload = json!({ "text": "x", "stream": true });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;

        let (s_post_p, f_post_p) = breaker_counts(&prefill);
        let (s_post_d, f_post_d) = breaker_counts(&decode);
        assert!(
            f_post_p > f_pre_p,
            "PD prefill 5xx must record at least one failure. failures {}→{}",
            f_pre_p,
            f_post_p
        );
        assert_eq!(
            s_post_p - s_pre_p,
            0,
            "PD prefill 5xx must not record success"
        );
        assert_eq!(
            (s_post_d - s_pre_d, f_post_d - f_pre_d),
            (0, 0),
            "PD decode must be untouched when prefill fails. \
             successes {}→{}, failures {}→{}",
            s_pre_d,
            s_post_d,
            f_pre_d,
            f_post_d
        );

        ctx.shutdown().await;
    }

    /// http chat streaming, upstream connect failure BEFORE the `BreakerTrackedStream` is constructed.
    #[tokio::test]
    async fn test_http_chat_pre_stream_failure_records_breaker_streaming() {
        use smg::config::RetryConfig;

        let worker_port = 20310;

        // max_retries=1 keeps the assertion exact: one attempt → one
        // failure tick. Any larger value multiplies the count.
        let config = TestRouterConfig::round_robin_with_retry(
            4310,
            RetryConfig {
                max_retries: 1,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            },
        );
        let mut ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(worker_port)])
                .await;
        let app = ctx.create_app().await;
        let worker_url = format!("http://127.0.0.1:{}", worker_port);
        let worker = pin_worker(&ctx, &worker_url);
        let (s_pre, f_pre) = breaker_counts(&worker);

        // Stop the worker AFTER startup health check has marked it healthy.
        ctx.workers[0].stop().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.status().is_server_error(),
            "Expected 5xx after upstream connect failure, got {}",
            resp.status()
        );
        let _ = resp.into_body().collect().await;

        let (s_post, f_post) = breaker_counts(&worker);
        assert_eq!(
            (s_post - s_pre, f_post - f_pre),
            (0, 1),
            "http chat streaming: pre-stream upstream failure must record \
             exactly one breaker failure (no tracked stream was installed, \
             so the deferred-record path doesn't fire). \
             successes {}→{}, failures {}→{}",
            s_pre,
            s_post,
            f_pre,
            f_post
        );

        ctx.shutdown().await;
    }

    /// PD streaming, decode connect failure BEFORE the `BreakerTrackedStream` is constructed.
    #[tokio::test]
    async fn test_pd_decode_pre_stream_failure_records_breaker_streaming() {
        use smg::config::RetryConfig;

        let prefill_port = 20311;
        let decode_port = 20312;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4311)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 1,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            })
            .build_unchecked();
        let mut ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(prefill_port),
                TestWorkerConfig::decode(decode_port),
            ],
        )
        .await;
        let app = ctx.create_app().await;
        let decode_url = format!("http://127.0.0.1:{}", decode_port);
        let prefill_url = format!("http://127.0.0.1:{}", prefill_port);
        let decode_worker = pin_worker(&ctx, &decode_url);
        let prefill_worker = pin_worker(&ctx, &prefill_url);
        let (s_pre_decode, f_pre_decode) = breaker_counts(&decode_worker);
        let (_s_pre_prefill, f_pre_prefill) = breaker_counts(&prefill_worker);

        ctx.workers[1].stop().await;

        let payload = json!({ "text": "x", "stream": true });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.status().is_server_error(),
            "Expected 5xx after decode connect failure, got {}",
            resp.status()
        );
        let _ = resp.into_body().collect().await;

        let (s_post_decode, f_post_decode) = breaker_counts(&decode_worker);
        assert!(
            f_post_decode > f_pre_decode,
            "PD streaming: pre-stream decode failure must record at least \
             one breaker failure on the decode worker (no tracked stream \
             was installed). failures {}→{}",
            f_pre_decode,
            f_post_decode
        );
        assert_eq!(
            s_post_decode - s_pre_decode,
            0,
            "PD streaming pre-stream decode failure must not record a \
             success on the decode breaker. successes {}→{}",
            s_pre_decode,
            s_post_decode
        );

        // Prefill stayed up and its `send()` returned 2xx.
        let (_s_post_prefill, f_post_prefill) = breaker_counts(&prefill_worker);
        assert_eq!(
            f_post_prefill - f_pre_prefill,
            0,
            "PD streaming: healthy prefill must not be penalised when only \
             decode fails. prefill failures {}→{}",
            f_pre_prefill,
            f_post_prefill
        );

        ctx.shutdown().await;
    }
}
