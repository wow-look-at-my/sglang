// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Concurrent-state invariants for the worker/registry/breaker layer.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use sgl_router::workers::{Worker, WorkerRegistry};

/// HalfOpen state must admit at most one probe at a time even under high concurrency.
#[tokio::test(start_paused = true)]
async fn breaker_half_open_admits_only_one_probe_concurrently() {
    let cb = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
        cool_down: Duration::from_millis(50),
    }));

    // Trip into Open.
    cb.record_failure();
    assert!(!cb.allow(), "must be Open immediately after a failure");

    // Advance the paused clock past cool_down so the next `allow()` will attempt the Open → HalfOpen transition.
    tokio::time::advance(Duration::from_millis(60)).await;

    let admitted = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..32 {
        let cb = cb.clone();
        let admitted = admitted.clone();
        handles.push(tokio::spawn(async move {
            if cb.allow() {
                admitted.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(
        admitted.load(Ordering::Relaxed),
        1,
        "exactly one probe must be admitted in HalfOpen",
    );
}

/// Concurrent `add_with_cb` (upsert) and `remove` from many threads on the same WorkerId must not panic, must not deadlock.
#[test]
fn registry_concurrent_add_remove_keeps_indexes_consistent() {
    let r = Arc::new(WorkerRegistry::default());
    let model = ModelId("m".into());

    let mut handles = Vec::new();
    for i in 0..8 {
        let r = r.clone();
        let model = model.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..200 {
                let _ = r.add(WorkerSpec {
                    id: WorkerId(format!("w{i}")),
                    url: format!("http://w{i}:30000"),
                    mode: WorkerMode::Plain,
                    model_ids: vec![model.clone()],
                    bootstrap_port: None,
                });
                let snapshot = r.workers_for(&model);
                for w in &snapshot {
                    // Cross-index invariant: an entry surfaced via
                    // `by_model[m]` must come from a Worker whose own
                    // `model_ids` includes `m`.
                    assert!(
                        w.model_ids.contains(&model),
                        "cross-index drift: by_model[{model:?}] surfaced \
                         {:?} whose own model_ids = {:?}",
                        w.id,
                        w.model_ids,
                    );
                }
                r.remove(&WorkerId(format!("w{i}")));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // After every thread finishes, every removed worker must be gone.
    assert!(
        r.workers_for(&model).is_empty(),
        "registry must be empty after all threads finished their add/remove cycles",
    );
}

/// `LoadGuard` must decrement the counter during a panic-unwind, not on a normal scope exit.
#[test]
fn load_guard_decrements_on_panic_unwind() {
    let w = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("w".into()),
        url: "http://x:30000".into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }));
    assert_eq!(w.router_inflight_load(), 0);

    let w_inner = w.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _g = w_inner.load_guard();
        assert_eq!(w_inner.router_inflight_load(), 1);
        panic!("synthetic panic to exercise Drop on unwind");
    }));
    assert!(result.is_err(), "the closure must have panicked");
    assert_eq!(
        w.router_inflight_load(),
        0,
        "LoadGuard's Drop must decrement even when the holder panics",
    );
}
