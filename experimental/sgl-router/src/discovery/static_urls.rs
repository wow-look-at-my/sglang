// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-URL discovery backend.

use crate::config::StaticUrlsDiscoveryConfig;
use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use anyhow::Result;
use tokio::sync::mpsc;

/// Spawn the static-URLs producer task and return its `JoinHandle`.
///
/// Returns `Result` for parity with [`crate::discovery::k8s::spawn`] (which
/// can fail to construct a `kube::Client`).
/// infallible.
pub async fn spawn(
    cfg: StaticUrlsDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        for url in cfg.urls {
            let spec = WorkerSpec {
                id: WorkerId(url.clone()),
                url,
                mode: WorkerMode::Plain,
                model_ids: Vec::new(),
                bootstrap_port: None,
            };
            if tx.send(DiscoveryEvent::Added(spec)).await.is_err() {
                tracing::info!(
                    "static_urls discovery: event channel closed during fan-out; exiting"
                );
                return;
            }
        }
        tracing::debug!(
            "static_urls discovery: initial fan-out complete; parking until channel closes"
        );
        // After fan-out the static backend has no further work —.
        tx.closed().await;
        tracing::info!(
            "static_urls discovery: event channel closed by receiver \
             (worker manager dropped its end, or shutdown abort raced); exiting"
        );
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task exits cleanly when the consumer drops the receiver mid-fanout.
    #[tokio::test]
    async fn exits_when_receiver_dropped() {
        let cfg = StaticUrlsDiscoveryConfig {
            urls: (0..10).map(|i| format!("http://w{i}:30000")).collect(),
        };
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let h = spawn(cfg, tx).await.unwrap();
        // No panic, no hang — task exits on the first send error.
        h.await.unwrap();
    }

    /// After fan-out the task must STAY ALIVE so the critical-task supervisor (`server::supervisor::supervise_critical_tasks`).
    #[tokio::test]
    async fn stays_alive_after_fanout_until_receiver_dropped() {
        use std::time::Duration;

        let cfg = StaticUrlsDiscoveryConfig {
            urls: vec!["http://w0:30000".into(), "http://w1:30000".into()],
        };
        let (tx, mut rx) = mpsc::channel(8);
        let h = spawn(cfg, tx).await.unwrap();

        // Drain the fan-out so the task is past the for-loop.
        for _ in 0..2 {
            let _ = rx.recv().await.expect("fan-out event");
        }

        // Now give the task a long-by-test-standards moment to exit post-fanout.
        let mut handle = h;
        let exited = tokio::time::timeout(Duration::from_millis(200), &mut handle).await;
        let still_running = exited.is_err();
        if !still_running {
            panic!(
                "static_urls task exited after fan-out (joined as {exited:?}); \
                 this trips `supervise_critical_tasks` → mark_unready and the pod \
                 becomes /readyz 503. The task must park until the receiver is dropped."
            );
        }
        // Clean shutdown: dropping the receiver closes the channel.
        drop(rx);
        let joined = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("task must exit promptly after the receiver is dropped");
        joined.expect("task panicked during clean shutdown");
    }
}
