//! Shared streaming helpers used by the HTTP / OpenAI / PD routers.
//!

use std::{
    fmt::Display,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use bytes::Bytes;
use futures_util::Stream;
use tracing::error;

use crate::core::Worker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terminal {
    /// Stream is still in flight, or was dropped before terminating.
    Active,
    /// Stream returned `None` or the caller marked it complete.
    Completed,
    /// Stream yielded an `Err` item.
    Errored,
}

/// Wraps a `Stream<Item = Result<Bytes, E>>` so that the circuit breaker on
/// `worker` is updated exactly once on drop.
#[must_use = "BreakerTrackedStream must be polled to completion (or pre-marked) \
              and then dropped for the circuit breaker to record an outcome; \
              discarding it immediately records nothing"]
pub struct BreakerTrackedStream<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    worker: Arc<dyn Worker>,
    log_url: String,
    terminal: Terminal,
}

impl<E> BreakerTrackedStream<E> {
    pub fn new<S>(inner: S, worker: Arc<dyn Worker>, log_url: String) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            worker,
            log_url,
            terminal: Terminal::Active,
        }
    }

    /// Mark the stream as cleanly completed.
    pub fn mark_completed(&mut self) {
        if self.terminal == Terminal::Active {
            self.terminal = Terminal::Completed;
        }
    }

    /// Pre-tag the stream as terminally errored.
    pub fn mark_errored(&mut self) {
        self.terminal = Terminal::Errored;
    }
}

impl<E: Display> Stream for BreakerTrackedStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                error!("Upstream stream error from worker {}: {}", self.log_url, e);
                self.terminal = Terminal::Errored;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if self.terminal == Terminal::Active {
                    self.terminal = Terminal::Completed;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<E> Drop for BreakerTrackedStream<E> {
    fn drop(&mut self) {
        match self.terminal {
            Terminal::Completed => self.worker.circuit_breaker().record_success(),
            Terminal::Errored => self.worker.circuit_breaker().record_failure(),
            // Client disconnected before we knew the worker's verdict.
            Terminal::Active => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt, sync::Arc};

    use bytes::Bytes;
    use futures_util::StreamExt;

    use super::BreakerTrackedStream;
    use crate::core::{BasicWorkerBuilder, Worker};

    /// Lightweight error type for tests — keeps the wrapper generic so we don't need.
    #[derive(Debug)]
    struct TestErr(&'static str);

    impl fmt::Display for TestErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://test-worker").build())
    }

    fn breaker_counters(w: &Arc<dyn Worker>) -> (u64, u64) {
        let cb = w.circuit_breaker();
        (cb.total_successes(), cb.total_failures())
    }

    #[tokio::test]
    async fn drop_while_active_records_nothing() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 0));
    }

    #[tokio::test]
    async fn clean_stream_records_one_success() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn stream_error_records_one_failure() {
        let w = worker();
        let inner =
            futures_util::stream::iter(vec![Ok(Bytes::from_static(b"a")), Err(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn errored_is_absorbing_across_subsequent_polls() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Err::<Bytes, _>(TestErr("boom")),
            Ok(Bytes::from_static(b"after")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_then_drop_records_success() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn mark_errored_then_clean_end_still_records_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Ok::<_, TestErr>(Bytes::from_static(b"a"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_errored();
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_does_not_overwrite_errored() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Err::<Bytes, _>(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    // PD's [DONE] handler calls mark_completed before the underlying byte stream finishes.
    #[tokio::test]
    async fn mark_completed_then_later_err_escalates_to_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(TestErr("trailing transport error")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        // Caller observes the [DONE] chunk and pre-marks completed...
        assert!(tracked.next().await.is_some());
        tracked.mark_completed();
        // ...but a trailing poll surfaces a transport error.
        assert!(matches!(tracked.next().await, Some(Err(_))));
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }
}
