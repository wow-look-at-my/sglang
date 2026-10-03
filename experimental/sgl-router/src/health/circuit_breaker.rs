// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

// NOTE: `opened_at` uses `tokio::time::Instant` rather than `std::time::Instant`.

use std::num::NonZeroU32;
use std::sync::Mutex;
use std::time::Duration;
use tokio::time::Instant;

/// Consistent `(admit, state_code)` pair read under a single breaker lock. See [`CircuitBreaker::snapshot`].
#[derive(Debug, Clone, Copy)]
pub struct CircuitSnapshot {
    /// Would the breaker admit a request right now (`would_allow` semantics).
    pub admit: bool,
    pub state_code: u8,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub threshold: NonZeroU32,
    pub cool_down: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            threshold: NonZeroU32::new(3).expect("3 is non-zero"),
            cool_down: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum State {
    Closed,
    Open { opened_at: Instant },
    HalfOpen { probe_in_flight: bool },
}

#[derive(Debug)]
struct Inner {
    state: State,
    consecutive_failures: u32,
    probe_generation: u64,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    inner: Mutex<Inner>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: State::Closed,
                consecutive_failures: 0,
                probe_generation: 0,
            }),
            config,
        }
    }

    /// Non-mutating predicate: would [`allow`] return `true` if called right
    /// now?
    pub fn would_allow(&self) -> bool {
        let g = self.inner.lock().unwrap();
        match g.state {
            State::Closed => true,
            State::Open { opened_at } => opened_at.elapsed() >= self.config.cool_down,
            State::HalfOpen { probe_in_flight } => !probe_in_flight,
        }
    }

    /// Single-lock snapshot of `(admit, state_code)` for the `/metrics`
    /// scrape path.
    pub fn snapshot(&self) -> CircuitSnapshot {
        let g = self.inner.lock().unwrap();
        let (admit, state_code) = match g.state {
            State::Closed => (true, 0),
            State::Open { opened_at } => (opened_at.elapsed() >= self.config.cool_down, 1),
            State::HalfOpen { probe_in_flight } => (!probe_in_flight, 2),
        };
        CircuitSnapshot { admit, state_code }
    }

    /// Admit a caller that will explicitly record its outcome.
    pub fn allow(&self) -> bool {
        self.acquire().map(|permit| permit.disarm()).is_some()
    }

    /// Claim admission, releasing an unfinished recovery probe on cancellation.
    pub fn acquire(&self) -> Option<CircuitPermit<'_>> {
        let mut g = self.inner.lock().unwrap();
        let generation = match g.state {
            State::Closed => None,
            State::Open { opened_at } if opened_at.elapsed() < self.config.cool_down => {
                return None;
            }
            State::HalfOpen {
                probe_in_flight: true,
            } => return None,
            _ => {
                g.probe_generation = g.probe_generation.wrapping_add(1);
                g.state = State::HalfOpen {
                    probe_in_flight: true,
                };
                Some(g.probe_generation)
            }
        };
        Some(CircuitPermit {
            breaker: self,
            generation,
        })
    }

    pub fn record_success(&self) {
        let mut g = self.inner.lock().unwrap();
        g.consecutive_failures = 0;
        g.state = State::Closed;
    }

    pub fn record_failure(&self) {
        let mut g = self.inner.lock().unwrap();
        match g.state {
            State::Closed | State::HalfOpen { .. } => {
                g.consecutive_failures += 1;
                if g.consecutive_failures >= self.config.threshold.get() {
                    g.state = State::Open {
                        opened_at: Instant::now(),
                    };
                }
            }
            State::Open { .. } => {
            }
        }
    }

    pub fn record_backpressure(&self) {
        let mut g = self.inner.lock().unwrap();
        if matches!(g.state, State::HalfOpen { .. }) {
            g.consecutive_failures = 0;
            g.state = State::Closed;
        }
    }
}

/// Releases only the recovery probe claimed by this admission.
pub struct CircuitPermit<'a> {
    breaker: &'a CircuitBreaker,
    generation: Option<u64>,
}

impl CircuitPermit<'_> {
    /// Leave outcome accounting to the caller or streaming completion hook.
    pub fn disarm(mut self) {
        self.generation = None;
    }
}

impl Drop for CircuitPermit<'_> {
    fn drop(&mut self) {
        let Some(generation) = self.generation else {
            return;
        };
        let mut g = self.breaker.inner.lock().unwrap();
        if g.probe_generation == generation {
            if let State::HalfOpen { probe_in_flight } = &mut g.state {
                *probe_in_flight = false;
            }
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb(threshold: u32, cool_down_secs: u64) -> CircuitBreaker {
        CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(threshold).unwrap(),
            cool_down: Duration::from_secs(cool_down_secs),
        })
    }

    #[test]
    fn cancelled_permits_release_only_their_own_probe() {
        let b = cb(1, 0);
        let closed = b.acquire().unwrap();
        b.record_failure();
        let old_probe = b.acquire().unwrap();
        b.record_success();
        b.record_failure();
        let current_probe = b.acquire().unwrap();
        drop((closed, old_probe));
        assert!(!b.would_allow());
        drop(current_probe);
        assert!(b.would_allow());
        assert_eq!(b.snapshot().state_code, 2);
    }

    #[test]
    fn state_code_is_closed_by_default() {
        assert_eq!(CircuitBreaker::new().snapshot().state_code, 0);
    }

    #[test]
    fn state_code_reports_open_only_after_threshold() {
        let b = cb(2, 30);
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            0,
            "1 failure < threshold 2 stays closed",
        );
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            1,
            "reaching threshold opens the breaker",
        );
    }

    #[tokio::test(start_paused = true)]
    async fn state_code_reports_half_open_after_cooldown_probe() {
        let b = cb(1, 10);
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            1,
            "threshold=1 opens on first failure"
        );
        tokio::time::advance(Duration::from_secs(11)).await;
        // `allow()` claims the probe slot, transitioning Open -> HalfOpen.
        assert!(b.allow());
        assert_eq!(b.snapshot().state_code, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_reports_open_but_admittable_after_cooldown() {
        // The contract the scrape path depends on: a single read can show an Open breaker (state_code=1) that nonetheless admits (admit=true).
        let b = cb(1, 10);
        b.record_failure();
        let s = b.snapshot();
        assert!(!s.admit, "open within cooldown must not admit");
        assert_eq!(s.state_code, 1);

        tokio::time::advance(Duration::from_secs(11)).await;
        let s = b.snapshot();
        assert!(s.admit, "open past cooldown admits a probe");
        assert_eq!(s.state_code, 1, "...but is still reported as open");
    }

    #[tokio::test(start_paused = true)]
    async fn backpressure_resolves_half_open_probe() {
        let b = cb(1, 10);
        b.record_failure(); // Open
        assert_eq!(b.snapshot().state_code, 1);
        tokio::time::advance(Duration::from_secs(11)).await;
        assert!(
            b.allow(),
            "cooldown elapsed → claims the probe slot (HalfOpen)"
        );
        assert_eq!(b.snapshot().state_code, 2);

        b.record_backpressure();
        assert_eq!(
            b.snapshot().state_code,
            0,
            "a 503 probe answer must close the breaker, not leave it wedged half-open",
        );
        assert!(
            b.would_allow(),
            "worker must admit again after the probe resolves"
        );
    }

    #[test]
    fn backpressure_in_closed_state_preserves_failure_streak() {
        // Unlike record_success, record_backpressure must NOT reset an in-progress streak.
        let b = cb(3, 30);
        b.record_failure();
        b.record_failure();
        b.record_backpressure();
        assert_eq!(
            b.snapshot().state_code,
            0,
            "2 faults < threshold 3, still closed"
        );
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            1,
            "the 503 must not have reset the streak; the 3rd fault opens the breaker",
        );
    }

    #[test]
    fn backpressure_alone_never_opens_a_closed_breaker() {
        let b = cb(3, 30);
        for _ in 0..10 {
            b.record_backpressure();
        }
        assert_eq!(
            b.snapshot().state_code,
            0,
            "backpressure alone must never open the breaker, regardless of volume",
        );
    }
}
