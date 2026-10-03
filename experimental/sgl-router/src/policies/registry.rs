// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-model PD pool resolution.

use crate::discovery::{ModelId, WorkerMode};
use crate::workers::{Worker, WorkerRegistry};
use std::sync::Arc;

/// Multiplier over the median decode-pool load above which a same-host decode peer is considered "too hot" — we fall back.
const AFFINITY_LOAD_TOLERANCE: f64 = 2.0;

/// Resolution result for a single request route.
#[derive(Debug)]
pub enum PdPools {
    /// Non-PD deployment: the model is served by plain workers.
    Plain { workers: Vec<Arc<Worker>> },
    /// PD-disaggregation deployment: the model has prefill and/or decode
    /// workers.
    Pd {
        prefill: Vec<Arc<Worker>>,
        decode: Vec<Arc<Worker>>,
    },
}

/// Reason the resolver could not satisfy a request — exposed so the handler can map to the right HTTP error code.
#[derive(Debug, PartialEq, Eq)]
pub enum PdResolveError {
    /// The model has no workers registered at all, healthy or not.
    NoHealthyWorkers,
    /// PD-mode deployment whose prefill pool is empty (all breakers-open or no prefill workers ever registered).
    NoPrefillWorkersAvailable,
    /// PD-mode deployment whose decode pool is empty.
    NoDecodeWorkersAvailable,
}

/// Thin façade over [`WorkerRegistry`] that returns the per-pool candidate sets for a model.
#[derive(Debug, Clone)]
pub struct PdPoolResolver {
    workers: Arc<WorkerRegistry>,
}

impl PdPoolResolver {
    pub fn new(workers: Arc<WorkerRegistry>) -> Self {
        Self { workers }
    }

    /// Classify a model and return its pool partition over healthy
    /// workers. Workers whose circuit breaker is open are filtered out
    /// at this layer so the policy never has to re-check.
    ///
    /// Returns `Err(NoHealthyWorkers)` only when the model has zero
    /// **registered** workers (healthy or not).
    /// registered as PD but every PD worker is currently unhealthy
    /// (any failure path that flips `breaker.allow()` to false),
    /// returns `Ok(Pd { prefill: [], decode.
    /// `prefill_candidates` / `decode_candidates` can surface the more
    /// specific `NoPrefillWorkersAvailable` / `NoDecodeWorkersAvailable`
    /// transient health state.
    pub fn resolve(&self, model: &ModelId) -> Result<PdPools, PdResolveError> {
        let all = self.workers.healthy_workers_for(model);
        if all.is_empty() {
            // No healthy workers — distinguish "model never registered" (true 404-ish, operator misconfiguration).
            let registered = self.workers.workers_for(model);
            let pd_intent = registered
                .iter()
                .any(|w| matches!(w.mode(), WorkerMode::Prefill | WorkerMode::Decode));
            return if pd_intent {
                Ok(PdPools::Pd {
                    prefill: Vec::new(),
                    decode: Vec::new(),
                })
            } else {
                Err(PdResolveError::NoHealthyWorkers)
            };
        }
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut plain = Vec::new();
        for w in all {
            match w.mode() {
                WorkerMode::Prefill => prefill.push(w),
                WorkerMode::Decode => decode.push(w),
                WorkerMode::Plain => plain.push(w),
            }
        }
        // PD-mode iff any prefill OR any decode worker exists.
        if !prefill.is_empty() || !decode.is_empty() {
            Ok(PdPools::Pd { prefill, decode })
        } else {
            Ok(PdPools::Plain { workers: plain })
        }
    }

    /// Convenience for the prefill dispatch path. Returns the prefill
    /// pool for a PD model, or the full plain pool for a non-PD model.
    /// Errors when the relevant pool is empty.
    pub fn prefill_candidates(&self, model: &ModelId) -> Result<Vec<Arc<Worker>>, PdResolveError> {
        match self.resolve(model)? {
            PdPools::Plain { workers } => Ok(workers),
            PdPools::Pd { prefill, .. } => {
                if prefill.is_empty() {
                    Err(PdResolveError::NoPrefillWorkersAvailable)
                } else {
                    Ok(prefill)
                }
            }
        }
    }

    /// Convenience for the decode dispatch path. Mirror of
    /// [`Self::prefill_candidates`].
    pub fn decode_candidates(&self, model: &ModelId) -> Result<Vec<Arc<Worker>>, PdResolveError> {
        match self.resolve(model)? {
            PdPools::Plain { workers } => Ok(workers),
            PdPools::Pd { decode, .. } => {
                if decode.is_empty() {
                    Err(PdResolveError::NoDecodeWorkersAvailable)
                } else {
                    Ok(decode)
                }
            }
        }
    }

    /// Pick a decode worker for a PD-mode handoff with **host affinity** to
    /// the prefill worker. Resolves the decode pool for `model`, then applies
    /// the affinity rules in [`select_decode_with_affinity`].
    pub fn decode_with_affinity(
        &self,
        model: &ModelId,
        prefill_url: &str,
    ) -> Result<Arc<Worker>, PdResolveError> {
        let candidates = self.decode_candidates(model)?;
        select_decode_with_affinity(prefill_url, &candidates)
            .ok_or(PdResolveError::NoDecodeWorkersAvailable)
    }
}

/// Pick a decode worker from `candidates` preferring the one whose URL shares a host with `prefill_url`. **Same-host preference.** Parse the host portion of both URLs
///    (`url::Url::host_str`). If any candidate shares the host AND has
///    a closed circuit breaker AND has `router_inflight_load <=
///    AFFINITY_LOAD_TOLERANCE × median(decode_pool_load)`, return it.
/// 2. **Fallback: min-load among closed-breaker candidates.** No
///    same-host peer, or the same-host peer was filtered by rule 1's
///    health/load gates.
/// 3. **Last resort: min-load over ALL candidates.** Every candidate
///    has its breaker open; the next dispatch will likely fail too,
///    but a min-load fallback keeps the selection function total.
///    Callers should observe the breaker-open error and surface it as
///    `BreakerOpen`, not silently retry.
pub fn select_decode_with_affinity(
    prefill_url: &str,
    candidates: &[Arc<Worker>],
) -> Option<Arc<Worker>> {
    if candidates.is_empty() {
        return None;
    }
    let prefill_host = host_of(prefill_url);

    // Build the closed-breaker subset once; both the affinity branch and the
    // fallback branch read from it.
    let healthy: Vec<&Arc<Worker>> = candidates
        .iter()
        .filter(|w| w.breaker.would_allow())
        .collect();

    // Compute the median load over the closed-breaker subset.
    let load_tolerance = if healthy.is_empty() {
        0
    } else {
        let mut loads: Vec<usize> = healthy.iter().map(|w| w.router_inflight_load()).collect();
        loads.sort_unstable();
        let median = loads[loads.len() / 2];
        ((median as f64) * AFFINITY_LOAD_TOLERANCE).ceil() as usize
    };

    if let Some(host) = prefill_host.as_deref() {
        let affinity_peer = healthy.iter().find(|w| {
            host_of(&w.url).as_deref() == Some(host)
                && (load_tolerance == 0 || w.router_inflight_load() <= load_tolerance)
        });
        if let Some(w) = affinity_peer {
            return Some(Arc::clone(w));
        }
    }

    if let Some(w) = healthy.iter().min_by_key(|w| w.router_inflight_load()) {
        return Some(Arc::clone(w));
    }

    candidates
        .iter()
        .min_by_key(|w| w.router_inflight_load())
        .cloned()
}

/// Parse the host portion of a worker URL.
fn host_of(worker_url: &str) -> Option<String> {
    url::Url::parse(worker_url)
        .ok()?
        .host_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};

    fn spec(id: &str, mode: WorkerMode, model: &str) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode,
            model_ids: vec![ModelId(model.into())],
            bootstrap_port: None,
        }
    }

    fn registry(specs: &[WorkerSpec]) -> Arc<WorkerRegistry> {
        let r = Arc::new(WorkerRegistry::default());
        for s in specs {
            let _ = r.add(s.clone());
        }
        r
    }

    /// Model with only Plain workers → Plain partition.
    #[test]
    fn plain_mode_returns_all_plain_workers() {
        let r = registry(&[
            spec("w1", WorkerMode::Plain, "m"),
            spec("w2", WorkerMode::Plain, "m"),
        ]);
        let res = PdPoolResolver::new(r)
            .resolve(&ModelId("m".into()))
            .unwrap();
        match res {
            PdPools::Plain { workers } => assert_eq!(workers.len(), 2),
            PdPools::Pd { .. } => panic!("expected Plain"),
        }
    }

    /// Model with prefill + decode → Pd partition, both pools populated.
    #[test]
    fn pd_mode_returns_distinct_pools() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("d1", WorkerMode::Decode, "m"),
            spec("d2", WorkerMode::Decode, "m"),
        ]);
        let res = PdPoolResolver::new(r)
            .resolve(&ModelId("m".into()))
            .unwrap();
        match res {
            PdPools::Pd { prefill, decode } => {
                assert_eq!(prefill.len(), 1);
                assert_eq!(decode.len(), 2);
                // No cross-contamination: each worker carries the right mode.
                assert!(prefill.iter().all(|w| w.mode() == WorkerMode::Prefill));
                assert!(decode.iter().all(|w| w.mode() == WorkerMode::Decode));
            }
            PdPools::Plain { .. } => panic!("expected Pd"),
        }
    }

    /// Unknown model → NoHealthyWorkers.
    #[test]
    fn unknown_model_returns_no_healthy_workers() {
        let r = Arc::new(WorkerRegistry::default());
        let err = PdPoolResolver::new(r)
            .resolve(&ModelId("ghost".into()))
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoHealthyWorkers);
    }

    /// Gap closer #: PD mode with no prefill workers → resolve() returns a Pd partition with an empty prefill pool.
    #[test]
    fn pd_mode_with_no_prefill_errors_on_prefill_dispatch() {
        let r = registry(&[
            spec("d1", WorkerMode::Decode, "m"),
            spec("d2", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        // resolve() succeeds — we have decode workers.
        match resolver.resolve(&model).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert!(prefill.is_empty());
                assert_eq!(decode.len(), 2);
            }
            other => panic!("expected Pd, got {other:?}"),
        }
        // prefill_candidates errors.
        let err = resolver.prefill_candidates(&model).unwrap_err();
        assert_eq!(err, PdResolveError::NoPrefillWorkersAvailable);
        // decode_candidates succeeds.
        let decode = resolver.decode_candidates(&model).unwrap();
        assert_eq!(decode.len(), 2);
    }

    /// PD mode where every breaker is open (e.g. the upstream pool went hard down) must NOT collapse.
    #[test]
    fn pd_mode_all_breakers_open_keeps_per_pool_codes() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("d1", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        // Trip both breakers.
        for w in resolver.workers.workers_for(&model) {
            while w.breaker.allow() {
                w.breaker.record_failure();
            }
        }
        // resolve() still returns a PD shape (both pools empty) — the
        // PD intent is preserved across the breaker-open state.
        match resolver.resolve(&model).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert!(prefill.is_empty());
                assert!(decode.is_empty());
            }
            other => panic!("expected Pd, got {other:?}"),
        }
        // prefill dispatch → NoPrefillWorkersAvailable (not NoHealthyWorkers).
        assert_eq!(
            resolver.prefill_candidates(&model).unwrap_err(),
            PdResolveError::NoPrefillWorkersAvailable,
        );
        // decode dispatch → NoDecodeWorkersAvailable (not NoHealthyWorkers).
        assert_eq!(
            resolver.decode_candidates(&model).unwrap_err(),
            PdResolveError::NoDecodeWorkersAvailable,
        );
    }

    /// Symmetric: PD mode with no decode workers → decode dispatch errors.
    #[test]
    fn pd_mode_with_no_decode_errors_on_decode_dispatch() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("p2", WorkerMode::Prefill, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        let err = resolver.decode_candidates(&model).unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);
    }

    /// Carry-forward: separate models don't cross-contaminate.
    #[test]
    fn distinct_models_isolated_across_pd_and_plain() {
        let r = registry(&[
            spec("plain1", WorkerMode::Plain, "plainmodel"),
            spec("p1", WorkerMode::Prefill, "pdmodel"),
            spec("d1", WorkerMode::Decode, "pdmodel"),
        ]);
        let resolver = PdPoolResolver::new(r);
        match resolver.resolve(&ModelId("plainmodel".into())).unwrap() {
            PdPools::Plain { workers } => assert_eq!(workers.len(), 1),
            _ => panic!("plainmodel should resolve to Plain"),
        }
        match resolver.resolve(&ModelId("pdmodel".into())).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert_eq!(prefill.len(), 1);
                assert_eq!(decode.len(), 1);
            }
            _ => panic!("pdmodel should resolve to Pd"),
        }
    }

    /// Plain-mode prefill_candidates returns the plain pool.
    #[test]
    fn plain_mode_prefill_candidates_returns_plain_pool() {
        let r = registry(&[spec("w1", WorkerMode::Plain, "m")]);
        let resolver = PdPoolResolver::new(r);
        let v = resolver.prefill_candidates(&ModelId("m".into())).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].mode(), WorkerMode::Plain);
    }

    // === Decoder affinity (Task C) ===

    /// Build a `WorkerSpec` with an explicit URL — the affinity tests
    /// distinguish workers by host, so they care about the URL string
    /// directly, not the generated `http://{id}` form.
    fn spec_with_url(id: &str, url: &str, mode: WorkerMode, model: &str) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId(model.into())],
            bootstrap_port: None,
        }
    }

    /// Same-host affinity: a request that lands on `prefill@host_a` picks `decode@host_a` even when `decode@host_b` has lower load. Pin.
    #[test]
    fn decoder_picks_same_host_when_available() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let prefill_url = "http://host_a:30000";

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), prefill_url)
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_a:30001",
            "same-host decode peer must win over remote peer",
        );
    }

    /// Affinity peer's breaker is open → fall back to the remote healthy peer.
    #[test]
    fn decoder_falls_back_when_affinity_peer_breaker_open() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        // Trip d1's breaker by saturating record_failure() against the
        // default config (threshold = 3).
        let d1 = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .find(|w| w.url == "http://host_a:30001")
            .unwrap();
        for _ in 0..3 {
            d1.breaker.record_failure();
        }
        assert!(!d1.breaker.allow(), "d1 breaker must be open");

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_b:30001",
            "breaker-open affinity peer must fall back to the remote healthy peer",
        );
    }

    /// Affinity peer is overloaded (load > 2× median) → fall back to the remote lower-load peer.
    #[test]
    fn decoder_falls_back_when_affinity_peer_load_imbalance() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
            spec_with_url("d3", "http://host_c:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        let decode_pool = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        let d1 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_a:30001")
            .unwrap();
        let d2 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_b:30001")
            .unwrap();
        let d3 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_c:30001")
            .unwrap();
        let mut guards = Vec::new();
        for _ in 0..20 {
            guards.push(d1.load_guard());
        }
        for _ in 0..2 {
            guards.push(d2.load_guard());
            guards.push(d3.load_guard());
        }

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert!(
            chosen.url == "http://host_b:30001" || chosen.url == "http://host_c:30001",
            "overloaded affinity peer must fall back to a remote min-load peer, got: {}",
            chosen.url,
        );
        // Drop guards explicitly so the test cleanup doesn't depend on RAII order against the resolver / registry.
        drop(guards);
    }

    /// No same-host decode peer exists → fall back to min-load remote.
    #[test]
    fn decoder_falls_back_when_no_same_host_peer() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_b:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_c:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        let pool = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        let d1 = pool
            .iter()
            .find(|w| w.url == "http://host_b:30001")
            .unwrap();
        let _g = d1.load_guard();

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_c:30001",
            "no same-host peer → min-load fallback over remote candidates",
        );
    }

    /// Empty decode pool → `NoDecodeWorkersAvailable`.
    #[test]
    fn decoder_with_affinity_returns_error_when_pool_empty() {
        let r = registry(&[spec_with_url(
            "p1",
            "http://host_a:30000",
            WorkerMode::Prefill,
            "m",
        )]);
        let resolver = PdPoolResolver::new(r);
        let err = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);
    }

    /// Prefill URL is malformed (no host) → still picks a min-load decode peer.
    #[test]
    fn decoder_handles_malformed_prefill_url_via_min_load_fallback() {
        let r = registry(&[
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "not-a-url")
            .unwrap();
        // The assertion is only that the function returns Some, not None /
        // panic.
        assert!(
            chosen.url == "http://host_a:30001" || chosen.url == "http://host_b:30001",
            "unexpected decode worker chosen: {}",
            chosen.url,
        );
    }

    /// All decode peers' breakers are open → `decode_with_affinity` surfaces `NoDecodeWorkersAvailable` (the per-pool variant).
    #[test]
    fn decoder_with_affinity_errors_when_all_breakers_open() {
        let r = registry(&[
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let pool = resolver
            .workers
            .workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        // Trip every decode breaker; loop on `allow()` for threshold
        // resilience.
        for w in &pool {
            while w.breaker.allow() {
                w.breaker.record_failure();
            }
        }
        // resolver path: healthy_workers_for returns empty, but the model is
        // registered as PD (decode peers exist).
        let err = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);

        // helper path with a non-empty (but all-breaker-open) slice returns Some via the last-resort branch — selection function stays total.
        let any = select_decode_with_affinity("http://host_a:30000", &pool).unwrap();
        assert!(
            any.url == "http://host_a:30001" || any.url == "http://host_b:30001",
            "last-resort path must return some candidate, got: {}",
            any.url,
        );
    }
}
