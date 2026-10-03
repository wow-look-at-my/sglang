// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lightweight in-process Prometheus exposition.

use crate::config::PolicyKind;
use crate::proxy::sse::{StreamEnd, StreamEndReason};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Histogram bucket upper bounds (seconds) for
/// `sgl_router_request_duration_seconds`.
const REQUEST_DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Histogram bucket upper bounds (seconds) for `sgl_router_ttft_seconds`.
const TTFT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, // router-only sub-100 ms head
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0,
];

/// Histogram bucket upper bounds (blocks) for
/// `sgl_router_diverted_overlap_blocks`.
const OVERLAP_BLOCK_BUCKETS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
];

/// Recordable outcome for a request — narrowed to a handful of variants so the label cardinality stays bounded.
#[derive(Debug, Clone, Copy)]
pub enum RequestOutcome {
    Success,
    ClientError,
    Backpressure,
    /// The worker failed to serve the request: a 5xx fault, a transport failure, a timeout.
    Error,
    /// The router cancelled the request itself — today only the stale-request deadline.
    Cancelled,
}

impl RequestOutcome {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::ClientError => "client_error",
            Self::Backpressure => "backpressure",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Derive the bounded [`RequestOutcome`] label from the client-visible HTTP
/// status.
pub fn outcome_from_status(status: u16) -> RequestOutcome {
    match status {
        200..=299 => RequestOutcome::Success,
        // Responsive but at capacity. Listed before the 4xx arm so lands here rather than in `ClientError`.
        429 | 503 => RequestOutcome::Backpressure,
        400..=499 => RequestOutcome::ClientError,
        _ => RequestOutcome::Error,
    }
}

/// Routing context a handler attaches to its `Response` (via response extensions).
#[derive(Debug, Clone)]
pub struct RequestLogContext {
    /// The worker the client-visible response came from.
    pub worker_url: String,
    pub model_id: String,
    /// Whether the client asked for an SSE stream.
    pub streaming: bool,
    /// The outcome the handler recorded for this request.
    pub outcome: RequestOutcome,
    /// Router-minted engine ID, logged beside the caller's correlation ID.
    pub engine_rid: Option<String>,
}

/// Final outcome of a 2xx SSE stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamOutcome {
    /// Stream ended without errors.
    Ok,
    /// The engine sent a `data: {"error"...}` SSE event.
    StreamErrorEvent,
    /// The upstream byte stream failed.
    UpstreamError,
    /// The client disconnected before the stream finished.
    ClientDisconnect,
    /// The router's stale-request deadline aborted the stream.
    Expired,
}

pub(crate) fn classify_stream_end(end: StreamEnd) -> StreamOutcome {
    match end.reason {
        StreamEndReason::Expired => StreamOutcome::Expired,
        StreamEndReason::UpstreamError
        | StreamEndReason::IdleTimeout
        | StreamEndReason::PumpPanicked => StreamOutcome::UpstreamError,
        _ if end.saw_error_event => StreamOutcome::StreamErrorEvent,
        StreamEndReason::ClientDisconnect => StreamOutcome::ClientDisconnect,
        StreamEndReason::Completed => StreamOutcome::Ok,
    }
}

impl StreamOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::StreamErrorEvent => "stream_error_event",
            Self::UpstreamError => "upstream_error",
            Self::ClientDisconnect => "client_disconnect",
            Self::Expired => "expired",
        }
    }
}

/// Worker dispatch mode label — narrowed to the modes the policy resolver distinguishes.
#[derive(Debug, Clone, Copy)]
pub enum WorkerModeLabel {
    Prefill,
    Decode,
    Plain,
}

impl WorkerModeLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Plain => "plain",
        }
    }
}

/// Decode-affinity outcome — see `select_decode_with_affinity` for the reasons the affinity may not be honored.
#[derive(Debug, Clone, Copy)]
pub enum DecodeAffinityOutcome {
    SameHostPicked,
    FallbackBreaker,
    FallbackLoadImbalance,
}

impl DecodeAffinityOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::SameHostPicked => "same_host_picked",
            Self::FallbackBreaker => "fallback_breaker",
            Self::FallbackLoadImbalance => "fallback_load_imbalance",
        }
    }
}

/// Sticky-policy selection outcome — see `StickyPolicy::select` for the branches.
#[derive(Debug, Clone, Copy)]
pub enum StickyOutcome {
    /// Routing key found and its assigned worker is still healthy.
    Hit,
    /// Routing key seen for the first time — a worker was assigned.
    Assigned,
    /// Routing key's assigned worker left the healthy set — remapped.
    Remap,
    /// Request carried no routing key — delegated to the fallback policy.
    NoRoutingKey,
}

impl StickyOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Assigned => "assigned",
            Self::Remap => "remap",
            Self::NoRoutingKey => "no_routing_key",
        }
    }
}

/// Stale-request outcome label.
#[derive(Debug, Clone, Copy)]
pub enum StaleRequestOutcome {
    Expired,
}

impl StaleRequestOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Expired => "expired",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PolicySelectionFailureReason {
    PrefillAdmissionExhausted,
    CacheCandidatesExhausted,
    ProposalEmpty,
}

/// Final cache-aware routing decision, one per prefill selection.
#[derive(Debug, Clone, Copy)]
pub enum CacheAwareDecision {
    CacheHit,
    CacheMiss,
    CacheWorkerQueued,
    AllQueued,
}

impl CacheAwareDecision {
    fn as_str(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::CacheMiss => "cache_miss",
            Self::CacheWorkerQueued => "cache_worker_queued",
            Self::AllQueued => "all_queued",
        }
    }
}

/// Whether a dispatched chat request carried router-rendered `input_ids` to the engine, and why not otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputIdsForwarding {
    Forwarded,
    Disabled,
    Ineligible,
    IneligibleMultimodal,
    TokenizeFailed,
}

impl InputIdsForwarding {
    fn as_str(self) -> &'static str {
        match self {
            Self::Forwarded => "forwarded",
            Self::Disabled => "disabled",
            Self::Ineligible => "ineligible",
            Self::IneligibleMultimodal => "ineligible_multimodal",
            Self::TokenizeFailed => "tokenize_failed",
        }
    }
}

impl PolicySelectionFailureReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PrefillAdmissionExhausted => "prefill_admission_exhausted",
            Self::CacheCandidatesExhausted => "cache_candidates_exhausted",
            Self::ProposalEmpty => "proposal_empty",
        }
    }
}

/// Active-load kind label — separates the axes of per-worker load.
#[derive(Debug, Clone, Copy)]
pub enum RouterInflightLoadKind {
    PrefillTokens,
    DecodeBlocks,
}

impl RouterInflightLoadKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrefillTokens => "prefill_tokens",
            Self::DecodeBlocks => "decode_blocks",
        }
    }
}

/// The shared metrics registry, held on `AppContext`.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    // Edge counters (recorded at the app.rs middleware): intake at entry, responses at exit.
    requests_total: Mutex<HashMap<EdgeKey, Arc<AtomicU64>>>,
    responses_total: Mutex<HashMap<EdgeResponseKey, Arc<AtomicU64>>>,
    // Per-worker dispatch outcomes.
    worker_requests_total: Mutex<HashMap<RequestKey, Arc<AtomicU64>>>,
    // Keyed by `model_id` only: a model's pool is either all-plain or all-PD (the registry rejects mixed pools).
    request_duration: Mutex<HashMap<String, Histogram>>,
    ttft_seconds: Mutex<HashMap<String, Histogram>>,
    stream_outcome_total: Mutex<HashMap<StreamOutcomeKey, Arc<AtomicU64>>>,
    router_inflight_load: Mutex<HashMap<RouterInflightLoadKey, Arc<AtomicI64>>>,
    stale_requests_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    decode_affinity_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    sticky_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    policy_decisions_total: Mutex<HashMap<PolicyDecisionKey, Arc<AtomicU64>>>,
    policy_selection_failures_total: Mutex<HashMap<PolicyDecisionKey, Arc<AtomicU64>>>,
    cache_admission_evaluated_total: AtomicU64,
    cache_admission_rejected_total: AtomicU64,
    cache_pressure_guard_compared_total: AtomicU64,
    cache_pressure_guard_override_total: AtomicU64,
    cache_monitor_decisions_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    cache_aware_decisions_total: Mutex<HashMap<CacheAwareDecisionKey, Arc<AtomicU64>>>,
    diverted_overlap_blocks: Mutex<HashMap<String, Histogram>>,
    ingress_tokenize_errors_total: Mutex<HashMap<String, Arc<AtomicU64>>>,
    input_ids_forwarding_total: Mutex<HashMap<InputIdsForwardingKey, Arc<AtomicU64>>>,
    sampling_contract_rejections_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct RequestKey {
    worker_url: String,
    model_id: String,
    mode: &'static str,
    outcome: &'static str,
}

/// Labels for the edge `requests_total` (intake) counter.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct EdgeKey {
    route: String,
    method: String,
}

/// Labels for the edge `responses_total` counter: `EdgeKey` + final HTTP status.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct EdgeResponseKey {
    route: String,
    method: String,
    status_code: u16,
}

/// Labels for `sgl_router_stream_outcome_total`.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Clone)]
struct StreamOutcomeKey {
    worker_url: String,
    model_id: String,
    outcome: &'static str,
}

/// Per-worker state sampled from the [`crate::workers::WorkerRegistry`] at scrape time and rendered.
#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub worker_url: String,
    /// `"plain"`, `"prefill"`, or `"decode"`.
    pub mode: &'static str,
    /// Circuit breaker would currently admit a request (`would_allow`).
    pub healthy: bool,
    pub cb_state: u8,
    /// In-flight request count for this worker (`Worker::router_inflight_load`).
    pub inflight: i64,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct RouterInflightLoadKey {
    worker_url: String,
    kind: &'static str,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct PolicyDecisionKey {
    policy: String,
    reason: String,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct CacheAwareDecisionKey {
    model_id: String,
    decision: &'static str,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct InputIdsForwardingKey {
    model_id: String,
    outcome: &'static str,
}

#[derive(Debug)]
struct Histogram {
    /// Bucket upper bounds this histogram observes against.
    bounds: &'static [f64],
    /// One counter per boundary in `bounds`, plus one for `+Inf`.
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        debug_assert!(
            bounds.windows(2).all(|w| w[0] <= w[1]),
            "histogram bounds must be ascending; `observe` relies on first-match placement",
        );
        Self {
            bounds,
            buckets: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        let mut placed = false;
        for (i, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.buckets[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            // +Inf bucket
            let last = self.buckets.len() - 1;
            self.buckets[last] += 1;
        }
        self.sum += value;
        self.count += 1;
    }
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Bump the edge intake counter `requests_total{route,method}`. Called at the
    /// middleware before worker pick, so it sees pre-dispatch drops.
    pub fn record_ingress(&self, route: &str, method: &str) {
        let key = EdgeKey {
            route: route.to_owned(),
            method: method.to_owned(),
        };
        let mut guard = self.requests_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `worker_requests_total`. Recorded after dispatch — see `record_ingress`
    /// for true intake.
    pub fn record_worker_request(
        &self,
        worker_url: &str,
        model_id: &str,
        mode: WorkerModeLabel,
        outcome: RequestOutcome,
    ) {
        let key = RequestKey {
            worker_url: worker_url.to_owned(),
            model_id: model_id.to_owned(),
            mode: mode.as_str(),
            outcome: outcome.as_str(),
        };
        let mut guard = self.worker_requests_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe end-to-end request latency (seconds) for
    /// `sgl_router_request_duration_seconds`. Recorded once the upstream
    /// outcome is known.
    /// still latency the operator cares about.
    pub fn observe_request_duration(&self, model_id: &str, seconds: f64) {
        // Drop non-finite observations before touching the map: a NaN would
        // poison the series `sum` permanently.
        if !seconds.is_finite() {
            return;
        }
        let mut guard = self.request_duration.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(REQUEST_DURATION_BUCKETS));
        hist.observe(seconds);
    }

    /// Observe time-to-first-token (seconds) for `sgl_router_ttft_seconds`
    /// — the interval from request receipt to the first response chunk
    /// arriving from the upstream worker.
    pub fn observe_ttft(&self, model_id: &str, seconds: f64) {
        // See `observe_request_duration` — drop non-finite before the map.
        if !seconds.is_finite() {
            return;
        }
        let mut guard = self.ttft_seconds.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(TTFT_BUCKETS));
        hist.observe(seconds);
    }

    /// Record the final outcome of a 2xx stream.
    pub fn record_stream_outcome(&self, worker_url: &str, model_id: &str, outcome: StreamOutcome) {
        let key = StreamOutcomeKey {
            worker_url: worker_url.to_owned(),
            model_id: model_id.to_owned(),
            outcome: outcome.as_str(),
        };
        let counter = self
            .stream_outcome_total
            .lock()
            .entry(key)
            .or_default()
            .clone();
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the edge counter `responses_total{route,method,status_code}`. Called
    /// at the middleware, so it captures every response — including early-exit
    /// 400/413/503s that never reach a handler.
    pub fn record_response(&self, route: &str, method: &str, status_code: u16) {
        let key = EdgeResponseKey {
            route: route.to_owned(),
            method: method.to_owned(),
            status_code,
        };
        let mut guard = self.responses_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Set `sgl_router_active_load` for the given worker + kind. Replaces the
    /// previous value (gauge semantics).
    pub fn set_router_inflight_load(
        &self,
        worker_url: &str,
        kind: RouterInflightLoadKind,
        value: i64,
    ) {
        let key = RouterInflightLoadKey {
            worker_url: worker_url.to_owned(),
            kind: kind.as_str(),
        };
        let mut guard = self.router_inflight_load.lock();
        let gauge = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicI64::new(0)))
            .clone();
        drop(guard);
        gauge.store(value, Ordering::Relaxed);
    }

    /// Bump `sgl_router_stale_requests_total{outcome}`.
    pub fn record_stale_request(&self, outcome: StaleRequestOutcome) {
        let mut guard = self.stale_requests_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_decode_affinity_total{outcome}`.
    pub fn record_decode_affinity(&self, outcome: DecodeAffinityOutcome) {
        let mut guard = self.decode_affinity_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_sticky_total{outcome}`.
    pub fn record_sticky(&self, outcome: StickyOutcome) {
        let mut guard = self.sticky_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the final Prefill policy decision.
    pub fn record_policy_decision(&self, policy: &str, reason: &str) {
        let key = PolicyDecisionKey {
            policy: policy.to_owned(),
            reason: reason.to_owned(),
        };
        let mut guard = self.policy_decisions_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_policy_selection_failure(
        &self,
        policy: PolicyKind,
        reason: PolicySelectionFailureReason,
    ) {
        let key = PolicyDecisionKey {
            policy: policy.to_string(),
            reason: reason.as_str().to_owned(),
        };
        let mut guard = self.policy_selection_failures_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Cache-Aware candidates evaluated by hard admission.
    pub fn record_cache_admission_evaluations(&self, count: u64) {
        self.cache_admission_evaluated_total
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Cache-Aware candidates rejected by hard admission.
    pub fn record_cache_admission_rejections(&self, count: u64) {
        self.cache_admission_rejected_total
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Pressure-guard pairs compared and overridden with complete monitor data.
    pub fn record_cache_pressure_guard(&self, compared: u64, overrides: u64) {
        self.cache_pressure_guard_compared_total
            .fetch_add(compared, Ordering::Relaxed);
        self.cache_pressure_guard_override_total
            .fetch_add(overrides, Ordering::Relaxed);
    }

    /// Load source used for a Cache-Aware decision. Benchmarks reject
    /// `router_local` results to verify that monitor data affected selection.
    pub fn record_cache_monitor_decision(&self, source: &'static str) {
        let mut guard = self.cache_monitor_decisions_total.lock();
        let counter = guard
            .entry(source)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the final cache-aware routing decision for one prefill
    /// selection — exactly one call per cache-aware request, so the labels
    /// sum to the cache-aware request rate.
    pub fn record_cache_aware_decision(&self, model_id: &str, decision: CacheAwareDecision) {
        let key = CacheAwareDecisionKey {
            model_id: model_id.to_owned(),
            decision: decision.as_str(),
        };
        let mut guard = self.cache_aware_decisions_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe the matched-prefix depth (blocks) a queue-gate diversion gave
    /// up.
    /// candidate set (`cache_worker_queued`).
    /// histogram measures sacrifice rather than traffic.
    pub fn observe_diverted_overlap_blocks(&self, model_id: &str, blocks: u64) {
        let mut guard = self.diverted_overlap_blocks.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(OVERLAP_BLOCK_BUCKETS));
        hist.observe(blocks as f64);
    }

    /// Bump `sgl_router_ingress_tokenize_errors_total{model_id}`.
    ///
    /// Count formatter failures only for chats eligible for `input_ids`
    /// forwarding. Requests excluded by the guard are expected fallbacks.
    /// Pairs with the per-model WARN log in `encode_chat`.
    pub fn record_ingress_tokenize_error(&self, model_id: &str) {
        let mut guard = self.ingress_tokenize_errors_total.lock();
        let counter = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_input_ids_forwarding_total{model_id,outcome}`
    /// — exactly one call per dispatched chat request.
    pub fn record_input_ids_forwarding(&self, model_id: &str, outcome: InputIdsForwarding) {
        let key = InputIdsForwardingKey {
            model_id: model_id.to_owned(),
            outcome: outcome.as_str(),
        };
        let mut guard = self.input_ids_forwarding_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_sampling_contract_rejections_total{param}`.
    /// `--sampling-param-conflict reject`.
    pub fn record_sampling_contract_rejection(&self, param: &'static str) {
        let mut guard = self.sampling_contract_rejections_total.lock();
        let counter = guard
            .entry(param)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        self.render_with_workers(&[])
    }

    /// Render the full exposition, sampling the supplied per-worker
    /// [`WorkerSnapshot`]s into the `sgl_router_workers` /
    /// `sgl_router_worker_*` gauge families.
    pub fn render_with_workers(&self, workers: &[WorkerSnapshot]) -> String {
        let mut out = String::new();

        // requests_total — edge intake (every request, counted before dispatch)
        out.push_str(
            "# HELP sgl_router_requests_total Total requests received at the router HTTP edge, counted before worker dispatch (true intake).\n",
        );
        out.push_str("# TYPE sgl_router_requests_total counter\n");
        let guard = self.requests_total.lock();
        let mut entries: Vec<(&EdgeKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.route, &a.0.method).cmp(&(&b.0.route, &b.0.method)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_requests_total{{route=\"{}\",method=\"{}\"}} {}\n",
                escape_label(&key.route),
                escape_label(&key.method),
                value,
            ));
        }
        drop(guard);

        // worker_requests_total — per-worker dispatch outcomes
        out.push_str(
            "# HELP sgl_router_worker_requests_total Chat-completions requests dispatched to a worker, by dispatch outcome.\n",
        );
        out.push_str("# TYPE sgl_router_worker_requests_total counter\n");
        let guard = self.worker_requests_total.lock();
        // Sort for stable output — easier for tests.
        let mut entries: Vec<(&RequestKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| {
            (&a.0.worker_url, &a.0.model_id, a.0.mode, a.0.outcome).cmp(&(
                &b.0.worker_url,
                &b.0.model_id,
                b.0.mode,
                b.0.outcome,
            ))
        });
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_worker_requests_total{{worker_url=\"{}\",model_id=\"{}\",mode=\"{}\",outcome=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                escape_label(&key.model_id),
                key.mode,
                key.outcome,
                value,
            ));
        }
        drop(guard);

        // request_duration histogram
        out.push_str(
            "# HELP sgl_router_request_duration_seconds End-to-end latency of chat-completions requests dispatched to a worker, in seconds (streaming requests are measured to stream completion).\n",
        );
        out.push_str("# TYPE sgl_router_request_duration_seconds histogram\n");
        let guard = self.request_duration.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(
                &mut out,
                "sgl_router_request_duration_seconds",
                &label_body,
                hist,
            );
        }
        drop(guard);

        // ttft histogram
        out.push_str(
            "# HELP sgl_router_ttft_seconds Time to first token (first upstream response chunk) for streaming requests, in seconds.\n",
        );
        out.push_str("# TYPE sgl_router_ttft_seconds histogram\n");
        let guard = self.ttft_seconds.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(&mut out, "sgl_router_ttft_seconds", &label_body, hist);
        }
        drop(guard);

        // Final outcomes observed after a 2xx stream's headers are committed.
        out.push_str("# HELP sgl_router_stream_outcome_total Final outcome of a 2xx stream.\n");
        out.push_str("# TYPE sgl_router_stream_outcome_total counter\n");
        let guard = self.stream_outcome_total.lock();
        let mut entries: Vec<(&StreamOutcomeKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort();
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_stream_outcome_total{{worker_url=\"{}\",model_id=\"{}\",outcome=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                escape_label(&key.model_id),
                key.outcome,
                value,
            ));
        }
        drop(guard);

        out.push_str(
            "# HELP sgl_router_responses_total Responses returned at the router HTTP edge, by route, method and HTTP status code.\n",
        );
        out.push_str("# TYPE sgl_router_responses_total counter\n");
        let guard = self.responses_total.lock();
        let mut entries: Vec<(&EdgeResponseKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| {
            (&a.0.route, &a.0.method, a.0.status_code).cmp(&(
                &b.0.route,
                &b.0.method,
                b.0.status_code,
            ))
        });
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_responses_total{{route=\"{}\",method=\"{}\",status_code=\"{}\"}} {}\n",
                escape_label(&key.route),
                escape_label(&key.method),
                key.status_code,
                value,
            ));
        }
        drop(guard);

        // router_inflight_load gauge
        out.push_str(
            "# HELP sgl_router_active_load Per-worker active load (prefill_tokens or decode_blocks).\n",
        );
        out.push_str("# TYPE sgl_router_active_load gauge\n");
        let guard = self.router_inflight_load.lock();
        let mut entries: Vec<(&RouterInflightLoadKey, i64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.worker_url, a.0.kind).cmp(&(&b.0.worker_url, b.0.kind)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_active_load{{worker_url=\"{}\",kind=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                key.kind,
                value,
            ));
        }
        drop(guard);

        // Worker gauges — sampled from the live registry snapshot passed in, not stored.

        // workers (pool size by mode).
        out.push_str("# HELP sgl_router_workers Registered workers by mode.\n");
        out.push_str("# TYPE sgl_router_workers gauge\n");
        for mode in ["plain", "prefill", "decode"] {
            let count = workers.iter().filter(|w| w.mode == mode).count();
            out.push_str(&format!(
                "sgl_router_workers{{mode=\"{}\"}} {}\n",
                mode, count,
            ));
        }

        // Sort the per-worker series by URL for stable output (tests + diffs).
        let mut sorted: Vec<&WorkerSnapshot> = workers.iter().collect();
        sorted.sort_by(|a, b| a.worker_url.cmp(&b.worker_url));

        out.push_str(
            "# HELP sgl_router_worker_health Worker health: 1 = circuit breaker admits requests, 0 = rejecting (open within cooldown, or half-open with a probe in flight). May read 1 while sgl_router_worker_cb_state=1 (open but cooldown elapsed).\n",
        );
        out.push_str("# TYPE sgl_router_worker_health gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_health{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                u8::from(w.healthy),
            ));
        }

        out.push_str(
            "# HELP sgl_router_worker_cb_state Circuit breaker state per worker (0=closed, 1=open, 2=half_open).\n",
        );
        out.push_str("# TYPE sgl_router_worker_cb_state gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_cb_state{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                w.cb_state,
            ));
        }

        // worker_inflight_requests (in-flight request count per worker)
        out.push_str(
            "# HELP sgl_router_worker_inflight_requests In-flight requests currently dispatched to each worker.\n",
        );
        out.push_str("# TYPE sgl_router_worker_inflight_requests gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_inflight_requests{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                w.inflight,
            ));
        }

        // stale_requests_total
        out.push_str(
            "# HELP sgl_router_stale_requests_total Total stale-request cancellations fired by the janitor.\n",
        );
        out.push_str("# TYPE sgl_router_stale_requests_total counter\n");
        let guard = self.stale_requests_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_stale_requests_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // decode_affinity_total
        out.push_str(
            "# HELP sgl_router_decode_affinity_total Decode-affinity outcomes from select_decode_with_affinity.\n",
        );
        out.push_str("# TYPE sgl_router_decode_affinity_total counter\n");
        let guard = self.decode_affinity_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_decode_affinity_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // sticky_total
        out.push_str(
            "# HELP sgl_router_sticky_total Sticky-session selection outcomes from StickyPolicy.\n",
        );
        out.push_str("# TYPE sgl_router_sticky_total counter\n");
        let guard = self.sticky_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_sticky_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // policy_decisions_total
        out.push_str(
            "# HELP sgl_router_policy_decisions_total Final Prefill policy decisions by policy and bounded reason.\n",
        );
        out.push_str("# TYPE sgl_router_policy_decisions_total counter\n");
        let guard = self.policy_decisions_total.lock();
        let mut entries: Vec<(&PolicyDecisionKey, u64)> = guard
            .iter()
            .map(|(key, value)| (key, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.policy, &a.0.reason).cmp(&(&b.0.policy, &b.0.reason)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_policy_decisions_total{{policy=\"{}\",reason=\"{}\"}} {}\n",
                escape_label(&key.policy),
                escape_label(&key.reason),
                value,
            ));
        }
        drop(guard);

        // policy_selection_failures_total
        out.push_str(
            "# HELP sgl_router_policy_selection_failures_total Failed Prefill policy selections by policy and bounded reason.\n",
        );
        out.push_str("# TYPE sgl_router_policy_selection_failures_total counter\n");
        let guard = self.policy_selection_failures_total.lock();
        let mut entries: Vec<(&PolicyDecisionKey, u64)> = guard
            .iter()
            .map(|(key, value)| (key, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.policy, &a.0.reason).cmp(&(&b.0.policy, &b.0.reason)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_policy_selection_failures_total{{policy=\"{}\",reason=\"{}\"}} {}\n",
                escape_label(&key.policy),
                escape_label(&key.reason),
                value,
            ));
        }
        drop(guard);

        out.push_str(
            "# HELP sgl_router_cache_admission_evaluated_total Cache-Aware candidates evaluated by hard admission.\n",
        );
        out.push_str("# TYPE sgl_router_cache_admission_evaluated_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_admission_evaluated_total {}\n",
            self.cache_admission_evaluated_total.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_admission_rejected_total Cache-Aware candidates rejected by hard admission.\n",
        );
        out.push_str("# TYPE sgl_router_cache_admission_rejected_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_admission_rejected_total {}\n",
            self.cache_admission_rejected_total.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_pressure_guard_compared_total Complete fresh Cache-Aware candidate pairs evaluated by the pressure guard.\n",
        );
        out.push_str("# TYPE sgl_router_cache_pressure_guard_compared_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_pressure_guard_compared_total {}\n",
            self.cache_pressure_guard_compared_total
                .load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_pressure_guard_override_total Pressure-guard comparisons whose outcome differs from cache/work ordering without the guard.\n",
        );
        out.push_str("# TYPE sgl_router_cache_pressure_guard_override_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_pressure_guard_override_total {}\n",
            self.cache_pressure_guard_override_total
                .load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_monitor_decisions_total Cache-Aware candidate resolutions by actual load source.\n",
        );
        out.push_str("# TYPE sgl_router_cache_monitor_decisions_total counter\n");
        let guard = self.cache_monitor_decisions_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(source, value)| (source, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|entry| *entry.0);
        for (source, value) in entries {
            out.push_str(&format!(
                "sgl_router_cache_monitor_decisions_total{{source=\"{}\"}} {}\n",
                source, value,
            ));
        }
        drop(guard);

        // cache_aware_decisions_total
        out.push_str(
            "# HELP sgl_router_cache_aware_decisions_total Final Cache-Aware routing decisions, one per selection that resolved a worker: cache_hit = prefix owner won; cache_miss = no usable owner (a tree miss books here even under saturation, because the gate never fired); cache_worker_queued = queue gate diverted the request off its prefix; all_queued = queue gate fired but every worker is queueing (fleet-saturation signal, not a cache hit).\n",
        );
        out.push_str("# TYPE sgl_router_cache_aware_decisions_total counter\n");
        let guard = self.cache_aware_decisions_total.lock();
        let mut entries: Vec<(&CacheAwareDecisionKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.decision).cmp(&(&b.0.model_id, b.0.decision)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_cache_aware_decisions_total{{model_id=\"{}\",decision=\"{}\"}} {}\n",
                escape_label(&key.model_id),
                key.decision,
                value,
            ));
        }
        drop(guard);

        // diverted_overlap_blocks histogram
        out.push_str(
            "# HELP sgl_router_diverted_overlap_blocks Matched-prefix depth (blocks) given up by queue-gate diversions (decision=cache_worker_queued). Read against the overlap of all selections: a curve skewing high means the gate is trading large cached prefixes for short waits.\n",
        );
        out.push_str("# TYPE sgl_router_diverted_overlap_blocks histogram\n");
        let guard = self.diverted_overlap_blocks.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(
                &mut out,
                "sgl_router_diverted_overlap_blocks",
                &label_body,
                hist,
            );
        }
        drop(guard);

        // ingress_tokenize_errors_total
        out.push_str(
            "# HELP sgl_router_ingress_tokenize_errors_total Plain text chat requests on a chat-formatter model whose ingress rendering or tokenization failed, silently falling back to engine-side tokenization (the input_ids offload was defeated).\n",
        );
        out.push_str("# TYPE sgl_router_ingress_tokenize_errors_total counter\n");
        let guard = self.ingress_tokenize_errors_total.lock();
        let mut entries: Vec<(&String, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (model_id, value) in entries {
            out.push_str(&format!(
                "sgl_router_ingress_tokenize_errors_total{{model_id=\"{}\"}} {}\n",
                escape_label(model_id),
                value,
            ));
        }
        drop(guard);

        // input_ids_forwarding_total
        out.push_str(
            "# HELP sgl_router_input_ids_forwarding_total Dispatched chat requests by whether router-rendered input_ids were forwarded to the engine (outcome=forwarded) or why not (disabled, ineligible_multimodal, ineligible, tokenize_failed).\n",
        );
        out.push_str("# TYPE sgl_router_input_ids_forwarding_total counter\n");
        let guard = self.input_ids_forwarding_total.lock();
        let mut entries: Vec<(&InputIdsForwardingKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.outcome).cmp(&(&b.0.model_id, b.0.outcome)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_input_ids_forwarding_total{{model_id=\"{}\",outcome=\"{}\"}} {}\n",
                escape_label(&key.model_id),
                key.outcome,
                value,
            ));
        }
        drop(guard);

        // sampling_contract_rejections_total
        out.push_str(
            "# HELP sgl_router_sampling_contract_rejections_total Requests refused by the fleet-wide sampling contract (--override-sampling-params under --sampling-param-conflict reject), by parameter.\n",
        );
        out.push_str("# TYPE sgl_router_sampling_contract_rejections_total counter\n");
        let guard = self.sampling_contract_rejections_total.lock();
        let mut entries: Vec<(&str, u64)> = guard
            .iter()
            .map(|(k, v)| (*k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|entry| entry.0);
        for (param, value) in entries {
            out.push_str(&format!(
                "sgl_router_sampling_contract_rejections_total{{param=\"{}\"}} {}\n",
                escape_label(param),
                value,
            ));
        }
        drop(guard);

        out
    }
}

/// Render one labelled histogram family (`<name>_bucket` / `_sum` /
/// `_count`) into `out`. emitted verbatim — callers escape their own
/// label values.
/// rendered cumulatively per the Prometheus histogram contract.
/// final `+Inf` bucket.
fn render_histogram(out: &mut String, name: &str, label_body: &str, hist: &Histogram) {
    let mut cumulative: u64 = 0;
    for (i, &bound) in hist.bounds.iter().enumerate() {
        cumulative += hist.buckets[i];
        out.push_str(&format!(
            "{name}_bucket{{{label_body},le=\"{bound}\"}} {cumulative}\n"
        ));
    }
    cumulative += hist.buckets[hist.bounds.len()];
    out.push_str(&format!(
        "{name}_bucket{{{label_body},le=\"+Inf\"}} {cumulative}\n"
    ));
    out.push_str(&format!("{name}_sum{{{label_body}}} {}\n", hist.sum));
    out.push_str(&format!("{name}_count{{{label_body}}} {}\n", hist.count));
}

/// Prometheus label-value escape rule per
/// https://prometheus.io/docs/instrumenting/exposition_formats/. We
/// only escape `\`, `"`.
/// reference parser rejects unescaped.
pub(crate) fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str(r"\\"),
            '"' => out.push_str(r#"\""#),
            '\n' => out.push_str(r"\n"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_metric_line(output: &str, expected: &str) {
        assert!(
            output.lines().any(|line| line == expected),
            "missing metric line `{expected}`; got:\n{output}"
        );
    }

    #[test]
    fn empty_registry_renders_only_help_lines() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        // Should at least carry HELP / TYPE for every metric family.
        assert!(out.contains("# TYPE sgl_router_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_request_duration_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_ttft_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_responses_total counter"));
        assert!(out.contains("# TYPE sgl_router_active_load gauge"));
        assert!(out.contains("# TYPE sgl_router_workers gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_health gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_cb_state gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_inflight_requests gauge"));
        assert!(out.contains("# TYPE sgl_router_stale_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_decode_affinity_total counter"));
        assert!(out.contains("# TYPE sgl_router_sticky_total counter"));
        assert!(out.contains("# TYPE sgl_router_policy_decisions_total counter"));
        assert!(out.contains("# TYPE sgl_router_ingress_tokenize_errors_total counter"));
        assert!(out.contains(r#"sgl_router_workers{mode="plain"} 0"#));
        assert!(out.contains(r#"sgl_router_workers{mode="prefill"} 0"#));
        assert!(out.contains(r#"sgl_router_workers{mode="decode"} 0"#));
    }

    #[test]
    fn observe_request_duration_writes_buckets_sum_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_request_duration("tiny", 0.025);
        reg.observe_request_duration("tiny", 0.12);
        reg.observe_request_duration("tiny", 0.6);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_count{model_id="tiny"} 3"#),
            "expected count=3; got:\n{out}",
        );
        assert!(
            out.contains(
                r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="0.025"} 1"#
            ),
            "expected le=0.025 bucket = 1; got:\n{out}",
        );
        // le=1 is cumulative over all observations.
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="1"} 3"#),
            "expected le=1 bucket = 3; got:\n{out}",
        );
        assert!(out.contains(
            r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="+Inf"} 3"#
        ));
    }

    #[test]
    fn request_duration_separates_by_model() {
        let reg = MetricsRegistry::new();
        reg.observe_request_duration("a", 0.01);
        reg.observe_request_duration("b", 0.01);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="a"} 1"#));
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="b"} 1"#));
    }

    #[test]
    fn request_duration_overflow_lands_in_plus_inf_bucket_only() {
        let reg = MetricsRegistry::new();
        // 45s is beyond the top finite bound (30s).
        reg.observe_request_duration("m", 45.0);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="m",le="30"} 0"#),
            "45s must not fall in the le=30 bucket; got:\n{out}",
        );
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="m",le="+Inf"} 1"#)
        );
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="m"} 1"#));
        assert!(out.contains(r#"sgl_router_request_duration_seconds_sum{model_id="m"} 45"#));
    }

    #[test]
    fn observe_request_duration_ignores_non_finite() {
        let reg = MetricsRegistry::new();
        reg.observe_request_duration("m", f64::NAN);
        reg.observe_request_duration("m", f64::INFINITY);
        let out = reg.render();
        // Nothing recorded — no count series for the model (sum stays uncorrupted).
        assert!(
            !out.contains(r#"sgl_router_request_duration_seconds_count{model_id="m"}"#),
            "non-finite observations must be dropped, not bucketed; got:\n{out}",
        );
    }

    #[test]
    fn observe_ttft_writes_buckets_sum_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_ttft("tiny", 0.04);
        reg.observe_ttft("tiny", 0.2);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_ttft_seconds_count{model_id="tiny"} 2"#),
            "expected ttft count=2; got:\n{out}",
        );
        assert!(
            out.contains(r#"sgl_router_ttft_seconds_bucket{model_id="tiny",le="0.05"} 1"#),
            "expected le=0.05 bucket = 1; got:\n{out}",
        );
        assert!(out.contains(r#"sgl_router_ttft_seconds_bucket{model_id="tiny",le="0.2"} 2"#));
    }

    #[test]
    fn ttft_buckets_align_with_engine_grid() {
        let reg = MetricsRegistry::new();
        reg.observe_ttft("m", 0.5);
        let out = reg.render();
        for le in [
            "0.1", "0.2", "0.4", "0.6", "0.8", "1", "2", "4", "6", "8", "10", "20", "40", "60",
            "80", "100", "200", "400",
        ] {
            assert!(
                out.contains(&format!(
                    r#"sgl_router_ttft_seconds_bucket{{model_id="m",le="{le}"}}"#
                )),
                "missing engine-aligned TTFT bucket le={le}; got:\n{out}",
            );
        }
    }

    #[test]
    fn stream_outcome_precedence() {
        use StreamOutcome::*;

        for (reason, expected) in [
            (StreamEndReason::Completed, Ok),
            (StreamEndReason::ClientDisconnect, ClientDisconnect),
            (StreamEndReason::UpstreamError, UpstreamError),
            (StreamEndReason::IdleTimeout, UpstreamError),
            (StreamEndReason::PumpPanicked, UpstreamError),
            (StreamEndReason::Expired, Expired),
        ] {
            for saw_error_event in [false, true] {
                let end = StreamEnd {
                    reason,
                    saw_error_event,
                };
                let expected = if saw_error_event && matches!(expected, Ok | ClientDisconnect) {
                    StreamErrorEvent
                } else {
                    expected
                };
                assert_eq!(classify_stream_end(end), expected, "{end:?}");
            }
        }
    }

    #[test]
    fn record_stream_outcome_emits_labelled_counter_lines() {
        let reg = MetricsRegistry::new();
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::Ok);
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::Ok);
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::StreamErrorEvent);
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::UpstreamError);
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::ClientDisconnect);
        reg.record_stream_outcome("http://w:30000", "tiny", StreamOutcome::Expired);
        let out = reg.render();
        for expected in [
            r#"sgl_router_stream_outcome_total{worker_url="http://w:30000",model_id="tiny",outcome="ok"} 2"#,
            r#"sgl_router_stream_outcome_total{worker_url="http://w:30000",model_id="tiny",outcome="stream_error_event"} 1"#,
            r#"sgl_router_stream_outcome_total{worker_url="http://w:30000",model_id="tiny",outcome="upstream_error"} 1"#,
            r#"sgl_router_stream_outcome_total{worker_url="http://w:30000",model_id="tiny",outcome="client_disconnect"} 1"#,
            r#"sgl_router_stream_outcome_total{worker_url="http://w:30000",model_id="tiny",outcome="expired"} 1"#,
        ] {
            assert_metric_line(&out, expected);
        }
    }

    #[test]
    fn stream_outcome_absent_until_recorded() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        assert!(out.contains("# TYPE sgl_router_stream_outcome_total counter"));
        assert!(
            !out.contains("sgl_router_stream_outcome_total{"),
            "no series until an outcome is recorded; got:\n{out}",
        );
    }

    #[test]
    fn record_response_counts_by_route_method_status_code() {
        let reg = MetricsRegistry::new();
        reg.record_response("/v1/chat/completions", "POST", 200);
        reg.record_response("/v1/chat/completions", "POST", 200);
        reg.record_response("/v1/chat/completions", "POST", 502);
        reg.record_response("/v1/chat/completions", "POST", 504);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 2"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="502"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="504"} 1"#
        ));
    }

    #[test]
    fn record_ingress_counts_by_route_method() {
        let reg = MetricsRegistry::new();
        reg.record_ingress("/v1/chat/completions", "POST");
        reg.record_ingress("/v1/chat/completions", "POST");
        reg.record_ingress("/v1/models", "GET");
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_requests_total{route="/v1/chat/completions",method="POST"} 2"#
        ));
        assert!(out.contains(r#"sgl_router_requests_total{route="/v1/models",method="GET"} 1"#));
    }

    #[test]
    fn render_with_workers_emits_per_worker_gauges_and_pool_size() {
        let reg = MetricsRegistry::new();
        let workers = vec![
            WorkerSnapshot {
                worker_url: "http://p0:30000".into(),
                mode: "prefill",
                healthy: true,
                cb_state: 0,
                inflight: 5,
            },
            WorkerSnapshot {
                worker_url: "http://d0:30000".into(),
                mode: "decode",
                healthy: false,
                cb_state: 1,
                inflight: 0,
            },
        ];
        let out = reg.render_with_workers(&workers);
        // Pool size by mode.
        assert!(out.contains(r#"sgl_router_workers{mode="prefill"} 1"#));
        assert!(out.contains(r#"sgl_router_workers{mode="decode"} 1"#));
        assert!(out.contains(r#"sgl_router_workers{mode="plain"} 0"#));
        assert!(out.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(out.contains(r#"sgl_router_worker_health{worker_url="http://d0:30000"} 0"#));
        // Circuit breaker state codes.
        assert!(out.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(out.contains(r#"sgl_router_worker_cb_state{worker_url="http://d0:30000"} 1"#));
        // In-flight request counts.
        assert!(
            out.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 5"#)
        );
        assert!(
            out.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://d0:30000"} 0"#)
        );
    }

    #[test]
    fn render_without_workers_emits_no_per_worker_series() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        // Headers present, but no per-worker series lines.
        assert!(out.contains("# TYPE sgl_router_worker_health gauge"));
        assert!(!out.contains("sgl_router_worker_health{"));
        assert!(!out.contains("sgl_router_worker_cb_state{"));
        assert!(!out.contains("sgl_router_worker_inflight_requests{"));
    }

    #[test]
    fn record_worker_request_emits_labelled_counter_line() {
        let reg = MetricsRegistry::new();
        reg.record_worker_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        reg.record_worker_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_worker_requests_total{worker_url="http://worker-a:30000",model_id="tiny",mode="prefill",outcome="success"} 2"#),
            "render did not include the expected counter line; got:\n{out}",
        );
    }

    #[test]
    fn set_active_load_gauge_overwrites() {
        let reg = MetricsRegistry::new();
        reg.set_router_inflight_load("http://w:30000", RouterInflightLoadKind::PrefillTokens, 100);
        reg.set_router_inflight_load("http://w:30000", RouterInflightLoadKind::PrefillTokens, 250);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 250"#,
        ));
        // First write must NOT appear.
        assert!(!out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 100"#,
        ));
    }

    #[test]
    fn stale_request_counter_increments() {
        let reg = MetricsRegistry::new();
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_stale_requests_total{outcome="expired"} 3"#));
    }

    #[test]
    fn decode_affinity_counter_emits_three_outcomes() {
        let reg = MetricsRegistry::new();
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackBreaker);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackLoadImbalance);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="same_host_picked"} 2"#));
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="fallback_breaker"} 1"#));
        assert!(out
            .contains(r#"sgl_router_decode_affinity_total{outcome="fallback_load_imbalance"} 1"#,));
    }

    #[test]
    fn sticky_counter_emits_all_outcomes() {
        let reg = MetricsRegistry::new();
        reg.record_sticky(StickyOutcome::Hit);
        reg.record_sticky(StickyOutcome::Hit);
        reg.record_sticky(StickyOutcome::Assigned);
        reg.record_sticky(StickyOutcome::Remap);
        reg.record_sticky(StickyOutcome::NoRoutingKey);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="hit"} 2"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="assigned"} 1"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="remap"} 1"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="no_routing_key"} 1"#));
    }

    #[test]
    fn policy_decisions_are_keyed_by_policy_and_reason() {
        let reg = MetricsRegistry::new();
        reg.record_policy_decision("session_aware", "session_primary");
        reg.record_policy_decision("session_aware", "session_primary");
        reg.record_policy_decision("cache_aware", "cache_candidate");

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_policy_decisions_total{policy="cache_aware",reason="cache_candidate"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="session_primary"} 2"#
        ));
    }

    #[test]
    fn policy_selection_failures_are_keyed_by_policy_and_reason() {
        let reg = MetricsRegistry::new();
        reg.record_policy_selection_failure(
            PolicyKind::SessionAware,
            PolicySelectionFailureReason::PrefillAdmissionExhausted,
        );
        reg.record_policy_selection_failure(
            PolicyKind::CacheAware,
            PolicySelectionFailureReason::CacheCandidatesExhausted,
        );
        reg.record_policy_selection_failure(
            PolicyKind::RoundRobin,
            PolicySelectionFailureReason::ProposalEmpty,
        );

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="session_aware",reason="prefill_admission_exhausted"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="cache_aware",reason="cache_candidates_exhausted"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="round_robin",reason="proposal_empty"} 1"#
        ));
    }

    #[test]
    fn cache_monitor_and_guard_counters_are_exposed() {
        let reg = MetricsRegistry::new();
        reg.record_cache_monitor_decision("estimated_prefill_queue_ms");
        reg.record_cache_admission_evaluations(3);
        reg.record_cache_admission_rejections(2);
        reg.record_cache_pressure_guard(3, 1);

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_cache_monitor_decisions_total{source="estimated_prefill_queue_ms"} 1"#
        ));
        assert!(out.contains("sgl_router_cache_admission_evaluated_total 3"));
        assert!(out.contains("sgl_router_cache_admission_rejected_total 2"));
        assert!(out.contains("sgl_router_cache_pressure_guard_compared_total 3"));
        assert!(out.contains("sgl_router_cache_pressure_guard_override_total 1"));
    }

    #[test]
    fn cache_aware_decisions_and_diverted_overlap_render() {
        let reg = MetricsRegistry::new();
        reg.record_cache_aware_decision("tiny", CacheAwareDecision::CacheHit);
        reg.record_cache_aware_decision("tiny", CacheAwareDecision::CacheWorkerQueued);
        reg.record_cache_aware_decision("tiny", CacheAwareDecision::CacheWorkerQueued);
        reg.record_cache_aware_decision("tiny", CacheAwareDecision::AllQueued);
        reg.observe_diverted_overlap_blocks("tiny", 40);

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_cache_aware_decisions_total{model_id="tiny",decision="cache_hit"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_cache_aware_decisions_total{model_id="tiny",decision="cache_worker_queued"} 2"#
        ));
        assert!(out.contains(
            r#"sgl_router_cache_aware_decisions_total{model_id="tiny",decision="all_queued"} 1"#
        ));
        // The saturation label must not be absorbed by a `cache_hit.*` hit-rate query.
        assert!(!out.contains(r#"decision="cache_hit_all_queued""#));
        assert!(
            out.contains(r#"sgl_router_diverted_overlap_blocks_count{model_id="tiny"} 1"#),
            "expected one diverted observation; got:\n{out}"
        );
        assert!(
            out.contains(r#"sgl_router_diverted_overlap_blocks_bucket{model_id="tiny",le="64"} 1"#)
        );
        assert!(
            out.contains(r#"sgl_router_diverted_overlap_blocks_bucket{model_id="tiny",le="32"} 0"#)
        );
    }

    #[test]
    fn ingress_tokenize_error_counter_increments_per_model() {
        let reg = MetricsRegistry::new();
        reg.record_ingress_tokenize_error("tiny");
        reg.record_ingress_tokenize_error("tiny");
        reg.record_ingress_tokenize_error("other");
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_ingress_tokenize_errors_total{model_id="tiny"} 2"#),
            "expected tiny=2; got:\n{out}",
        );
        assert!(
            out.contains(r#"sgl_router_ingress_tokenize_errors_total{model_id="other"} 1"#),
            "expected other=1; got:\n{out}",
        );
    }

    #[test]
    fn input_ids_forwarding_counter_labels_outcome() {
        let reg = MetricsRegistry::new();
        reg.record_input_ids_forwarding("tiny", InputIdsForwarding::Forwarded);
        reg.record_input_ids_forwarding("tiny", InputIdsForwarding::Forwarded);
        reg.record_input_ids_forwarding("tiny", InputIdsForwarding::Ineligible);
        let out = reg.render();
        assert!(out.contains("# TYPE sgl_router_input_ids_forwarding_total counter"));
        for series in [
            r#"sgl_router_input_ids_forwarding_total{model_id="tiny",outcome="forwarded"} 2"#,
            r#"sgl_router_input_ids_forwarding_total{model_id="tiny",outcome="ineligible"} 1"#,
        ] {
            assert!(out.contains(series), "missing {series}; got:\n{out}");
        }
    }

    #[test]
    fn ingress_tokenize_error_absent_until_recorded() {
        // Healthy operation never calls the recorder.
        let reg = MetricsRegistry::new();
        let out = reg.render();
        assert!(out.contains("# TYPE sgl_router_ingress_tokenize_errors_total counter"));
        assert!(
            !out.contains("sgl_router_ingress_tokenize_errors_total{"),
            "no per-model series until an error is recorded; got:\n{out}",
        );
    }

    #[test]
    fn label_values_escape_quotes_and_backslashes() {
        let reg = MetricsRegistry::new();
        reg.record_worker_request(
            r#"http://"weird":30000"#,
            r"back\slash",
            WorkerModeLabel::Plain,
            RequestOutcome::Error,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"worker_url="http://\"weird\":30000""#),
            "render did not escape double-quote; got:\n{out}",
        );
        assert!(
            out.contains(r#"model_id="back\\slash""#),
            "render did not escape backslash; got:\n{out}",
        );
    }
    /// The contract's rollout gauge: absent until a request is refused, then keyed by the parameter that refused it.
    #[test]
    fn sampling_contract_rejections_are_keyed_by_param() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        assert!(out.contains("# TYPE sgl_router_sampling_contract_rejections_total counter"));
        assert!(
            !out.contains("sgl_router_sampling_contract_rejections_total{"),
            "must emit no series before the first rejection"
        );

        reg.record_sampling_contract_rejection("temperature");
        reg.record_sampling_contract_rejection("temperature");
        reg.record_sampling_contract_rejection("top_p");
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_sampling_contract_rejections_total{param="temperature"} 2"#),
            "got:\n{out}"
        );
        assert!(
            out.contains(r#"sgl_router_sampling_contract_rejections_total{param="top_p"} 1"#),
            "got:\n{out}"
        );
    }

    /// The status → outcome mapping is the single definition shared by the access log and `worker_requests_total`.
    #[test]
    fn outcome_from_status_maps_every_class() {
        let cases = [
            (200, "success"),
            (204, "success"),
            (299, "success"),
            (429, "backpressure"),
            (503, "backpressure"),
            (400, "client_error"),
            (404, "client_error"),
            (499, "client_error"),
            (500, "error"),
            (502, "error"),
            (504, "error"),
            (199, "error"),
            (300, "error"),
        ];
        for (status, want) in cases {
            assert_eq!(
                outcome_from_status(status).as_str(),
                want,
                "status {status} must map to `{want}`",
            );
        }
    }
}
