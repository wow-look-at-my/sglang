// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::buckets_reorg::BucketResolver;
use crate::config::Config;
use crate::discovery::ModelId;

use crate::policies::buckets::BucketSelector;
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::policies::PolicyRegistry;
use crate::proxy::Proxy;
use crate::server::inflight::InflightHttp;
use crate::server::metrics::MetricsRegistry;
use crate::state::kv_events::{BlockSizeOracle, KvIndexMetrics};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// `/readyz` readiness as a one-way door: `NOT_READY -> READY -> DRAINING`, and never backwards.
const READINESS_NOT_READY: u8 = 0;
const READINESS_READY: u8 = 1;
const READINESS_DRAINING: u8 = 2;

/// Routing implementation used by the standard chat-completions endpoint.
#[derive(Debug, Default)]
pub enum ChatRouting {
    #[default]
    Legacy,
    Reorg(HashMap<ModelId, BucketResolver>),
}

pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    pub registry: Arc<WorkerRegistry>,
    pub policies: Arc<PolicyRegistry>,
    /// Converts static Bucket configuration into request candidate domains.
    pub bucket_selector: Arc<BucketSelector>,
    /// Select legacy policies or model-specific bucket-first routing.
    pub chat_routing: ChatRouting,
    /// Per-worker active-load bookkeeping shared by the proxy, policies, timeout janitor, and metrics.
    pub router_inflight_load: Arc<RouterInflightLoadRegistry>,
    /// Lightweight Prometheus-format metrics registry served via `/metrics`.
    pub metrics: Arc<MetricsRegistry>,
    /// Shared Engine LoadStat table; ingress captures one immutable snapshot per request.
    pub engine_reported_load: Arc<EngineReportedLoadTable>,
    pub prefix_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    pub radix_tree_prefix_provider: Option<RadixTreePrefixProvider>,
    pub block_size_oracle: Arc<BlockSizeOracle>,
    /// Read-only handles `/metrics` pulls the KV storage-tier series from on scrape.
    pub kv_metrics: Option<KvIndexMetrics>,
    /// Open HTTP exchanges, on every route.
    pub inflight_http: Arc<InflightHttp>,
    readiness: AtomicU8,
}

impl AppContext {
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_router_inflight_load(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            RouterInflightLoadRegistry::with_defaults(),
        )
    }

    /// Construct an [`AppContext`] with an explicit [`RouterInflightLoadRegistry`].
    /// Production wires the default (5-minute timeout, SystemTimeClock)
    /// via [`Self::new`]; tests that exercise the janitor pass a registry
    /// built with a `MockClock`.
    pub fn with_router_inflight_load(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        router_inflight_load: Arc<RouterInflightLoadRegistry>,
    ) -> Self {
        let metrics = MetricsRegistry::new();
        // Wire the per-worker active-load gauge so `sgl_router_active_load` mirrors the live counter.
        router_inflight_load.attach_metrics(Arc::clone(&metrics));
        // The metrics registry is built after the policy registry.
        policies.attach_metrics(Arc::clone(&metrics));
        let bucket_selector = Arc::new(BucketSelector::new(config.model.bucket_config.clone()));
        Self {
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            bucket_selector,
            chat_routing: ChatRouting::Legacy,
            router_inflight_load,
            metrics,
            prefix_index: None,
            radix_tree_prefix_provider: None,
            block_size_oracle: BlockSizeOracle::new(),
            kv_metrics: None,
            engine_reported_load: EngineReportedLoadTable::new(),
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
        }
    }

    /// Report bootstrap as finished, unless the pod has already begun
    /// draining.
    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state.
        let _ = self.readiness.compare_exchange(
            READINESS_NOT_READY,
            READINESS_READY,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    pub fn mark_not_ready(&self) {
        self.readiness.store(READINESS_DRAINING, Ordering::Relaxed);
    }

    /// Whether bootstrap finished — only ONE term of the `/readyz`
    /// predicate.
    pub fn is_ready(&self) -> bool {
        self.readiness.load(Ordering::Relaxed) == READINESS_READY
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                    ..Default::default()
                },
                observability: Default::default(),
                model: crate::config::ModelConfig {
                    id: "stub-model".into(),
                    tokenizer_path: "stub".into(),
                    disable_input_ids_forwarding: false,
                    policy: crate::config::PolicyKind::RoundRobin,
                    decode_policy: Default::default(),
                    bucket_config: None,
                    circuit_breaker: None,
                    cache_aware: None,
                    sticky: None,
                    affinity: None,
                    fused: None,
                    eligibility: None,
                    sampling_overrides: Default::default(),
                },
                discovery: crate::config::DiscoveryBackend::StaticUrls(
                    crate::config::StaticUrlsDiscoveryConfig {
                        urls: vec!["http://placeholder:0".into()],
                    },
                ),
                proxy: crate::config::ProxyConfig::default(),
                router_inflight_load: crate::config::InflightLoadConfig::default(),
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(Proxy::new(std::time::Duration::from_secs(60)).expect("stub proxy")),
            registry: Arc::new(WorkerRegistry::default()),
            policies: Arc::new(PolicyRegistry::default()),
            bucket_selector: Arc::new(BucketSelector::new(None)),
            chat_routing: ChatRouting::Legacy,
            router_inflight_load: RouterInflightLoadRegistry::with_defaults(),
            metrics: MetricsRegistry::new(),
            prefix_index: None,
            radix_tree_prefix_provider: None,
            block_size_oracle: BlockSizeOracle::new(),
            kv_metrics: None,
            engine_reported_load: EngineReportedLoadTable::new(),
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_not_ready_flips_readiness_back_off() {
        let ctx = AppContext::stub();
        // stub starts not-ready; mark_ready is the readiness on-switch.
        ctx.mark_ready();
        assert!(ctx.is_ready(), "mark_ready must report ready");
        ctx.mark_not_ready();
        assert!(
            !ctx.is_ready(),
            "mark_not_ready must flip readiness back off",
        );
    }

    /// The safety-critical direction.
    #[test]
    fn readiness_does_not_come_back_once_draining() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_not_ready();

        ctx.mark_ready();
        assert!(
            !ctx.is_ready(),
            "mark_ready must not re-ready a pod that has begun draining",
        );
    }

    /// `mark_ready` is idempotent: the compare-exchange failing because the state is already READY must not be mistaken.
    #[test]
    fn mark_ready_is_idempotent() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_ready();
        assert!(
            ctx.is_ready(),
            "a second mark_ready must keep the pod ready"
        );
    }
}
