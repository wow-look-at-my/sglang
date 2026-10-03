//! Per-worker, per-DP-rank ZMQ subscriber for SGLang's `ZmqEventPublisher`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

use super::discovery::EventConfig;
use super::tree::KvWorkerId;
use super::wire::{decode_event_batch, KvEventBatch};
use crate::state::load_monitor::engine_reported_load::{decode_load_stat, LoadStat};

/// Maximum number of consecutive `recv()` errors before the subscriber gives up and exits its task.
const RECV_ERROR_CEILING: u32 = 64;

/// Bounded retry configuration for the initial connect + subscribe handshake.
const CONNECT_MAX_ATTEMPTS: u32 = 5;
const CONNECT_BACKOFF_BASE: Duration = Duration::from_millis(50);
const CONNECT_BACKOFF_CAP: Duration = Duration::from_secs(2);

/// Sentinel sequence number meaning "publisher is shutting down".
const END_SEQ_SENTINEL: i64 = -1;

/// Which topic a subscriber task listens on, and therefore what kind of [`WorkerEvent`] it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubKind {
    /// Cache-delta topic (`BlockStored` / `BlockRemoved` / `AllBlocksCleared`).
    Kv,
    /// Load-snapshot topic (`LoadStat`).
    Load,
}

/// Message forwarded from a per-worker subscriber task to the pump.
#[derive(Debug)]
pub enum WorkerEvent {
    /// A normal decoded event batch.
    Batch {
        /// Identity of the SGLang worker (DP rank) that produced this batch.
        worker: KvWorkerId,
        /// 8-byte big-endian sequence number from the publisher's monotonic counter.
        seq: i64,
        /// Decoded batch payload.
        batch: KvEventBatch,
    },
    /// A runtime load snapshot from the load topic. Carries no sequence
    /// number: load is a gauge, applied last-value-wins with no dedup.
    Load {
        /// Identity of the SGLang worker (DP rank) that produced this load.
        worker: KvWorkerId,
        /// Latest load snapshot for this `(worker, dp_rank)`.
        load: LoadStat,
    },
    PublisherReset { worker: KvWorkerId },
}

impl WorkerEvent {
    /// The worker that produced this event, regardless of variant.
    pub fn worker(&self) -> &KvWorkerId {
        match self {
            Self::Batch { worker, .. } => worker,
            Self::Load { worker, .. } => worker,
            Self::PublisherReset { worker } => worker,
        }
    }
}

/// Internal handle for one running per-(worker, dp_rank) subscriber task.
struct SubscriberHandle {
    cancel: CancellationToken,
    join: JoinHandle<()>,
}

/// Shared inner state for [`KvEventSubscriberRegistry`].
struct Inner {
    tx: mpsc::Sender<WorkerEvent>,
    /// Keyed by `(worker_url, dp_rank)`.
    handles: Mutex<HashMap<KvWorkerId, SubscriberHandle>>,
}

/// Owns one ZMQ SUB connection per `(worker_url, dp_rank)`.
pub struct KvEventSubscriberRegistry {
    inner: Arc<Inner>,
    kind: SubKind,
}

impl KvEventSubscriberRegistry {
    /// Build an empty KV-cache registry. `tx` is where decoded events flow
    /// out; the channel buffer capacity is the caller's choice.
    pub fn new(tx: mpsc::Sender<WorkerEvent>) -> Self {
        Self::with_kind(tx, SubKind::Kv)
    }

    /// Build an empty registry of the given kind.
    pub fn with_kind(tx: mpsc::Sender<WorkerEvent>, kind: SubKind) -> Self {
        Self {
            inner: Arc::new(Inner {
                tx,
                handles: Mutex::new(HashMap::new()),
            }),
            kind,
        }
    }

    /// Open one SUB connection per `dp_rank` in `0..cfg.dp_size`, connecting
    /// to `tcp://{cfg.host}:{cfg.port_base + dp_rank}`. Spawns background
    /// tasks. Idempotent: a second `add_worker` for the same `(worker_url,
    /// dp_rank)` pair is a no-op. `worker_url` is the HTTP URL the gateway
    /// uses for routing (e.g., `"http://10.0.0.1:30000"`).
    /// worker's `/server_info` introspection (or filled it from a global
    /// fallback). # Errors If `cfg.port_base + dp_rank` overflows `u16`.
    /// rank is skipped with a `warn!` log and the remaining ranks proceed.
    pub async fn add_worker(&self, worker_url: &str, cfg: &EventConfig) {
        // (port_base, topic) depend on this registry's kind. KV uses the cache
        // socket + configured topic; Load uses its own advertised socket +
        // topic. Refuse an incomplete load descriptor rather than subscribe-all:
        // the load wire is a distinct contract and a future mixed-use socket
        // must not feed unrelated payloads into the load decoder.
        let (port_base, topic) = match self.kind {
            SubKind::Kv => (cfg.port_base, cfg.topic.clone()),
            SubKind::Load => match (&cfg.load_port_base, &cfg.load_topic) {
                (Some(port), Some(topic)) => (*port, topic.clone()),
                _ => {
                    debug!(
                        worker_url = %worker_url,
                        "kv-events: worker lacks a complete load descriptor; skipping load subscribers"
                    );
                    return;
                }
            },
        };
        let mut handles = self.inner.handles.lock().await;
        for dp_rank in 0..cfg.dp_size {
            let id = KvWorkerId {
                url: worker_url.to_string(),
                dp_rank,
            };
            if handles.contains_key(&id) {
                debug!(
                    worker_url = %worker_url,
                    dp_rank,
                    "subscriber already registered; skipping"
                );
                continue;
            }
            let port = match u16::try_from(port_base as u32 + dp_rank) {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        worker_url = %worker_url,
                        dp_rank,
                        port_base,
                        kind = ?self.kind,
                        "ZMQ event port overflows u16; skipping this rank"
                    );
                    continue;
                }
            };
            let endpoint = format!("tcp://{}:{}", cfg.host, port);
            let cancel = CancellationToken::new();
            let join = spawn_subscriber_task(
                id.clone(),
                endpoint,
                topic.clone(),
                self.kind,
                self.inner.tx.clone(),
                cancel.clone(),
            );
            handles.insert(id, SubscriberHandle { cancel, join });
        }
    }

    /// Cancel all subscribers for `worker_url` and await their shutdown.
    pub async fn remove_worker(&self, worker_url: &str) {
        let drained: Vec<SubscriberHandle> = {
            let mut handles = self.inner.handles.lock().await;
            // Pull out every entry whose URL matches; leave the others.
            let to_drop: Vec<KvWorkerId> = handles
                .keys()
                .filter(|k| k.url == worker_url)
                .cloned()
                .collect();
            to_drop
                .into_iter()
                .filter_map(|k| handles.remove(&k))
                .collect()
        };
        for h in drained {
            h.cancel.cancel();
            // A panicked task surfaces here; we log and continue so one
            // poisoned subscriber cannot stall the registry.
            if let Err(e) = h.join.await {
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    "subscriber task did not join cleanly"
                );
            }
        }
    }

    /// Sync cancellation: triggers every per-worker token without awaiting
    /// the join handles.
    pub fn cancel_all(&self) {
        if let Ok(handles) = self.inner.handles.try_lock() {
            for h in handles.values() {
                h.cancel.cancel();
            }
        }
    }

    /// Cancel everything and await shutdown. Caller is responsible for
    /// draining any remaining events on the receiver side.
    pub async fn shutdown(&self) {
        let drained: Vec<(KvWorkerId, SubscriberHandle)> = {
            let mut handles = self.inner.handles.lock().await;
            handles.drain().collect()
        };
        for (id, h) in drained {
            h.cancel.cancel();
            if let Err(e) = h.join.await {
                warn!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    error = %e,
                    "subscriber task did not join cleanly during shutdown"
                );
            }
        }
    }
}

/// Spawn the background task that owns one SUB socket and forwards
/// decoded batches.
fn spawn_subscriber_task(
    id: KvWorkerId,
    endpoint: String,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_subscriber(id, endpoint, topic, kind, tx, cancel).await;
    })
}

/// Inner subscriber loop. Returns when.
///   * the cancellation token fires, OR
///   * the downstream mpsc receiver is dropped, OR
///   * the initial connect/subscribe fails after [`CONNECT_MAX_ATTEMPTS`]
///     attempts with exponential backoff, OR
///   * `recv()` returns errors [`RECV_ERROR_CEILING`] times in a row
///     (escalated to `error!` so the silent stall is detectable).
async fn run_subscriber(
    id: KvWorkerId,
    endpoint: String,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) {
    debug!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        topic = %topic,
        kind = ?kind,
        "starting kv-event subscriber"
    );

    let mut sub = match connect_with_backoff(&id, &endpoint, &topic, &cancel).await {
        Some(s) => s,
        None => return,
    };

    let mut errors_in_a_row = 0u32;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                debug!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    "subscriber cancelled"
                );
                return;
            }
            res = sub.recv() => {
                match res {
                    Ok(msg) => {
                        errors_in_a_row = 0;
                        if let Some(event) = decode_message(&id, msg, kind) {
                            if tx.send(event).await.is_err() {
                                // The pump (or the entire index) is gone.
                                warn!(
                                    worker_url = %id.url,
                                    dp_rank = id.dp_rank,
                                    "downstream mpsc receiver dropped; exiting"
                                );
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        errors_in_a_row += 1;
                        if errors_in_a_row >= RECV_ERROR_CEILING {
                            error!(
                                worker_url = %id.url,
                                dp_rank = id.dp_rank,
                                endpoint = %endpoint,
                                error = %e,
                                consecutive_errors = errors_in_a_row,
                                "SUB socket has produced {RECV_ERROR_CEILING} consecutive recv errors; giving up on this subscriber"
                            );
                            return;
                        }
                        // SubSocket auto-reconnects internally; transient errors should resume once a new peer.
                        warn!(
                            worker_url = %id.url,
                            dp_rank = id.dp_rank,
                            error = %e,
                            consecutive_errors = errors_in_a_row,
                            "recv error from SUB socket; continuing"
                        );
                        tokio::task::yield_now().await;
                    }
                }
            }
        }
    }
}

/// Open a `SubSocket`, connect to `endpoint`, and subscribe to the
/// supplied `topic` prefix (empty string = receive every message,
/// matching the prior all-topics behavior).
///
/// Retries with exponential backoff up to [`CONNECT_MAX_ATTEMPTS`] times
/// so a worker that just booted (publisher not yet bound) doesn't
/// permanently disable its KV-event subscriber.
///
/// Returns `None` if cancelled or if every attempt fails.
/// the backoff.
async fn connect_with_backoff(
    id: &KvWorkerId,
    endpoint: &str,
    topic: &str,
    cancel: &CancellationToken,
) -> Option<SubSocket> {
    let mut delay = CONNECT_BACKOFF_BASE;
    for attempt in 1..=CONNECT_MAX_ATTEMPTS {
        let mut sub = SubSocket::new();
        let connect_res = tokio::select! {
            _ = cancel.cancelled() => {
                debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled before connect");
                return None;
            }
            res = sub.connect(endpoint) => res,
        };
        if let Err(e) = connect_res {
            warn!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                endpoint = %endpoint,
                attempt,
                error = %e,
                "kv-events: connect SUB socket failed; retrying"
            );
        } else {
            let subscribe_res = tokio::select! {
                _ = cancel.cancelled() => {
                    debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled before subscribe");
                    return None;
                }
                res = sub.subscribe(topic) => res,
            };
            match subscribe_res {
                Ok(()) => return Some(sub),
                Err(e) => warn!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    endpoint = %endpoint,
                    attempt,
                    topic = %topic,
                    error = %e,
                    "kv-events: SUB subscribe failed; retrying"
                ),
            }
        }
        if attempt == CONNECT_MAX_ATTEMPTS {
            break;
        }
        tokio::select! {
            _ = cancel.cancelled() => {
                debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled during connect backoff");
                return None;
            }
            _ = tokio::time::sleep(delay) => {}
        }
        delay = (delay * 2).min(CONNECT_BACKOFF_CAP);
    }
    error!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        attempts = CONNECT_MAX_ATTEMPTS,
        "kv-events: gave up establishing SUB socket after {CONNECT_MAX_ATTEMPTS} attempts; this worker's cache-aware routing is disabled until next add_worker"
    );
    None
}

/// Validate, parse, and decode a single 3-frame multipart ZMQ message.
/// Returns `None` (with logging) for any non-event input (bad frame
/// count, sentinel sequence, or msgpack decode error). `kind` selects
/// whether to emit a KV [`WorkerEvent::Batch`] or a [`WorkerEvent::Load`].
fn decode_message(id: &KvWorkerId, msg: ZmqMessage, kind: SubKind) -> Option<WorkerEvent> {
    if msg.len() != 3 {
        warn!(
            worker_url = %id.url,
            dp_rank = id.dp_rank,
            frames = msg.len(),
            "dropping ZMQ message with unexpected frame count (expected 3)"
        );
        return None;
    }

    let seq_frame = msg.get(1)?;
    let payload = msg.get(2)?;

    // Decode the 8-byte BE seq. Frames smaller or larger than several
    // bytes mean a malformed publisher; log and drop.
    let seq_bytes: [u8; 8] = match seq_frame.as_ref().try_into() {
        Ok(b) => b,
        Err(_) => {
            warn!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq_len = seq_frame.len(),
                "dropping message with non-8-byte sequence frame"
            );
            return None;
        }
    };
    let seq = i64::from_be_bytes(seq_bytes);

    if seq == END_SEQ_SENTINEL {
        match kind {
            SubKind::Kv => {
                info!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    "publisher signalled shutdown (END_SEQ); forwarding cursor reset"
                );
                return Some(WorkerEvent::PublisherReset { worker: id.clone() });
            }
            // Load has no cursor / replay state to reset — drop.
            SubKind::Load => return None,
        }
    }

    // Decode by kind: the cache topic carries `KvEventBatch`es, the load topic
    // carries bare `LoadStat` snapshots — independent wire formats on
    // independent sockets.
    match kind {
        SubKind::Kv => {
            let batch = match decode_event_batch(payload.as_ref()) {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        worker_url = %id.url,
                        dp_rank = id.dp_rank,
                        seq,
                        error = %e,
                        "failed to decode KV event batch payload; dropping"
                    );
                    return None;
                }
            };
            trace!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq,
                n_events = batch.events.len(),
                "decoded KV event batch"
            );
            Some(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch,
            })
        }
        SubKind::Load => {
            let load = match decode_load_stat(payload.as_ref()) {
                Ok(l) => l,
                Err(e) => {
                    warn!(
                        worker_url = %id.url,
                        dp_rank = id.dp_rank,
                        seq,
                        error = %e,
                        "failed to decode load snapshot payload; dropping"
                    );
                    return None;
                }
            };
            trace!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq,
                "decoded load snapshot"
            );
            Some(WorkerEvent::Load {
                worker: id.clone(),
                load,
            })
        }
    }
}

/// Pull the host out of a routing URL like `http://10.0.0.1:30000` or `https://[::1]:30000`.
#[cfg(test)]
fn extract_host(worker_url: &str) -> Option<String> {
    let parsed = url::Url::parse(worker_url).ok()?;
    parsed.host_str().map(|s| s.to_string())
}

// --------------------------------------------------------------------------- Tests — bind real PUB sockets to ephemeral ports.

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use bytes::Bytes;
    use tokio::time::timeout;
    use zeromq::{Endpoint, PubSocket, Socket, SocketSend, ZmqMessage};

    use crate::state::kv_events::wire::KvCacheEvent;

    mod helpers {
        use super::*;
        use rmp::encode as mp;

        /// Bind a PUB socket to an OS-assigned localhost port and return
        /// `(socket, port)`.
        pub async fn make_pub_bound() -> (PubSocket, u16) {
            let mut sock = PubSocket::new();
            let endpoint = sock
                .bind("tcp://127.0.0.1:0")
                .await
                .expect("bind PUB socket");
            let port = match endpoint {
                Endpoint::Tcp(_, p) => p,
                other => panic!("unexpected endpoint: {other:?}"),
            };
            (sock, port)
        }

        /// Build a minimal [`EventConfig`] for test fixtures: take the host
        /// from `worker_url` (matches the pre-discovery behavior) and fill
        /// the rest with reasonable defaults.
        pub fn cfg_for(worker_url: &str, port_base: u16, dp_size: u32) -> EventConfig {
            EventConfig {
                host: extract_host(worker_url).unwrap_or_else(|| "127.0.0.1".to_string()),
                port_base,
                topic: String::new(),
                load_port_base: None,
                load_topic: None,
                block_size: 64,
                dp_size,
                is_bigram: false,
            }
        }

        /// Encode a minimal AllBlocksCleared batch with the given ts and
        /// optional dp_rank, in the same array layout msgspec emits.
        pub fn encode_all_blocks_cleared_batch(ts: f64, attn_dp_rank: Option<u32>) -> Vec<u8> {
            let mut buf = Vec::new();
            // Outer batch array: [ts, [event], dp_rank?]
            mp::write_array_len(&mut buf, 3).unwrap();
            mp::write_f64(&mut buf, ts).unwrap();
            mp::write_array_len(&mut buf, 1).unwrap();
            // Events use msgspec's tagged-map encoding: {"type": "AllBlocksCleared"}.
            mp::write_map_len(&mut buf, 1).unwrap();
            mp::write_str(&mut buf, "type").unwrap();
            mp::write_str(&mut buf, "AllBlocksCleared").unwrap();
            match attn_dp_rank {
                Some(v) => {
                    mp::write_uint(&mut buf, v as u64).unwrap();
                }
                None => mp::write_nil(&mut buf).unwrap(),
            }
            buf
        }

        /// Encode a LoadStat batch `[ts, [["LoadStat", running, waiting,
        /// num_tokens, max_total]], dp_rank?]` in msgspec's array layout.
        /// Encode a bare LoadStat msgpack array `["LoadStat", running, waiting,
        /// num_tokens, max_total, attn_dp_rank]` — the payload on the load
        /// socket (no EventBatch envelope).
        pub fn encode_load_stat(
            running: u64,
            waiting: u64,
            num_tokens: u64,
            max_total: u64,
            attn_dp_rank: u32,
        ) -> Vec<u8> {
            let mut buf = Vec::new();
            mp::write_array_len(&mut buf, 6).unwrap();
            mp::write_str(&mut buf, "LoadStat").unwrap();
            mp::write_uint(&mut buf, running).unwrap();
            mp::write_uint(&mut buf, waiting).unwrap();
            mp::write_uint(&mut buf, num_tokens).unwrap();
            mp::write_uint(&mut buf, max_total).unwrap();
            mp::write_uint(&mut buf, attn_dp_rank as u64).unwrap();
            buf
        }

        /// Build a 3-frame multipart with topic="", the given seq (BE i64),
        /// and the given payload bytes.
        pub fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
            build_multipart_with_topic(b"", seq, payload)
        }

        /// Build a 3-frame multipart with an explicit topic frame.
        pub fn build_multipart_with_topic(topic: &[u8], seq: i64, payload: Vec<u8>) -> ZmqMessage {
            let mut msg = ZmqMessage::from(Bytes::copy_from_slice(topic));
            msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
            msg.push_back(Bytes::from(payload));
            msg
        }

        /// Allows the local PUB/SUB handshake to complete in concurrent tests.
        pub async fn settle() {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        /// Destructure a `WorkerEvent::Batch`, panicking on any other
        /// variant. Keeps test assertions terse.
        pub fn expect_batch(ev: WorkerEvent) -> (KvWorkerId, i64, KvEventBatch) {
            match ev {
                WorkerEvent::Batch { worker, seq, batch } => (worker, seq, batch),
                WorkerEvent::Load { worker, .. } => {
                    panic!("expected Batch, got Load for {worker:?}")
                }
                WorkerEvent::PublisherReset { worker } => {
                    panic!("expected Batch, got PublisherReset for {worker:?}")
                }
            }
        }
    }

    /// Single subscriber: publish one batch, see one batch.
    #[tokio::test]
    async fn single_subscriber_receives_one_event() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        helpers::settle().await;

        let payload = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let msg = helpers::build_multipart(7, payload);
        pub_sock.send(msg).await.expect("send");

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv timed out")
            .expect("channel closed");
        let (worker, seq, batch) = helpers::expect_batch(event);

        assert_eq!(seq, 7);
        assert_eq!(worker.dp_rank, 0);
        assert_eq!(worker.url, "http://127.0.0.1:30000");
        assert_eq!(batch.events.len(), 1);
        assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));

        let shutdown_done = timeout(Duration::from_millis(500), registry.shutdown()).await;
        assert!(shutdown_done.is_ok(), "shutdown should return promptly");
    }

    /// When the worker advertises a non-empty topic in `EventConfig.topic`, the SUB socket must filter on that prefix.
    #[tokio::test]
    async fn subscriber_filters_by_configured_topic() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        let worker_url = "http://127.0.0.1:30100";
        let mut cfg = helpers::cfg_for(worker_url, port, 1);
        cfg.topic = "match".into();
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        // Publish matched first, then `other`.
        let payload_matched = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let payload_other = helpers::encode_all_blocks_cleared_batch(2.0, Some(0));
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"match",
                22,
                payload_matched,
            ))
            .await
            .unwrap();
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"other",
                11,
                payload_other,
            ))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out waiting for matched event")
            .expect("channel closed");
        let (_, seq, _) = helpers::expect_batch(event);
        assert_eq!(seq, 22, "matched message must arrive; got seq={seq}");

        // The load-bearing assertion: no second message in 200ms.
        let stray = timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(
            stray.is_err(),
            "second message with topic=`other` must NOT pass the filter \
             (got {stray:?}); cfg.topic is being ignored at subscribe()",
        );

        registry.shutdown().await;
    }

    /// The # load stream has its own advertised topic.
    #[tokio::test]
    async fn load_subscriber_filters_by_advertised_topic() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::with_kind(tx, SubKind::Load);

        let worker_url = "http://127.0.0.1:30101";
        let mut cfg = helpers::cfg_for(worker_url, port, 1);
        cfg.load_port_base = Some(port);
        cfg.load_topic = Some("load".into());
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        let payload = helpers::encode_load_stat(5, 2, 100, 1000, 0);
        pub_sock
            .send(helpers::build_multipart_with_topic(b"load", 3, payload))
            .await
            .unwrap();
        let other_payload = helpers::encode_load_stat(99, 0, 0, 0, 0);
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"other",
                4,
                other_payload,
            ))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out waiting for load event")
            .expect("channel closed");
        match event {
            WorkerEvent::Load { load, .. } => assert_eq!(load.num_running_reqs, 5),
            other => panic!("expected Load, got {other:?}"),
        }
        assert!(
            timeout(Duration::from_millis(200), rx.recv())
                .await
                .is_err(),
            "unmatched load topic must not reach the subscriber"
        );

        registry.shutdown().await;
    }

    /// DP rank fan-out: PUB sockets, distinct events, all delivered.
    #[tokio::test]
    async fn dp_rank_fan_out() {
        let (mut pub0, p0) = helpers::make_pub_bound().await;
        let (mut pub1, p1) = helpers::make_pub_bound().await;
        let (mut pub2, p2) = helpers::make_pub_bound().await;
        // We need contiguous ports for `base_port + dp_rank` to land on each PUB socket.
        let url0 = "http://127.0.0.1:30000";
        let url1 = "http://127.0.0.1:30001";
        let url2 = "http://127.0.0.1:30002";

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(16);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(url0, &helpers::cfg_for(url0, p0, 1))
            .await;
        registry
            .add_worker(url1, &helpers::cfg_for(url1, p1, 1))
            .await;
        registry
            .add_worker(url2, &helpers::cfg_for(url2, p2, 1))
            .await;
        helpers::settle().await;

        let payload0 = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let payload1 = helpers::encode_all_blocks_cleared_batch(2.0, Some(1));
        let payload2 = helpers::encode_all_blocks_cleared_batch(3.0, Some(2));

        pub0.send(helpers::build_multipart(10, payload0))
            .await
            .unwrap();
        pub1.send(helpers::build_multipart(20, payload1))
            .await
            .unwrap();
        pub2.send(helpers::build_multipart(30, payload2))
            .await
            .unwrap();

        let mut seq_by_url: HashMap<String, i64> = HashMap::new();
        for _ in 0..3 {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            let (worker, seq, _batch) = helpers::expect_batch(event);
            seq_by_url.insert(worker.url, seq);
        }

        assert_eq!(seq_by_url.len(), 3);
        assert_eq!(seq_by_url[url0], 10);
        assert_eq!(seq_by_url[url1], 20);
        assert_eq!(seq_by_url[url2], 30);

        registry.shutdown().await;
    }

    /// True per-DP fan-out behind a single worker URL: bind PUB sockets on contiguous ports and subscribe.
    #[tokio::test]
    async fn dp_size_three_per_worker() {
        // Pick a single base port and keep retrying until the next ports are also free.
        let mut attempt = 0;
        let (pub0, pub1, pub2, base_port) = loop {
            attempt += 1;
            assert!(attempt < 256, "could not find 3 contiguous free ports");

            // Bind PUB at OS-assigned port to learn what's free, then try to bind the next ports explicitly.
            let mut p0 = PubSocket::new();
            let ep0 = p0.bind("tcp://127.0.0.1:0").await.unwrap();
            let base = match ep0 {
                Endpoint::Tcp(_, p) => p,
                _ => unreachable!(),
            };

            let mut p1 = PubSocket::new();
            let ep1 = p1.bind(&format!("tcp://127.0.0.1:{}", base + 1)).await;
            if ep1.is_err() {
                continue;
            }

            let mut p2 = PubSocket::new();
            let ep2 = p2.bind(&format!("tcp://127.0.0.1:{}", base + 2)).await;
            if ep2.is_err() {
                continue;
            }
            break (p0, p1, p2, base);
        };

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(16);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", base_port, 3),
            )
            .await;
        helpers::settle().await;

        let mut pub0 = pub0;
        let mut pub1 = pub1;
        let mut pub2 = pub2;
        pub0.send(helpers::build_multipart(
            100,
            helpers::encode_all_blocks_cleared_batch(1.0, Some(0)),
        ))
        .await
        .unwrap();
        pub1.send(helpers::build_multipart(
            200,
            helpers::encode_all_blocks_cleared_batch(2.0, Some(1)),
        ))
        .await
        .unwrap();
        pub2.send(helpers::build_multipart(
            300,
            helpers::encode_all_blocks_cleared_batch(3.0, Some(2)),
        ))
        .await
        .unwrap();

        let mut by_rank: HashMap<u32, i64> = HashMap::new();
        for _ in 0..3 {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            let (worker, seq, _batch) = helpers::expect_batch(event);
            assert_eq!(worker.url, "http://127.0.0.1:30000");
            by_rank.insert(worker.dp_rank, seq);
        }
        assert_eq!(by_rank.get(&0), Some(&100));
        assert_eq!(by_rank.get(&1), Some(&200));
        assert_eq!(by_rank.get(&2), Some(&300));

        registry.shutdown().await;
    }

    /// 8-rank multi-publisher fan-out: a worker that publishes to contiguous ZMQ ports (one per DP rank).
    #[tokio::test]
    async fn dp_size_eight_per_worker() {
        const N: usize = 8;
        let mut attempt = 0;
        let mut publishers: Vec<PubSocket> = Vec::new();
        let base_port: u16 = loop {
            attempt += 1;
            assert!(attempt < 64, "could not find 8 contiguous free ports");
            publishers.clear();

            let mut p0 = PubSocket::new();
            let ep0 = p0.bind("tcp://127.0.0.1:0").await.unwrap();
            let base = match ep0 {
                Endpoint::Tcp(_, p) => p,
                _ => unreachable!(),
            };
            // Ensure `base + N - 1` fits in u16 *and* we can bind every
            // contiguous port.
            if u32::from(base) + (N as u32) > u32::from(u16::MAX) {
                continue;
            }
            publishers.push(p0);
            let mut ok = true;
            for offset in 1..N as u16 {
                let mut p = PubSocket::new();
                let res = p.bind(&format!("tcp://127.0.0.1:{}", base + offset)).await;
                if res.is_err() {
                    ok = false;
                    break;
                }
                publishers.push(p);
            }
            if ok {
                break base;
            }
        };

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(64);
        let registry = KvEventSubscriberRegistry::new(tx);
        let worker_url = "http://127.0.0.1:30000";
        registry
            .add_worker(
                worker_url,
                &helpers::cfg_for(worker_url, base_port, N as u32),
            )
            .await;
        helpers::settle().await;

        let mut by_rank: HashMap<u32, i64> = HashMap::new();
        for _ in 0..20 {
            for (rank, pubsock) in publishers.iter_mut().enumerate() {
                pubsock
                    .send(helpers::build_multipart(
                        1000 + rank as i64,
                        helpers::encode_all_blocks_cleared_batch(rank as f64, Some(rank as u32)),
                    ))
                    .await
                    .unwrap();
            }

            for _ in 0..N {
                let Ok(Some(event)) = timeout(Duration::from_millis(50), rx.recv()).await else {
                    break;
                };
                let (worker, seq, _batch) = helpers::expect_batch(event);
                assert_eq!(worker.url, worker_url);
                by_rank.insert(worker.dp_rank, seq);
            }
            if by_rank.len() == N {
                break;
            }
        }
        assert_eq!(by_rank.len(), N, "every rank must produce an event");
        for rank in 0..N as u32 {
            assert_eq!(
                by_rank.get(&rank),
                Some(&(1000 + rank as i64)),
                "rank {rank} missing or wrong seq",
            );
        }

        registry.shutdown().await;
    }

    /// Bad msgpack payload is logged and dropped; subsequent valid event still arrives.
    #[tokio::test]
    async fn decoding_error_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        // Garbage payload (not msgpack).
        pub_sock
            .send(helpers::build_multipart(1, vec![0xff, 0xfe, 0xfd]))
            .await
            .unwrap();
        // Then a valid one.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(2, payload))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, batch) = helpers::expect_batch(event);
        assert_eq!(seq, 2);
        // We must NOT have received the bad message.
        assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));

        registry.shutdown().await;
    }

    /// 2-frame and 4-frame messages are dropped; valid 3-frame still works.
    #[tokio::test]
    async fn wrong_frame_count_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        // 2-frame: topic + payload.
        let mut bad2 = ZmqMessage::from(Bytes::new());
        bad2.push_back(Bytes::from_static(b"junk"));
        pub_sock.send(bad2).await.unwrap();

        // 4-frame: topic + seq + payload + extra.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        let mut bad4 = helpers::build_multipart(99, payload.clone());
        bad4.push_back(Bytes::from_static(b"extra"));
        pub_sock.send(bad4).await.unwrap();

        // Valid 3-frame.
        pub_sock
            .send(helpers::build_multipart(42, payload))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, _batch) = helpers::expect_batch(event);
        assert_eq!(seq, 42);

        registry.shutdown().await;
    }

    /// END_SEQ sentinel (-) is forwarded as a `PublisherReset` so the downstream pump can clear its cursor.
    #[tokio::test]
    async fn sequence_number_sentinel_propagates_as_reset() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        pub_sock
            .send(helpers::build_multipart(-1, b"ignored".to_vec()))
            .await
            .unwrap();
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(5, payload))
            .await
            .unwrap();

        let first = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert!(
            matches!(first, WorkerEvent::PublisherReset { .. }),
            "END_SEQ must surface as PublisherReset, got {first:?}",
        );

        let second = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, _batch) = helpers::expect_batch(second);
        assert_eq!(seq, 5);

        registry.shutdown().await;
    }

    /// `remove_worker` cancels the task; further publishes are not received.
    #[tokio::test]
    async fn remove_worker_cancels() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        helpers::settle().await;

        // First event arrives.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(1, payload.clone()))
            .await
            .unwrap();
        let _ = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("first event timed out");

        // Remove and verify the handle map empties.
        registry.remove_worker("http://127.0.0.1:30000").await;
        {
            let handles = registry.inner.handles.lock().await;
            assert!(
                handles.is_empty(),
                "handles map should be empty after remove"
            );
        }

        // Publish more — receiver should see nothing.
        pub_sock
            .send(helpers::build_multipart(2, payload))
            .await
            .unwrap();
        let res = timeout(Duration::from_millis(150), rx.recv()).await;
        assert!(
            res.is_err(),
            "no event should arrive after remove_worker (got {:?})",
            res.unwrap()
        );
    }

    /// Calling `add_worker` twice for the same `(url, dp_rank)` pair must not double-spawn.
    #[tokio::test]
    async fn add_worker_idempotent() {
        let (_pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, _rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;

        {
            let handles = registry.inner.handles.lock().await;
            assert_eq!(handles.len(), 1, "expected 1 entry, got {}", handles.len());
        }

        registry.shutdown().await;
    }

    /// `cancel_all` signals every per-worker token without awaiting.
    #[tokio::test]
    async fn cancel_all_then_shutdown_is_clean() {
        let (_pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, _rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 2),
            )
            .await;
        helpers::settle().await;

        // Sync cancel — must not block, must not panic.
        registry.cancel_all();

        // shutdown should still join cleanly even though the per-worker tokens were already fired by cancel_all.
        let done = timeout(Duration::from_millis(500), registry.shutdown()).await;
        assert!(done.is_ok(), "shutdown after cancel_all must not hang");
    }

    /// Direct unit test of [`extract_host`] — no socket required.
    #[test]
    fn extract_host_handles_common_urls() {
        assert_eq!(
            extract_host("http://10.0.0.1:30000").as_deref(),
            Some("10.0.0.1")
        );
        assert_eq!(
            extract_host("https://my.host.example:443").as_deref(),
            Some("my.host.example")
        );
        // url crate strips brackets from IPv6 literals in host_str().
        assert_eq!(extract_host("http://[::1]:30000").as_deref(), Some("[::1]"));
        assert!(extract_host("not a url").is_none());
    }

    /// Direct unit test of [`decode_message`] — exercises sentinel and bad-frame paths without involving sockets.
    #[test]
    fn decode_message_unit() {
        let id = KvWorkerId {
            url: "http://x".to_string(),
            dp_rank: 0,
        };

        // Wrong frame count.
        let one_frame = ZmqMessage::from(Bytes::from_static(b"only"));
        assert!(decode_message(&id, one_frame, SubKind::Kv).is_none());

        let sentinel = helpers::build_multipart(-1, b"ignored".to_vec());
        let reset = decode_message(&id, sentinel, SubKind::Kv).expect("END_SEQ forwards");
        assert!(matches!(reset, WorkerEvent::PublisherReset { .. }));

        // Bad seq frame length.
        let mut bad_seq = ZmqMessage::from(Bytes::new());
        bad_seq.push_back(Bytes::from_static(b"abc"));
        bad_seq.push_back(Bytes::from_static(b""));
        assert!(decode_message(&id, bad_seq, SubKind::Kv).is_none());

        // Bad payload.
        let bad_payload = helpers::build_multipart(1, vec![0xff, 0xfe]);
        assert!(decode_message(&id, bad_payload, SubKind::Kv).is_none());

        // Happy path.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        let good = helpers::build_multipart(7, payload);
        let event = decode_message(&id, good, SubKind::Kv).expect("should decode");
        let (worker, seq, _batch) = helpers::expect_batch(event);
        assert_eq!(seq, 7);
        assert_eq!(worker, id);
    }

    /// A `SubKind::Load` subscriber decodes a bare LoadStat frame into `WorkerEvent::Load`.
    #[test]
    fn decode_message_load_kind() {
        let id = KvWorkerId {
            url: "http://x".to_string(),
            dp_rank: 1,
        };

        // END_SEQ is dropped for the load topic.
        let sentinel = helpers::build_multipart(-1, b"ignored".to_vec());
        assert!(decode_message(&id, sentinel, SubKind::Load).is_none());

        // A bare LoadStat frame becomes WorkerEvent::Load.
        let payload = helpers::encode_load_stat(5, 2, 100, 1000, 1);
        let msg = helpers::build_multipart(3, payload);
        let event = decode_message(&id, msg, SubKind::Load).expect("should decode load");
        match event {
            WorkerEvent::Load { worker, load } => {
                assert_eq!(worker, id);
                assert_eq!(load.num_running_reqs, 5);
                assert_eq!(load.num_waiting_reqs, 2);
                assert_eq!(load.num_tokens, 100);
                assert_eq!(load.max_total_num_tokens, 1000);
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// Restart-resume contract: after a worker is removed and then re-added to the same endpoint.
    #[tokio::test]
    async fn restart_after_remove_picks_up_new_events() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let worker_url = "http://127.0.0.1:30000";
        let cfg = helpers::cfg_for(worker_url, port, 1);

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        // First incarnation: publish + drain.
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;
        let payload_a = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(1, payload_a))
            .await
            .unwrap();
        let event_a = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv before remove timed out")
            .expect("channel closed");
        let (_, seq_a, _) = helpers::expect_batch(event_a);
        assert_eq!(seq_a, 1);

        // Detach the subscriber while the publisher keeps going.
        registry.remove_worker(worker_url).await;

        // This batch is sent while no subscriber is attached.
        let payload_b = helpers::encode_all_blocks_cleared_batch(2.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(2, payload_b))
            .await
            .unwrap();
        // Verify rx has nothing buffered.
        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "no event must arrive while the worker is detached",
        );

        // Re-attach the SAME worker at the SAME endpoint.
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        // Fresh event from the publisher → must surface on the new subscriber.
        let payload_c = helpers::encode_all_blocks_cleared_batch(3.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(3, payload_c))
            .await
            .unwrap();
        let event_c = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv after re-add timed out")
            .expect("channel closed");
        let (worker_c, seq_c, _) = helpers::expect_batch(event_c);
        assert_eq!(seq_c, 3);
        assert_eq!(worker_c.url, worker_url);

        registry.shutdown().await;
    }
}
