//! ZMQ-based KV-cache event indexer for cache-aware routing.

pub mod block_size_oracle;
pub mod discovery;
pub mod hash;
pub mod index;
pub mod subscriber;
pub mod tally;
pub mod tree;
pub mod wire;

pub use block_size_oracle::BlockSizeOracle;
pub(crate) use discovery::classify_bigram;
pub use discovery::{fetch_event_config, EventConfig};
pub use hash::{compute_block_hashes, compute_block_hashes_bigram, sha256_to_i64};
pub use index::{KvEventIndex, KvIndexMetrics};
pub use subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
pub use tally::{EventKind, EventTally, TallyRow};
pub use tree::{
    HashTree, KvWorkerId, MatchResult, TierCounts, Tiers, ACCOUNTING_REASONS, TIER_SLOT_COUNT,
};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};
