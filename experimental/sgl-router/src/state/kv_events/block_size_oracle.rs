// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-shared KV-cache block size, sourced from the workers.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

/// Tri-state for the bigram flag: distinguishes "not yet reported" from an established `false`.
const BIGRAM_UNKNOWN: u8 = 0;
const BIGRAM_UNIGRAM: u8 = 1;
const BIGRAM_BIGRAM: u8 = 2;

/// First-wins, idempotent block-size publisher. Internally an `AtomicU32` where means "not yet known".
#[derive(Debug, Default)]
pub struct BlockSizeOracle {
    value: AtomicU32,
    bigram: AtomicU8,
}

/// Returned by [`BlockSizeOracle::try_set`] when the candidate disagrees with the already-established value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizeMismatch {
    pub established: u32,
    pub candidate: u32,
}

impl BlockSizeOracle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the established block size, or `None` if no worker has
    /// reported one yet.
    pub fn get(&self) -> Option<u32> {
        let v = self.value.load(Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// Publish whether the cluster's workers use bigram (EAGLE-family) KV-block
    /// hashing. Called from `KvEventIndex::add_worker` alongside `try_set`.
    /// First-wins: the first worker establishes the mode.
    /// disagrees is logged (not silently honored), since the query-hashing mode
    /// is process-wide and one mismatched worker would zero out cache-aware
    /// routing for the cluster.
    pub fn set_bigram(&self, is_bigram: bool) {
        let candidate = if is_bigram {
            BIGRAM_BIGRAM
        } else {
            BIGRAM_UNIGRAM
        };
        match self.bigram.compare_exchange(
            BIGRAM_UNKNOWN,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(existing) if existing == candidate => {}
            Err(existing) => {
                tracing::warn!(
                    established_bigram = existing == BIGRAM_BIGRAM,
                    worker_bigram = is_bigram,
                    "kv-events: worker hashing mode (bigram/EAGLE) disagrees with the \
                     established cluster value; keeping the first. A heterogeneous \
                     EAGLE/non-EAGLE cluster will silently never match cache for the \
                     minority workers — check that all workers run the same model.",
                );
            }
        }
    }

    /// Whether query hashing should use the bigram variant
    /// ([`super::hash::compute_block_hashes_bigram`]).
    pub fn is_bigram(&self) -> bool {
        self.bigram.load(Ordering::Relaxed) == BIGRAM_BIGRAM
    }

    /// Publish a candidate block size. Returns the established value on
    /// success (idempotent: same candidate as already set is `Ok`);
    /// returns `Err(BlockSizeMismatch)` when the candidate disagrees.
    ///
    /// yet known" sentinel.
    pub fn try_set(&self, candidate: u32) -> Result<u32, BlockSizeMismatch> {
        if candidate == 0 {
            return Err(BlockSizeMismatch {
                established: self.value.load(Ordering::Relaxed),
                candidate,
            });
        }
        match self
            .value
            .compare_exchange(0, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => Ok(candidate),
            Err(existing) if existing == candidate => Ok(existing),
            Err(existing) => Err(BlockSizeMismatch {
                established: existing,
                candidate,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_oracle_returns_none() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.get(), None);
    }

    #[test]
    fn first_set_establishes_the_value() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn matching_set_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn bigram_flag_defaults_false_first_wins_and_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert!(
            !oracle.is_bigram(),
            "unknown (no worker reported yet) reads as non-bigram"
        );
        oracle.set_bigram(true);
        assert!(oracle.is_bigram(), "first worker establishes the mode");
        oracle.set_bigram(true); // idempotent agreement
        assert!(oracle.is_bigram());
        // Independent of block_size establishment.
        assert_eq!(oracle.get(), None);
        // First-wins: a conflicting later worker is logged, not honored.
        oracle.set_bigram(false);
        assert!(
            oracle.is_bigram(),
            "a disagreeing worker must not flip the established mode"
        );
    }

    #[test]
    fn bigram_flag_establishes_false_first_wins() {
        let oracle = BlockSizeOracle::new();
        oracle.set_bigram(false);
        assert!(!oracle.is_bigram(), "established as unigram");
        oracle.set_bigram(true); // conflicting; first (unigram) wins
        assert!(!oracle.is_bigram());
    }

    #[test]
    fn mismatching_set_fails_without_changing_state() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).unwrap();
        assert_eq!(
            oracle.try_set(128),
            Err(BlockSizeMismatch {
                established: 64,
                candidate: 128
            })
        );
        assert_eq!(
            oracle.get(),
            Some(64),
            "mismatched candidate must not overwrite established value"
        );
    }

    #[test]
    fn zero_candidate_is_rejected() {
        let oracle = BlockSizeOracle::new();
        assert!(oracle.try_set(0).is_err());
        assert_eq!(oracle.get(), None);
    }
}
