// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Counters for the KV-cache event stream the pump consumes.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use tracing::warn;

use super::tree::Tiers;

/// Medium labels, in the order [`EventTally`] stores them.
pub const MEDIUM_LABELS: [&str; 6] = [
    Tiers::WIRE_MEDIA[0].0,
    Tiers::WIRE_MEDIA[1].0,
    Tiers::WIRE_MEDIA[2].0,
    Tiers::WIRE_MEDIA[3].0,
    "untagged",
    "unknown",
];
const UNTAGGED: usize = 4;
const UNKNOWN: usize = 5;
const _: () = assert!(
    Tiers::WIRE_MEDIA.len() == UNTAGGED,
    "MEDIUM_LABELS lists every WIRE_MEDIA entry before the folds",
);

/// Cap on the distinct unrecognised `medium` strings remembered for warn-once.
const MAX_REMEMBERED_UNKNOWN_MEDIA: usize = 16;

/// Which event a tally entry describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    BlockStored,
    BlockRemoved,
    AllBlocksCleared,
}

impl EventKind {
    /// Every kind, in storage order.
    pub const ALL: [EventKind; 3] = [
        Self::BlockStored,
        Self::BlockRemoved,
        Self::AllBlocksCleared,
    ];

    const fn slot(self) -> usize {
        match self {
            Self::BlockStored => 0,
            Self::BlockRemoved => 1,
            Self::AllBlocksCleared => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::BlockStored => "block_stored",
            Self::BlockRemoved => "block_removed",
            Self::AllBlocksCleared => "all_blocks_cleared",
        }
    }
}

// `slot()` indexes the counter arrays, so it must be a permutation of
// `0..ALL.len()`. Pinned here rather than trusted.
const _: () = {
    let mut i = 0;
    while i < EventKind::ALL.len() {
        assert!(
            EventKind::ALL[i].slot() == i,
            "EventKind::slot must match position in EventKind::ALL",
        );
        i += 1;
    }
};

/// One rendered cell of the tally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TallyRow {
    pub event: &'static str,
    pub medium: &'static str,
    /// Events applied.
    pub events: u64,
    pub blocks: u64,
}

/// Lock-free counters, written by the single pump task and read on scrape.
#[derive(Debug, Default)]
pub struct EventTally {
    events: [[AtomicU64; MEDIUM_LABELS.len()]; EventKind::ALL.len()],
    blocks: [[AtomicU64; MEDIUM_LABELS.len()]; EventKind::ALL.len()],
    /// Batches the transport lost, inferred from gaps in the publisher's dense sequence.
    batches_lost: AtomicU64,
    /// Unrecognised `medium` strings already warned about, so an engine that adds a tier logs once per string rather than once.
    warned_unknown_media: Mutex<HashSet<String>>,
}

impl EventTally {
    pub fn new() -> Self {
        Self::default()
    }

    fn medium_slot(&self, medium: Option<&str>) -> usize {
        let Some(m) = medium else {
            return UNTAGGED;
        };
        match MEDIUM_LABELS[..UNTAGGED]
            .iter()
            .position(|known| *known == m)
        {
            Some(slot) => slot,
            None => {
                // Only an unrecognised medium takes the lock, so the common path stays allocation- and lock-free.
                self.warn_unknown_medium(m);
                UNKNOWN
            }
        }
    }

    /// Log the first sighting of each unrecognised `medium`. This is the
    /// "the engine added a storage tier, upgrade the router" line: the tree
    /// drops a store tagged with one ([`Tiers::for_store`]) and would
    /// otherwise do it in silence.
    /// bug this module exists to prevent recurring.
    fn warn_unknown_medium(&self, medium: &str) {
        let mut seen = self.warned_unknown_media.lock();
        if seen.contains(medium) {
            return;
        }
        if seen.len() < MAX_REMEMBERED_UNKNOWN_MEDIA {
            seen.insert(medium.to_owned());
        }
        drop(seen);
        warn!(
            medium,
            known = ?MEDIUM_LABELS[..UNTAGGED],
            "kv-events: unrecognised storage medium; blocks stored on it are routable but rank below every known tier, and a removal tagged with it clears every tier",
        );
    }

    /// Book one applied event carrying `blocks` block hashes.
    pub fn record(&self, event: EventKind, medium: Option<&str>, blocks: usize) {
        let (e, m) = (event.slot(), self.medium_slot(medium));
        self.events[e][m].fetch_add(1, Ordering::Relaxed);
        self.blocks[e][m].fetch_add(blocks as u64, Ordering::Relaxed);
    }

    pub fn record_lost_batches(&self, count: u64) {
        self.batches_lost.fetch_add(count, Ordering::Relaxed);
    }

    pub fn batches_lost(&self) -> u64 {
        self.batches_lost.load(Ordering::Relaxed)
    }

    /// Every cell in (event, medium) order, zeros included.
    pub fn snapshot(&self) -> Vec<TallyRow> {
        let mut rows = Vec::with_capacity(EventKind::ALL.len() * MEDIUM_LABELS.len());
        for event in EventKind::ALL {
            for (m, medium) in MEDIUM_LABELS.iter().enumerate() {
                rows.push(TallyRow {
                    event: event.label(),
                    medium,
                    events: self.events[event.slot()][m].load(Ordering::Relaxed),
                    blocks: self.blocks[event.slot()][m].load(Ordering::Relaxed),
                });
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell<'a>(rows: &'a [TallyRow], event: &str, medium: &str) -> &'a TallyRow {
        rows.iter()
            .find(|r| r.event == event && r.medium == medium)
            .expect("every (event, medium) cell is rendered")
    }

    #[test]
    fn records_by_kind_and_medium_and_folds_the_rest() {
        let t = EventTally::new();
        t.record(EventKind::BlockStored, Some("GPU"), 3);
        t.record(EventKind::BlockStored, Some("CPU_PINNED"), 3);
        t.record(EventKind::BlockRemoved, Some("GPU"), 1);
        t.record(EventKind::BlockRemoved, None, 2);
        t.record(EventKind::BlockStored, Some("NVLINK_PEER"), 5);
        t.record(EventKind::AllBlocksCleared, None, 0);

        let rows = t.snapshot();
        assert_eq!(rows.len(), EventKind::ALL.len() * MEDIUM_LABELS.len());
        assert_eq!(cell(&rows, "block_stored", "GPU").blocks, 3);
        assert_eq!(cell(&rows, "block_stored", "CPU_PINNED").events, 1);
        assert_eq!(cell(&rows, "block_removed", "GPU").blocks, 1);
        assert_eq!(cell(&rows, "block_removed", "untagged").blocks, 2);
        assert_eq!(cell(&rows, "block_stored", "unknown").blocks, 5);
        assert_eq!(cell(&rows, "all_blocks_cleared", "untagged").events, 1);
        // Zero cells are present, not omitted.
        assert_eq!(cell(&rows, "block_removed", "CPU_PINNED").events, 0);
    }

    /// A publisher inventing media must not grow router memory without bound; past the cap the warning repeats instead.
    #[test]
    fn remembered_unknown_media_are_bounded() {
        let t = EventTally::new();
        for i in 0..MAX_REMEMBERED_UNKNOWN_MEDIA * 3 {
            t.record(EventKind::BlockStored, Some(&format!("MEDIUM_{i}")), 1);
        }
        assert_eq!(
            t.warned_unknown_media.lock().len(),
            MAX_REMEMBERED_UNKNOWN_MEDIA,
        );
        assert_eq!(
            cell(&t.snapshot(), "block_stored", "unknown").events,
            (MAX_REMEMBERED_UNKNOWN_MEDIA * 3) as u64,
            "every unknown medium is still counted, only the warning is capped",
        );
    }

    #[test]
    fn lost_batches_accumulate() {
        let t = EventTally::new();
        assert_eq!(t.batches_lost(), 0);
        t.record_lost_batches(3);
        t.record_lost_batches(1);
        assert_eq!(t.batches_lost(), 4);
    }
}
