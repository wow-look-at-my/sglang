// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Concurrent-mutation stress test for `HashTree`.

use std::sync::Arc;
use std::thread;

use sgl_router::state::kv_events::{HashTree, KvWorkerId};

fn worker(i: usize) -> KvWorkerId {
    KvWorkerId {
        url: format!("http://w{i}:30000"),
        dp_rank: 0,
    }
}

/// mutator threads × ops + reader threads × match queries.
#[test]
fn tree_survives_concurrent_inserts_removes_and_matches() {
    let tree = Arc::new(HashTree::new());

    let mut handles = Vec::new();

    for tid in 0..8 {
        let tree = tree.clone();
        handles.push(thread::spawn(move || {
            let w = worker(tid);
            for round in 0..200_u64 {
                // Each round uses a fresh chain so different mutators do not
                // trample each other's nodes.
                let chain: Vec<i64> = (0..4)
                    .map(|i| ((tid as i64) << 32) | ((round as i64) << 8) | i as i64)
                    .collect();
                tree.insert(&w, None, &chain);

                let m = tree.match_prefix(None, &chain);
                assert!(
                    m.matched_blocks <= chain.len(),
                    "match must never exceed query length",
                );

                // Half the rounds use remove(&chain); the rest use
                // clear_worker — both must leave a consistent tree.
                if round % 2 == 0 {
                    tree.remove(&w, &chain);
                } else {
                    tree.clear_worker(&w);
                }
            }
            // Final blanket clear in case the last iteration used `remove` on only part of the chain.
            tree.clear_worker(&w);
        }));
    }

    for tid in 0..4 {
        let tree = tree.clone();
        handles.push(thread::spawn(move || {
            for round in 0..500_u64 {
                let probe: Vec<i64> = (0..3)
                    .map(|i| ((tid as i64) << 40) | ((round as i64) << 8) | i as i64)
                    .collect();
                // Readers must never block-walk and must never panic.
                let _ = tree.match_prefix(None, &probe);
            }
        }));
    }

    for h in handles {
        h.join()
            .expect("worker thread panicked under concurrent load");
    }

    assert_eq!(
        tree.node_count(),
        0,
        "tree must be empty after every worker was cleared; \
         residual nodes indicate a missed clear_worker path",
    );

    // The arena and the reverse index must agree: zero non-root nodes means
    // zero `by_hash` entries.
    assert_eq!(
        tree.reverse_index_size(),
        0,
        "by_hash reverse index must be empty when no non-root nodes remain",
    );
}

/// A mutator races `clear_worker` against a reader that is mid-`match_prefix` on a deep chain.
#[test]
fn match_prefix_is_consistent_with_concurrent_clear() {
    let tree = Arc::new(HashTree::new());
    let w = worker(0);
    let chain: Vec<i64> = (0..32).map(|i| 1_000 + i).collect();

    // Pre-populate so the reader has something to walk.
    tree.insert(&w, None, &chain);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mutator = {
        let tree = tree.clone();
        let stop = stop.clone();
        let w = w.clone();
        let chain = chain.clone();
        thread::spawn(move || {
            let mut round = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if round.is_multiple_of(2) {
                    tree.clear_worker(&w);
                } else {
                    tree.insert(&w, None, &chain);
                }
                round += 1;
            }
        })
    };

    for _ in 0..2_000 {
        let m = tree.match_prefix(None, &chain);
        // Either the worker was present (matched_blocks == chain.len(),
        // workers set contains w) or it was cleared mid-walk.
        if m.matched_blocks == chain.len() {
            assert!(
                m.holds(&w),
                "full match must include worker; got {:?}",
                m.workers(),
            );
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    mutator.join().unwrap();
}

/// Reader storm concurrent with a writer hammering insert/remove on DISTINCT chain roots.
#[test]
fn readers_unaffected_by_writer_on_distinct_roots() {
    let tree = Arc::new(HashTree::new());

    let warm = KvWorkerId {
        url: "http://warm:30000".into(),
        dp_rank: 0,
    };
    const WARM_CHAINS: i64 = 16;
    let warm_chain = |c: i64| -> Vec<i64> { vec![c * 1_000, c * 1_000 + 1, c * 1_000 + 2] };
    for c in 0..WARM_CHAINS {
        tree.insert(&warm, None, &warm_chain(c));
    }

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Scratch chains on distinct roots far from the warm space, insert then
    // immediate remove. Runs until the readers are done rather than for a
    // fixed count the writer could burn through early, leaving the test with
    // no contention at all. Every round is a complete pair, so stopping at
    // any point leaves no residue.
    let writer = {
        let tree = tree.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let scratch = KvWorkerId {
                url: "http://scratch:30000".into(),
                dp_rank: 0,
            };
            let mut round = 0i64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                // Cycled so the arithmetic cannot drift into the warm space.
                let base = 1_000_000 + (round % 100_000) * 7;
                let chain = [base, base + 1, base + 2, base + 3];
                tree.insert(&scratch, None, &chain);
                tree.remove(&scratch, &chain);
                round = round.wrapping_add(1);
            }
        })
    };

    // Each reader asserts the warm worker is present at full depth on every warm chain, regardless of writer churn.
    let mut readers = Vec::new();
    for _ in 0..4 {
        let tree = tree.clone();
        let warm = warm.clone();
        readers.push(thread::spawn(move || {
            for _ in 0..2_000 {
                for c in 0..WARM_CHAINS {
                    let chain = warm_chain(c);
                    let m = tree.match_prefix(None, &chain);
                    assert_eq!(
                        m.matched_blocks,
                        chain.len(),
                        "warm chain {c} must always fully match despite writer churn",
                    );
                    assert!(
                        m.holds(&warm),
                        "warm worker must always hold its own untouched chain",
                    );
                }
            }
        }));
    }

    for r in readers {
        r.join().expect("reader thread panicked under contention");
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writer
        .join()
        .expect("writer thread panicked under contention");

    // Only the warm chains remain: chains x nodes = 48 non-root nodes.
    assert_eq!(
        tree.node_count(),
        (WARM_CHAINS * 3) as usize,
        "writer's scratch churn must leave no residual nodes",
    );
    for c in 0..WARM_CHAINS {
        let m = tree.match_prefix(None, &warm_chain(c));
        assert_eq!(m.matched_blocks, 3);
        assert!(m.holds(&warm));
    }
}
