// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Independent `KvEventIndex` instances subscribed to the same PUB socket — the in-process surrogate.

use std::sync::Arc;
use std::time::Duration;

use zeromq::SocketSend;

use sgl_router::state::kv_events::discovery::EventConfig;
use sgl_router::state::kv_events::{compute_block_hashes, KvEventIndex, KvWorkerId};

use super::zmq_helpers::{
    build_multipart, encode_block_stored_event, encode_event_batch, make_pub_bound,
};

#[tokio::test]
async fn two_independent_subscribers_converge_to_same_tree_state() {
    // 1. One PUB socket — the worker. Both router surrogates connect to it.
    let (mut publisher, port) = make_pub_bound().await;
    let worker_url = "http://127.0.0.1:30000";
    let block_size = 4u32;
    let cfg = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port,
        topic: String::new(),
        block_size,
        dp_size: 1,
        load_port_base: None,
        load_topic: None,
        is_bigram: false,
    };

    let router_a = KvEventIndex::new();
    let router_b = KvEventIndex::new();
    router_a.add_worker(worker_url, Some(cfg.clone())).await;
    router_b.add_worker(worker_url, Some(cfg.clone())).await;

    // 3. Build a deterministic, multi-block event chain.
    let tokens: Vec<u32> = (0..16).collect();
    let hashes = compute_block_hashes(&tokens, block_size as usize);
    assert!(
        hashes.len() >= 3,
        "test needs ≥3 blocks; got {}",
        hashes.len()
    );
    let event_bytes = encode_block_stored_event(&hashes, None, &tokens, block_size);
    let payload = encode_event_batch(0.0, vec![event_bytes], Some(0));
    // 4. Republish until both subscribers observe the event.
    let target = hashes.len();
    let key = KvWorkerId {
        url: worker_url.into(),
        dp_rank: 0,
    };
    let start = std::time::Instant::now();
    let mut sequence = 1i64;
    loop {
        publisher
            .send(build_multipart(sequence, payload.clone()))
            .await
            .expect("publish BlockStored");
        let ma = router_a.tree().match_prefix(None, &hashes);
        let mb = router_b.tree().match_prefix(None, &hashes);
        let converged = ma.matched_blocks == target
            && mb.matched_blocks == target
            && ma.holds(&key)
            && mb.holds(&key);
        if converged {
            // Both trees agree on count AND on the worker that holds the
            // prefix.
            assert_eq!(
                ma.matched_blocks, mb.matched_blocks,
                "subscribers disagreed on matched_blocks",
            );
            assert_eq!(
                ma.workers(),
                mb.workers(),
                "subscribers disagreed on worker set",
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(3) {
            panic!(
                "subscribers did not converge within 3s: \
                 router_a={{matched={}, workers={:?}}}, \
                 router_b={{matched={}, workers={:?}}}, target={target}",
                ma.matched_blocks,
                ma.workers(),
                mb.matched_blocks,
                mb.workers(),
            );
        }
        sequence += 1;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let unseen: Vec<i64> = vec![999_999_999_001, 999_999_999_002, 999_999_999_003];
    let na = router_a.tree().match_prefix(None, &unseen);
    let nb = router_b.tree().match_prefix(None, &unseen);
    assert_eq!(na.matched_blocks, 0, "router_a leaked unpublished key");
    assert_eq!(nb.matched_blocks, 0, "router_b leaked unpublished key");

    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_a).shutdown()).await;
    assert!(r.is_ok(), "router_a shutdown hung");

    let t = std::time::Instant::now();
    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_b).shutdown()).await;
    assert!(r.is_ok(), "router_b shutdown hung");
    let elapsed = t.elapsed();
    assert!(
        elapsed < Duration::from_millis(100),
        "router_b shutdown after router_a drained took {elapsed:?}; \
         expected <100ms (no shared-resource contention)",
    );
}

/// PUB sockets ( workers) + `KvEventIndex` instances ( routers).
#[tokio::test]
async fn two_subscribers_merge_events_from_two_publishers() {
    let (mut pub_x, port_x) = make_pub_bound().await;
    let (mut pub_y, port_y) = make_pub_bound().await;
    let worker_x = "http://127.0.0.1:30001";
    let worker_y = "http://127.0.0.1:30002";
    let block_size = 4u32;
    let cfg_x = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port_x,
        topic: String::new(),
        block_size,
        dp_size: 1,
        load_port_base: None,
        load_topic: None,
        is_bigram: false,
    };
    let cfg_y = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port_y,
        topic: String::new(),
        block_size,
        dp_size: 1,
        load_port_base: None,
        load_topic: None,
        is_bigram: false,
    };

    // Both routers subscribe to BOTH workers — the production fan-out.
    let router_a = KvEventIndex::new();
    let router_b = KvEventIndex::new();
    router_a.add_worker(worker_x, Some(cfg_x.clone())).await;
    router_a.add_worker(worker_y, Some(cfg_y.clone())).await;
    router_b.add_worker(worker_x, Some(cfg_x.clone())).await;
    router_b.add_worker(worker_y, Some(cfg_y.clone())).await;

    // Non-overlapping token streams → distinct hash chains.
    let tokens_x: Vec<u32> = (0..16).collect();
    let tokens_y: Vec<u32> = (1000..1016).collect();
    let hashes_x = compute_block_hashes(&tokens_x, block_size as usize);
    let hashes_y = compute_block_hashes(&tokens_y, block_size as usize);
    assert!(hashes_x.len() >= 3 && hashes_y.len() >= 3);

    let payload_x = encode_event_batch(
        0.0,
        vec![encode_block_stored_event(
            &hashes_x, None, &tokens_x, block_size,
        )],
        Some(0),
    );
    let payload_y = encode_event_batch(
        0.0,
        vec![encode_block_stored_event(
            &hashes_y, None, &tokens_y, block_size,
        )],
        Some(0),
    );
    let key_x = KvWorkerId {
        url: worker_x.into(),
        dp_rank: 0,
    };
    let key_y = KvWorkerId {
        url: worker_y.into(),
        dp_rank: 0,
    };
    let target_x = hashes_x.len();
    let target_y = hashes_y.len();

    let start = std::time::Instant::now();
    let mut sequence = 1i64;
    loop {
        pub_x
            .send(build_multipart(sequence, payload_x.clone()))
            .await
            .expect("publish on pub_x");
        pub_y
            .send(build_multipart(sequence, payload_y.clone()))
            .await
            .expect("publish on pub_y");
        let ax = router_a.tree().match_prefix(None, &hashes_x);
        let ay = router_a.tree().match_prefix(None, &hashes_y);
        let bx = router_b.tree().match_prefix(None, &hashes_x);
        let by = router_b.tree().match_prefix(None, &hashes_y);
        let converged = ax.matched_blocks == target_x
            && ay.matched_blocks == target_y
            && bx.matched_blocks == target_x
            && by.matched_blocks == target_y
            && ax.holds(&key_x)
            && ay.holds(&key_y)
            && bx.holds(&key_x)
            && by.holds(&key_y);
        if converged {
            // Negative attribution: prefix X must not be attributed to
            // worker_y in either tree, and vice versa.
            assert!(
                !ax.holds(&key_y),
                "router_a cross-attributed worker_y to prefix X: {:?}",
                ax.workers(),
            );
            assert!(
                !ay.holds(&key_x),
                "router_a cross-attributed worker_x to prefix Y: {:?}",
                ay.workers(),
            );
            assert!(
                !bx.holds(&key_y),
                "router_b cross-attributed worker_y to prefix X: {:?}",
                bx.workers(),
            );
            assert!(
                !by.holds(&key_x),
                "router_b cross-attributed worker_x to prefix Y: {:?}",
                by.workers(),
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(3) {
            panic!(
                "trees did not converge within 3s:\n  \
                 router_a: X={{matched={}, workers={:?}}}, Y={{matched={}, workers={:?}}}\n  \
                 router_b: X={{matched={}, workers={:?}}}, Y={{matched={}, workers={:?}}}\n  \
                 targets: X={target_x}, Y={target_y}",
                ax.matched_blocks,
                ax.workers(),
                ay.matched_blocks,
                ay.workers(),
                bx.matched_blocks,
                bx.workers(),
                by.matched_blocks,
                by.workers(),
            );
        }
        sequence += 1;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_a).shutdown()).await;
    assert!(r.is_ok(), "router_a shutdown hung");
    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_b).shutdown()).await;
    assert!(r.is_ok(), "router_b shutdown hung");
}
