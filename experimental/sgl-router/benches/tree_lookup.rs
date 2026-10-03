// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware tree-lookup microbench.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sgl_router::state::kv_events::tree::{HashTree, KvWorkerId};

fn build_tree(num_workers: usize, blocks_per_worker: usize, seed: u64) -> HashTree {
    let tree = HashTree::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for w in 0..num_workers {
        let worker = KvWorkerId::new(format!("http://w{w}:30000"), 0);
        // Each worker holds a distinct (random) prefix so the trees fan out — this is the realistic case.
        let hashes: Vec<i64> = (0..blocks_per_worker).map(|_| rng.gen::<i64>()).collect();
        tree.insert(&worker, None, &hashes);
    }
    tree
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_insert");
    for &n_blocks in &[8usize, 32, 128, 512] {
        group.throughput(Throughput::Elements(n_blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_blocks), &n_blocks, |b, &n| {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE);
            let hashes: Vec<i64> = (0..n).map(|_| rng.gen::<i64>()).collect();
            b.iter_batched(
                HashTree::new,
                |tree| {
                    let worker = KvWorkerId::new("http://w:30000".to_string(), 0);
                    tree.insert(&worker, None, black_box(&hashes));
                    tree
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Cost of an insert that carries a parent, against one that does not.
/// `route_insert` early-returns on `parent_hash = None` and writes a single
/// shard; a `Some(p)` takes a read lock on every shard to find which already
/// holds `p`. The pump passes `Some` for every block after a sequence's
/// first, so that is the steady-state write path.
/// suite would time only the shape the pump rarely sends. The scan is real
/// but small next to the descent it precedes (~36ns against ~800ns measured),
/// and the unsharded tree shows no gap between both cases at all, having
/// no shards to scan.
fn bench_insert_continuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_insert_continuation");
    let tree = build_tree(64, 64, 0xDEADBEEF);
    let worker = KvWorkerId::new("http://w0:30000".to_string(), 0);
    let chain: Vec<i64> = {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        (0..64).map(|_| rng.gen::<i64>()).collect()
    };
    let (head, tail) = chain.split_at(32);
    let parent_hash = head[head.len() - 1];

    group.throughput(Throughput::Elements(tail.len() as u64));
    group.bench_function("parent_none", |b| {
        b.iter(|| tree.insert(&worker, None, black_box(head)))
    });
    group.bench_function("parent_some", |b| {
        b.iter(|| tree.insert(&worker, Some(black_box(parent_hash)), black_box(tail)))
    });
    group.finish();
}

fn bench_match_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_match_prefix");
    // (workers, blocks_per_worker, query_len) cases that span the realistic
    // operating window: small fleet w/ moderate prefixes.
    let cases = [
        (4usize, 32usize, 8usize),
        (16, 64, 32),
        (64, 128, 64),
        (128, 256, 128),
    ];
    for (workers, bpw, query_len) in cases {
        let label = format!("w{workers}_bpw{bpw}_q{query_len}");
        group.throughput(Throughput::Elements(query_len as u64));
        let tree = build_tree(workers, bpw, 0xDEADBEEF);
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let probe: Vec<i64> = (0..query_len).map(|_| rng.gen::<i64>()).collect();
        group.bench_function(label, |b| {
            b.iter(|| {
                let m = tree.match_prefix(None, black_box(&probe));
                black_box(m.matched_blocks)
            });
        });
    }
    group.finish();
}

/// How the background writer shapes its inserts.
#[derive(Clone, Copy)]
enum WriterShape {
    Rooted,
    Continuation,
}

impl WriterShape {
    fn label(self) -> &'static str {
        match self {
            Self::Rooted => "reader_under_write_pressure",
            Self::Continuation => "reader_under_continuation_write_pressure",
        }
    }
}

/// Reader `match_prefix` throughput while a background writer hammers
/// `insert` / `remove` — the read-vs-write contention sharding is built
/// for.
fn bench_contended_match(c: &mut Criterion) {
    for shape in [WriterShape::Rooted, WriterShape::Continuation] {
        bench_contended_match_with(c, shape);
    }
}

fn bench_contended_match_with(c: &mut Criterion, shape: WriterShape) {
    let mut group = c.benchmark_group("hashtree_contended_match");
    let tree = Arc::new(build_tree(64, 64, 0xDEADBEEF));
    let warm_chain: Vec<i64> = {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        (0..64).map(|_| rng.gen::<i64>()).collect()
    };

    // One background writer, insert + remove of a fresh 4-block chain per round, so it keeps taking write locks.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let tree = tree.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let scratch = KvWorkerId::new("http://scratch:30000".to_string(), 0);
            let mut round = 0i64;
            while !stop.load(Ordering::Relaxed) {
                // Cycled, so a long run cannot drift the scratch roots into the warm chain's space.
                let base = 1_000_000 + (round % 100_000) * 7;
                let chain = [base, base + 1, base + 2, base + 3];
                match shape {
                    WriterShape::Rooted => tree.insert(&scratch, None, &chain),
                    // Root the chain, then extend it the way the pump does:
                    // only a sequence's first block carries no parent.
                    WriterShape::Continuation => {
                        tree.insert(&scratch, None, &chain[..1]);
                        tree.insert(&scratch, Some(base), &chain[1..]);
                    }
                }
                tree.remove(&scratch, &chain);
                round = round.wrapping_add(1);
            }
        })
    };
    // Signals the writer on the way out however we leave — a panic in the bench body unwinds past any explicit store.
    let _stop_writer = StopOnDrop(stop);

    group.throughput(Throughput::Elements(warm_chain.len() as u64));
    group.bench_function(shape.label(), |b| {
        b.iter(|| {
            let m = tree.match_prefix(None, black_box(&warm_chain));
            black_box(m.matched_blocks)
        });
    });

    drop(_stop_writer);
    writer.join().expect("bench writer thread panicked");
    group.finish();
}

/// Sets its flag on drop, so a background thread parked on it is stopped by an unwind as reliably.
struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

criterion_group!(
    benches,
    bench_insert,
    bench_insert_continuation,
    bench_match_prefix,
    bench_contended_match
);
criterion_main!(benches);
