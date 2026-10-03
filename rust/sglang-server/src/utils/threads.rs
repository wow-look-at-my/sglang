//! Thread-group machinery: CPU-core partitioning and the pinned-thread spawners (for [`Runnable`] stages) used.

use std::thread::JoinHandle;
use std::time::Duration;

use core_affinity::CoreId;

use super::runtime::Runnable;
use crate::message::config::RuntimeConfig;

/// Cores reserved for both TokenizerManager router threads (`to-scheduler`, `from-scheduler`) — light.
const TM_CORES: usize = 2;

/// Partition the machine's cores into disjoint sets: the I/O-bound API
/// pool, the CPU-bound tokenizer.
pub(super) struct CorePlan {
    pub(super) api: Vec<CoreId>,
    pub(super) tok: Vec<CoreId>,
    pub(super) detok: Vec<CoreId>,
    pub(super) tm: Vec<CoreId>,
}

pub(super) fn plan_cores(cfg: &RuntimeConfig) -> Option<CorePlan> {
    // `cores` carries the pinning decision: `None`/empty → run unpinned.
    let cores: Vec<CoreId> = match &cfg.rust_server_args.cores {
        Some(ids) if !ids.is_empty() => ids.iter().map(|&id| CoreId { id }).collect(),
        _ => return None,
    };
    if cores.len()
        < cfg.rust_server_args.http_api_worker_num
            + cfg.server_args.tokenizer_worker_num
            + cfg.server_args.detokenizer_worker_num
    {
        tracing::warn!(
            available = cores.len(),
            "not enough cores to pin all pools; running unpinned"
        );
        return None;
    }
    let mut it = cores.into_iter();
    let api: Vec<CoreId> = it
        .by_ref()
        .take(cfg.rust_server_args.http_api_worker_num)
        .collect();
    let tok = it
        .by_ref()
        .take(cfg.server_args.tokenizer_worker_num)
        .collect();
    let detok = it
        .by_ref()
        .take(cfg.server_args.detokenizer_worker_num)
        .collect();
    // The TM router threads get up to `TM_CORES` leftover cores.
    let mut tm: Vec<CoreId> = it.by_ref().take(TM_CORES).collect();
    if tm.is_empty() {
        tm = api.clone();
    }
    Some(CorePlan {
        api,
        tok,
        detok,
        tm,
    })
}

/// Pin the calling thread to `core` if one was assigned (no-op otherwise).
fn pin_current(core: Option<CoreId>) {
    if let Some(c) = core {
        core_affinity::set_for_current(c);
    }
}

/// Pick the pinned core for worker `i` from an optional pool core set.
pub(super) fn pool_core(cores: &Option<Vec<CoreId>>, i: usize) -> Option<CoreId> {
    cores.as_ref().and_then(|c| c.get(i).copied())
}

/// Spawn a single [`Runnable`] stage on a named thread, optionally pinned.
/// Used by [`spawn_pool`]; every group goes through the pool spawner now.
fn spawn_stage(
    name: &str,
    core: Option<CoreId>,
    stage: impl Runnable,
    threads: &mut Vec<JoinHandle<()>>,
) {
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            pin_current(core);
            stage.run();
        })
        .expect("spawn stage");
    threads.push(handle);
}

/// Spawn a pool of `count` [`Runnable`] workers, each pinned to `cores[i]` (when
/// available) and named `{name}-{i}`. `build(i)` constructs worker `i` — cloning
/// shared handles, or moving a per-worker resource out of a captured iterator.
pub(super) fn spawn_pool<R, F>(
    name: &str,
    cores: Option<Vec<CoreId>>,
    count: usize,
    threads: &mut Vec<JoinHandle<()>>,
    mut build: F,
) where
    R: Runnable,
    F: FnMut(usize) -> R,
{
    for i in 0..count {
        let core = pool_core(&cores, i);
        spawn_stage(&format!("{name}-{i}"), core, build(i), threads);
    }
}

/// Join every handle, giving up after `timeout`.
pub(super) fn join_all_with_timeout(handles: Vec<JoinHandle<()>>, timeout: Duration) -> bool {
    if handles.is_empty() {
        return true;
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        for h in handles {
            let _ = h.join();
        }
        let _ = done_tx.send(());
    });
    done_rx.recv_timeout(timeout).is_ok()
}
