//! Abort-on-disconnect guard for in-flight requests.

use std::collections::HashSet;

use crate::message::ids::Rid;
use crate::tokenizer_manager::wiring::{AbortSource, Senders};

/// Aborts still-in-flight rids on drop. Each rid is disarmed on natural finish;
/// whatever remains at drop is aborted.
pub(super) struct AbortGuard {
    senders: Senders,
    /// Rids still in flight.
    rids: HashSet<Rid>,
}

impl AbortGuard {
    pub(super) fn new(senders: Senders, rid: Rid) -> Self {
        Self {
            senders,
            rids: HashSet::from([rid]),
        }
    }

    /// Guard covering no rids yet — a batch arms each as it is submitted.
    pub(super) fn new_empty(senders: Senders) -> Self {
        Self {
            senders,
            rids: HashSet::new(),
        }
    }

    /// Track a request for abort-on-drop.
    pub(super) fn arm(&mut self, rid: Rid) {
        self.rids.insert(rid);
    }

    /// Request finished naturally — don't abort it on drop.
    pub(super) fn disarm(&mut self, rid: &Rid) {
        self.rids.remove(rid);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        // Report the abort and nothing more.
        for rid in self.rids.drain() {
            let _ = self.senders.abort_tx.send(AbortSource::Guard(rid));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn senders_with_abort(abort: flume::Sender<AbortSource>) -> Senders {
        Senders {
            tok_manager_tx: flume::unbounded().0,
            abort_tx: abort,
            tokenizer_tx: flume::unbounded().0,
            detokenizer_tx: vec![],
        }
    }

    /// A batch guard aborts exactly the rids still armed at drop — the ones whose requests never reached a terminal —.
    #[test]
    fn guard_aborts_only_the_rids_still_armed() {
        let (abort_tx, abort_rx) = flume::unbounded();
        let done: Rid = "done".into();
        let mut guard = AbortGuard::new(senders_with_abort(abort_tx), done.clone());
        guard.arm("aborted".into());
        guard.disarm(&done); // finished naturally
        drop(guard);

        assert!(
            matches!(abort_rx.try_recv().unwrap(), AbortSource::Guard(r) if r.as_str() == "aborted")
        );
        assert!(
            abort_rx.try_recv().is_err(),
            "a disarmed rid must not be aborted"
        );
    }

    /// An armed guard aborts its rid on drop — exactly the cleanup a busy-skipped `/health_generate` probe relies on.
    #[test]
    fn armed_guard_aborts_on_drop() {
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(AbortGuard::new(senders_with_abort(tm_tx), "r7".into()));
        assert!(
            matches!(tm_rx.try_recv(), Ok(AbortSource::Guard(rid)) if rid.as_str() == "r7"),
            "armed guard must abort its rid on drop",
        );
        assert!(tm_rx.try_recv().is_err(), "exactly one abort");
    }

    /// A disarmed rid (finished naturally) is not aborted on drop.
    #[test]
    fn disarmed_guard_does_not_abort() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let id = Rid::from("r9");
        let mut guard = AbortGuard::new(senders_with_abort(tm_tx), "r9".into());
        guard.disarm(&id);
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }
}
