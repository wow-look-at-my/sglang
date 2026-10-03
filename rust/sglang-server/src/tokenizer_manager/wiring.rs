//! The flume fabric between stages: the request-loop inbox ([`TmEvent`]), the abort lane ([`AbortSource`]), the producer-side handles.

use crate::message::detok::DetokMsg;
use crate::message::ids::Rid;
use crate::message::request::Request;

/// Blocking receive that also wakes on shutdown: returns `None` when `rx` closes
/// *or* the `shutdown` sender is dropped.
pub fn recv<T>(rx: &flume::Receiver<T>, shutdown: &flume::Receiver<()>) -> Option<T> {
    flume::Selector::new()
        .recv(rx, |r| r.ok())
        .recv(shutdown, |_| None)
        .wait()
}

/// Events into the TokenizerManager request loop.
pub enum TmEvent {
    /// A freshly received request from the API server.
    Intake(Request),
    /// A request back from the tokenizer pool: `PreSendValidating` (ids filled) on success, or `Failed`.
    Tokenized(Request),
    /// An MM worker finished a request parked in `Encoding`: `input_ids` are the final placeholder-expanded prompt ids.
    MmEncoded { rid: Rid, input_ids: Vec<i32> },
    /// An MM worker rejected a request parked in `Encoding`.
    MmFailed { rid: Rid, message: String },
}

/// The source of the abort request.
#[derive(Clone, Debug)]
pub enum AbortSource {
    /// From an `AbortGuard` drop. Owns the release.
    Guard(Rid),
    /// From a detokenizer terminal path. Aborts the scheduler work.
    Detok(Rid),
}

impl AbortSource {
    pub fn rid(&self) -> &Rid {
        match self {
            Self::Guard(rid) | Self::Detok(rid) => rid,
        }
    }
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager loop.
    pub tok_manager_tx: flume::Sender<TmEvent>,
    /// → the same loop, but UNBOUNDED and abort-only.
    pub abort_tx: flume::Sender<AbortSource>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tokenizer_tx: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `Rid::shard(detok.len())`.
    pub detokenizer_tx: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, rid: &Rid) -> &flume::Sender<DetokMsg> {
        &self.detokenizer_tx[rid.shard(self.detokenizer_tx.len())]
    }
}
