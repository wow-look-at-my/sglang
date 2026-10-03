//! Messages to a Detokenizer shard.

use super::ids::Rid;
use super::response::{ChunkEvent, ResponseSink};

/// Messages to a Detokenizer shard. `Register` carries the per-request sink
/// for the shard's local `rid -> sink` map.
pub enum DetokMsg {
    Register {
        /// Client-visible rid string — kept in `DetokState` so the shard can emit `TmEvent::Abort(rid)`.
        rid: Rid,
        sink: ResponseSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
    },
    /// One decode step's chunks for *this shard*. Batched because `from-scheduler` blocks per send.
    Chunks(Vec<ChunkEvent>),
    /// Decode a complete token-id sequence — the backend of [`RequestKind::Detokenize`](super::RequestKind::Detokenize).
    Decode { rid: Rid, token_ids: Vec<u32> },
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result { rid: Rid, payload: bytes::Bytes },
    Fail { rid: Rid, message: String },
    /// Drop the `rid -> sink` entry for a request rejected before the scheduler (the rejecting stage already answered the client).
    Deregister { rid: Rid },
}
