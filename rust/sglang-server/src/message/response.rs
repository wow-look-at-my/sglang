//! The response direction: the per-request back-channel the API handler drains ([`ResponseSink`] / [`ResponseItem`]).

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::finish_reason::FinishReason;
use super::types::TokenIds;
use crate::message::ids::Rid;
use crate::utils::error::Error;

/// Per-request back-channel the detok shard writes decode frames to and the API handler drains for SSE; bounded.
#[derive(Clone, Debug)]
pub enum ResponseSink {
    Local(mpsc::Sender<ResponseItem>),
}

/// Why an [`ResponseSink::try_send`] failed: `Full` = client backpressure, `Closed` = client gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkError {
    Full,
    Closed,
}

impl ResponseSink {
    /// Non-blocking send. `Err(Full)` = backpressure, `Err(Closed)` = client gone.
    pub fn try_send(&self, item: ResponseItem) -> Result<(), SinkError> {
        match self {
            ResponseSink::Local(tx) => tx.try_send(item).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => SinkError::Full,
                mpsc::error::TrySendError::Closed(_) => SinkError::Closed,
            }),
        }
    }
}

#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type ResponseSource = mpsc::Receiver<ResponseItem>;

/// What the connection handler receives on the decode stream: a detok-decoded [`ChunkEvent`] (handler formats it).
#[derive(Debug)]
pub enum ResponseItem {
    /// An intermediate streamed generation step (only sent for streaming reqs).
    Frame(ChunkEvent),
    /// The final generation step.
    Done(ChunkEvent),
    /// A control-request result: one verbatim payload (e.g. `/server_info`), delivered as-is.
    Control(Bytes),
    /// Reply to an internal service request (`RequestKind::Detokenize`): raw bytes for the SUBMITTER to consume.
    Data(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// Response frame tag (first byte, prepended Rust-side.
pub const DISPATCH_TAG_RESULT: u8 = 1;
/// A whole decode batch: msgpack columnar header + one concatenated raw buffer.
pub const DISPATCH_TAG_BATCH: u8 = 2;
/// A per-request failure `[rid, message]`: the Python drain couldn't decode a header.
pub const DISPATCH_TAG_ERROR: u8 = 3;

/// Read `n` little-endian f32s from `data` at `*off`, advancing `*off`.
/// `None` when the range runs past the buffer (a malformed /
/// positional-ABI-drifted frame): the caller rejects the whole frame.
fn take_f32(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<f32>> {
    let start = *off;
    let end = start.checked_add(n.checked_mul(4)?)?;
    let out = data
        .get(start..end)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    Some(out)
}

/// Read `n` little-endian i32s from `data` at `*off`, advancing `*off`. `None` past
/// the buffer end (see [`take_f32`]).
fn take_i32(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<i32>> {
    let start = *off;
    let end = start.checked_add(n.checked_mul(4)?)?;
    let out = data
        .get(start..end)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    Some(out)
}

/// Frame a decode batch: `[BATCH tag][u32 header len][header][data cols…]`. The
/// caller's `data_cols` are concatenated straight into the frame (one copy, no
/// `b"".join`); `header` is the msgpack [`BatchHeader`].
pub fn frame_decode_batch_cols(header: &[u8], data_cols: &[&[u8]]) -> Bytes {
    let data_len: usize = data_cols.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(1 + 4 + header.len() + data_len);
    buf.push(DISPATCH_TAG_BATCH);
    buf.extend_from_slice(&(header.len() as u32).to_le_bytes());
    buf.extend_from_slice(header);
    for col in data_cols {
        buf.extend_from_slice(col);
    }
    Bytes::from(buf)
}

/// Columnar scalar header for a whole decode batch.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BatchHeader {
    /// Request ids, as the same strings Python holds (`Req.rid`, uuid hex) — hashed back to the internal routing key in `decode_one`.
    pub rids: Vec<String>,
    pub finish_reasons: Vec<Option<FinishReason>>,
    pub prompt_tokens: Vec<u32>,
    pub tok_lens: Vec<u32>,
    #[serde(default)]
    pub out_lp_lens: Vec<u32>,
    #[serde(default)]
    pub in_lp_lens: Vec<u32>,
    #[serde(default)]
    pub out_top_reqlens: Vec<u32>,
    #[serde(default)]
    pub out_top_poslens: Vec<u32>,
    #[serde(default)]
    pub in_top_reqlens: Vec<u32>,
    #[serde(default)]
    pub in_top_poslens: Vec<u32>,
    #[serde(default)]
    pub out_tokids_lp_reqlens: Vec<u32>,
    #[serde(default)]
    pub out_tokids_lp_poslens: Vec<u32>,
    #[serde(default)]
    pub in_tokids_lp_reqlens: Vec<u32>,
    #[serde(default)]
    pub in_tokids_lp_poslens: Vec<u32>,
    #[serde(default)]
    pub hidden_reqlens: Vec<u32>,
    #[serde(default)]
    pub hidden_poslens: Vec<u32>,
}

/// Read a request's flat logprob column (`l` val/idx pairs) from `data` at
/// cursors `cv`/`ci`, advancing them.
fn take_flat(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    l: usize,
) -> Option<(Vec<f32>, Vec<i32>)> {
    Some((take_f32(data, cv, l)?, take_i32(data, ci, l)?))
}

/// Read `np` per-position lengths from `poslens` at `*pcur`, advancing it.
fn take_poslens(poslens: &[u32], pcur: &mut usize, np: usize) -> Option<Vec<u32>> {
    let start = *pcur;
    let end = start.checked_add(np)?;
    let lens = poslens.get(start..end)?.to_vec();
    *pcur = end;
    Some(lens)
}

/// Read a request's ragged logprob column (`np` positions): its per-position `lens`
/// from `poslens` (advancing `pcur`), then that many val/idx from `cv`/`ci`. `None`
/// if either read runs past its buffer (see [`take_poslens`], [`take_f32`]).
fn take_ragged(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    np: usize,
) -> Option<(Vec<f32>, Vec<i32>, Vec<u32>)> {
    let lens = take_poslens(poslens, pcur, np)?;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, take_i32(data, ci, nv)?, lens))
}

/// Like [`take_ragged`] but for hidden states — a val column + row `poslens`, no
/// idx column. `None` if either read runs past its buffer (see [`take_poslens`],
/// [`take_f32`]).
fn take_hidden(
    data: &[u8],
    cv: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    nr: usize,
) -> Option<(Vec<f32>, Vec<u32>)> {
    let lens = take_poslens(poslens, pcur, nr)?;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, lens))
}

/// Decode a batch frame (tag stripped), calling `route` with each request's
/// [`ChunkEvent`] as it's decoded — one pass, no intermediate `Vec`, peak memory
/// one request. Column order matches `push_generation`.
///
/// `ok == false` means the frame was rejected. The caller discards everything it
/// routed and fails the frame's requests instead of forwarding a partial fan-out
/// (see `tokenizer_manager::from_scheduler`), so a rejected frame delivers nothing —
/// `rids` exists precisely so those requests can be failed rather than left
/// waiting for a `Done` that no longer exists.
pub fn for_each_chunk(body: &[u8], mut route: impl FnMut(ChunkEvent)) -> Decoded {
    let mut decoded = Decoded::default();
    if body.len() < 4 {
        return decoded;
    }
    let hlen = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let Some(header) = body.get(4..4 + hlen) else {
        return decoded;
    };
    // Every rejection past the header slice names its requests by re-reading
    // the rid column, rather than the whole frame paying a clone.
    macro_rules! reject {
        () => {{
            decoded.rids = recover_rids(header);
            return decoded;
        }};
    }

    let data = &body[4 + hlen..];
    let mut h = match rmp_serde::from_slice::<BatchHeader>(header) {
        Ok(h) => h,
        Err(_) => reject!(),
    };
    let n = h.rids.len();
    // Every column must agree with `rids`.
    if h.finish_reasons.len() != n || h.prompt_tokens.len() != n || h.tok_lens.len() != n {
        reject!()
    }
    // The per-request extras columns are either absent (no request asked) or one entry per request — never partial.
    let per_req_ok = |c: &[u32]| c.is_empty() || c.len() == n;
    if !per_req_ok(&h.out_lp_lens)
        || !per_req_ok(&h.in_lp_lens)
        || !per_req_ok(&h.out_top_reqlens)
        || !per_req_ok(&h.in_top_reqlens)
        || !per_req_ok(&h.out_tokids_lp_reqlens)
        || !per_req_ok(&h.in_tokids_lp_reqlens)
        || !per_req_ok(&h.hidden_reqlens)
    {
        reject!()
    }
    let sum = |v: &[u32]| v.iter().map(|&x| x as usize).sum::<usize>();
    // Each ragged family's per-request counts must consume its position column
    // EXACTLY.
    if sum(&h.out_top_reqlens) != h.out_top_poslens.len()
        || sum(&h.in_top_reqlens) != h.in_top_poslens.len()
        || sum(&h.out_tokids_lp_reqlens) != h.out_tokids_lp_poslens.len()
        || sum(&h.in_tokids_lp_reqlens) != h.in_tokids_lp_poslens.len()
        || sum(&h.hidden_reqlens) != h.hidden_poslens.len()
    {
        reject!()
    }

    // Per-column byte cursors, advanced per request — no whole-column read.
    let mut base = 0usize;
    let mut col = |count: usize| -> usize {
        let start = base;
        base += count * 4;
        start
    };
    // Each val/idx column pair shares one element count — sum it once.
    let n_ids = sum(&h.tok_lens);
    let n_olp = sum(&h.out_lp_lens);
    let n_ilp = sum(&h.in_lp_lens);
    let n_ot = sum(&h.out_top_poslens);
    let n_it = sum(&h.in_top_poslens);
    let n_od = sum(&h.out_tokids_lp_poslens);
    let n_id = sum(&h.in_tokids_lp_poslens);
    let n_h = sum(&h.hidden_poslens);
    let mut c_ids = col(n_ids);
    let mut c_olp_v = col(n_olp);
    let mut c_olp_i = col(n_olp);
    let mut c_ilp_v = col(n_ilp);
    let mut c_ilp_i = col(n_ilp);
    let mut c_ot_v = col(n_ot);
    let mut c_ot_i = col(n_ot);
    let mut c_it_v = col(n_it);
    let mut c_it_i = col(n_it);
    let mut c_od_v = col(n_od);
    let mut c_od_i = col(n_od);
    let mut c_id_v = col(n_id);
    let mut c_id_i = col(n_id);
    let mut c_h_v = col(n_h);

    // `col` summed every column's span into `base`.
    if base != data.len() {
        reject!()
    }

    // Mirror of Python's `has_extra` guard: checking once per frame lets the
    // per-request loop skip the extras machinery entirely on a plain decode frame.
    let has_extras = !(h.out_lp_lens.is_empty()
        && h.in_lp_lens.is_empty()
        && h.out_top_reqlens.is_empty()
        && h.in_top_reqlens.is_empty()
        && h.out_tokids_lp_reqlens.is_empty()
        && h.in_tokids_lp_reqlens.is_empty()
        && h.hidden_reqlens.is_empty());

    // Position cursors into the header's per-request `poslens` (ragged + hidden).
    let (mut p_ot, mut p_it, mut p_od, mut p_id, mut p_h) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let lens_i = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0) as usize;

    // Decode one request's slice of every column, advancing the cursors. `None` if a
    // read overruns `data` (belt-and-suspenders past the upfront check, which already
    // covers every column's span — so this is unreachable today). It stops the loop
    // before slicing out of bounds, but requests decoded earlier in the frame are
    // already routed; making that abort atomic would mean buffering the whole frame,
    // which is exactly what the streaming decode avoids.
    let mut decode_one = |i: usize| -> Option<ChunkEvent> {
        let token_ids = take_i32(data, &mut c_ids, lens_i(&h.tok_lens, i))?;

        // Plain decode frame (no request in the batch asked for logprobs/hidden):
        // the extras columns are all zero-width, so skip reading them entirely.
        let extras = if !has_extras {
            None
        } else {
            let (out_lp_val, out_lp_idx) =
                take_flat(data, &mut c_olp_v, &mut c_olp_i, lens_i(&h.out_lp_lens, i))?;
            let (in_lp_val, in_lp_idx) =
                take_flat(data, &mut c_ilp_v, &mut c_ilp_i, lens_i(&h.in_lp_lens, i))?;
            let (out_top_val, out_top_idx, out_top_lens) = take_ragged(
                data,
                &mut c_ot_v,
                &mut c_ot_i,
                &h.out_top_poslens,
                &mut p_ot,
                lens_i(&h.out_top_reqlens, i),
            )?;
            let (in_top_val, in_top_idx, in_top_lens) = take_ragged(
                data,
                &mut c_it_v,
                &mut c_it_i,
                &h.in_top_poslens,
                &mut p_it,
                lens_i(&h.in_top_reqlens, i),
            )?;
            let (out_tid_val, out_tid_idx, out_tid_lens) = take_ragged(
                data,
                &mut c_od_v,
                &mut c_od_i,
                &h.out_tokids_lp_poslens,
                &mut p_od,
                lens_i(&h.out_tokids_lp_reqlens, i),
            )?;
            let (in_tid_val, in_tid_idx, in_tid_lens) = take_ragged(
                data,
                &mut c_id_v,
                &mut c_id_i,
                &h.in_tokids_lp_poslens,
                &mut p_id,
                lens_i(&h.in_tokids_lp_reqlens, i),
            )?;
            let (hidden_val, hidden_lens) = take_hidden(
                data,
                &mut c_h_v,
                &h.hidden_poslens,
                &mut p_h,
                lens_i(&h.hidden_reqlens, i),
            )?;

            // Even in an extras batch, most requests carry none — box only
            // if this does, so its `ChunkEvent` stays the small common frame.
            let ex = ChunkExtras {
                out_lp_val,
                out_lp_idx,
                in_lp_val,
                in_lp_idx,
                out_top_val,
                out_top_idx,
                out_top_lens,
                in_top_val,
                in_top_idx,
                in_top_lens,
                out_tid_val,
                out_tid_idx,
                out_tid_lens,
                in_tid_val,
                in_tid_idx,
                in_tid_lens,
                hidden_val,
                hidden_lens,
                // Explicit, NOT `..Default::default()` — same reason as `ChunkEvent` below: a new column must fail to compile here.
                out_lp_txt: Vec::new(),
                in_lp_txt: Vec::new(),
                out_top_txt: Vec::new(),
                in_top_txt: Vec::new(),
                out_tid_txt: Vec::new(),
                in_tid_txt: Vec::new(),
            };
            (!ex.is_empty()).then(|| Box::new(ex))
        };

        Some(ChunkEvent {
            // Any string is a valid rid; hash to the routing key.
            rid: std::mem::take(&mut h.rids[i]).into(),
            token_ids,
            finish_reason: h.finish_reasons.get(i).cloned().flatten(),
            prompt_tokens: h.prompt_tokens.get(i).copied().unwrap_or(0),
            extras,
            // Listed explicitly, NOT `..Default::default()`: a new column added to `ChunkEvent` and wired into the response must fail to compile here.
            text: String::new(),
            completion_tokens: 0,
        })
    };

    for i in 0..n {
        let Some(ev) = decode_one(i) else { reject!() };
        route(ev);
    }
    decoded.ok = true;
    decoded
}

/// Recover the rid column straight from the header bytes. Read through `rmpv`
/// `Err(LengthMismatch(1))` every time and named nobody.
fn recover_rids(header: &[u8]) -> Vec<Rid> {
    rmpv::decode::read_value(&mut &header[..])
        .ok()
        .and_then(|v| match v {
            rmpv::Value::Array(cols) => cols.into_iter().next(),
            _ => None,
        })
        .and_then(|c| match c {
            rmpv::Value::Array(rids) => Some(
                rids.iter()
                    .filter_map(|r| r.as_str().map(Rid::from))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

/// Outcome of [`for_each_chunk`]: whether the frame was accepted, plus the rids it named.
#[derive(Debug, Default)]
pub struct Decoded {
    pub ok: bool,
    pub rids: Vec<Rid>,
}

/// Frame a control result `[rid, payload]` for the response ring (tag prepended).
pub fn frame_control_result(rid: &str, payload: &[u8]) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::Binary(payload.to_vec())]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 8);
    buf.push(DISPATCH_TAG_RESULT);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// Frame a per-request failure `[rid, message]` for the response — routes a
/// terminal error back to the owning request (→ HTTP 400) instead of crashing.
pub fn frame_error(rid: &str, message: &str) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::from(message)]);
    let mut buf = Vec::with_capacity(1 + rid.len() + message.len() + 8);
    buf.push(DISPATCH_TAG_ERROR);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// One scheduler output increment — the common, always-present frame.
#[derive(Debug, Clone, Default)]
pub struct ChunkEvent {
    /// Client-visible rid — the request's IDENTITY.
    pub rid: Rid,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: TokenIds,
    /// `None` while streaming, the [`FinishReason`] on the final chunk.
    pub finish_reason: Option<FinishReason>,
    /// Prompt token count for this request (constant across its chunks).
    pub prompt_tokens: u32,
    /// Decoded text **delta** for this chunk (empty in skip mode / on partial UTF-8), filled by the detok shard.
    pub text: String,
    pub completion_tokens: u64,
    /// Logprob + hidden-state columns — `None` unless the request asked for them.
    pub extras: Option<Box<ChunkExtras>>,
}

/// Logprob + hidden-state columns for a [`ChunkEvent`], allocated only.
#[derive(Debug, Clone, Default)]
pub struct ChunkExtras {
    /// Output-token logprobs (parallel `val`/`idx`, one entry per new output token).
    pub out_lp_val: Vec<f32>,
    pub out_lp_idx: Vec<i32>,
    /// Input (prefill) token logprobs, sent once on the first chunk.
    pub in_lp_val: Vec<f32>,
    pub in_lp_idx: Vec<i32>,
    pub out_top_val: Vec<f32>,
    pub out_top_idx: Vec<i32>,
    pub out_top_lens: Vec<u32>,
    pub in_top_val: Vec<f32>,
    pub in_top_idx: Vec<i32>,
    pub in_top_lens: Vec<u32>,
    /// Token-ids logprobs (same ragged layout); set only when `token_ids_logprob` was.
    pub out_tid_val: Vec<f32>,
    pub out_tid_idx: Vec<i32>,
    pub out_tid_lens: Vec<u32>,
    pub in_tid_val: Vec<f32>,
    pub in_tid_idx: Vec<i32>,
    pub in_tid_lens: Vec<u32>,
    /// Hidden states (dense f32): flat buffer + per-row lengths.
    pub hidden_val: Vec<f32>,
    pub hidden_lens: Vec<u32>,
    /// Decoded logprob token text (`return_text_in_logprobs`), parallel to the `*_idx` buffers.
    pub out_lp_txt: Vec<String>,
    pub in_lp_txt: Vec<String>,
    pub out_top_txt: Vec<String>,
    pub in_top_txt: Vec<String>,
    pub out_tid_txt: Vec<String>,
    pub in_tid_txt: Vec<String>,
}

impl ChunkExtras {
    /// True when no logprob / hidden column carries data — lets the decoder skip the
    /// box allocation for the common (extras-free) frame.
    fn is_empty(&self) -> bool {
        self.out_lp_val.is_empty()
            && self.in_lp_val.is_empty()
            && self.out_top_lens.is_empty()
            && self.in_top_lens.is_empty()
            && self.out_tid_lens.is_empty()
            && self.in_tid_lens.is_empty()
            && self.hidden_lens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::{FinishKind, Matched};

    #[test]
    fn batch_cols_match_single_joined_buffer() {
        let header = [1u8, 2, 3];
        let a = [10u8, 11];
        let b = [12u8, 13, 14];
        let multi = frame_decode_batch_cols(&header, &[&a[..], &b[..]]);
        let joined: Vec<u8> = a.iter().chain(&b).copied().collect();
        let single = frame_decode_batch_cols(&header, &[joined.as_slice()]);
        assert_eq!(multi, single);
        assert_eq!(multi[0], DISPATCH_TAG_BATCH);
        assert_eq!(
            u32::from_le_bytes([multi[1], multi[2], multi[3], multi[4]]),
            3
        );
        assert_eq!(&multi[5..8], &header); // header
        assert_eq!(&multi[8..], &[10, 11, 12, 13, 14]); // columns end-to-end
    }

    /// A batch frame (the fast path) decodes into per-request ChunkEvents.
    #[test]
    fn decodes_batch_frame() {
        use rmpv::Value;
        // Requests: rids "1","2","3".
        let stop = Value::Map(vec![
            (Value::from("type"), Value::from("stop")),
            (Value::from("matched"), Value::from(5)),
        ]);
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1"), Value::from("2"), Value::from("3")]),
            Value::Array(vec![Value::Nil, stop, Value::Nil]),
            Value::Array(vec![
                Value::from(4u32),
                Value::from(5u32),
                Value::from(6u32),
            ]),
            Value::Array(vec![
                Value::from(2u32),
                Value::from(0u32),
                Value::from(1u32),
            ]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [10i32, 11, 12]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();

        let framed = frame_decode_batch_cols(&header, &[&data]);
        assert_eq!(framed[0], DISPATCH_TAG_BATCH);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].rid, Rid::from("1"));
        assert_eq!(events[0].token_ids, vec![10, 11]);
        assert_eq!(events[0].prompt_tokens, 4);
        assert!(events[0].finish_reason.is_none());
        assert_eq!(events[1].rid, Rid::from("2"));
        assert!(events[1].token_ids.is_empty());
        // The whole reason survives msgpack (type + matched), not the type.
        assert_eq!(
            events[1].finish_reason,
            Some(
                FinishKind::Stop {
                    matched: Some(Matched::Token(5))
                }
                .into()
            )
        );
        assert_eq!(events[2].rid, Rid::from("3"));
        assert_eq!(events[2].token_ids, vec![12]);
        assert_eq!(events[2].prompt_tokens, 6);
        // A plain decode frame carries no extras columns at all.
        assert!(events.iter().all(|e| e.extras.is_none()));
    }

    /// A header whose column lengths exceed the data buffer (a Python/Rust positional-ABI drift, or a truncated frame) is rejected.
    #[test]
    fn rejects_frame_with_lengths_past_data() {
        use rmpv::Value;
        // The old clamp-only-`end` code advanced the cursor past `len`, then sliced
        // `data[40..4]` (start > end) and panicked.
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]),   // rids
            Value::Array(vec![Value::Nil]),         // finish_reasons
            Value::Array(vec![Value::from(0u32)]),  // prompt_tokens
            Value::Array(vec![Value::from(10u32)]), // tok_lens (claims many
            Value::Array(vec![Value::from(1u32)]),  // out_lp_lens (base now past data)
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i32].iter().flat_map(|x| x.to_le_bytes()).collect(); // A few

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        let decoded = for_each_chunk(&framed[1..], |_| routed += 1);
        assert!(!decoded.ok, "malformed frame must be rejected, not decoded");
        assert_eq!(routed, 0, "no request may be routed from a rejected frame");
        // The rid is still reported, so the caller can fail the request that was waiting on this frame instead.
        assert_eq!(decoded.rids, vec![Rid::from("1")]);
    }

    /// A header whose `reqlens` claim more positions than `poslens` carries is rejected, not truncated.
    #[test]
    fn rejects_poslens_shorter_than_reqlens_claims() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        let rids = Value::Array(vec![Value::from("1"), Value::from("2")]);
        let finish = Value::Array(vec![Value::Nil, Value::Nil]);

        let header_arr = Value::Array(vec![
            rids.clone(),
            finish.clone(),
            arr_u(&[0, 0]), // prompt
            arr_u(&[1, 1]), // tok_lens
            arr_u(&[0, 0]), // out_lp_lens
            arr_u(&[0, 0]), // in_lp_lens
            arr_u(&[2, 1]), // out_top_reqlens — positions claimed
            arr_u(&[2, 2]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20])); // token_ids
        data.extend(f(&[-0.1, -0.2, -0.3, -0.4])); // out_top_val (sum of poslens = 4)
        data.extend(i(&[1, 2, 3, 4])); // out_top_idx
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        assert!(
            !for_each_chunk(&framed[1..], |_| routed += 1).ok,
            "ragged poslens drift must be rejected, not truncated"
        );

        let header_arr = Value::Array(vec![
            rids,
            finish,
            arr_u(&[0, 0]), // prompt
            arr_u(&[1, 1]), // tok_lens
            arr_u(&[0, 0]), // out_lp_lens
            arr_u(&[0, 0]), // in_lp_lens
            arr_u(&[0, 0]), // out_top_reqlens
            arr_u(&[]),     // out_top_poslens
            arr_u(&[0, 0]), // in_top_reqlens
            arr_u(&[]),     // in_top_poslens
            arr_u(&[0, 0]), // out_tokids_lp_reqlens
            arr_u(&[]),     // out_tokids_lp_poslens
            arr_u(&[0, 0]), // in_tokids_lp_reqlens
            arr_u(&[]),     // in_tokids_lp_poslens
            arr_u(&[2, 0]),
            arr_u(&[3]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20])); // token_ids
        data.extend(f(&[0.1, 0.2, 0.3])); // hidden_val (sum of poslens = 3)
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        assert!(
            !for_each_chunk(&framed[1..], |_| routed += 1).ok,
            "hidden poslens drift must be rejected, not truncated"
        );
    }

    /// A frame that fails at request buckets nothing.
    #[test]
    fn rejected_frame_still_reports_all_its_rids() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("r0"), Value::from("r1")]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
            Value::Array(vec![Value::from(9u32), Value::from(9u32)]), // tok_lens past data
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let framed = frame_decode_batch_cols(&header, &[&[0u8; 4][..]]);
        let mut routed = 0usize;
        let decoded = for_each_chunk(&framed[1..], |_| routed += 1);
        assert!(!decoded.ok);
        assert_eq!(routed, 0, "fails at request 0 — nothing bucketed");
        assert_eq!(
            decoded.rids,
            vec![Rid::from("r0"), Rid::from("r1")],
            "both requests must be nameable so the caller can fail them"
        );
    }

    /// A data buffer LONGER than the header's columns means the disagree — a producer-side val/idx mismatch.
    #[test]
    fn frame_longer_than_its_columns_is_rejected() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]),
            Value::Array(vec![Value::Nil]),
            Value::Array(vec![Value::from(0u32)]),
            Value::Array(vec![Value::from(1u32)]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = vec![0u8; 8]; // A few bytes too
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let decoded = for_each_chunk(&framed[1..], |_| {});
        assert!(!decoded.ok, "header and data must agree exactly");
    }

    /// Request/response rid agreement: the rid decoded from the frame must be the one Python sent.
    #[test]
    fn uuid_rid_decodes_to_the_same_rid_and_shard() {
        use rmpv::Value;
        let rid = Rid::from("9f86d081884c7d659a2feaa0c55ad015");
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from(rid.to_string())]), // rids
            Value::Array(vec![Value::Nil]),                   // finish_reasons
            Value::Array(vec![Value::from(1u32)]),            // prompt_tokens
            Value::Array(vec![Value::from(1u32)]),            // tok_lens
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i32].iter().flat_map(|x| x.to_le_bytes()).collect();

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 1);
        // The equality below reuses `from_rid`, so on its own it would hold
        // for any decoder that hashed *something*.
        assert_eq!(
            events[0].rid, rid,
            "the rid IS the identity — carried, not hashed"
        );
        assert_ne!(events[0].rid, Rid::from("0"));
        // Same string, independently constructed → same shard, on any shard count.
        for shards in [1usize, 3, 8, 64] {
            assert_eq!(
                events[0].rid.shard(shards),
                Rid::from("9f86d081884c7d659a2feaa0c55ad015").shard(shards),
                "the partition key must not depend on WHERE the Rid was built"
            );
        }
        assert_ne!(
            events[0].rid,
            Rid::from("another-rid"),
            "distinct rids stay distinct — identity is the string, not a digest"
        );
    }

    /// A batch frame carrying the numeric columns (extras path): requests, req0 with output logprobs + top-k + hidden.
    #[test]
    fn decodes_batch_frame_with_extras() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        // header: rids, finish, prompt, tok_lens, out_lp_lens, in_lp_lens,
        //   out_top_reqlens, out_top_poslens, in_top_*, out_tokids_lp_*,
        //   in_tokids_lp_*,
        //   hidden_reqlens, hidden_poslens
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1"), Value::from("2")]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),             // finish
            arr_u(&[3, 4]),                                         // prompt
            arr_u(&[1, 1]),                                         // tok_lens
            arr_u(&[2, 0]),                                         // out_lp_lens
            arr_u(&[0, 0]),                                         // in_lp_lens
            arr_u(&[1, 0]),
            arr_u(&[2]),    // out_top_poslens (that pos: k=2)
            arr_u(&[0, 0]), // in_top_reqlens
            arr_u(&[]),     // in_top_poslens
            arr_u(&[0, 0]), // out_tokids_lp_reqlens
            arr_u(&[]),     // out_tokids_lp_poslens
            arr_u(&[0, 0]), // in_tokids_lp_reqlens
            arr_u(&[]),     // in_tokids_lp_poslens
            arr_u(&[1, 0]),
            arr_u(&[3]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20]));
        data.extend(f(&[-0.5, -0.6]));
        data.extend(i(&[10, 99])); // out_lp_idx
        data.extend(f(&[-0.1, -0.2]));
        data.extend(i(&[10, 11])); // out_top_idx
        data.extend(f(&[0.1, 0.2, 0.3]));

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].token_ids, vec![10]);
        let ex0 = events[0]
            .extras
            .as_deref()
            .expect("req0 has logprob/hidden extras");
        assert_eq!(ex0.out_lp_val, vec![-0.5, -0.6]);
        assert_eq!(ex0.out_lp_idx, vec![10, 99]);
        assert_eq!(ex0.out_top_val, vec![-0.1, -0.2]);
        assert_eq!(ex0.out_top_lens, vec![2]);
        assert_eq!(ex0.hidden_val, vec![0.1, 0.2, 0.3]);
        assert_eq!(ex0.hidden_lens, vec![3]);
        // req1 has a token id but no numeric columns → no extras box allocated.
        assert_eq!(events[1].token_ids, vec![20]);
        assert!(events[1].extras.is_none());
    }

    /// An inactive extras family may arrive as EMPTY columns or as `batch_size` zeros.
    #[test]
    fn empty_and_zero_filled_inactive_families_decode_alike() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());

        // Only `out_lp` is active; the other families are inactive.
        // `zeros` picks how they are spelled: the producer's per-request
        // zeros, or the gated producer's empty column.
        let build = |zeros: bool| {
            let reqlens = if zeros { arr_u(&[0, 0]) } else { arr_u(&[]) };
            let header_arr = Value::Array(vec![
                Value::Array(vec![Value::from("1"), Value::from("2")]), // rids
                Value::Array(vec![Value::Nil, Value::Nil]),             // finish
                arr_u(&[3, 4]),                                         // prompt
                arr_u(&[1, 1]),                                         // tok_lens
                arr_u(&[2, 0]),                                         // out_lp_lens (ACTIVE)
                reqlens.clone(),                                        // in_lp_lens
                reqlens.clone(),                                        // out_top_reqlens
                arr_u(&[]),                                             // out_top_poslens
                reqlens.clone(),                                        // in_top_reqlens
                arr_u(&[]),                                             // in_top_poslens
                reqlens.clone(),                                        // out_tokids_lp_reqlens
                arr_u(&[]),                                             // out_tokids_lp_poslens
                reqlens.clone(),                                        // in_tokids_lp_reqlens
                arr_u(&[]),                                             // in_tokids_lp_poslens
                reqlens,                                                // hidden_reqlens
                arr_u(&[]),                                             // hidden_poslens
            ]);
            let mut header = Vec::new();
            rmpv::encode::write_value(&mut header, &header_arr).unwrap();
            let mut data = Vec::new();
            data.extend(i(&[10, 20])); // token_ids
            data.extend(f(&[-0.5, -0.6])); // out_lp_val (req0)
            data.extend(i(&[10, 99])); // out_lp_idx
            let framed = frame_decode_batch_cols(&header, &[&data]);
            let mut events = Vec::new();
            assert!(
                for_each_chunk(&framed[1..], |ev| events.push(ev)).ok,
                "frame rejected (zeros={zeros})"
            );
            events
        };

        let old = build(true);
        let new = build(false);
        assert_eq!(old.len(), 2);
        assert_eq!(old.len(), new.len());
        for (o, n) in old.iter().zip(&new) {
            assert_eq!(o.rid, n.rid);
            assert_eq!(o.token_ids, n.token_ids);
            match (o.extras.as_deref(), n.extras.as_deref()) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.out_lp_val, b.out_lp_val);
                    assert_eq!(a.out_lp_idx, b.out_lp_idx);
                    // Every inactive family stays empty in BOTH spellings.
                    assert!(a.in_lp_val.is_empty() && b.in_lp_val.is_empty());
                    assert!(a.out_top_lens.is_empty() && b.out_top_lens.is_empty());
                    assert!(a.in_top_lens.is_empty() && b.in_top_lens.is_empty());
                    assert!(a.out_tid_lens.is_empty() && b.out_tid_lens.is_empty());
                    assert!(a.in_tid_lens.is_empty() && b.in_tid_lens.is_empty());
                    assert!(a.hidden_lens.is_empty() && b.hidden_lens.is_empty());
                }
                (None, None) => {}
                _ => panic!("extras presence differs between the two spellings"),
            }
        }
        // req0 did carry its active family through both spellings.
        assert_eq!(
            new[0]
                .extras
                .as_deref()
                .expect("req0 has out_lp")
                .out_lp_val,
            vec![-0.5, -0.6]
        );
    }

    /// All extras families in one frame, each with a distinct length AND distinct values.
    #[test]
    fn decodes_all_extras_families_without_transposition() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());

        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]), // rids
            Value::Array(vec![Value::Nil]),       // finish
            arr_u(&[9]),                          // prompt
            arr_u(&[1]),                          // tok_lens
            arr_u(&[2]),
            arr_u(&[1]),
            arr_u(&[1]),                          // out_top_reqlens (position…
            arr_u(&[2]),                          // out_top_poslens …k=2)
            arr_u(&[1]),                          // in_top_reqlens (position…
            arr_u(&[1]),                          // in_top_poslens …k=1)
            arr_u(&[1]),
            arr_u(&[2]),
            arr_u(&[1]),
            arr_u(&[1]),
            arr_u(&[1]),                          // hidden_reqlens (a couple
            arr_u(&[3]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();

        // Concatenated in `for_each_chunk`'s column order.
        let mut data = Vec::new();
        data.extend(i(&[100])); // token_ids
        data.extend(f(&[-1.1, -1.2])); // out_lp_val
        data.extend(i(&[11, 12])); // out_lp_idx
        data.extend(f(&[-2.1])); // in_lp_val
        data.extend(i(&[21])); // in_lp_idx
        data.extend(f(&[-3.1, -3.2])); // out_top_val
        data.extend(i(&[31, 32])); // out_top_idx
        data.extend(f(&[-4.1])); // in_top_val
        data.extend(i(&[41])); // in_top_idx
        data.extend(f(&[-5.1, -5.2])); // out_tid_val
        data.extend(i(&[51, 52])); // out_tid_idx
        data.extend(f(&[-6.1])); // in_tid_val
        data.extend(i(&[61])); // in_tid_idx
        data.extend(f(&[7.1, 7.2, 7.3])); // hidden_val

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 1);
        let ex = events[0].extras.as_deref().expect("extras present");

        assert_eq!(ex.out_lp_val, vec![-1.1, -1.2]);
        assert_eq!(ex.out_lp_idx, vec![11, 12]);
        assert_eq!(ex.in_lp_val, vec![-2.1]);
        assert_eq!(ex.in_lp_idx, vec![21]);
        assert_eq!(ex.out_top_val, vec![-3.1, -3.2]);
        assert_eq!(ex.out_top_idx, vec![31, 32]);
        assert_eq!(ex.out_top_lens, vec![2]);
        assert_eq!(ex.in_top_val, vec![-4.1]);
        assert_eq!(ex.in_top_idx, vec![41]);
        assert_eq!(ex.in_top_lens, vec![1]);
        assert_eq!(ex.out_tid_val, vec![-5.1, -5.2]);
        assert_eq!(ex.out_tid_idx, vec![51, 52]);
        assert_eq!(ex.out_tid_lens, vec![2]);
        assert_eq!(ex.in_tid_val, vec![-6.1]);
        assert_eq!(ex.in_tid_idx, vec![61]);
        assert_eq!(ex.in_tid_lens, vec![1]);
        assert_eq!(ex.hidden_val, vec![7.1, 7.2, 7.3]);
        assert_eq!(ex.hidden_lens, vec![3]);
        // Every byte of the data buffer was consumed by exactly one family.
        assert_eq!(events[0].token_ids, vec![100]);
    }

    /// DISTINCT rids that hash to the same shard must stay separate requests.
    #[test]
    fn colliding_rids_stay_distinct_requests() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("alice"), Value::from("bob")]),
            Value::Array(vec![Value::Nil, Value::Nil]),
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
            Value::Array(vec![Value::from(1u32), Value::from(1u32)]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [7i32, 8].iter().flat_map(|x| x.to_le_bytes()).collect();
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        // Each chunk carries its OWN rid — the value a shard keys its table on.
        assert_eq!(events[0].rid, Rid::from("alice"));
        assert_eq!(events[1].rid, Rid::from("bob"));
        assert_ne!(events[0].rid, events[1].rid);
    }

    /// The common frame must stay small: logprob/hidden columns are boxed behind `ChunkExtras`.
    #[test]
    fn chunk_event_frame_stays_small() {
        let sz = std::mem::size_of::<ChunkEvent>();
        assert!(
            sz <= 144,
            "ChunkEvent grew to {sz} bytes; keep rare columns behind ChunkExtras"
        );
    }
}

#[cfg(test)]
mod rid_recovery_tests {
    use super::*;

    /// A header this build cannot type-decode (Python appended a column) must still name its requests.
    #[test]
    fn rid_recovery_works_at_every_header_arity() {
        use rmpv::Value;
        for extra_cols in [0usize, 3, 15, 16] {
            let mut cols = vec![Value::Array(vec![Value::from("a"), Value::from("b")])];
            // Columns of a type this build would reject (strings where u32 is expected).
            cols.extend((0..extra_cols).map(|_| Value::from("unexpected")));
            let mut header = Vec::new();
            rmpv::encode::write_value(&mut header, &Value::Array(cols)).unwrap();
            let framed = frame_decode_batch_cols(&header, &[]);
            let decoded = for_each_chunk(&framed[1..], |_| {});
            assert!(!decoded.ok, "arity {extra_cols}: must reject");
            assert_eq!(
                decoded.rids,
                vec![Rid::from("a"), Rid::from("b")],
                "arity {extra_cols}: rids must survive so the caller can fail them"
            );
        }
    }
}
