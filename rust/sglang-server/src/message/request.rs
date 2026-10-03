//! The `/generate` request path: the HTTP body and its per-request fan-out ([`GenerateBody`] → [`GenerateRequest`]s).

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::LazyLock;

use bytes::Bytes;
use itertools::izip;
use serde::de::{DeserializeOwned, SeqAccess, Visitor, value::SeqAccessDeserializer};
use serde::{Deserialize, Deserializer};

use super::io_struct::{ControlRequest, TokenizedGenerateReqInput};
use super::multimodal::{self, MmDataInput, MmItem};
use super::response::ResponseSink;
use super::sampling::{SamplingParams, SamplingParamsInput};
use super::types::{OneOrMany, OneOrManyItem, TokenIds};
use crate::message::ids::Rid;
use crate::utils::fsm::RequestState;
use crate::utils::{environ::env_i64, error::Error};

/// Hard cap on how many scheduler requests one `/generate` HTTP call may
/// expand into.
static MAX_BATCH_REQS_PER_HTTP_REQ: LazyLock<i64> =
    LazyLock::new(|| env_i64("SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ", 4096));

fn batch_size_exceeds_limit(batch_size: usize, limit: i64) -> bool {
    limit >= 0 && batch_size as u128 > limit as u128
}

/// Hard cap on the total bytes a broadcast value may clone into the batch (see the `One` arms of the fan-out).
const MAX_BROADCAST_CLONE_BYTES: usize = 64 << 20;

/// Live heap per byte of serialized JSON.
const JSON_TO_HEAP_FACTOR: usize = 8;

/// Top-level fields in this namespace belong to the selected multimodal processor.
const PROCESSOR_EXTENSION_PREFIX: &str = "multimodal_";

/// Model-owned request fields.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct ProcessorExtensions(BTreeMap<String, rmpv::Value>);

impl ProcessorExtensions {
    /// Deserialize the model-agnostic value tree directly into the selected
    /// processor's schema. This does not encode or decode MessagePack bytes.
    pub fn deserialize<T: DeserializeOwned>(self) -> Result<T, String> {
        let fields = self
            .0
            .into_iter()
            .map(|(name, value)| (rmpv::Value::from(name), value))
            .collect();
        rmpv::ext::from_value(rmpv::Value::Map(fields))
            .map_err(|error| format!("invalid processor extensions: {error}"))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn values(&self) -> impl Iterator<Item = &rmpv::Value> {
        self.0.values()
    }
}

impl FromIterator<(String, rmpv::Value)> for ProcessorExtensions {
    fn from_iter<T: IntoIterator<Item = (String, rmpv::Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// The `/generate` wire body before batch splitting: `text`/`input_ids`/`sampling_params` each scalar-or-list, fanned.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerateBody {
    /// Optional client-supplied request id(s): a single string (a batch fans it out as `{rid}_{i}`, mirroring Python `_normalize_batch`).
    pub rid: Option<OneOrMany<String>>,
    pub text: Option<OneOrMany<String>>,
    #[serde(default, deserialize_with = "deserialize_input_ids")]
    pub input_ids: Option<OneOrMany<TokenIds>>,
    #[serde(default)]
    pub stream: bool,
    /// One params object (broadcast) or a list of them (per item); see [`SamplingParamsInput`].
    pub sampling_params: Option<SamplingParamsInput>,
    /// Logprob / hidden-state options: a scalar broadcasts to every prompt.
    pub return_logprob: Option<OneOrMany<bool>>,
    pub logprob_start_len: Option<OneOrMany<i64>>,
    pub top_logprobs_num: Option<OneOrMany<i64>>,
    /// Token ids to report logprobs for: one list (broadcast to every prompt) or one list per prompt.
    pub token_ids_logprob: Option<OneOrMany<TokenIds>>,
    pub return_hidden_states: Option<OneOrMany<bool>>,
    /// Scalar-only in Python too (`return_text_in_logprobs: bool`).
    pub return_text_in_logprobs: Option<bool>,
    // PD-disaggregation routing, injected per request by the PD router (mini_lb / sgl-model-gateway): a scalar for a single prompt.
    pub bootstrap_host: Option<OneOrMany<Option<String>>>,
    pub bootstrap_port: Option<OneOrMany<Option<i64>>>,
    /// `bootstrap_room` fits in i64: the PD routers draw it from `[0, 2^63)`.
    pub bootstrap_room: Option<OneOrMany<Option<i64>>>,
    pub bootstrap_pair_key: Option<OneOrMany<Option<String>>>,
    pub decode_tp_size: Option<OneOrMany<Option<i64>>>,
    /// DP routing hints — per-request scalars even for batches, as in Python.
    pub routed_dp_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    // Multimodal inputs (Python `MultimodalDataInputFormat`), fanned out per request by `multimodal::fan_out`.
    pub image_data: Option<MmDataInput>,
    /// Caller-supplied per-item content hashes (hex) overriding the computed ones.
    pub mm_hashes: Option<OneOrMany<Vec<String>>>,
    pub video_data: Option<MmDataInput>,
    pub audio_data: Option<MmDataInput>,
    /// Model-specific multimodal fields, retained without teaching the shared request schema their contents.
    #[serde(flatten)]
    processor_extensions: ProcessorExtensions,
}

/// Decode `input_ids` directly into token vectors. The first element selects
/// flat vs. batched input, avoiding the untagged enum's intermediate value tree.
fn deserialize_input_ids<'de, D>(deserializer: D) -> Result<Option<OneOrMany<TokenIds>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct InputIds(OneOrMany<TokenIds>);

    enum FirstElement {
        Token(i32),
        Tokens(TokenIds),
    }

    impl<'de> Deserialize<'de> for FirstElement {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct FirstVisitor;

            impl<'de> Visitor<'de> for FirstVisitor {
                type Value = FirstElement;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("an i32 token or a list of i32 tokens")
                }

                fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                    i32::try_from(value)
                        .map(FirstElement::Token)
                        .map_err(E::custom)
                }

                fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                    i32::try_from(value)
                        .map(FirstElement::Token)
                        .map_err(E::custom)
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    sequence: A,
                ) -> Result<Self::Value, A::Error> {
                    TokenIds::deserialize(SeqAccessDeserializer::new(sequence))
                        .map(FirstElement::Tokens)
                }
            }

            deserializer.deserialize_any(FirstVisitor)
        }
    }

    impl<'de> Deserialize<'de> for InputIds {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct TokensVisitor;

            impl<'de> Visitor<'de> for TokensVisitor {
                type Value = InputIds;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a token list or a list of token lists")
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut sequence: A,
                ) -> Result<Self::Value, A::Error> {
                    let tokens = match sequence.next_element::<FirstElement>()? {
                        // Preserve OneOrMany's first-variant choice for [].
                        None => OneOrMany::One(Vec::new()),
                        Some(FirstElement::Token(first)) => {
                            let mut tokens = vec![first];
                            while let Some(token) = sequence.next_element::<i32>()? {
                                tokens.push(token);
                            }
                            OneOrMany::One(tokens)
                        }
                        Some(FirstElement::Tokens(first)) => {
                            let mut prompts = vec![first];
                            while let Some(tokens) = sequence.next_element::<TokenIds>()? {
                                prompts.push(tokens);
                            }
                            OneOrMany::Many(prompts)
                        }
                    };
                    Ok(InputIds(tokens))
                }
            }

            deserializer.deserialize_seq(TokensVisitor)
        }
    }

    Option::<InputIds>::deserialize(deserializer).map(|value| value.map(|tokens| tokens.0))
}

impl GenerateBody {
    /// Merge operator-provided sampling defaults beneath request values,
    /// matching Python TokenizerManager's preferred/request precedence.
    pub fn apply_preferred_sampling(&mut self, preferred: &serde_json::Value) -> Result<(), Error> {
        match &mut self.sampling_params {
            Some(params) => params.apply_preferred(preferred),
            None => SamplingParamsInput::from_preferred(preferred).map(|params| {
                self.sampling_params = Some(params);
            }),
        }
        .map_err(|e| Error::Validation(format!("invalid preferred_sampling_params: {e}")))
    }

    /// Validate, normalize and fan the body into one [`GenerateRequest`] per
    /// prompt + `is_batch` (list form — a 1-element list is still a batch → JSON
    /// array response). The Rust counterpart of Python
    /// `GenerateReqInput.normalize_batch_and_arguments`; an invalid/inconsistent
    /// batch is [`Error::Validation`].
    /// variant's own status (400).
    pub fn into_requests(self) -> Result<(Vec<GenerateRequest>, bool), Error> {
        let GenerateBody {
            rid,
            text,
            input_ids,
            stream,
            sampling_params,
            return_logprob,
            logprob_start_len,
            top_logprobs_num,
            token_ids_logprob,
            return_hidden_states,
            return_text_in_logprobs,
            bootstrap_host,
            bootstrap_port,
            bootstrap_room,
            bootstrap_pair_key,
            decode_tp_size,
            routed_dp_rank,
            disagg_prefill_dp_rank,
            image_data,
            video_data,
            audio_data,
            mm_hashes,
            processor_extensions,
        } = self;

        // Cap the batch BEFORE the columns below allocate anything.
        let declared_n = match (&text, &input_ids) {
            (Some(OneOrMany::Many(v)), None) => v.len(),
            (None, Some(OneOrMany::Many(v))) => v.len(),
            _ => 1,
        };
        if batch_size_exceeds_limit(declared_n, *MAX_BATCH_REQS_PER_HTTP_REQ) {
            return Err(Error::Validation(format!(
                "batch size {declared_n} exceeds the maximum of {}",
                *MAX_BATCH_REQS_PER_HTTP_REQ
            )));
        }

        // Per-item (text, input_ids) columns + whether the input used list form.
        type Columns = (Vec<Option<String>>, Vec<Option<TokenIds>>, bool);
        // Exactly one of text / input_ids (Python `_validate_inputs`), and no
        // empty id list (Python `_determine_batch_size`).
        let (texts, id_lists, is_batch): Columns = match (text, input_ids) {
            (Some(_), Some(_)) => {
                return Err(Error::Validation(
                    "provide either `text` or `input_ids`, not both".into(),
                ));
            }
            (None, None) => {
                return Err(Error::Validation(
                    "either `text` or `input_ids` must be provided".into(),
                ));
            }
            (Some(OneOrMany::One(s)), None) => (vec![Some(s)], vec![None], false),
            (Some(OneOrMany::Many(v)), None) => {
                let n = v.len();
                (v.into_iter().map(Some).collect(), vec![None; n], true)
            }
            // `[]` parses as `One(vec![])` (one prompt with no ids), so the
            // `n == 0` guard below never sees it — reject it here, as Python's
            // `_determine_batch_size` does.
            (None, Some(OneOrMany::One(x))) => {
                if x.is_empty() {
                    return Err(Error::Validation("input_ids cannot be empty".into()));
                }
                (vec![None], vec![Some(x)], false)
            }
            (None, Some(OneOrMany::Many(vv))) => {
                if vv.iter().any(|ids| ids.is_empty()) {
                    return Err(Error::Validation(
                        "input_ids cannot be empty for any prompt in the batch".into(),
                    ));
                }
                let n = vv.len();
                (vec![None; n], vv.into_iter().map(Some).collect(), true)
            }
        };
        let n = texts.len();
        if n == 0 {
            return Err(Error::Validation(
                "batch must contain at least one item".into(),
            ));
        }

        // A list is per-item; a single object broadcasts to every item.
        let sps: Vec<SamplingParams> = match sampling_params {
            None => vec![SamplingParams::default(); n],
            Some(SamplingParamsInput::Many(v)) => {
                if v.len() != n {
                    return Err(Error::Validation(format!(
                        "sampling_params list length {} does not match batch size {n}",
                        v.len()
                    )));
                }
                v
            }
            Some(SamplingParamsInput::One(sp)) => {
                // Broadcasting deep-clones the client's params once per
                // prompt, heap and all — `stop`, `logit_bias` and
                // `custom_params` (arbitrary JSON) are still unnormalized
                // client data here.
                if n > 1 {
                    let per_clone = serde_json::to_string(&*sp)
                        .map_or(0, |s| s.len())
                        .saturating_mul(JSON_TO_HEAP_FACTOR);
                    check_broadcast_budget(per_clone, n, "sampling_params")?;
                }
                vec![*sp; n]
            }
        };

        // rid: absent → mint one uuid per item here, so every request carries its
        // final rid from this point on; a single string fans out as `{rid}_{i}`
        // for a batch (Python `_normalize_batch`); a list is per-item.
        //
        // Every CLIENT-supplied rid goes through `Rid::from_client`, which appends a
        // uniquifier so concurrent requests sharing an rid cannot collide on the
        // detok table. `client_facing` strips it back off for `meta_info.id`, so the
        // client sees exactly what it sent. Minted rids (`Rid::default`) are already
        // unique and are left bare.
        let rids: Vec<Rid> = match rid {
            None => (0..n).map(|_| Rid::default()).collect(),
            Some(OneOrMany::One(r)) if !is_batch => vec![Rid::from_client(&r)],
            Some(OneOrMany::One(r)) => {
                check_broadcast_budget(r.len(), n, "rid")?;
                // Uniquify AFTER the `_{i}` split, so the split index stays part of
                // the rid the client gets back.
                (0..n)
                    .map(|i| Rid::from_client(&format!("{r}_{i}")))
                    .collect()
            }
            Some(OneOrMany::Many(v)) => {
                if !is_batch || v.len() != n {
                    return Err(Error::Validation(format!(
                        "rid list length {} does not match batch size {n}",
                        v.len()
                    )));
                }
                // Python `_validate_rid_uniqueness`. `from_client` below would make even these unique, so this is parity rather than safety: Python 400s a request that names one id twice, and echoing the same `meta_info.id` on entries of one
                // batch response is useless to the client regardless.
                {
                    let mut seen = HashSet::with_capacity(v.len());
                    let duplicates: Vec<&String> = v.iter().filter(|r| !seen.insert(*r)).collect();
                    if !duplicates.is_empty() {
                        return Err(Error::Validation(format!(
                            "duplicate request IDs detected within the request: {duplicates:?}"
                        )));
                    }
                }
                v.iter().map(|r| Rid::from_client(r)).collect()
            }
        };

        // Fans out exactly like the scalar options: one list broadcasts.
        let tid_logprobs = fan_out(token_ids_logprob, n, "token_ids_logprob")?;

        // Each logprob/hidden opt: absent → None for every item, a scalar broadcasts, a list is per-item.
        let return_logprobs = fan_out(return_logprob, n, "return_logprob")?;
        let logprob_start_lens = fan_out(logprob_start_len, n, "logprob_start_len")?;
        let top_logprobs_nums = fan_out(top_logprobs_num, n, "top_logprobs_num")?;
        let return_hidden = fan_out(return_hidden_states, n, "return_hidden_states")?;

        // PD fields fan out like Python `_normalize_bootstrap_params`: scalars broadcast — except a scalar `bootstrap_room`.
        let bootstrap_hosts = flatten_column(fan_out(bootstrap_host, n, "bootstrap_host")?);
        let bootstrap_ports = flatten_column(fan_out(bootstrap_port, n, "bootstrap_port")?);
        let bootstrap_rooms = match bootstrap_room {
            // `wrapping_add`, not `checked_`: rooms are drawn from `[, ^)`.
            Some(OneOrMany::One(Some(room))) => {
                (0..n).map(|i| Some(room.wrapping_add(i as i64))).collect()
            }
            other => flatten_column(fan_out(other, n, "bootstrap_room")?),
        };
        let bootstrap_pair_keys =
            flatten_column(fan_out(bootstrap_pair_key, n, "bootstrap_pair_key")?);
        let decode_tp_sizes = flatten_column(fan_out(decode_tp_size, n, "decode_tp_size")?);
        // `mm_hashes` has no batch form: honoring it only here would give both
        // servers different prefix-cache keys for the same body. Reject instead of
        // dropping it silently as Python does — the field exists to align a
        // caller's keys, so ignoring it returns subtly wrong ones.
        let mm_hashes: Vec<String> = match mm_hashes {
            None => Vec::new(),
            Some(OneOrMany::One(hashes)) if hashes.is_empty() => Vec::new(),
            Some(_) if is_batch => {
                return Err(Error::Validation(
                    "mm_hashes is not supported for batch requests; send one request per prompt"
                        .into(),
                ));
            }
            Some(OneOrMany::One(hashes)) => hashes,
            Some(OneOrMany::Many(_)) => {
                return Err(Error::Validation(
                    "mm_hashes must be a flat list of hex strings for a single request".into(),
                ));
            }
        };
        // Multimodal columns; see `multimodal::fan_out` for the Python parity rules.
        let images = multimodal::fan_out(image_data, n, is_batch, "image_data")?;
        let videos = multimodal::fan_out(video_data, n, is_batch, "video_data")?;
        let audios = multimodal::fan_out(audio_data, n, is_batch, "audio_data")?;
        let processor_extensions = split_extension_columns(processor_extensions, n, is_batch)?;

        // Every column above is exactly `n` long, so zip them by value: each
        // request takes ownership of its cell, with no indexing or bounds checks.
        let mut requests: Vec<GenerateRequest> = izip!(
            rids,
            texts,
            id_lists,
            sps,
            return_logprobs,
            logprob_start_lens,
            top_logprobs_nums,
            tid_logprobs,
            return_hidden,
            bootstrap_hosts,
            bootstrap_ports,
            bootstrap_rooms,
            bootstrap_pair_keys,
            decode_tp_sizes,
            images,
            videos,
            audios,
            processor_extensions,
        )
        .map(
            |(
                rid,
                text,
                input_ids,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                return_hidden_states,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                image_data,
                video_data,
                audio_data,
                processor_extensions,
            )| GenerateRequest {
                rid,
                text,
                input_ids,
                // Plain text prompts keep the post-processor specials; the chat flow sets this explicitly.
                skip_special_tokens: false,
                sampling_params,
                stream,
                // Python `GenerateReqInput` defaults.
                return_logprob: return_logprob.unwrap_or(false),
                logprob_start_len: logprob_start_len.unwrap_or(-1),
                top_logprobs_num: top_logprobs_num.unwrap_or(0),
                // `Some` here means "these ids were requested", so an empty list collapses to None.
                token_ids_logprob: token_ids_logprob.filter(|ids| !ids.is_empty()),
                return_sampling_mask: false, // TODO: port Python's `return_sampling_mask`
                return_hidden_states: return_hidden_states.unwrap_or(false),
                return_text_in_logprobs,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                routed_dp_rank,
                disagg_prefill_dp_rank,
                mm: pack_mm(image_data, video_data, audio_data, processor_extensions),
            },
        )
        .collect();
        // Single requests only (batches rejected above).
        if let Some(mm) = requests.first_mut().and_then(|req| req.mm.as_deref_mut()) {
            mm.mm_hashes = mm_hashes;
        }
        Ok((requests, is_batch))
    }
}

/// Box the per-item mm values, `None` when the item has none — the common
/// text-only case keeps `GenerateRequest` slim.
fn pack_mm(
    image_data: Vec<MmItem>,
    video_data: Vec<MmItem>,
    audio_data: Vec<MmItem>,
    processor_extensions: ProcessorExtensions,
) -> Option<Box<MmData>> {
    if image_data.is_empty()
        && video_data.is_empty()
        && audio_data.is_empty()
        && processor_extensions.is_empty()
    {
        return None;
    }
    Some(Box::new(MmData {
        image_data,
        video_data,
        audio_data,
        processor_extensions,
        ..Default::default()
    }))
}

fn split_extension_columns(
    fields: ProcessorExtensions,
    n: usize,
    is_batch: bool,
) -> Result<Vec<ProcessorExtensions>, Error> {
    let mut requests = vec![ProcessorExtensions::default(); n];
    for (name, value) in fields.0 {
        if !name.starts_with(PROCESSOR_EXTENSION_PREFIX) || value.is_nil() {
            continue;
        }
        if !is_batch {
            requests[0].0.insert(name, value);
            continue;
        }
        let rmpv::Value::Array(values) = value else {
            return Err(Error::Validation(format!(
                "{name} must be a list for batch processing"
            )));
        };
        if values.is_empty() {
            for request in &mut requests {
                request
                    .0
                    .insert(name.clone(), rmpv::Value::Array(Vec::new()));
            }
            continue;
        }
        if values.len() != n {
            return Err(Error::Validation(format!(
                "{name} list length {} does not match batch size {n}",
                values.len()
            )));
        }
        for (request, value) in requests.iter_mut().zip(values) {
            request.0.insert(name.clone(), value);
        }
    }
    Ok(requests)
}

fn extension_value_present(value: &rmpv::Value) -> bool {
    match value {
        rmpv::Value::Nil => false,
        rmpv::Value::Array(values) => values.iter().any(extension_value_present),
        _ => true,
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result, plus the owned inputs.
#[derive(Debug)]
pub struct MmRequest {
    pub rid: Rid,
    pub work: MmWorkItem,
}

/// The parked request's fields the MM worker owns.
#[derive(Debug, Default)]
pub struct MmWorkItem {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub image_data: Vec<MmItem>,
    pub video_data: Vec<MmItem>,
    pub audio_data: Vec<MmItem>,
    pub processor_extensions: ProcessorExtensions,
    /// See [`MmData::prefetched`].
    pub prefetched: Vec<Bytes>,
    /// See [`GenerateBody::mm_hashes`].
    pub mm_hashes: Vec<String>,
}

/// The owned request as it travels request stages (single owner, so `state` is mutated lock-free).
#[derive(Debug)]
pub struct Request {
    /// Client-visible request id (uuid hex) — what the scheduler wire and `meta_info.id` carry.
    pub rid: Rid,
    pub state: RequestState,
    /// Back-channel to the client connection for response frames.
    pub sink: ResponseSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// One to_scheduler channel entry, split columnar: the scalar `header` (msgpack, `input_ids` omitted) + the raw int64 `ids` cell.
#[derive(Debug)]
pub struct SchedulerRequest {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Request variant — selects the request branch, scheduler wire message, and response shape.
#[derive(Debug)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(Box<GenerateRequest>),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization.
    Control(Box<ControlRequest>),
    /// Internal service call: decode a complete token-id sequence to text.
    Detokenize { token_ids: TokenIds },
}

/// A single in-flight `/generate` request (per-item from [`GenerateBody::into_requests`]), serialized.
#[derive(Debug, Default)]
pub struct GenerateRequest {
    /// This item's final rid: the client's (normalized per item by `into_requests`) or a uuid minted there when none was sent. A [`Rid`].
    pub rid: Rid,
    pub text: Option<String>,
    /// Client-supplied token ids, or filled by the Tokenizer stage.
    pub input_ids: Option<TokenIds>,
    /// Template-rendered prompts (chat) already contain their role/special tokens.
    pub skip_special_tokens: bool,
    /// Sampling params (defaults when the client sent none, as in Python).
    pub sampling_params: SamplingParams,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
    /// Logprob / hidden-state options.
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    /// This request's `token_ids_logprob` ids, fanned out by `into_requests` and collapsed to `None` when empty.
    pub token_ids_logprob: Option<TokenIds>,
    pub return_sampling_mask: bool,
    pub return_hidden_states: bool,
    /// Decode logprob token ids to text in each `[logprob, token_id, text]` tuple (default leaves the text slot null).
    pub return_text_in_logprobs: Option<bool>,
    /// PD-disaggregation routing, forwarded verbatim to the scheduler.
    pub bootstrap_host: Option<String>,
    pub bootstrap_port: Option<i64>,
    pub bootstrap_room: Option<i64>,
    pub bootstrap_pair_key: Option<String>,
    pub decode_tp_size: Option<i64>,
    /// DP routing hints.
    pub routed_dp_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    /// Multimodal inputs.
    pub mm: Option<Box<MmData>>,
}

/// The multimodal fields of one request (see [`GenerateRequest::mm`]), each modality already fanned out.
#[derive(Debug, Default)]
pub struct MmData {
    pub image_data: Vec<MmItem>,
    pub video_data: Vec<MmItem>,
    pub audio_data: Vec<MmItem>,
    pub processor_extensions: ProcessorExtensions,
    /// Bytes of `image_data`'s I/O-backed sources, resolved by `api_server::prefetch` in `payload::io_sources` order.
    pub prefetched: Vec<bytes::Bytes>,
    /// See [`GenerateBody::mm_hashes`]; applied by the MM worker.
    pub mm_hashes: Vec<String>,
}

impl GenerateRequest {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// True when the request carries a usable multimodal payload — the mirror of
    /// Python `GenerateReqInput.contains_mm_input()`.
    pub fn has_multimodal(&self) -> bool {
        self.mm.as_ref().is_some_and(|mm| {
            !mm.image_data.is_empty()
                || !mm.video_data.is_empty()
                || !mm.audio_data.is_empty()
                || mm
                    .processor_extensions
                    .values()
                    .any(extension_value_present)
        })
    }

    /// Carve out the MM worker's inputs: `text` is cloned (the scheduler header
    /// still needs it), `input_ids` is taken (the expanded ids replace it).
    /// the mm values move wholesale.
    pub fn take_mm_work(&mut self) -> MmWorkItem {
        let mut work = MmWorkItem {
            text: self.text.clone(),
            input_ids: self.input_ids.take(),
            ..Default::default()
        };
        if let Some(m) = self.mm.as_deref_mut() {
            work.image_data = std::mem::take(&mut m.image_data);
            work.video_data = std::mem::take(&mut m.video_data);
            work.audio_data = std::mem::take(&mut m.audio_data);
            work.processor_extensions = std::mem::take(&mut m.processor_extensions);
            work.prefetched = std::mem::take(&mut m.prefetched);
            work.mm_hashes = std::mem::take(&mut m.mm_hashes);
        }
        work
    }

    pub fn encode_header(&self) -> Result<Bytes, Error> {
        TokenizedGenerateReqInput::from(self).encode()
    }

    /// `input_ids` widened to raw little-endian int64 bytes (the scheduler's
    /// `array("q")` columnar cell — rides the to-scheduler channel outside
    /// msgpack). Empty when not tokenized.
    pub fn encode_data_buf(&self) -> Bytes {
        let ids = self.input_ids.as_deref().unwrap_or(&[]);
        let mut buf = Vec::with_capacity(ids.len() * 8);
        for &id in ids {
            buf.extend_from_slice(&(id as i64).to_le_bytes());
        }
        Bytes::from(buf)
    }
}

/// Fan one scalar-or-list option out to `n` per-item values: absent →
/// `None` each, a scalar broadcasts, a list must match the batch size.
pub(super) trait HeapBytes {
    fn heap_bytes(&self) -> usize;
}
impl HeapBytes for bool {
    fn heap_bytes(&self) -> usize {
        0
    }
}
impl HeapBytes for i64 {
    fn heap_bytes(&self) -> usize {
        0
    }
}
impl HeapBytes for String {
    fn heap_bytes(&self) -> usize {
        self.len()
    }
}
impl HeapBytes for TokenIds {
    fn heap_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<i32>()
    }
}
impl<T: HeapBytes> HeapBytes for Option<T> {
    fn heap_bytes(&self) -> usize {
        self.as_ref().map_or(0, HeapBytes::heap_bytes)
    }
}

/// Collapse `fan_out`'s nullable-element output: outer `None` (field absent /
/// scalar broadcast of nothing) and inner `None`.
fn flatten_column<T>(column: Vec<Option<Option<T>>>) -> Vec<Option<T>> {
    column.into_iter().map(Option::flatten).collect()
}

/// Reject a broadcast whose clones would exceed [`MAX_BROADCAST_CLONE_BYTES`].
pub(super) fn check_broadcast_budget(per_clone: usize, n: usize, name: &str) -> Result<(), Error> {
    // `n == 1` is not a broadcast — there is one value and one prompt, so nothing
    // is duplicated. Charging it here rejected ordinary single requests with a
    // message about a batch they never sent.
    if n > 1 && per_clone.saturating_mul(n) > MAX_BROADCAST_CLONE_BYTES {
        return Err(Error::Validation(format!(
            "{name} ({per_clone} bytes) broadcast to {n} prompts would allocate more \
             than the {MAX_BROADCAST_CLONE_BYTES}-byte limit; send a shorter {name} \
             or a smaller batch"
        )));
    }
    Ok(())
}

fn fan_out<T: OneOrManyItem + Clone + HeapBytes>(
    value: Option<OneOrMany<T>>,
    n: usize,
    name: &str,
) -> Result<Vec<Option<T>>, Error> {
    match value {
        None => Ok(vec![None; n]),
        Some(OneOrMany::One(v)) => {
            // Same budget as the `sampling_params` broadcast: `vec![Some(v); n]` deep-clones client data once per prompt.
            check_broadcast_budget(v.heap_bytes(), n, name)?;
            Ok(vec![Some(v); n])
        }
        Some(OneOrMany::Many(v)) => {
            if v.len() != n {
                return Err(Error::Validation(format!(
                    "{name} list length {} does not match batch size {n}",
                    v.len()
                )));
            }
            Ok(v.into_iter().map(Some).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestProcessorExtensions {
        multimodal_custom: TestProcessorExtension,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct TestProcessorExtension {
        value: i64,
    }

    /// Vocab size for tests that aren't about the vocab bound (see `sampling::tests::TEST_VOCAB`).
    const TEST_VOCAB: u64 = 1000;

    fn requests(body: &str) -> Result<(Vec<GenerateRequest>, bool), Error> {
        serde_json::from_str::<GenerateBody>(body)
            .unwrap()
            .into_requests()
    }

    /// Scalar `text` → one item, not a batch (response stays a single object).
    #[test]
    fn scalar_text_is_single() {
        let (ps, is_batch) = requests(r#"{"text": "hi"}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
    }

    /// List `text` → batch (even length); each prompt becomes its own payload.
    #[test]
    fn list_text_is_batch() {
        let (ps, is_batch) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0].text.as_deref(), Some("a"));
        assert_eq!(ps[1].text.as_deref(), Some("b"));

        let (ps, is_batch) = requests(r#"{"text": ["only"]}"#).unwrap();
        assert!(is_batch, "single-element list is still a batch");
        assert_eq!(ps.len(), 1);
    }

    /// Scalar `sampling_params` broadcasts to every item; a list maps per item.
    #[test]
    fn sampling_params_broadcast_and_per_item() {
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "sampling_params": {"temperature": 0.5}}"#).unwrap();
        assert_eq!(ps[0].sampling_params, ps[1].sampling_params);
        assert_eq!(ps[0].sampling_params.temperature, 0.5);

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "sampling_params": [{"temperature": 0.1}, {"temperature": 0.9}]}"#,
        )
        .unwrap();
        assert_ne!(ps[0].sampling_params, ps[1].sampling_params);
    }

    /// A per-item `sampling_params` list whose length ≠ batch size is a.
    #[test]
    fn sampling_params_length_mismatch_errors() {
        let err = requests(r#"{"text": ["a", "b"], "sampling_params": [{}]}"#).unwrap_err();
        assert!(err.to_string().contains("length"), "{err}");
    }

    /// `input_ids` batch (list of lists) fans out; scalar (list of ints) is single.
    #[test]
    fn input_ids_scalar_vs_batch() {
        let (ps, is_batch) = requests(r#"{"input_ids": [1, 2, 3]}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps[0].input_ids, Some(vec![1, 2, 3]));

        let (ps, is_batch) = requests(r#"{"input_ids": [[1, 2], [3]]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[1].input_ids, Some(vec![3]));
    }

    #[test]
    fn input_ids_deserialization_matches_untagged() {
        assert!(
            serde_json::from_str::<GenerateBody>(r#"{"text":"hi"}"#)
                .unwrap()
                .input_ids
                .is_none()
        );
        for value in [
            "null",
            "[]",
            "[0]",
            "[-2147483648,2147483647]",
            "[[]]",
            "[[],[1,-2]]",
            "[[1,2],[3]]",
            "[[-2147483648],[2147483647]]",
            "1",
            "true",
            "{}",
            r#""1""#,
            "[null]",
            "[true]",
            r#"["1"]"#,
            "[1.0]",
            "[1e0]",
            "[-0]",
            "[2147483648]",
            "[-2147483649]",
            "[18446744073709551616]",
            "[0,2147483648]",
            "[0,-0]",
            "[1,[2]]",
            "[[1],2]",
            "[[[1]]]",
            "[[1.0]]",
            "[[0],[2147483648]]",
        ] {
            let expected = serde_json::from_str::<Option<OneOrMany<TokenIds>>>(value);
            let body = format!(r#"{{"input_ids":{value}}}"#);
            let actual = serde_json::from_str::<GenerateBody>(&body).map(|body| body.input_ids);
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => assert_eq!(actual, expected, "{value}"),
                (Err(_), Err(_)) => {}
                (expected, actual) => panic!("{value}: expected {expected:?}, got {actual:?}"),
            }
        }

        for (body, path) in [
            (r#"{"input_ids":[true]}"#, "input_ids[0]"),
            (r#"{"input_ids":[[1,true]]}"#, "input_ids[0][1]"),
            (r#"{"input_ids":[[1],[2,true]]}"#, "input_ids[1][1]"),
        ] {
            let error = axum::Json::<GenerateBody>::from_bytes(body.as_bytes()).unwrap_err();
            assert!(error.body_text().contains(path), "{error}");
        }
        for body in [
            r#"{"input_ids":null,"input_ids":[1]}"#,
            r#"{"input_ids":[1],"input_ids":[2]}"#,
            r#"{"input_ids":[1]} {}"#,
        ] {
            assert!(axum::Json::<GenerateBody>::from_bytes(body.as_bytes()).is_err());
        }
    }

    /// Both / neither of text+input_ids is a.
    #[test]
    fn split_validates_inputs() {
        assert!(requests(r#"{"text": "a", "input_ids": [1]}"#).is_err());
        assert!(requests(r#"{"stream": true}"#).is_err());
        // Parallel sampling is rejected where Python reads it — in the params, at normalization, not here.
        let (mut ps, _) = requests(r#"{"text": "a", "sampling_params": {"n": 2}}"#).unwrap();
        assert!(ps[0].sampling_params.normalize(false, TEST_VOCAB).is_err());
    }

    /// Unported `GenerateReqInput` fields are IGNORED, not rejected.
    #[test]
    fn unported_generate_req_input_fields_are_ignored() {
        for field in [
            r#""priority": 3"#,
            r#""extra_key": "k""#,
            r#""session_id": "s""#,
            r#""session_params": {"a": 1}"#,
            r#""return_sampling_mask": true"#,
            r#""custom_logit_processor": "cls""#,
            r#""lora_path": "adapter""#,
            r#""image_data": "base64""#,
            r#""return_routed_experts": true"#,
            r#""bootstrap_host": "h""#,
            // Python has no top-level `n` either, and ignores it the same.
            r#""n": 1"#,
            r#""totally_made_up": 1"#,
        ] {
            let body = format!(r#"{{"text": "hi", {field}}}"#);
            let (ps, _) = requests(&body)
                .unwrap_or_else(|e| panic!("{field} must be ignored, not rejected: {e}"));
            assert_eq!(ps.len(), 1, "{field}");
            assert_eq!(ps[0].text.as_deref(), Some("hi"), "{field}");
        }
    }

    /// Client-supplied rid semantics mirror Python's `_normalize_batch`.
    #[test]
    fn split_rid_matches_python_normalize() {
        let (ps, _) = requests(r#"{"text": "a", "rid": "r"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "r");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "base"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "base_0");
        assert_eq!(ps[1].rid.client_facing(), "base_1");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "x");
        assert_eq!(ps[1].rid.client_facing(), "y");

        let (ps, _) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        // Absent → `into_requests` mints one uuid per item, all distinct.
        assert_eq!(ps[0].rid.len(), 32);
        assert_ne!(ps[0].rid, ps[1].rid);

        assert!(
            requests(r#"{"text": ["a", "b"], "rid": ["x"]}"#).is_err(),
            "rid list length must match batch size"
        );
        assert!(
            requests(r#"{"text": "a", "rid": ["x"]}"#).is_err(),
            "rid list with a single (non-batch) prompt is rejected"
        );
    }

    /// The native `bench_serving` payload (a `GenerateReqInput` superset) parses.
    #[test]
    fn accepts_bench_serving_payload() {
        let (ps, is_batch) = requests(
            r#"{"text": "hi", "sampling_params": {"max_new_tokens": 8},
                "stream": true, "lora_path": null, "return_logprob": false,
                "return_routed_experts": false, "logprob_start_len": -1,
                "image_data": null}"#,
        )
        .unwrap();
        assert!(!is_batch);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
        assert!(ps[0].stream);
        assert!(!ps[0].has_multimodal());
    }

    /// Mm columns fan out per Python `_normalize_{image,video}_data`: a single request keeps its items.
    #[test]
    fn split_mm_fanout_matches_python_normalize() {
        let src = |s: &str| MmItem::Source(s.to_owned());
        let images_of = |p: &GenerateRequest| p.mm.as_ref().unwrap().image_data.clone();

        // Single request: one item, or a flat list, kept as sent.
        let (ps, _) = requests(r#"{"text": "a", "image_data": "http://x/i.jpg"}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("http://x/i.jpg")]);
        assert!(ps[0].has_multimodal());
        let (ps, _) = requests(r#"{"text": "a", "image_data": ["u1", {"url": "u2"}]}"#).unwrap();
        assert_eq!(
            images_of(&ps[0]),
            vec![src("u1"), MmItem::Ref { url: "u2".into() }]
        );

        // Batch + scalar image: broadcast, one image per item.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": "u"}"#).unwrap();
        for p in &ps {
            assert_eq!(images_of(p), vec![src("u")]);
            assert!(p.has_multimodal());
        }

        // Batch + per-item list: element i goes to item i; nested lists are per-item lists.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": ["u1", "u2"]}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("u1")]);
        assert_eq!(images_of(&ps[1]), vec![src("u2")]);
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "image_data": [["u1", "u2"], null]}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("u1"), src("u2")]);
        assert!(!ps[1].has_multimodal());

        assert!(requests(r#"{"text": ["a", "b"], "image_data": ["u1"]}"#).is_err());
        assert!(requests(r#"{"text": "a", "image_data": [["u1"]]}"#).is_err());

        // null / [] mean "no multimodal input".
        let (ps, _) = requests(r#"{"text": "a", "image_data": null}"#).unwrap();
        assert!(!ps[0].has_multimodal());
        let (ps, _) = requests(r#"{"text": "a", "image_data": []}"#).unwrap();
        assert!(!ps[0].has_multimodal());

        // Batch + scalar video broadcasts too.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "video_data": "v"}"#).unwrap();
        assert_eq!(ps[1].mm.as_ref().unwrap().video_data, vec![src("v")]);
        assert!(ps[1].has_multimodal());
    }

    #[test]
    fn multimodal_extensions_follow_request_batch_shape() {
        let single = r#"{"input_ids":[9],"image_data":"u","multimodal_placeholders":[{"type":"image","token_index":0,"item_index":0}]}"#;
        let (reqs, is_batch) = requests(single).unwrap();
        assert!(!is_batch);
        let value = reqs[0]
            .mm
            .as_ref()
            .unwrap()
            .processor_extensions
            .0
            .get("multimodal_placeholders")
            .unwrap();
        assert_eq!(value.as_array().unwrap().len(), 1);

        let batched = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_placeholders":[[{"type":"image","token_index":0,"item_index":0}],[{"type":"image","token_index":0,"item_index":0}]]}"#;
        let (reqs, is_batch) = requests(batched).unwrap();
        assert!(is_batch);
        assert_eq!(reqs.len(), 2);
        assert!(reqs.iter().all(GenerateRequest::has_multimodal));
        assert!(reqs.iter().all(|request| {
            request
                .mm
                .as_ref()
                .and_then(|mm| mm.processor_extensions.0.get("multimodal_placeholders"))
                .and_then(rmpv::Value::as_array)
                .is_some_and(|placeholders| placeholders.len() == 1)
        }));

        let invalid = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_placeholders":[{"type":"image","token_index":0,"item_index":0}]}"#;
        assert!(requests(invalid).is_err());

        let generic = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_custom":[{"value":1},{"value":2}]}"#;
        let (reqs, _) = requests(generic).unwrap();
        assert_eq!(
            reqs[1]
                .mm
                .as_ref()
                .unwrap()
                .processor_extensions
                .0
                .get("multimodal_custom")
                .unwrap()
                .as_map()
                .unwrap()[0]
                .1
                .as_i64(),
            Some(2)
        );

        let extensions: TestProcessorExtensions =
            requests(r#"{"input_ids":[9],"multimodal_custom":{"value":3}}"#)
                .unwrap()
                .0
                .pop()
                .unwrap()
                .mm
                .unwrap()
                .processor_extensions
                .deserialize()
                .unwrap();
        assert_eq!(extensions.multimodal_custom.value, 3);

        for fields in [
            r#"{"multimodal_custom":{"value":true}}"#,
            r#"{"multimodal_custom":{"value":"3"}}"#,
            r#"{"multimodal_custom":{"value":3,"unknown":0}}"#,
            r#"{"multimodal_custom":{}}"#,
        ] {
            let extensions: ProcessorExtensions = serde_json::from_str(fields).unwrap();
            assert!(
                extensions.deserialize::<TestProcessorExtensions>().is_err(),
                "{fields}"
            );
        }

        let (reqs, _) = requests(r#"{"text":"hi","totally_made_up":1}"#).unwrap();
        assert!(reqs[0].mm.is_none());

        let (reqs, _) = requests(r#"{"input_ids":[9],"multimodal_custom":null}"#).unwrap();
        assert!(!reqs[0].has_multimodal());
    }

    /// A scalar broadcast is budget-checked before the deep clones ( MiB × prompts would be GiB and an abort).
    #[test]
    fn oversized_mm_broadcast_rejected() {
        let big = MmItem::Source("x".repeat(MAX_BROADCAST_CLONE_BYTES / 2 + 1));
        let err = multimodal::fan_out(Some(MmDataInput::One(big.clone())), 2, true, "image_data")
            .err()
            .unwrap();
        assert!(err.to_string().contains("broadcast"), "{err}");
        // A per-item list of the same total size moves, not clones: accepted.
        let list = MmDataInput::Many(vec![Some(big), Some(MmItem::Source("y".into()))]);
        assert!(multimodal::fan_out(Some(list), 2, true, "image_data").is_ok());
        // Small scalars broadcast fine.
        let small = MmDataInput::One(MmItem::Source("u1".into()));
        assert!(multimodal::fan_out(Some(small), 2, true, "audio_data").is_ok());
    }

    /// `mm_hashes` rides only on single requests (Python `__getitem__` parity: batches drop it) and moves.
    #[test]
    fn mm_hashes_single_only() {
        let (mut ps, _) =
            requests(r#"{"text": "a", "image_data": "u", "mm_hashes": ["a1b2", "0xff"]}"#).unwrap();
        assert_eq!(ps[0].mm.as_ref().unwrap().mm_hashes, vec!["a1b2", "0xff"]);
        assert_eq!(ps[0].take_mm_work().mm_hashes, vec!["a1b2", "0xff"]);
        assert!(ps[0].mm.as_ref().unwrap().mm_hashes.is_empty());

        // A batch cannot carry hashes (Python drops them), so it is rejected,
        // as is the nested batch shape on a single request...
        for body in [
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": [["x"], ["y"]]}"#,
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": ["x", "y"]}"#,
            r#"{"text": "a", "image_data": "u", "mm_hashes": [["x"]]}"#,
        ] {
            let err = requests(body).err().unwrap();
            assert!(matches!(err, Error::Validation(_)), "{body}: {err:?}");
        }
        // ...while an absent or empty field is not a payload and must still pass.
        for body in [
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": null}"#,
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": []}"#,
        ] {
            assert!(requests(body).is_ok(), "{body}");
        }
    }

    /// `take_mm_work` clones `text` (the scheduler header still needs it) and moves everything the worker owns out.
    #[test]
    fn mm_work_item_takes_owned_fields() {
        let (mut ps, _) =
            requests(r#"{"text": "hi", "image_data": ["u1", "u2"], "audio_data": "a"}"#).unwrap();
        let work = ps[0].take_mm_work();
        assert_eq!(work.text.as_deref(), Some("hi"));
        assert!(work.input_ids.is_none());
        assert_eq!(work.image_data.len(), 2);
        assert!(work.video_data.is_empty());
        assert_eq!(work.audio_data, vec![MmItem::Source("a".into())]);
        // Moved out, not cloned; `text` survives for the header.
        assert!(ps[0].mm.as_ref().unwrap().image_data.is_empty());
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
    }

    /// The body limit is disabled, so an unbounded batch turns a small body into an unbounded allocation.
    #[test]
    fn oversized_batches_are_rejected_before_allocating() {
        let cap = usize::try_from(*MAX_BATCH_REQS_PER_HTTP_REQ).unwrap();
        let texts: Vec<String> = (0..cap + 1).map(|i| i.to_string()).collect();
        let body = serde_json::json!({ "text": texts }).to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("exceeds the maximum"), "{err}");

        // At the cap it is accepted.
        let texts: Vec<String> = (0..cap).map(|i| i.to_string()).collect();
        let (reqs, _) = requests(&serde_json::json!({ "text": texts }).to_string()).unwrap();
        assert_eq!(reqs.len(), cap);

        // A small batch with a huge broadcast `custom_params` is the quadratic case: few items.
        let blob = "x".repeat(1 << 20);
        let body = serde_json::json!({
            "text": vec!["hi"; 200],
            "sampling_params": { "custom_params": { "k": blob } },
        })
        .to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("would allocate more than"), "{err}");
    }

    #[test]
    fn negative_batch_limit_disables_the_item_cap() {
        assert!(!batch_size_exceeds_limit(usize::MAX, -1));
        assert!(batch_size_exceeds_limit(11, 10));
        assert!(!batch_size_exceeds_limit(10, 10));
    }

    /// `token_ids_logprob` mirrors Python `_normalize_batch`'s nested-structure branch: a flat list broadcasts to every prompt.
    #[test]
    fn token_ids_logprob_broadcasts_flat_and_splits_nested() {
        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [1, 2]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![1, 2]));
        assert_eq!(ps[1].token_ids_logprob, Some(vec![1, 2]));

        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[1], [2, 3]]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![1]));
        assert_eq!(ps[1].token_ids_logprob, Some(vec![2, 3]));

        let err = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[1]]}"#).unwrap_err();
        assert!(
            err.to_string().contains("does not match batch size"),
            "{err}"
        );

        let (ps, _) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);
    }

    /// An empty `token_ids_logprob` means "none requested" and must reach the scheduler as None.
    #[test]
    fn empty_token_ids_logprob_collapses_to_none() {
        let (ps, _) = requests(r#"{"text": "a", "token_ids_logprob": []}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);

        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": []}"#).unwrap();
        assert!(ps.iter().all(|p| p.token_ids_logprob.is_none()));

        let (ps, _) =
            requests(r#"{"text": ["a", "b", "c", "d"], "token_ids_logprob": [[], [], [], []]}"#)
                .unwrap();
        assert!(ps.iter().all(|p| p.token_ids_logprob.is_none()));

        // Nested and mixed: only the empty cell collapses.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[], [7]]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);
        assert_eq!(ps[1].token_ids_logprob, Some(vec![7]));

        // A non-empty list is untouched.
        let (ps, _) = requests(r#"{"text": "a", "token_ids_logprob": [7]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![7]));
    }

    /// The logprob/hidden options take Python's batch form too (`Union[List[T], T]`): a scalar broadcasts.
    #[test]
    fn logprob_options_broadcast_scalar_and_split_list() {
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "return_logprob": true, "top_logprobs_num": 3}"#)
                .unwrap();
        assert!(ps[0].return_logprob);
        assert_eq!(ps[1].top_logprobs_num, 3);

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "return_logprob": [true, false],
                "logprob_start_len": [0, 2], "return_hidden_states": [false, true]}"#,
        )
        .unwrap();
        assert!(ps[0].return_logprob);
        assert!(!ps[1].return_logprob);
        assert_eq!(ps[0].logprob_start_len, 0);
        assert_eq!(ps[1].logprob_start_len, 2);
        assert!(ps[1].return_hidden_states);

        let err = requests(r#"{"text": ["a", "b"], "return_logprob": [true]}"#).unwrap_err();
        assert!(
            err.to_string().contains("does not match batch size"),
            "{err}"
        );
    }

    /// `{"input_ids": []}` parses as one prompt with no ids, so the batch-size guard misses it.
    #[test]
    fn empty_input_ids_is_rejected() {
        let err = requests(r#"{"input_ids": []}"#).unwrap_err();
        assert!(
            err.to_string().contains("input_ids cannot be empty"),
            "{err}"
        );

        let err = requests(r#"{"input_ids": [[1, 2], []]}"#).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"), "{err}");

        assert!(requests(r#"{"input_ids": [1, 2]}"#).is_ok());
        assert!(requests(r#"{"input_ids": [[1], [2]]}"#).is_ok());
    }

    /// items in one request cannot share an rid.
    #[test]
    fn duplicate_rids_within_one_request_are_rejected() {
        let err = requests(r#"{"text": ["a", "b"], "rid": ["x", "x"]}"#).unwrap_err();
        assert!(err.to_string().contains("duplicate request IDs"), "{err}");

        assert!(requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).is_ok());
        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "x"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "x_0");
        assert_eq!(ps[1].rid.client_facing(), "x_1");
    }

    /// The collision this whole scheme exists to prevent: CONCURRENT requests naming the same rid.
    #[test]
    fn concurrent_requests_sharing_an_rid_get_distinct_internal_rids() {
        let (a, _) = requests(r#"{"text": "a", "rid": "same"}"#).unwrap();
        let (b, _) = requests(r#"{"text": "b", "rid": "same"}"#).unwrap();
        assert_ne!(
            a[0].rid, b[0].rid,
            "a shared client rid must not become a shared internal rid"
        );
        assert_eq!(a[0].rid.client_facing(), "same");
        assert_eq!(b[0].rid.client_facing(), "same");
    }

    /// PD bootstrap fields fan out like Python `_normalize_bootstrap_params`: scalars broadcast.
    #[test]
    fn bootstrap_fields_fan_out() {
        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": "h", "bootstrap_port": 8998,
                "bootstrap_room": 7, "routed_dp_rank": 1}"#,
        )
        .unwrap();
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(p.bootstrap_host.as_deref(), Some("h"));
            assert_eq!(p.bootstrap_port, Some(8998));
            assert_eq!(p.bootstrap_room, Some(7 + i as i64));
            assert_eq!(p.routed_dp_rank, Some(1));
        }

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": ["h1", "h2"],
                "bootstrap_room": [10, 20]}"#,
        )
        .unwrap();
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("h1"));
        assert_eq!(ps[1].bootstrap_host.as_deref(), Some("h2"));
        assert_eq!(ps[0].bootstrap_room, Some(10));
        assert_eq!(ps[1].bootstrap_room, Some(20));

        let err = requests(r#"{"text": ["a", "b"], "bootstrap_room": [1, 2, 3]}"#).unwrap_err();
        assert!(err.to_string().contains("bootstrap_room"), "{err}");
    }

    /// The PD router (mini_lb) and PD-warmup payload shapes must parse.
    #[test]
    fn accepts_pd_router_and_warmup_payloads() {
        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": ["h", "h"],
                "bootstrap_port": [null, null],
                "bootstrap_room": [123456789, 987654321]}"#,
        )
        .unwrap();
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("h"));
        assert_eq!(ps[0].bootstrap_port, None);
        assert_eq!(ps[1].bootstrap_room, Some(987654321));

        let (ps, is_batch) = requests(
            r#"{"sampling_params": {"temperature": 0.0, "max_new_tokens": 8,
                                    "ignore_eos": true},
                "bootstrap_host": "2.2.2.2", "bootstrap_room": 0,
                "input_ids": [10, 11, 12, 13], "routed_dp_rank": 0}"#,
        )
        .unwrap();
        assert!(!is_batch);
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("2.2.2.2"));
        assert_eq!(ps[0].bootstrap_room, Some(0));
        assert_eq!(ps[0].routed_dp_rank, Some(0));
    }
}
