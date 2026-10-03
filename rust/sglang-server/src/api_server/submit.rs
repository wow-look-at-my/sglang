//! Request submission into the to_scheduler pipeline, shared by every endpoint module: mint the client-visible rid.

use axum::{http::StatusCode, response::Response};
use tokio::sync::mpsc;

use super::app::AppState;
use super::native_api::native_error;
use crate::message::ids::Rid;
use crate::message::request::{Request, RequestKind};
use crate::message::response::{ResponseItem, ResponseSink};
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::fsm::RequestState;

/// Submit one request; returns its rid and the response receiver.
/// `into_requests` (or the `HEALTH_CHECK_<uuid>` the health probe sets).
/// control request from its constructor — so this only echoes it back.
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
    // `stream`: the client is reading an SSE stream.
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<ResponseItem>), Response> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the client's, or minted one.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().into(),
        // Internal service call — no client-facing rid; mint a fresh one.
        RequestKind::Detokenize { .. } => Rid::new(),
    };
    let (tx, rx) = mpsc::channel::<ResponseItem>(state.response_buf);
    let request = Request {
        rid: rid.clone(),
        state: RequestState::Received,
        sink: ResponseSink::Local(tx),
        kind,
    };
    match state
        .senders
        .tok_manager_tx
        .send_async(TmEvent::Intake(request))
        .await
    {
        Ok(()) => Ok((rid, rx)),
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            Err(native_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                stream,
            ))
        }
    }
}
