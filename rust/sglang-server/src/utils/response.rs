//! Shared HTTP error-response shaping for the api-server endpoint modules.

use std::convert::Infallible;

use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};

/// The native error body — the same `{"error": {...}}` object every native
/// path emits, not bare text, which a client parsing JSON chokes on.
pub fn error_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "code": code } })
}

/// Unary native-shape error response: `code` + [`error_value`] body.
pub fn json_error(code: StatusCode, message: &str) -> Response {
    error_response(code, error_value(code.as_u16(), message), false)
}

/// Form an error in the shape the client committed to: unary → `code` plus
/// the JSON `body`.
pub fn error_response(code: StatusCode, body: serde_json::Value, stream: bool) -> Response {
    if !stream {
        return (code, Json(body)).into_response();
    }
    sse_error_response(body)
}

/// Shared by every
/// endpoint family: the native API and the OpenAI
/// frontend's `openai_error_response`.
pub fn sse_error_response(body: serde_json::Value) -> Response {
    let frames = [body.to_string(), "[DONE]".to_string()];
    Sse::new(futures::stream::iter(
        frames.map(|data| Ok::<_, Infallible>(Event::default().data(data))),
    ))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python-parity pins for the native error shapes (`generate_request`): unary errors are a 4xx/5xx with a JSON `{"error": ...}` body.
    #[tokio::test]
    async fn error_responses_match_python_shape() {
        let unary = error_response(
            StatusCode::BAD_REQUEST,
            error_value(400, "bad input"),
            false,
        );
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(unary.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["code"], 400);

        let streamed = error_response(StatusCode::BAD_REQUEST, error_value(400, "bad input"), true);
        assert_eq!(
            streamed.status(),
            StatusCode::OK,
            "the stream itself is 200"
        );
        let body = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains(r#""code":400"#),
            "carries the status in-band: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "terminated: {text}"
        );
    }
}
