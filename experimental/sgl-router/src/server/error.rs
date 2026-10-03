// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

pub const X_ROUTER_ERROR_CODE: HeaderName = HeaderName::from_static("x-router-error-code");

/// Carries the worker's *own* HTTP status when the router had to synthesize a
/// status of its own over a worker that did respond.
pub const X_ROUTER_UPSTREAM_STATUS: HeaderName =
    HeaderName::from_static("x-router-upstream-status");

/// Coarse failure class for a router-originated error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorClass {
    BadRequest,
    NotFound,
    Upstream,
    NoTarget,
    Timeout,
    Internal,
}

impl ErrorClass {
    fn status(self) -> StatusCode {
        match self {
            ErrorClass::BadRequest => StatusCode::BAD_REQUEST,
            ErrorClass::NotFound => StatusCode::NOT_FOUND,
            ErrorClass::Upstream => StatusCode::BAD_GATEWAY,
            ErrorClass::NoTarget => StatusCode::SERVICE_UNAVAILABLE,
            ErrorClass::Timeout => StatusCode::GATEWAY_TIMEOUT,
            ErrorClass::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// A request refused by the fleet-wide sampling contract.
    #[error("{param} violates this deployment's sampling contract: {detail}")]
    SamplingContract { param: &'static str, detail: String },

    /// Could not reach the upstream worker (connect refused, DNS, TLS, request build error).
    #[error("upstream unreachable: worker {worker}")]
    UpstreamUnreachable {
        worker: reqwest::Url,
        #[source]
        source: anyhow::Error,
    },

    /// The worker started a response (status + headers received) but failed to deliver the full body — mid-body socket drop, framing error.
    #[error("upstream response body incomplete after status {status}")]
    UpstreamStatus { status: StatusCode },

    /// Wall-clock timeout exceeded while waiting for the upstream worker's response (per-request `request_timeout`).
    #[error("upstream timed out: worker {worker}")]
    UpstreamTimeout { worker: reqwest::Url },

    /// No healthy worker is available for `model`: either none were ever registered.
    #[error("no healthy workers for model {model}")]
    NoHealthyWorkers { model: String },

    /// PD-mode deployment whose prefill pool has zero healthy workers.
    #[error("no prefill workers available for model {model}")]
    NoPrefillWorkersAvailable { model: String },

    /// PD-mode deployment whose decode pool has zero healthy workers. Mirror of [`Self::NoPrefillWorkersAvailable`].
    #[error("no decode workers available for model {model}")]
    NoDecodeWorkersAvailable { model: String },

    /// A request whose lifetime exceeded `stale_request_timeout` — the active-load janitor force-expired the in-flight bookkeeping.
    #[error("stale request expired for model {model}")]
    StaleRequestExpired { model: String },

    /// The per-model policy returned `None` despite the candidate set being non-empty.
    #[error("policy selected no worker for model {model}")]
    PolicySelectionFailed { model: String },

    /// The worker's circuit breaker was open at the moment of dispatch.
    #[error("worker circuit breaker open: {worker}")]
    BreakerOpen { worker: String },

    /// The worker URL emitted by discovery failed to parse.
    #[error("worker misconfigured: {worker}")]
    WorkerMisconfigured {
        worker: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    /// The failure class — the sole determinant of the HTTP status. Grouping by
    /// class is what makes the status non-divergent.
    /// [`ErrorClass::Timeout`], so a per-request timeout and a stale-deadline
    /// cancel are guaranteed the same status, and no future variant can quietly
    /// pick a different one.
    fn class(&self) -> ErrorClass {
        match self {
            ApiError::BadRequest(_) => ErrorClass::BadRequest,
            ApiError::SamplingContract { .. } => ErrorClass::BadRequest,
            ApiError::ModelNotFound(_) => ErrorClass::NotFound,
            ApiError::UpstreamUnreachable { .. } => ErrorClass::Upstream,
            ApiError::UpstreamStatus { .. } => ErrorClass::Upstream,
            ApiError::UpstreamTimeout { .. } => ErrorClass::Timeout,
            ApiError::NoHealthyWorkers { .. } => ErrorClass::NoTarget,
            ApiError::NoPrefillWorkersAvailable { .. } => ErrorClass::NoTarget,
            ApiError::NoDecodeWorkersAvailable { .. } => ErrorClass::NoTarget,
            ApiError::StaleRequestExpired { .. } => ErrorClass::Timeout,
            ApiError::PolicySelectionFailed { .. } => ErrorClass::NoTarget,
            ApiError::BreakerOpen { .. } => ErrorClass::NoTarget,
            ApiError::WorkerMisconfigured { .. } => ErrorClass::NoTarget,
            ApiError::Internal(_) => ErrorClass::Internal,
        }
    }

    /// Stable, machine-readable `x-router-error-code` — the authoritative signal
    /// a gateway converts on. Distinct per condition even when conditions
    fn error_code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::SamplingContract { .. } => "sampling_contract_violation",
            ApiError::ModelNotFound(_) => "model_not_found",
            ApiError::UpstreamUnreachable { .. } => "upstream_unreachable",
            // Mid-body drop: headers (incl. a status) arrived, then the body didn't.
            ApiError::UpstreamStatus { .. } => "upstream_body_incomplete",
            ApiError::UpstreamTimeout { .. } => "upstream_timeout",
            ApiError::NoHealthyWorkers { .. } => "no_healthy_workers",
            ApiError::NoPrefillWorkersAvailable { .. } => "no_prefill_workers_available",
            ApiError::NoDecodeWorkersAvailable { .. } => "no_decode_workers_available",
            ApiError::StaleRequestExpired { .. } => "stale_request_expired",
            ApiError::PolicySelectionFailed { .. } => "policy_selection_failed",
            ApiError::BreakerOpen { .. } => "breaker_open",
            ApiError::WorkerMisconfigured { .. } => "worker_misconfigured",
            ApiError::Internal(_) => "internal_error",
        }
    }

    /// The worker's *own* status to echo in `x-router-upstream-status`, for the
    /// case where the router synthesized its own status over a worker that did
    /// respond. Today only the mid-body-drop (`UpstreamStatus`) carries one. This
    /// is an exhaustive.
    /// a future "synthesized over a responding worker" variant is forced to decide
    /// whether it echoes a status, rather than silently inheriting `None`.
    fn upstream_status(&self) -> Option<StatusCode> {
        match self {
            ApiError::UpstreamStatus { status } => Some(*status),
            ApiError::BadRequest(_)
            | ApiError::SamplingContract { .. }
            | ApiError::ModelNotFound(_)
            | ApiError::UpstreamUnreachable { .. }
            | ApiError::UpstreamTimeout { .. }
            | ApiError::NoHealthyWorkers { .. }
            | ApiError::NoPrefillWorkersAvailable { .. }
            | ApiError::NoDecodeWorkersAvailable { .. }
            | ApiError::StaleRequestExpired { .. }
            | ApiError::PolicySelectionFailed { .. }
            | ApiError::BreakerOpen { .. }
            | ApiError::WorkerMisconfigured { .. }
            | ApiError::Internal(_) => None,
        }
    }

    /// The HTTP status this error maps to — the same value the client
    /// receives via `into_response`.
    pub fn status_code(&self) -> StatusCode {
        self.class().status()
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    #[serde(rename = "type")]
    typ: &'static str,
    code: &'a str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.class().status();
        let code = self.error_code();
        let typ = match status.as_u16() {
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        // Pick a client-facing message that NEVER leaks worker URLs or raw
        // source chains; full structured details are logged server-side.
        let message = match &self {
            ApiError::Internal(e) => {
                // `{:#}` prints the anyhow chain (top error + sources) — `?e` would only show the outermost message.
                tracing::error!("internal error serving request: {e:#}");
                "internal error".to_string()
            }
            ApiError::UpstreamUnreachable { worker, source } => {
                tracing::warn!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "upstream worker unreachable",
                );
                "upstream unavailable".to_string()
            }
            ApiError::UpstreamStatus { status } => {
                tracing::warn!(
                    upstream_status = %status,
                    "upstream response body did not complete",
                );
                "upstream response was incomplete".to_string()
            }
            ApiError::UpstreamTimeout { worker } => {
                tracing::warn!(upstream = %worker, "upstream request timed out");
                "upstream request timed out".to_string()
            }
            ApiError::NoHealthyWorkers { model } => {
                tracing::warn!(model = %model, reason = "no_healthy_workers", "service unavailable");
                "no healthy workers for the requested model".to_string()
            }
            ApiError::NoPrefillWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_prefill_workers_available",
                    "service unavailable",
                );
                "no prefill workers available for the requested model".to_string()
            }
            ApiError::NoDecodeWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_decode_workers_available",
                    "service unavailable",
                );
                "no decode workers available for the requested model".to_string()
            }
            ApiError::StaleRequestExpired { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "stale_request_expired",
                    "stale-request janitor expired in-flight request",
                );
                "request expired before completion".to_string()
            }
            ApiError::PolicySelectionFailed { .. } => "service unavailable".to_string(),
            ApiError::BreakerOpen { worker } => {
                tracing::warn!(upstream = %worker, reason = "breaker_open", "service unavailable");
                "service unavailable".to_string()
            }
            ApiError::WorkerMisconfigured { worker, source } => {
                tracing::error!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "worker URL emitted by discovery is malformed",
                );
                "service unavailable".to_string()
            }
            ApiError::BadRequest(_)
            | ApiError::ModelNotFound(_)
            | ApiError::SamplingContract { .. } => self.to_string(),
        };
        let mut resp = (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody { typ, code, message },
            }),
        )
            .into_response();
        resp.headers_mut()
            .insert(X_ROUTER_ERROR_CODE, HeaderValue::from_static(code));
        // Preserve the worker's real status when we synthesized our own.
        if let Some(upstream) = self.upstream_status() {
            resp.headers_mut().insert(
                X_ROUTER_UPSTREAM_STATUS,
                HeaderValue::from(upstream.as_u16()),
            );
        }
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use serde::Deserialize;

    fn collect_body(resp: Response) -> String {
        let bytes = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BodyExt::collect(resp.into_body()).await.unwrap().to_bytes() });
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Pin the exact JSON envelope shape that clients see.
    #[derive(Deserialize)]
    struct ErrEnv {
        error: ErrField,
    }

    #[derive(Deserialize)]
    struct ErrField {
        #[serde(rename = "type")]
        typ: String,
        code: String,
        message: String,
    }

    fn parse_envelope(resp: Response) -> (StatusCode, Option<String>, ErrEnv) {
        let status = resp.status();
        let code_header = resp
            .headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let body_str = collect_body(resp);
        let env: ErrEnv = serde_json::from_str(&body_str)
            .unwrap_or_else(|e| panic!("envelope did not match expected shape: {e}: {body_str}"));
        (status, code_header, env)
    }

    #[test]
    fn upstream_unreachable_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let secret = "TLS_HANDSHAKE_FAILED at /etc/secret_ca.pem";
        let err = ApiError::UpstreamUnreachable {
            worker: worker.clone(),
            source: anyhow::anyhow!("{secret}"),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_unreachable"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_unreachable\""), "{body}");
        assert!(body.contains("\"type\":\"server_error\""), "{body}");
        assert!(
            !body.contains(worker_str) && !body.contains(secret),
            "client body must NOT leak worker URL or reqwest source chain; got: {body}",
        );
    }

    #[test]
    fn upstream_body_incomplete_preserves_worker_status_in_header() {
        let err = ApiError::UpstreamStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_body_incomplete"),
        );
        assert_eq!(
            resp.headers()
                .get("x-router-upstream-status")
                .and_then(|v| v.to_str().ok()),
            Some("500"),
            "the worker's real status must be preserved, not discarded",
        );
        let body = collect_body(resp);
        assert!(
            body.contains("\"code\":\"upstream_body_incomplete\""),
            "{body}"
        );
    }

    #[test]
    fn upstream_timeout_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let err = ApiError::UpstreamTimeout {
            worker: worker.clone(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_timeout"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_timeout\""), "{body}");
        assert!(
            !body.contains(worker_str),
            "client body must NOT leak worker URL; got: {body}",
        );
    }

    #[test]
    fn policy_selection_failed_envelope_is_unchanged() {
        let resp = ApiError::PolicySelectionFailed {
            model: "tiny".into(),
        }
        .into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(code_header.as_deref(), Some("policy_selection_failed"));
        assert_eq!(env.error.typ, "server_error");
        assert_eq!(env.error.code, "policy_selection_failed");
        assert_eq!(env.error.message, "service unavailable");
    }

    #[test]
    fn bad_request_envelope_has_expected_shape() {
        let msg = "invalid_request: body must be an object";
        let err = ApiError::BadRequest(msg.into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code_header.as_deref(), Some("bad_request"));
        assert_eq!(env.error.code, "bad_request");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "model_not_found");
    }

    #[test]
    fn model_not_found_envelope_has_expected_shape() {
        let err = ApiError::ModelNotFound("ghost-7b".into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(code_header.as_deref(), Some("model_not_found"));
        assert_eq!(env.error.code, "model_not_found");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "bad_request");
    }

    #[test]
    fn internal_error_response_sanitizes_anyhow_chain() {
        let secret_msg = "internal /opt/secret/credential.json missing";
        let err = ApiError::Internal(anyhow::anyhow!("{secret_msg}"));
        let resp = err.into_response();
        let body_str = collect_body(resp);
        // Generic to client:
        assert!(
            body_str.contains("\"code\":\"internal_error\""),
            "body: {body_str}"
        );
        assert!(
            body_str.contains("\"type\":\"server_error\""),
            "body: {body_str}"
        );
        // No leak of the anyhow message:
        assert!(
            !body_str.contains(secret_msg),
            "ApiError::Internal must not leak anyhow chain to client; got: {body_str}"
        );
    }
    /// A sampling-contract rejection must not be filed under `bad_request`.
    #[test]
    fn sampling_contract_has_a_distinct_code_from_other_bad_requests() {
        let err = ApiError::SamplingContract {
            param: "temperature",
            detail: "got 0.5, expected 1 (or omit the field)".into(),
        };
        let (status, code_header, env) = parse_envelope(err.into_response());
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code_header.as_deref(), Some("sampling_contract_violation"));
        assert_eq!(env.error.code, "sampling_contract_violation");
        assert_ne!(env.error.code, "bad_request");
        assert!(
            env.error.message.contains("temperature") && env.error.message.contains("0.5"),
            "got: {}",
            env.error.message
        );
    }

    /// Those client-facing signals of the router status-code contract: the
    /// HTTP status, `x-router-error-code`, and `x-router-upstream-status`.
    fn signals(err: ApiError) -> (StatusCode, Option<String>, Option<String>) {
        let resp = err.into_response();
        let status = resp.status();
        let header = |name: &str| {
            resp.headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        (
            status,
            header("x-router-error-code"),
            header("x-router-upstream-status"),
        )
    }

    /// Every router-*originated* condition maps to a fixed (status, error-code) pair.
    #[test]
    fn router_originated_scenarios_match_status_and_headers() {
        let worker = reqwest::Url::parse("http://host:1/").unwrap();
        // (label, error, expected status, expected x-router-error-code,
        //  expected x-router-upstream-status)
        let cases: Vec<(&str, ApiError, StatusCode, &str, Option<&str>)> = vec![
            (
                "mid-body drop preserves worker status",
                ApiError::UpstreamStatus {
                    status: StatusCode::OK,
                },
                StatusCode::BAD_GATEWAY,
                "upstream_body_incomplete",
                Some("200"),
            ),
            (
                "unreachable",
                ApiError::UpstreamUnreachable {
                    worker: worker.clone(),
                    source: anyhow::anyhow!("connect refused"),
                },
                StatusCode::BAD_GATEWAY,
                "upstream_unreachable",
                None,
            ),
            (
                "request timeout",
                ApiError::UpstreamTimeout {
                    worker: worker.clone(),
                },
                StatusCode::GATEWAY_TIMEOUT,
                "upstream_timeout",
                None,
            ),
            (
                "stale-deadline cancel",
                ApiError::StaleRequestExpired { model: "m".into() },
                StatusCode::GATEWAY_TIMEOUT,
                "stale_request_expired",
                None,
            ),
            (
                "no healthy workers",
                ApiError::NoHealthyWorkers { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_healthy_workers",
                None,
            ),
            (
                "bad request",
                ApiError::BadRequest("bad".into()),
                StatusCode::BAD_REQUEST,
                "bad_request",
                None,
            ),
            (
                "sampling contract rejection",
                ApiError::SamplingContract {
                    param: "temperature",
                    detail: "got 0.5, expected 1".into(),
                },
                StatusCode::BAD_REQUEST,
                "sampling_contract_violation",
                None,
            ),
            (
                "model not found",
                ApiError::ModelNotFound("ghost".into()),
                StatusCode::NOT_FOUND,
                "model_not_found",
                None,
            ),
            (
                "no prefill workers",
                ApiError::NoPrefillWorkersAvailable { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_prefill_workers_available",
                None,
            ),
            (
                "no decode workers",
                ApiError::NoDecodeWorkersAvailable { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_decode_workers_available",
                None,
            ),
            (
                "policy selection failed",
                ApiError::PolicySelectionFailed { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "policy_selection_failed",
                None,
            ),
            (
                "breaker open",
                ApiError::BreakerOpen {
                    worker: "http://w:1".into(),
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "breaker_open",
                None,
            ),
            (
                "worker misconfigured",
                ApiError::WorkerMisconfigured {
                    worker: "http://w:1".into(),
                    source: anyhow::anyhow!("unparsable url"),
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_misconfigured",
                None,
            ),
            (
                "internal",
                ApiError::Internal(anyhow::anyhow!("boom")),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                None,
            ),
        ];
        for (label, err, want_status, want_code, want_upstream) in cases {
            let (status, code, upstream) = signals(err);
            assert_eq!(status, want_status, "status mismatch for `{label}`");
            assert_eq!(
                code.as_deref(),
                Some(want_code),
                "x-router-error-code mismatch for `{label}`",
            );
            assert_eq!(
                upstream.as_deref(),
                want_upstream,
                "x-router-upstream-status mismatch for `{label}`",
            );
        }
    }
}
