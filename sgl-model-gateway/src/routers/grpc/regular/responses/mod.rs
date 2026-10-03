//! Regular gRPC Router `/v1/responses` endpoint.

mod common;
mod conversions;
mod handlers;
mod non_streaming;
mod streaming;

// Public exports
pub(crate) use handlers::route_responses;
