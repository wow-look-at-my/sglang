//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime.
pub mod app;
mod common;
mod disaggregation;
mod frame;
mod guard;
mod log;
mod native_api;
mod openai;
mod prefetch;
mod submit;
