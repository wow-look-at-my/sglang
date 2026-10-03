//! OpenAI-compatible responses handling.

mod accumulator;
mod common;
mod mcp;
mod non_streaming;
mod streaming;
mod tool_handler;
mod utils;

pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;
