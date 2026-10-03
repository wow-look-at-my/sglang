//! Harmony pipeline.

pub(crate) mod builder;
pub(crate) mod detector;
pub(crate) mod parser;
pub(crate) mod processor;
pub(crate) mod responses;
pub(crate) mod stages;
pub(crate) mod streaming;
pub(crate) mod types;

// Re-export types that are accessed via harmony::TypeName
pub(crate) use builder::HarmonyBuilder;
pub(crate) use detector::HarmonyDetector;
pub(crate) use parser::HarmonyParserAdapter;
pub(crate) use processor::{HarmonyResponseProcessor, ResponsesIterationResult};
pub(crate) use responses::{serve_harmony_responses, serve_harmony_responses_stream};
pub(crate) use streaming::HarmonyStreamingProcessor;
pub(crate) use types::HarmonyMessage;
