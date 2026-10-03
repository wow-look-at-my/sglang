//! Parser module for function calls and reasoning extraction This module provides parsing operations for model output, including.

mod handlers;

pub use handlers::{parse_function_call, parse_reasoning};
