//! sglang-mm: Rust-accelerated multimodal preprocessing for SGLang.

pub mod common;
pub mod driver;
pub mod dsv41;
pub mod inkling;
pub mod pipeline;
pub mod qwen_vl;
pub mod registry;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _multimodal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    common::register(m)?;
    inkling::register(m)?;
    dsv41::register(m)?;
    qwen_vl::register(m)?;
    Ok(())
}
