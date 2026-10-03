//! WebAssembly (WASM) module support.

// Re-export everything from smg-wasm
pub use smg_wasm::*;

// Local HTTP API routes (depends on app-specific types)
pub mod route;
