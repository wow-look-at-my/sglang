// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! sgl-router: slim KV-aware OpenAI-compatible router for SGLang workers.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod buckets_reorg;
pub mod config;
pub mod discovery;
pub mod health;
pub mod policies;
pub mod policies_reorg;
pub mod proxy;
pub mod server;
pub mod state;
pub mod tokenizer;
pub mod workers;
