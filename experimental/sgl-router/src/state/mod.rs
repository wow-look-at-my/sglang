// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared mutable state that selection reads: the KV-event cache index, load monitoring, and affinity assignments.

pub mod affinity_store;
pub mod kv_events;
pub mod load_monitor;

pub use affinity_store::AffinityStore;
