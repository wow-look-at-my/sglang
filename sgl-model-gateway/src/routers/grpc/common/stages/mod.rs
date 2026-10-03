//! Common pipeline stages shared across all endpoints and model types These stages are endpoint-agnostic and model-agnostic.

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::context::RequestContext;

/// Trait for pipeline stages that process requests
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Execute this stage, mutating the context Returns: - `Ok(None)` - Continue to next stage - `Ok(Some(response))` - Pipeline complete.
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;
}

mod client_acquisition;
mod dispatch_metadata;
pub(crate) mod helpers;
mod request_execution;
mod worker_selection;

// Export stage implementations
pub(crate) use client_acquisition::ClientAcquisitionStage;
pub(crate) use dispatch_metadata::DispatchMetadataStage;
pub(crate) use request_execution::{ExecutionMode, RequestExecutionStage};
pub(crate) use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
