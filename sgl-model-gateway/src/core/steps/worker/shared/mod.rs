//! Shared worker registration steps used by both local and external workflows.

mod activate;
mod register;
mod update_policies;

use std::sync::Arc;

pub use activate::ActivateWorkersStep;
pub use register::RegisterWorkersStep;
pub use update_policies::UpdatePoliciesStep;

use crate::core::Worker;

/// Type alias for a collection of workers in workflow context.
pub type WorkerList = Vec<Arc<dyn Worker>>;
