//! Per-component drivers; each receives the whole `UnifiedTreeCore` for backward access.
#![allow(unused_variables)]

use std::collections::HashMap;

use tch::Tensor;

use crate::node::{ChildKeyType, NodeArena, NodeIdx_, TreeCoreRuntimeError};
use crate::unified_tree_core::{
    CacheAction, CacheTransferPhase, DecLockRefParams, EvictLayer, IncLockRefResult, InsertParams,
    InsertResult, LRURefreshPhase, MatchPrefixParams, MatchResult, PoolTransfer,
    PoolTransferResult, UnifiedTreeCore,
};

mod full;
mod mamba;
mod swa;

pub use full::FullComponent;
pub use mamba::MambaComponent;
pub use swa::SwaComponent;

/// Whether `node_id` holds the component's data on `target`, checking its
/// device or host slot.
pub(crate) fn node_has_component_data<K: ChildKeyType>(
    arena: &NodeArena<K>,
    node_id: NodeIdx_,
    component_type: ComponentType,
    target: EvictLayer,
) -> bool {
    match target {
        EvictLayer::Device => arena.has_device_value(node_id, component_type),
        EvictLayer::Host => arena.has_host_value(node_id, component_type),
        EvictLayer::All => panic!("node_has_component_data: EvictLayer::All is not a single layer"),
    }
}

/// Every device value of the component across all roots, concatenated.
pub(crate) fn all_values_flatten<K: ChildKeyType>(
    tree_core: &UnifiedTreeCore<K>,
    component_type: ComponentType,
) -> Tensor {
    let mut values: Vec<Tensor> = Vec::new();
    let mut stack: Vec<NodeIdx_> = vec![tree_core.arena.root()];
    while let Some(node_id) = stack.pop() {
        let node = tree_core.arena.node(node_id);
        if let Some(value) = node.try_device_value(component_type) {
            values.push(value.shallow_clone());
        }
        stack.extend(node.children.values().copied());
    }
    if values.is_empty() {
        return tree_core.empty_device_indices.shallow_clone();
    }
    Tensor::cat(&values, 0)
}

/// A per-component lock/value/eviction driver over the shared `UnifiedTreeCore`.
pub trait TreeComponent<K: ChildKeyType> {
    /// The component this driver serves.
    fn component_type(&self) -> ComponentType;

    /// Whether this component has device data that still needs a host backup.
    fn needs_incremental_backup(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        _node_id: NodeIdx_,
    ) -> bool {
        false
    }

    /// Refresh this component's LRU position for `node_id` at the given walk phase.
    fn refresh_lru(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        phase: LRURefreshPhase,
        node_id: NodeIdx_,
    ) {
        // Python reference — base.py::TreeComponent.refresh_lru.
        unimplemented!("TreeComponent.refresh_lru")
    }

    /// Return a per-match stateful predicate deciding whether a node is a valid
    /// match boundary for this component.
    fn create_match_validator(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool>;

    /// Tree-side post-processing inside the match walk (no cache access).
    fn finalize_match_result_in_tree_core(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        result: MatchResult,
        _last_device_node_idx: NodeIdx_,
        _best_match_node_idx: NodeIdx_,
        params: &MatchPrefixParams<'_, K>,
        value_chunks: &[Tensor],
        best_value_len: usize,
    ) -> MatchResult {
        result
    }

    /// Called per-node when an insert's key overlaps an existing node.
    /// Returns the index within `value_slice` from which this component
    /// consumed (took ownership of) the underlying KV pool slots. Returns
    /// `prefix_len` if nothing was consumed (default).
    fn update_component_on_insert_overlap(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        prefix_len: usize,
        total_prefix_len: usize,
        value_slice: Tensor,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) -> usize {
        prefix_len
    }

    /// Called after `unevict_node_on_insert_` restores the base (Full) value
    /// on an evicted node. Aux components (e.g. SWA) override this to rebuild
    /// their own data from the freshly assigned base value when their entry
    /// is still tombstoned.
    fn recover_after_unevict(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        prefix_len: usize,
        total_prefix_len: usize,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
    }

    /// Finalize component data on the target (leaf) node after the insert
    /// walk completes.
    fn commit_insert_component_data(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        is_new_leaf: bool,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
    }

    /// Evict shallow device checkpoints beyond the per-path state cap on the
    /// tail's root path; only the Mamba component caps its states.
    fn evict_excess_path_states(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tail_node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }

    /// Redistribute component data between `new_parent` and `child` when a node is
    /// split; `new_parent` is the newly created prefix node.
    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    );

    /// Free this component's KV resources on a node being evicted; returns
    /// (device_freed, host_freed) token counts.
    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize);

    /// Eviction priority on this node type; higher = evicted later.
    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        0
    }

    /// Begin this component's device-eviction walk (build its cursor/heap).
    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize);

    /// Advance one eviction step and return a device leaf, if selected.
    ///
    /// Implementations must return after one allocator-relevant internal
    /// mutation so the caller can drain pending frees before continuing.
    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_>;

    /// Clear this component's device-eviction walk state.
    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>);

    /// Increment component lock refs, protecting nodes from eviction.
    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult;

    /// Decrement component lock refs, un-protecting nodes.
    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: &DecLockRefParams,
        lock_host: bool,
    );

    /// Early-release the SWA lock along [node, swa_uuid_for_lock] while leaving
    /// the other components' locks intact; only the SWA component supports it.
    fn release_window_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _node_id: NodeIdx_,
        _params: &DecLockRefParams,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        unimplemented!("release_window_lock is SWA-only")
    }

    /// Build transfer descriptors for this component in the given phase; None when
    /// the component has nothing to transfer.
    fn build_hicache_transfers(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        mamba_pool_idx: Option<Tensor>,
        host_indices: Option<Tensor>,
        token_ids: Option<&[i64]>,
        prefetch_tokens: usize,
        staging_tokens: usize,
        last_hash: Option<&str>,
    ) -> Result<Option<Vec<PoolTransfer>>, TreeCoreRuntimeError> {
        // Python reference — base.py::TreeComponent.build_hicache_transfers.
        unimplemented!("TreeComponent.build_hicache_transfers")
    }

    /// Build this component's direct device-to-external-store transfer for a node.
    fn build_external_linker_offload_transfer(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        _node_id: NodeIdx_,
    ) -> Option<PoolTransfer> {
        None
    }

    /// Post-transfer bookkeeping: store host indices, update LRU, etc.
    fn commit_hicache_transfer(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        transfers: Vec<PoolTransfer>,
        cache_actions: &mut Vec<CacheAction>,
        insert_result: Option<&mut InsertResult>,
        pool_storage_result: Option<&PoolTransferResult>,
    ) {
        // Python reference — base.py::TreeComponent.commit_hicache_transfer.
        unimplemented!("TreeComponent.commit_hicache_transfer")
    }

    /// Reclaim host values that coexist with device values before ordinary
    /// host eviction. Called only under the write-back policy.
    fn reclaim_coexisting_host_values(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _num_tokens: usize,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }

    /// Evict from this component's host-side resources.
    /// Called by HostPoolGroup when the host pool is full.
    /// Default no-op for components without host storage.
    fn drive_host_eviction(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _num_tokens: usize,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }
}

// Tree component types.

/// The tree components; discriminants define the per-component array indexes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ComponentType {
    Full = 0,
    Swa = 1,
    Mamba = 2,
}

/// Short call-site aliases for the component types.
pub const FULL: ComponentType = ComponentType::Full;
pub const SWA: ComponentType = ComponentType::Swa;
pub const MAMBA: ComponentType = ComponentType::Mamba;

/// The base component every tree runs; the others are auxiliary.
pub const BASE_COMPONENT_TYPE: ComponentType = ComponentType::Full;

/// Slots per tier — the arrays are sized to this, not the enabled subset.
pub const NUM_COMPONENT_TYPES: usize = ComponentType::Mamba as usize + 1;

/// A set of component types (bitmask over `ComponentType::idx`), e.g. the components an `inc_lock_ref` left untaken.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct ComponentSet(u8);

impl ComponentSet {
    pub const EMPTY: ComponentSet = ComponentSet(0);

    /// The set holding exactly one component.
    pub const fn of(component_type: ComponentType) -> ComponentSet {
        ComponentSet(1 << component_type.idx())
    }

    pub fn insert(&mut self, component_type: ComponentType) {
        self.0 |= 1 << component_type.idx();
    }

    pub const fn contains(self, component_type: ComponentType) -> bool {
        self.0 & (1 << component_type.idx()) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The members, in component-index order.
    pub fn iter(self) -> impl Iterator<Item = ComponentType> {
        (0..NUM_COMPONENT_TYPES)
            .filter(move |idx| self.0 & (1 << idx) != 0)
            .map(ComponentType::from_idx)
    }
}

impl FromIterator<ComponentType> for ComponentSet {
    fn from_iter<I: IntoIterator<Item = ComponentType>>(iter: I) -> Self {
        let mut set = ComponentSet::EMPTY;
        for component_type in iter {
            set.insert(component_type);
        }
        set
    }
}

impl ComponentType {
    /// Index into a per-component array.
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Whether the component stores one state slot per node (Mamba) instead of
    /// one row per key atom.
    pub fn single_value_per_node(self) -> bool {
        matches!(self, ComponentType::Mamba)
    }

    /// The component at a per-component array index; panics out of range.
    pub fn from_idx(idx: usize) -> ComponentType {
        match idx {
            0 => ComponentType::Full,
            1 => ComponentType::Swa,
            2 => ComponentType::Mamba,
            _ => panic!("from_idx: {idx} is not a component index"),
        }
    }
}
#[cfg(test)]
#[path = "../tests/components/base.rs"]
mod tests;
