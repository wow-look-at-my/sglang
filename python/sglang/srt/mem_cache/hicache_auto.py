"""Best-effort HiCache for an automatically enabled host tier.

``enable_hierarchical_cache`` left unset resolves to on only for
configurations HiCache supports (``arg_groups/hicache_hook.py``). What
resolution cannot see is decided here, identically on every rank: the built
device pools must be ones HiCache mirrors completely, the host must be able
to pin memory and hold a useful tier, and the pools are pinned best-effort --
a rank that cannot pin makes every rank release its pools and retry at half
the size, down to the useful minimum, and then serve without HiCache.
"""

from __future__ import annotations

import logging
import resource
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch

from sglang.srt.mem_cache.pool_host.base import (
    HostMemoryBudgetError,
    host_memory_budget_bytes,
    host_memory_budget_scope,
    ranks_per_host,
)
from sglang.srt.mem_cache.pool_host.common import (
    HostPinError,
    probe_host_registration,
    track_host_registrations,
    unregister_host_buffers,
)
from sglang.srt.runtime_context import get_context, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan

logger = logging.getLogger(__name__)

# Upper bound on the grown host tier: pinning is a startup cost that scales
# with its size (roughly seconds per 10 GiB). Arbitrary; not measured.
AUTO_HICACHE_MAX_RATIO = 8.0
# The host tier is inclusive of the device cache, so a tier below 1.5x the
# device pool adds under half a device pool of reusable prefixes. Arbitrary.
AUTO_HICACHE_MIN_RATIO = 1.5

_RECOVERABLE_ATTACH_ERRORS = (
    HostPinError,
    HostMemoryBudgetError,
    MemoryError,
    OSError,
)


class HiCacheAttachAborted(Exception):
    """Every rank dropped this attach attempt; nothing was applied to the cache."""


class AutoHiCacheSize(msgspec.Struct, frozen=True):
    """The agreed host tier: ``ratio`` of the device pools, or None when off."""

    ratio: Optional[float]
    budget_bytes: int
    reason: str


def _world_cpu_group():
    if not torch.distributed.is_initialized():
        return None
    return get_parallel().world_group.cpu_group


def _all_reduce(value: float, op) -> float:
    group = _world_cpu_group()
    if group is None:
        return value
    tensor = torch.tensor([value], dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=op, group=group)
    return tensor.item()


class AutoHiCacheAttachGate:
    """Run one host-pool build on every rank and commit it only if all succeeded.

    The attach applies the built pools to the cache only after this returns,
    so an abort leaves the cache exactly as it was.
    """

    def run(self, build: Callable[[], Any]) -> Any:
        failure: Optional[BaseException] = None
        with track_host_registrations() as registered:
            try:
                result = build()
            except _RECOVERABLE_ATTACH_ERRORS as error:
                failure, result = error, None
        failed_ranks = _all_reduce(
            0.0 if failure is None else 1.0, torch.distributed.ReduceOp.SUM
        )
        if failed_ranks == 0:
            return result
        unregister_host_buffers(registered)
        if failure is not None:
            logger.warning("HiCache host pools could not be built here: %s", failure)
        raise HiCacheAttachAborted(
            f"{int(failed_ranks)} rank(s) could not build their host pools"
        )


def unmirrored_state_reason(
    kvcache: Any, draft_pools: tuple[Any, ...]
) -> Optional[str]:
    """Why HiCache would leave some device cache state unmirrored, or None.

    Only pool types whose every buffer a host pool copies qualify; exact type
    checks keep a new subclass with extra state out until it is verified.
    """
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_memory_pool import (
        UnifiedHybridLinearKVPool,
        UnifiedMHATokenToKVPool,
        UnifiedQSATokenToKVPool,
    )

    plain = (MHATokenToKVPool, MLATokenToKVPool)
    # Unified MHA only: its host pool resolves kernel-facing ids per transfer,
    # and packed MTP drafts are backed up with their own (virtual) ids.
    hybrid_full = {
        HybridLinearKVPool: plain,
        QSATokenToKVPool: plain,
        UnifiedHybridLinearKVPool: (UnifiedMHATokenToKVPool,),
        UnifiedQSATokenToKVPool: (UnifiedMHATokenToKVPool,),
    }

    def reason(pool: Any, role: str) -> Optional[str]:
        kind = type(pool)
        if kind in plain:
            return None
        if kind in hybrid_full:
            if type(pool.full_kv_pool) in hybrid_full[kind]:
                return None
            kind = type(pool.full_kv_pool)
        elif kind is SWAKVPool:
            if type(pool.full_kv_pool) in plain and type(pool.swa_kv_pool) in plain:
                return None
        return f"the {role} KV pool {kind.__name__} is not verified as fully mirrored"

    found = reason(kvcache, "target")
    for draft in draft_pools:
        found = found or reason(draft, "draft")
    return found


def _memlock_note() -> str:
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        return "memlock unlimited"
    return f"memlock limit {soft / 1024**3:.1f} GiB"


def _local_size(
    params: CacheInitParams,
    draft_plan: Optional[HiCacheDraftPlan],
    blocker: Optional[str],
) -> tuple[float, int, Optional[str]]:
    from sglang.srt.mem_cache.hicache_auto_size import _estimate_hicache_bytes

    if blocker is not None:
        return 0.0, 0, blocker
    memory = get_memory()
    # Read even for an explicit size: the pools re-read it while building,
    # where an unreadable cgroup would be fatal instead of turning HiCache off.
    try:
        rank_budget = host_memory_budget_bytes()
    except (RuntimeError, ValueError, OSError) as error:
        return 0.0, 0, f"the host memory limit cannot be read ({error})"
    if memory.hicache_host_memory_fraction is None:
        # An explicit --hicache-ratio / --hicache-size is honored as given.
        return memory.hicache_ratio, 0, None
    budget = int(rank_budget * memory.hicache_host_memory_fraction)
    device_bytes = _estimate_hicache_bytes(params, draft_plan)
    ratio = auto_hicache_ratio(device_bytes=device_bytes, budget_bytes=budget)
    if ratio < AUTO_HICACHE_MIN_RATIO:
        return (
            ratio,
            budget,
            (
                f"{budget / 1024**3:.1f} GiB of host memory per rank holds only "
                f"{ratio:.2f}x the {device_bytes / 1024**3:.1f} GiB device cache "
                f"(minimum {AUTO_HICACHE_MIN_RATIO}x)"
            ),
        )
    return ratio, budget, None


def auto_hicache_ratio(*, device_bytes: int, budget_bytes: int) -> float:
    """The largest host/device ratio the budget holds, capped at the max ratio."""
    from sglang.srt.mem_cache.hicache_auto_size import _ALLOCATION_SLACK_FRACTION

    fit = budget_bytes * (1 - _ALLOCATION_SLACK_FRACTION) / max(device_bytes, 1)
    return min(AUTO_HICACHE_MAX_RATIO, fit)


def plan_auto_hicache_size(
    params: CacheInitParams,
    draft_plan: Optional[HiCacheDraftPlan],
    *,
    blocker: Optional[str],
) -> AutoHiCacheSize:
    """Agree on one host tier across ranks; any rank's blocker turns it off."""
    ratio, budget, reason = _local_size(params, draft_plan, blocker)
    local = ratio if reason is None else 0.0
    agreed = _all_reduce(local, torch.distributed.ReduceOp.MIN)
    if agreed <= 0:
        return AutoHiCacheSize(
            ratio=None,
            budget_bytes=budget,
            reason=reason or "another rank cannot host the tier",
        )
    return AutoHiCacheSize(ratio=agreed, budget_bytes=budget, reason="")


def build_tree_cache_with_auto_hicache(
    *,
    create_tree_cache: Callable[[], Any],
    attach_hicache: Callable[[Any, AutoHiCacheAttachGate], None],
    params: CacheInitParams,
    draft_plan: Optional[HiCacheDraftPlan],
) -> tuple[Any, bool]:
    """Build the tree cache, then attach HiCache best-effort.

    Returns the cache and whether HiCache is attached; the published config
    is overridden to match, so every later reader sees the outcome.
    """
    from sglang.srt.mem_cache.host_memory import host_memory_claim_lock
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    blocker = _runtime_blocker(params, draft_plan)
    with host_memory_claim_lock(leader=_is_host_leader()):
        size = plan_auto_hicache_size(params, draft_plan, blocker=blocker)
        tree_cache = create_tree_cache()
        reason = size.reason
        ratio = size.ratio
        if ratio is not None and not isinstance(tree_cache, UnifiedRadixCache):
            reason, ratio = f"{type(tree_cache).__name__} has no host tier", None
        # A fixed --hicache-size does not shrink with the ratio: one attempt.
        can_shrink = get_memory().hicache_size <= 0
        while ratio is not None:
            get_context().override("hicache.auto_size", hicache_ratio=ratio)
            try:
                with _budget_scope(size.budget_bytes):
                    attach_hicache(tree_cache, AutoHiCacheAttachGate())
            except HiCacheAttachAborted as aborted:
                reason = f"host pools could not be pinned ({aborted})"
                half = ratio / 2
                ratio = half if can_shrink and half >= AUTO_HICACHE_MIN_RATIO else None
                if ratio is not None:
                    logger.warning(
                        "HiCache auto: %s; retrying at %.2fx.", reason, ratio
                    )
                continue
            logger.info(
                "HiCache auto: on, host tier %.2fx the device cache "
                "(budget %.1f GiB per rank, %d rank(s) on this host, %s).",
                ratio,
                size.budget_bytes / 1024**3,
                ranks_per_host(),
                _memlock_note(),
            )
            return tree_cache, True
    logger.info("HiCache auto: off, %s.", reason)
    get_context().override("hicache.auto", enable_hierarchical_cache=False)
    # An aborted build's controller registered its layer counter on the pool.
    params.token_to_kv_pool_allocator.get_kvcache().register_layer_transfer_counter(
        None
    )
    return tree_cache, False


def _budget_scope(budget_bytes: int):
    # An explicit --hicache-ratio/--hicache-size has no snapshot: each pool
    # checks live host memory, as it does without auto mode.
    if budget_bytes <= 0:
        return nullcontext()
    return host_memory_budget_scope(budget_bytes)


def _runtime_blocker(
    params: CacheInitParams, draft_plan: Optional[HiCacheDraftPlan]
) -> Optional[str]:
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode

    kvcache = params.token_to_kv_pool_allocator.get_kvcache()
    reason = unmirrored_state_reason(kvcache, tuple(params.mtp_draft_device_pools))
    if reason is not None:
        return reason
    if draft_plan is not None and draft_plan.mode == HiCacheDraftMode.SIDECAR:
        return "a separate draft model's KV sidecar is not covered"
    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.cuda.current_device()).is_integrated
    ):
        # Unified memory (e.g. GB10): a host copy only duplicates device memory.
        return "the GPU shares host memory"
    pin_failure = probe_host_registration()
    if pin_failure is not None:
        return f"host memory cannot be pinned ({pin_failure})"
    return None


def _is_host_leader() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() % ranks_per_host() == 0
