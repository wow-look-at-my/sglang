"""CPU-only unit tests for the mamba pool ratio vs the prefill->decode peak.

Pins the sizing invariant behind MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO:
at the first cache_unfinished_req, a request still holds its admission-locked
matched-prefix mamba (protected) plus its own COW slot, and then allocates a
donated slot. With N distinct-prefix requests that peak is N own + N locked +
1 donated. An effective ratio of 2 (pool = 2N) leaves no evictable victim and
the donated alloc asserts; ratio 3 (pool = 3N) has headroom. Once decode's
skip_mamba leaves the matched prefix evictable, even ratio 2 recovers via
eviction -- which is why the peak, not the decode steady state, sets the floor.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N = 4  # concurrent distinct-prefix requests


class _BoundedMambaAllocator:
    """Fixed-capacity slot allocator; alloc returns None once exhausted."""

    def __init__(self, size: int):
        self.free_ids = list(range(size))

    def alloc(self, n: int):
        if len(self.free_ids) < n:
            return None
        return torch.tensor([self.free_ids.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, value: torch.Tensor):
        self.free_ids.extend(int(v) for v in value.tolist())


class _RatioCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self, pool_size: int):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.allocator = _BoundedMambaAllocator(pool_size)
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=self.allocator)
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.prefix_nodes = []
        self.alloc_evict_params = []

    def evict_for_alloc(self, params: EvictParams):
        self.alloc_evict_params.append(params)
        need = params.mamba_num
        for node in list(self.prefix_nodes):
            if need <= 0:
                break
            cd = node.component_data[ComponentType.MAMBA]
            if cd.lock_ref == 0 and cd.value is not None:
                self.allocator.free(cd.value)
                self.component_evictable_size_[ComponentType.MAMBA] -= len(cd.value)
                cd.value = None
                self.prefix_nodes.remove(node)
                need -= 1


def _build_peak(pool_size: int, lock_prefixes: bool):
    """N own slots + N matched-prefix snapshots, then return the component ready
    to allocate one donated slot. Prefix snapshots are locked (protected,
    prefill peak) or left evictable (decode steady state after skip_mamba)."""
    cache = _RatioCache(pool_size)
    component = object.__new__(MambaComponent)
    component.cache = cache
    # The TreeCore owns the tree member-var state the component reads through.
    component.tree_core = cache
    component.component_type = ComponentType.MAMBA

    owned = [cache.allocator.alloc(1) for _ in range(N)]
    assert all(s is not None for s in owned)

    for _ in range(N):
        node = UnifiedTreeNode(cache.tree_components)
        slot = cache.allocator.alloc(1)
        assert slot is not None
        node.component_data[ComponentType.MAMBA].value = slot
        cache.component_evictable_size_[ComponentType.MAMBA] += len(slot)
        cache.prefix_nodes.append(node)
        if lock_prefixes:
            component.acquire_component_lock(node, IncLockRefResult())

    return component, cache, owned


class TestMambaRatioEnvGate(unittest.TestCase):
    """SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK gates the pool ratio: off restores the
    original base 3 (overlap 5, lazy 4, no_buffer 3), on drops the base to 2
    (overlap 4, lazy 3) while no_buffer stays 3. Guards the flag wiring so the
    ratio can never drift out of sync with whether the decode lock is skipped."""

    @staticmethod
    def _ratio(*, extra_buffer, lazy, disable_overlap, skip):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        fake = SimpleNamespace(server_args=SimpleNamespace())
        # Every input is a published leaf now: the extra-buffer predicates read
        # the radix-cache strategy off the bags, so the fixture publishes the
        # strategy that produces the combination under test.
        strategy = (
            "extra_buffer_lazy"
            if lazy
            else "extra_buffer"
            if extra_buffer
            else "no_buffer"
        )
        from sglang.srt import runtime_context as rc

        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(skip):
            with rc.get_context().override_server_args(
                disable_radix_cache=False,
                disable_overlap_schedule=disable_overlap,
                mamba_radix_cache_strategy=strategy,
            ):
                return KVCacheConfigurator._calculate_mamba_ratio(fake)

    def test_flag_off_restores_original_ratios(self):
        def r(**kwargs):
            return self._ratio(skip=False, **kwargs)

        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 4
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 5
        )  # overlap

    def test_flag_on_drops_base_but_keeps_no_buffer(self):
        def r(**kwargs):
            return self._ratio(skip=True, **kwargs)

        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 3
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 4
        )  # overlap


class _RecordingComp:
    """Fake tree component: records the dec params it is asked to release with."""

    def __init__(self, component_type, priority):
        self.component_type = component_type
        self._priority = priority
        self.released = []

    def eviction_priority(self, is_leaf):
        return self._priority

    def release_component_lock(self, node, params):
        self.released.append(params)

    def release_window_lock(  # SWA only
        self, node, swa_uuid_for_lock, device_frees, host_frees
    ):
        pass


class TestDecSwaLockSkip(unittest.TestCase):
    """dec_swa_lock_only early-releases SWA plus co-located lower-tier (Mamba)
    locks. On a node whose acquire skipped Mamba (decode hold), the release
    must skip it too, else it drops a mamba lock it never took -- another
    request's, on a shared FULL+SWA+MAMBA node (Inkling). Guards the contract
    without booting a 3-component model."""

    def _run(self, skipped_lock_components):
        # internal-node priority: full=2 > swa=1 > mamba=0
        full = _RecordingComp(ComponentType.FULL, 2)
        swa = _RecordingComp(ComponentType.SWA, 1)
        mamba = _RecordingComp(ComponentType.MAMBA, 0)
        node = SimpleNamespace(id=7)
        tree_core = SimpleNamespace(
            root_node=object(),
            components=(full, swa, mamba),
            components_by_type={ComponentType.SWA: swa},
            node_by_id=lambda node_id: node,
            _assert_receipt_anchor=UnifiedTreeCore._assert_receipt_anchor,
        )
        UnifiedTreeCore.dec_swa_lock_only(
            tree_core,
            node.id,
            DecLockRefParams(
                node_id=node.id,
                skipped_lock_components=skipped_lock_components,
                component_lock_uuids={ComponentType.SWA: None},
            ),
        )
        return full, mamba

    def test_unlocked_mamba_is_not_released(self):
        full, mamba = self._run(skipped_lock_components=(ComponentType.MAMBA,))
        # mamba took no lock at acquire, so the early release skips it too
        self.assertEqual(mamba.released, [])
        # full (above swa) is never touched
        self.assertEqual(full.released, [])

    def test_lower_tier_released_when_locked(self):
        full, mamba = self._run(skipped_lock_components=())
        self.assertEqual(len(mamba.released), 1)
        self.assertEqual(full.released, [])


class TestMambaDonatedAllocRatio(unittest.TestCase):
    def test_prefill_peak_ratio2_exhausts_pool(self):
        # pool = 2N, all N prefixes admission-locked: no evictable victim.
        component, cache, _ = _build_peak(pool_size=2 * N, lock_prefixes=True)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            component._alloc_mamba_slot()
        self.assertEqual(
            cache.alloc_evict_params, [EvictParams(num_tokens=0, mamba_num=1)]
        )

    def test_prefill_peak_ratio3_has_headroom(self):
        # pool = 3N: N free slots remain after own + locked prefix.
        component, cache, _ = _build_peak(pool_size=3 * N, lock_prefixes=True)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(cache.component_protected_size_[ComponentType.MAMBA], N)

    def test_decode_steady_evictable_prefix_ratio2_ok(self):
        # pool = 2N but the matched prefixes are evictable (skip_mamba on decode):
        # eviction reclaims a victim, so even ratio 2 serves the donated alloc.
        component, cache, _ = _build_peak(pool_size=2 * N, lock_prefixes=False)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(len(cache.prefix_nodes), N - 1)
        self.assertEqual(
            cache.alloc_evict_params, [EvictParams(num_tokens=0, mamba_num=1)]
        )


class TestPPMambaPoolSizing(unittest.TestCase):
    """A PP rank only allocates mamba state for its own [start_layer, end_layer)
    slice, so charging it for the whole model's layers starves the pool. Sizing
    uses the largest per-stage share, which also keeps every rank on the same
    pool size (and hence the same max_running_requests / pp_max_micro_batch_size)
    without a collective."""

    # Kimi-K3 shaped: 93 layers, linear attention everywhere except every 4th and
    # the last, so the 69 mamba layers split unevenly over 8 stages (9 or 8 each).
    TOTAL_LAYERS = 93
    MAMBA_LAYERS = [i for i in range(93) if (i + 1) % 4 != 0 and i <= 90]
    BUDGET_GB = 8.0

    @classmethod
    def _pool_size(cls, pp_rank, pp_size):
        from sglang.srt import runtime_context as rc
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.distributed.utils import get_pp_indices
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_schedule

        shape = Mamba2StateShape(
            conv=[(4096, 3)],
            temporal=(64, 128, 128),
            intermediate_size=0,
            conv_dim=0,
            ssm_state_size=0,
            num_heads=0,
            head_dim=0,
            state_size=0,
            conv_kernel=0,
            num_k_heads_per_tp=8,
        )
        params = Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=list(cls.MAMBA_LAYERS),
        )
        start, end = get_pp_indices(cls.TOTAL_LAYERS, pp_rank, pp_size)
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=params),
            server_args=SimpleNamespace(),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            layer_info=SimpleNamespace(start_layer=start, end_layer=end),
            attn_dp_size=1,
            pp_size=pp_size,
            hybrid_gdn_config=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), num_hidden_layers=cls.TOTAL_LAYERS
            ),
        )
        with rc.get_context().override_server_args(
            disable_radix_cache=False,
            max_mamba_cache_size=None,
            max_running_requests=None,
            mamba_full_memory_ratio=0.5,
            enable_linear_replayssm_spec=False,
        ):
            KVCacheConfigurator._handle_max_mamba_cache(fake, cls.BUDGET_GB)
            return get_schedule().max_mamba_cache_size

    def test_stage_is_not_charged_for_the_whole_model(self):
        solo = self._pool_size(0, 1)
        staged = self._pool_size(0, 8)
        # The busiest stage holds 9 of the 69 mamba layers, so it should fit
        # roughly 69/9 more slots than a rank holding all of them. pp_size=1 is
        # unchanged: that rank does hold every layer.
        self.assertGreater(staged, solo * 5)

    def test_every_stage_agrees_on_the_pool_size(self):
        sizes = {self._pool_size(r, 8) for r in range(8)}
        self.assertEqual(
            len(sizes), 1, f"per-rank pool sizes diverged: {sorted(sizes)}"
        )


_GiB = 1 << 30
_MiB = 1 << 20
_KiB = 1 << 10


class TestDerivedMambaPoolSplit(unittest.TestCase):
    """The split SGLang picks when no ratio, state-pool size or model default
    was given: both pools fill together at context_len, but the state pool keeps
    at least MIN_STATE_POOL_SHARE so short requests still run concurrently."""

    @staticmethod
    def _split(*, context_len, max_running_requests=None, fixed_state_bytes=0.0):
        from sglang.srt.mem_cache.mamba_pool_split import derive_mamba_pool_split

        # Qwen3-Next-class GDN hybrid at TP2 on a 96 GB card: 27.9 MiB per state
        # slot, 12 KiB of KV per token, 4 slots per request (extra_buffer_lazy).
        return derive_mamba_pool_split(
            budget_bytes=16.56 * _GiB,
            state_bytes_per_request=4 * 27.9 * _MiB,
            fixed_state_bytes=fixed_state_bytes,
            kv_bytes_per_token=12 * _KiB,
            context_len=context_len,
            max_running_requests=max_running_requests,
            default_share=0.9 / 1.9,
        )

    def test_balanced_split_is_the_skill_ratio(self):
        from sglang.srt.mem_cache.mamba_pool_split import derive_mamba_pool_split

        # compute-mamba-ratio worked example: r* = S * token_equiv / L with
        # token_equiv = 56.4 MB / 13.8 KB, S = 4, L = 9216 -> ~1.8.
        state_slot, kv_token = 56.4e6, 13.8e3
        split = derive_mamba_pool_split(
            budget_bytes=100e9,
            state_bytes_per_request=4 * state_slot,
            fixed_state_bytes=0.0,
            kv_bytes_per_token=kv_token,
            context_len=9216,
            max_running_requests=None,
            default_share=0.9 / 1.9,
        )
        self.assertEqual(split.sized_by, "context_len")
        self.assertAlmostEqual(
            split.mamba_full_memory_ratio, 4 * (state_slot / kv_token) / 9216
        )
        self.assertAlmostEqual(split.state_requests, split.kv_context_len_requests)

    def test_long_context_gives_kv_the_budget_but_keeps_a_short_request_floor(self):
        from sglang.srt.mem_cache.mamba_pool_split import MIN_STATE_POOL_SHARE

        for context_len in (262144, 524288):
            with self.subTest(context_len=context_len):
                split = self._split(context_len=context_len)
                # The undecided 0.9 gave the state pool 47% of the budget here,
                # leaving KV for fewer than two 400K-token conversations.
                self.assertEqual(split.sized_by, "short_request_floor")
                self.assertAlmostEqual(split.state_share, MIN_STATE_POOL_SHARE)
                self.assertLess(split.balanced_share, MIN_STATE_POOL_SHARE)
                # The state pool never binds before KV at context_len.
                self.assertGreater(split.state_requests, split.kv_context_len_requests)
                kv_tokens = (1 - split.state_share) * 16.56 * _GiB / (12 * _KiB)
                self.assertGreater(kv_tokens / 400_000, 3)

    def test_short_context_is_not_capped_at_the_default_share(self):
        # At 8K context every request is small; the balanced split gives the
        # state pool more than the undecided 0.9 did, and must not be clipped to it.
        split = self._split(context_len=8192)
        self.assertEqual(split.sized_by, "context_len")
        self.assertGreater(split.state_share, 0.9 / 1.9)

    def test_requested_concurrency_is_capped_at_the_default_share(self):
        split = self._split(context_len=262144, max_running_requests=100_000)
        self.assertEqual(split.sized_by, "default_share_cap")
        self.assertAlmostEqual(split.state_share, 0.9 / 1.9)


class TestDerivedSplitThroughPoolSizer(unittest.TestCase):
    """End to end through _handle_max_mamba_cache: an unset ratio is replaced by
    the derived one, and a pool sized for --max-running-requests admits all of
    them after the padding slots and the speculative joint solve."""

    CONTEXT_LEN = 262144
    BUDGET_GB = 16.56

    @classmethod
    def _run(cls, *, explicitly_set, num_draft_tokens=None, **fields):
        import types

        from sglang.srt import runtime_context as rc
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_schedule

        shape = Mamba2StateShape(
            conv=[(4096, 3)],
            temporal=(16, 128, 128),
            intermediate_size=0,
            conv_dim=0,
            ssm_state_size=0,
            num_heads=0,
            head_dim=0,
            state_size=0,
            conv_kernel=0,
            num_k_heads_per_tp=8,
        )
        params = Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
            layers=list(range(36)),
        )
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=params, full_attention_layer_ids=[3, 7, 11]
            ),
            server_args=SimpleNamespace(),
            spec_algorithm=SimpleNamespace(is_none=lambda: num_draft_tokens is None),
            attn_dp_size=1,
            pp_size=1,
            is_hybrid_swa=False,
            hybrid_gdn_config=None,
            model_config=SimpleNamespace(context_len=cls.CONTEXT_LEN),
            # The KV cell size is the pool configurator's; pin it here.
            _kv_bytes_per_logical_token=lambda: 12 * _KiB,
        )
        for name in (
            "_can_derive_mamba_split",
            "_derive_mamba_full_memory_ratio",
            "_calculate_mamba_ratio",
        ):
            setattr(
                fake, name, types.MethodType(getattr(KVCacheConfigurator, name), fake)
            )
        with rc.get_context().override_server_args(
            disable_radix_cache=False,
            max_mamba_cache_size=None,
            enable_linear_replayssm_spec=False,
            speculative_num_draft_tokens=num_draft_tokens,
            _mamba_full_memory_ratio_explicitly_set=explicitly_set,
            **fields,
        ):
            KVCacheConfigurator._handle_max_mamba_cache(fake, cls.BUDGET_GB)
            schedule = get_schedule()
            return (
                schedule.max_mamba_cache_size,
                schedule.mamba_full_memory_ratio,
                fake._calculate_mamba_ratio(),
            )

    def test_pool_sized_for_max_running_requests_admits_all_of_them(self):
        for num_draft_tokens in (None, 4):
            with self.subTest(num_draft_tokens=num_draft_tokens):
                pool, _, slots_per_request = self._run(
                    explicitly_set=False,
                    num_draft_tokens=num_draft_tokens,
                    max_running_requests=16,
                )
                self.assertEqual(pool // slots_per_request, 16)

    def test_unset_ratio_is_derived_from_context_len(self):
        from sglang.srt.mem_cache.mamba_pool_split import MIN_STATE_POOL_SHARE

        _, ratio, _ = self._run(explicitly_set=False, max_running_requests=None)
        self.assertAlmostEqual(ratio, MIN_STATE_POOL_SHARE / (1 - MIN_STATE_POOL_SHARE))

    def test_unified_pool_labels_are_derived_too(self):
        """Under the unified pool the split floats, but the state label still
        sizes the request cap, the per-request draft-state scratch and the host
        tier's state share; the arbitrary 0.9 default gave the state half of each."""
        from sglang.srt.mem_cache.mamba_pool_split import MIN_STATE_POOL_SHARE

        _, ratio, _ = self._run(
            explicitly_set=False,
            max_running_requests=None,
            enable_unified_memory=True,
        )
        self.assertAlmostEqual(ratio, MIN_STATE_POOL_SHARE / (1 - MIN_STATE_POOL_SHARE))

    def test_a_given_ratio_is_left_alone(self):
        for explicitly_set in (True, None):
            with self.subTest(explicitly_set=explicitly_set):
                pool, ratio, _ = self._run(
                    explicitly_set=explicitly_set,
                    max_running_requests=None,
                    mamba_full_memory_ratio=0.5,
                )
                self.assertEqual(ratio, 0.5)
                self.assertGreater(pool, 16 * 5)


if __name__ == "__main__":
    unittest.main()
