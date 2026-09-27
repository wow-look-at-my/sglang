import fcntl
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto as auto
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache import host_memory
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base, common
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestHiCacheAutoSize(CustomTestCase):
    def test_hybrid_target_draft_uses_its_sidecar_slot_capacity(self):
        # EAGLE3 can pair a hybrid target with a plain MHA draft. Full drafts
        # follow full target slots; SWA drafts follow the smaller SWA capacity.
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=4096),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=None,
                get_kv_size_bytes=Mock(return_value=1024),
            ),
        )
        draft_mha = Mock(
            spec=MHATokenToKVPool,
            size=16,
            host_capacity_bytes=None,
            get_kv_size_bytes=Mock(return_value=(128, 128)),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        for draft, expected_sidecar_bytes in (
            (draft_mha, 2048),
            (Mock(spec=BaseSWAKVPool, swa_kv_pool=draft_mha), 512),
        ):
            with self.subTest(
                draft_type=type(draft).__name__, bytes=expected_sidecar_bytes
            ):
                plan = Mock(mode="sidecar", device_pools=(draft,))
                self.assertEqual(
                    sizing._estimate_hicache_bytes(params, plan),
                    4096 + 1024 + expected_sidecar_bytes,
                )

    def test_unified_views_are_sized_from_host_capacity(self):
        """Unified sub-pools answer get_kv_size_bytes with zero (UnifiedKVPool
        logs the shared buffer once) and publish host_capacity_bytes instead,
        the weight the explicit host-size split already uses. An estimate taken
        from get_kv_size_bytes is zero device bytes, and the default ratio then
        divides by it."""
        target = Mock(
            spec=SWAKVPool,
            size=128,
            full_kv_pool=Mock(
                size=128,
                host_capacity_bytes=4096,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
            swa_kv_pool=Mock(
                size=32,
                host_capacity_bytes=1024,
                get_kv_size_bytes=Mock(return_value=(0, 0)),
            ),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=target)),
            page_size=2,
        )
        self.assertEqual(sizing._estimate_hicache_bytes(params, None), 4096 + 1024)

    def test_default_ratio_fits_host_budget_and_pools_book_it(self):
        """With only --enable-hierarchical-cache the default ratio shrinks to the
        per-rank budget, pools book one snapshot, and an explicit ratio opts out."""
        pool = MHATokenToKVPool(
            size=128,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
            page_size=2,
        )
        rank_budget = 10_000
        # Four ranks per host (e.g. TP8 over two 4-GPU nodes) share what psutil reports.
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 4 * rank_budget
        with (
            get_context().override_server_args(enable_hierarchical_cache=True),
            patch.object(base, "ranks_per_host", return_value=4),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
            sizing.auto_size_hicache(params, None, enabled=True),
        ):
            ratio = get_memory().hicache_ratio
            self.assertLess(ratio, 2.0)
            host = MHATokenToKVPoolHost(
                pool, ratio, 0, 2, "layer_first", pin_memory=False, device="cpu"
            )
            self.assertLessEqual(host.size * host.size_per_token, 0.8 * rank_budget)
            with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                MHATokenToKVPoolHost(
                    pool, ratio, 0, 2, "layer_first", pin_memory=False, device="cpu"
                )
        self.assertIsNone(base._host_memory_budget.get())

        explicit = ServerArgs(model_path="dummy", hicache_ratio=2.0)
        explicit.resolve_once()
        self.assertIsNone(
            resolution_result(explicit, "hicache_host_memory_fraction", 0.8)
        )


class _FakeCudart:
    """cudaHostRegister that refuses any single registration above a size."""

    def __init__(self, max_bytes=None):
        self.max_bytes = max_bytes
        self.registered = {}

    def cudaHostRegister(self, ptr, size, flags):
        if self.max_bytes is not None and size > self.max_bytes:
            return 2
        self.registered[ptr] = size
        return 0

    def cudaHostUnregister(self, ptr):
        self.registered.pop(ptr)
        return 0

    def cudaGetErrorString(self, rc):
        return "out of memory"


def _mha_pool(size=1 << 16):
    # Large enough that the 2 MiB pin probe fits under the fake size limits.
    return MHATokenToKVPool(
        size=size,
        page_size=2,
        dtype=torch.float16,
        head_num=2,
        head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _params(pool):
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
        page_size=2,
    )


class TestAutoHiCacheSizing(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(enable_hierarchical_cache=True)
        override.install()
        self.addCleanup(override.restore)
        self.pool = _mha_pool()
        self.device_bytes = sum(self.pool.get_kv_size_bytes())

    def _plan(self, rank_budget):
        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + rank_budget
        with (
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
        ):
            return auto.plan_auto_hicache_size(_params(self.pool), None, blocker=None)

    def test_tier_grows_to_the_budget_up_to_the_cap(self):
        """Auto mode fills the host budget (not the 2x default), capped at 8x."""
        fraction = get_memory().hicache_host_memory_fraction
        budget_for = lambda ratio: int(  # noqa: E731
            ratio
            * self.device_bytes
            / (1 - sizing._ALLOCATION_SLACK_FRACTION)
            / fraction
        )
        self.assertAlmostEqual(self._plan(budget_for(3.5)).ratio, 3.5, places=2)
        self.assertEqual(self._plan(budget_for(20)).ratio, auto.AUTO_HICACHE_MAX_RATIO)

    def test_tier_below_the_useful_minimum_turns_hicache_off(self):
        fraction = get_memory().hicache_host_memory_fraction
        budget = int(1.2 * self.device_bytes / fraction)
        plan = self._plan(budget)
        self.assertIsNone(plan.ratio)
        self.assertIn("minimum", plan.reason)

    def test_unreadable_cgroup_limit_turns_hicache_off_without_failing(self):
        with patch.object(
            base,
            "available_host_memory_bytes",
            side_effect=RuntimeError("Cannot locate the process memory cgroup"),
        ):
            plan = auto.plan_auto_hicache_size(_params(self.pool), None, blocker=None)
        self.assertIsNone(plan.ratio)
        self.assertIn("cgroup", plan.reason)

    def test_unreadable_cgroup_turns_off_an_explicitly_sized_tier_too(self):
        """An explicit --hicache-ratio skips the budget, but the pools still
        read the cgroup while building; that failure must turn HiCache off
        before the build rather than abort startup inside it."""
        with (
            get_context().override_server_args(
                enable_hierarchical_cache=True,
                hicache_ratio=3.0,
                hicache_host_memory_fraction=None,
            ),
            patch.object(
                base,
                "available_host_memory_bytes",
                side_effect=RuntimeError("Cannot locate the process memory cgroup"),
            ),
        ):
            plan = auto.plan_auto_hicache_size(_params(self.pool), None, blocker=None)
        self.assertIsNone(plan.ratio)
        self.assertIn("cgroup", plan.reason)

    def test_one_rank_that_cannot_host_the_tier_turns_it_off_everywhere(self):
        """The MIN across ranks decides: a peer reporting 0 (blocked or too
        little memory) turns HiCache off on a rank that could host it."""
        with patch.object(auto, "_all_reduce", side_effect=lambda value, op: 0.0):
            plan = self._plan(1 << 40)
        self.assertIsNone(plan.ratio)
        self.assertIn("another rank", plan.reason)


class TestAutoHiCacheRuntimeBlockers(CustomTestCase):
    """Startup declines what resolution cannot see: pools HiCache would not
    mirror completely, a separate draft sidecar, unified-memory GPUs, and
    hosts that cannot pin."""

    def _blocker(self, kvcache, *, draft_plan=None, integrated=False, pin=None):
        params = _params(kvcache)
        with (
            patch.object(torch.cuda, "is_available", return_value=integrated),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=Mock(is_integrated=integrated),
            ),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(auto, "probe_host_registration", return_value=pin),
        ):
            return auto._runtime_blocker(params, draft_plan)

    def test_a_mirrored_pool_on_a_pinnable_discrete_gpu_is_not_blocked(self):
        self.assertIsNone(self._blocker(_mha_pool()))

    def test_unverified_pool_types_are_blocked(self):
        class _SubclassWithExtraState(MHATokenToKVPool):
            pass

        pool = _SubclassWithExtraState.__new__(_SubclassWithExtraState)
        self.assertIn("not verified", self._blocker(pool))

    def test_separate_draft_sidecar_is_blocked(self):
        from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode

        plan = Mock(mode=HiCacheDraftMode.SIDECAR, device_pools=())
        self.assertIn("draft", self._blocker(_mha_pool(), draft_plan=plan))

    def test_integrated_gpu_is_blocked(self):
        self.assertIn("shares host memory", self._blocker(_mha_pool(), integrated=True))

    def test_unpinnable_host_is_blocked(self):
        reason = self._blocker(_mha_pool(), pin="cudaHostRegister failed")
        self.assertIn("cannot be pinned", reason)


class TestMambaHostBudget(CustomTestCase):
    def test_mamba_host_pool_budget_miss_is_recoverable(self):
        """The auto attach retries on HostMemoryBudgetError only; a Mamba
        host pool raising a bare ValueError aborted startup instead."""
        from types import SimpleNamespace

        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        device_pool = SimpleNamespace(
            num_mamba_layers=1,
            size=64,
            host_capacity_tokens=None,
            mamba_cache=SimpleNamespace(
                conv=[torch.zeros(1, 65, 4, 4)],
                temporal=torch.zeros(1, 65, 2, 8, 8),
            ),
            slot_sibling_views=lambda: [],
        )
        with base.host_memory_budget_scope(1024):
            with self.assertRaises(base.HostMemoryBudgetError):
                MambaPoolHost(
                    device_pool, 2.0, 0, pin_memory=False, layout="page_first"
                )


class TestMambaHostWeight(CustomTestCase):
    def test_state_weight_excludes_the_draft_state_scratch(self):
        """The host tier mirrors the state slots only; weighting the state
        pool by its whole allocation counted the per-request speculative
        scratch (as large as the slots themselves) and shrank the KV share."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _device_pool_bytes,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool

        slots, layers = 6, 2
        pool = object.__new__(MambaPool)
        pool.size = slots
        pool.mamba_cache = MambaPool.SpeculativeState(
            conv=[torch.zeros(layers, slots + 1, 4, 3)],
            temporal=torch.zeros(layers, slots + 1, 2, 8, 8),
            intermediate_ssm=torch.zeros(layers, 49, 4, 2, 8, 8),
            intermediate_conv_window=[torch.zeros(layers, 49, 4, 4, 3)],
        )
        mirrored = (slots + 1) * layers * (4 * 3 + 2 * 8 * 8) * 4
        self.assertEqual(sizing._pool_bytes(pool), mirrored)
        self.assertEqual(_device_pool_bytes(pool), mirrored)


class TestAutoHiCacheAttach(CustomTestCase):
    """Pinning is best-effort: a failed attempt releases every pinned buffer
    on every rank, and the next attempt is half the size."""

    def setUp(self):
        override = get_context().override_server_args(enable_hierarchical_cache=True)
        override.install()
        self.addCleanup(override.restore)
        self.pool = _mha_pool()
        self.device_bytes = sum(self.pool.get_kv_size_bytes())

    def _pin(self, num_bytes):
        common._cuda_host_register(torch.empty(num_bytes, dtype=torch.uint8))

    def test_a_peer_failure_rolls_back_this_ranks_pinned_buffers(self):
        cudart = _FakeCudart()
        with (
            patch.object(torch.cuda, "cudart", return_value=cudart),
            # One other rank reports a failed build.
            patch.object(auto, "_all_reduce", side_effect=lambda value, op: value + 1),
        ):
            with self.assertRaises(auto.HiCacheAttachAborted):
                auto.AutoHiCacheAttachGate().run(lambda: [self._pin(64), self._pin(64)])
        self.assertEqual(cudart.registered, {})

    def _build(self, cudart, rank_budget):
        """Attach pins one buffer of ratio * device bytes, like a host pool."""
        attempts = []

        def attach(tree_cache, gate):
            ratio = get_memory().hicache_ratio
            attempts.append(ratio)
            gate.run(lambda: self._pin(int(ratio * self.device_bytes)))

        host_free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + rank_budget
        with (
            patch.object(torch.cuda, "cudart", return_value=cudart),
            patch.object(base, "ranks_per_host", return_value=1),
            patch.object(base, "available_host_memory_bytes", return_value=host_free),
        ):
            cache, enabled = auto.build_tree_cache_with_auto_hicache(
                create_tree_cache=lambda: Mock(spec=UnifiedRadixCache),
                attach_hicache=attach,
                params=_params(self.pool),
                draft_plan=None,
            )
        return attempts, enabled

    def test_pin_failure_retries_at_half_size(self):
        cudart = _FakeCudart(max_bytes=int(2.5 * self.device_bytes))
        attempts, enabled = self._build(cudart, rank_budget=1 << 40)
        self.assertTrue(enabled)
        self.assertEqual(attempts, [8.0, 4.0, 2.0])
        # Only the successful attempt's buffer stays pinned.
        self.assertEqual(
            list(cudart.registered.values()), [int(2.0 * self.device_bytes)]
        )
        self.assertTrue(get_memory().enable_hierarchical_cache)

    def test_pinning_nothing_useful_serves_without_hicache(self):
        cudart = _FakeCudart(max_bytes=int(1.9 * self.device_bytes))
        attempts, enabled = self._build(cudart, rank_budget=1 << 40)
        self.assertFalse(enabled)
        self.assertEqual(attempts, [8.0, 4.0, 2.0])
        self.assertEqual(cudart.registered, {})
        self.assertFalse(get_memory().enable_hierarchical_cache)


class TestHostMemoryClaimLock(CustomTestCase):
    def test_a_held_lock_delays_but_never_blocks_startup(self):
        path = Path(tempfile.mkdtemp()) / "claim.lock"
        self.addCleanup(shutil.rmtree, path.parent)
        holder = os.open(path, os.O_CREAT | os.O_RDWR)
        self.addCleanup(os.close, holder)
        fcntl.flock(holder, fcntl.LOCK_EX)
        with patch.object(host_memory, "_claim_lock_path", return_value=path):
            start = time.monotonic()
            with host_memory.host_memory_claim_lock(leader=True, timeout_s=0.3):
                pass
            self.assertGreaterEqual(time.monotonic() - start, 0.3)
            # Released by its holder, the lock is taken without waiting.
            fcntl.flock(holder, fcntl.LOCK_UN)
            with host_memory.host_memory_claim_lock(leader=True, timeout_s=60):
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
