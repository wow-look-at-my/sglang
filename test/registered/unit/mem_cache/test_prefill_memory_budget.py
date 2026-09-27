"""CPU regressions for allocator-owned prefill admission and pending demand."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shared_allocator(*, page_size=4, total_bytes=1024):
    return init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=4,
        v_head_dim=4,
        swa_head_num=1,
        swa_head_dim=4,
        swa_v_head_dim=4,
        page_size=page_size,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        total_bytes=total_bytes,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    ).token_to_kv_pool_allocator


def _cache():
    return SimpleNamespace(
        sliding_window_size=8,
        full_evictable_size=lambda: 0,
        swa_evictable_size=lambda: 0,
        is_chunk_cache=lambda: False,
    )


class TestSharedPrefillMemoryBudget(unittest.TestCase):
    def setUp(self):
        self.allocator = _shared_allocator()
        self.cache = _cache()
        self.budget = self.allocator.create_prefill_budget(self.cache)
        self.request = dict(
            extend_input_len=12,
            total_tokens=20,
            max_new_tokens=4,
            input_tokens=12,
            swa_host_hit_length=0,
            chunk_limit=16,
        )

    def test_pending_batch_cannot_spend_shared_bytes_twice(self):
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        self.budget.reserve(12, 4, chunk_limit=16)
        # Each side separately has room, but their combined reservation does not.
        self.assertGreater(self.budget.remaining_total, 20)
        self.assertGreater(self.budget.remaining_swa, 16)
        self.assertEqual(self.budget.check_prefill(**self.request), (False, None))
        self.assertEqual(self.allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(self.allocator.swa_attn_allocator.allocated_count(), 0)

    def test_mixed_decode_reserves_both_sides(self):
        budget = self.allocator.create_prefill_budget(
            self.cache, num_mixed_decode_tokens=4
        )
        budget.reserve(12, 4, chunk_limit=16)
        self.assertEqual(
            (budget.total_offset, budget.current_offset, budget.swa_offset),
            (24, 20, 20),
        )
        self.assertEqual(budget.check_prefill(**self.request), (False, None))

    def test_prefix_lock_changes_admission_without_rebuilding_budget(self):
        self.assertIsNotNone(self.allocator.alloc(24))
        self.cache.full_evictable_size = lambda: 24
        self.cache.swa_evictable_size = lambda: 24
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        # Locking the cached prefix removes its eviction credit.
        self.cache.full_evictable_size = lambda: 0
        self.cache.swa_evictable_size = lambda: 0
        self.assertEqual(self.budget.check_prefill(**self.request), (False, None))

    def test_final_chunk_reserves_decode_headroom(self):
        limit = self.budget.fit_chunk(
            extend_input_len=12, max_new_tokens=80, chunk_limit=16
        )
        self.assertEqual(limit, 8)
        self.budget.reserve(limit, 0, chunk_limit=16, is_chunked_continuation=True)
        self.assertIsNotNone(self.allocator.alloc(limit))

    def test_host_swa_load_is_part_of_joint_demand(self):
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        request = {**self.request, "swa_host_hit_length": 32}
        self.assertEqual(self.budget.check_prefill(**request), (False, None))

    def test_prompt_clipping_uses_the_empty_pool(self):
        kwargs = dict(token_capacity=1, sliding_window_size=8, chunk_size=16)
        limit = self.allocator.max_new_tokens_for_memory(12, 80, **kwargs)
        self.assertIsNotNone(limit)
        self.assertGreater(limit, 0)
        self.assertIsNotNone(self.allocator.alloc(24))
        self.assertEqual(
            self.allocator.max_new_tokens_for_memory(12, 80, **kwargs), limit
        )
        self.assertIsNone(self.allocator.max_new_tokens_for_memory(100, 0, **kwargs))

    def test_shared_stats_pair_available_tokens_with_current_capacity(self):
        self.assertIsNotNone(self.allocator.alloc(12))
        (full_capacity, full_free), (swa_capacity, swa_free) = (
            self.allocator.swa_capacity_and_available(full_capacity=1, swa_capacity=1)
        )
        self.assertEqual(full_capacity - full_free, 12)
        self.assertEqual(swa_capacity - swa_free, 12)

    def test_common_eviction_dispatches_joint_reclaim(self):
        self.cache.token_to_kv_pool_allocator = self.allocator
        self.allocator.evict_to_free_tokens = MagicMock()
        evict_from_tree_cache(self.cache, 8)
        self.allocator.evict_to_free_tokens.assert_called_once_with(self.cache, 8)


class TestSharedPrefillAdmission(unittest.TestCase):
    def _new_admission(self, page_size, pool_pages, *, ignore_eos=False):
        allocator = _shared_allocator(
            page_size=page_size, total_bytes=pool_pages * page_size * 16
        )
        req = Req(
            rid="unaligned-prompt",
            origin_input_text=None,
            origin_input_ids=array("q", [1] * (page_size + 1)),
            sampling_params=SamplingParams(max_new_tokens=1, ignore_eos=ignore_eos),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = 16 * page_size
        scheduler.max_total_num_tokens = allocator.size_full
        scheduler.page_size = page_size
        scheduler.max_new_tokens_limit = None
        scheduler.sliding_window_size = page_size
        scheduler.chunked_prefill_size = page_size
        scheduler.token_to_kv_pool_allocator = allocator
        with get_parallel().override(attn_dcp_size=1):
            scheduler.init_req_max_new_tokens(req)
        self.assertEqual(req.sampling_params.max_new_tokens, 1)
        req._refresh_fill_ids()

        cache = SimpleNamespace(
            sliding_window_size=page_size,
            disable=True,
            full_evictable_size=lambda: 0,
            swa_evictable_size=lambda: 0,
            is_chunk_cache=lambda: True,
            supports_mamba=lambda: False,
        )
        adder = PrefillAdder(
            page_size=page_size,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=16 * page_size,
            rem_chunk_tokens=page_size,
        )
        return allocator, req, adder

    def test_unaligned_final_chunk_makes_progress(self):
        for page_size in (4, 64):
            with self.subTest(page_size=page_size):
                allocator, req, adder = self._new_admission(page_size, pool_pages=7)
                req.prefix_indices = allocator.alloc(page_size)
                self.assertIsNotNone(req.prefix_indices)
                self.assertTrue(allocator.can_reserve(page_size + 2, page_size + 2))

                self.assertIsNone(adder.add_chunked_req(req))
                self.assertEqual(adder.can_run_list, [req])
                self.assertEqual(req.extend_range.length, 1)

    def test_unaligned_ignore_eos_enters_empty_pool(self):
        for page_size in (4, 64):
            with self.subTest(page_size=page_size):
                allocator, req, adder = self._new_admission(
                    page_size, pool_pages=6, ignore_eos=True
                )
                self.assertEqual(len(req.prefix_indices), 0)
                self.assertTrue(allocator.can_reserve(2 * page_size + 2, 2 * page_size))

                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertEqual(adder.can_run_list, [req])


class TestFixedPrefillMemoryBudget(unittest.TestCase):
    def _allocator(self, cls=SWATokenToKVPoolAllocator):
        allocator = object.__new__(cls)
        allocator.page_size = 4
        allocator._size_full = 128
        allocator._size_swa = 64
        allocator.full_available_size = lambda: 128
        allocator.swa_available_size = lambda: 64
        return allocator

    def test_ring_slot_reserved_once_and_evictable_tokens_give_no_credit(self):
        allocator = self._allocator()
        allocator._swa_req_ring = True
        allocator._swa_ring_cost = 32
        cache = _cache()
        cache.swa_evictable_size = lambda: 1000
        budget = allocator.create_prefill_budget(cache)
        budget.reserve(12, 4, chunk_limit=16)
        self.assertEqual(budget.remaining_swa, 32)
        budget.reserve(12, 4, chunk_limit=16, is_chunked_continuation=True)
        self.assertEqual(budget.remaining_swa, 32)
        # The last exact ring slot remains admissible.
        self.assertEqual(
            budget.check_prefill(
                extend_input_len=4,
                total_tokens=12,
                max_new_tokens=4,
                input_tokens=4,
                swa_host_hit_length=0,
                chunk_limit=16,
            ),
            (True, 16),
        )

    def test_pure_swa_budget_reads_swa_capacity(self):
        allocator = self._allocator(PureSWATokenToKVPoolAllocator)
        allocator.full_available_size = lambda: 0
        budget = allocator.create_prefill_budget(_cache())
        self.assertEqual(budget.remaining_total, 64)
        self.assertTrue(budget.has_capacity())

    def test_hisparse_budget_reads_wrapper_capacity(self):
        allocator = self._allocator(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.full_available_size = lambda: 12
        budget = allocator.create_prefill_budget(_cache())
        self.assertEqual(budget.remaining_total, 12)
        self.assertEqual(budget.remaining_swa, 64)

    def test_tri_pool_keeps_fixed_admission_and_clipping(self):
        allocator = self._allocator(UnifiedMambaSWATokenToKVPoolAllocator)
        allocator.can_reserve = MagicMock(
            side_effect=AssertionError("two-pool reservation")
        )
        budget = allocator.create_prefill_budget(_cache())
        self.assertIs(type(budget), SWAPrefillBudget)
        self.assertTrue(budget.has_capacity())
        self.assertEqual(
            allocator.max_new_tokens_for_memory(
                5,
                100,
                token_capacity=32,
                sliding_window_size=8,
                chunk_size=16,
            ),
            19,
        )

    def test_tri_pool_eviction_does_not_reenter_common(self):
        allocator = self._allocator(UnifiedMambaSWATokenToKVPoolAllocator)
        allocator.available_size = lambda: 0
        allocator.full_available_size = lambda: 0
        allocator.swa_available_size = lambda: 0
        cache = _cache()
        cache.token_to_kv_pool_allocator = allocator
        cache.evict_for_alloc = MagicMock()
        evict_from_tree_cache(cache, 8)
        cache.evict_for_alloc.assert_called_once()
        params = cache.evict_for_alloc.call_args.args[0]
        self.assertEqual((params.num_tokens, params.swa_num_tokens), (8, 8))


class TestLockedPrefixIsNotAdmissionCapacity(unittest.TestCase):
    """A prefix a live conversation holds is not spendable by another request.

    Failure mode: a capacity query that counts `protected_size()` as free, or
    a tree core that stops moving a locked prefix out of `evictable_size()`,
    lets a new request admit by reclaiming a conversation that is still
    decoding. The victim then re-prefills its whole history on the next turn.
    The unlocked half of the case keeps the guard honest in the other
    direction: a finished conversation's cache must stay reclaimable.
    """

    PREFIX_TOKENS = 64

    def _cache_with_one_conversation(self):
        kv_pool = MHATokenToKVPool(
            size=self.PREFIX_TOKENS,
            page_size=1,
            dtype=torch.float16,
            head_num=2,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=self.PREFIX_TOKENS,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=2,
                    max_context_len=self.PREFIX_TOKENS,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy="lru",
                tree_components=(ComponentType.FULL,),
            )
        )
        tokens = list(range(1, self.PREFIX_TOKENS + 1))
        values = allocator.alloc(len(tokens))
        self.assertIsNotNone(values)
        node = cache.insert(
            InsertParams(key=RadixKey(array("q", tokens)), value=values)
        ).last_device_node
        # The whole pool is this conversation's prefix, so it is the only
        # capacity a second request could admit against.
        self.assertEqual(allocator.available_size(), 0)
        return cache, allocator, node

    def _probe(self, budget):
        return budget.check_prefill(
            extend_input_len=32,
            total_tokens=32,
            max_new_tokens=1,
            input_tokens=32,
            swa_host_hit_length=0,
            chunk_limit=None,
        )

    def test_locked_prefix_is_not_admission_capacity(self):
        cache, allocator, node = self._cache_with_one_conversation()
        budget = allocator.create_prefill_budget(cache)

        self.assertEqual(cache.evictable_size(), self.PREFIX_TOKENS)
        self.assertEqual(budget.remaining_total, self.PREFIX_TOKENS)
        self.assertEqual(self._probe(budget), (True, None))
        self.assertTrue(budget.has_capacity())

        # The conversation is mid-decode: its prefix is protected, not free.
        receipt = cache.inc_lock_ref(node)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), self.PREFIX_TOKENS)
        self.assertEqual(budget.remaining_total, 0)
        self.assertFalse(budget.has_capacity())
        self.assertEqual(self._probe(budget), (False, None))
        self.assertFalse(
            budget.can_allocate_prefill(
                paged_input=1,
                extend_input_len=1,
                max_new_tokens=1,
                chunk_limit=None,
            )
        )

        # The conversation finishes, so the same prefix is dead cache again.
        cache.dec_lock_ref(node, receipt.to_dec_params())
        self.assertEqual(cache.evictable_size(), self.PREFIX_TOKENS)
        self.assertEqual(budget.remaining_total, self.PREFIX_TOKENS)
        self.assertTrue(budget.has_capacity())
        self.assertEqual(self._probe(budget), (True, None))


if __name__ == "__main__":
    unittest.main()
