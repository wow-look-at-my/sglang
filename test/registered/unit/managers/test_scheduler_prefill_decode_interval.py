"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, require_mlp_sync: bool) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler.prefill_decode_balancer = None
    return scheduler


_RUNNING = SimpleNamespace(is_empty=lambda: False, is_prefill_only=False)


def _make_batch(*, local_extend: bool, global_extend: bool):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(scheduler._should_defer_prefill(_RUNNING))

    def test_non_dp_interval_uses_local_forward_mode(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(scheduler._should_defer_prefill(_RUNNING))
        self.assertTrue(scheduler._should_defer_prefill(_RUNNING))
        self.assertFalse(scheduler._should_defer_prefill(_RUNNING))

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill(_RUNNING))

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)


class TestMixedChunkDecodeRows(unittest.TestCase):
    def test_only_the_chunk_that_ends_an_overlap_burst_mixes_decode_rows(self):
        """A decode token inside the first chunk of a two-chunk overlap burst
        splits one stall into two, doubling the long gaps every stream sees."""
        from sglang.srt.managers.scheduler_components.prefill_decode_balancer import (
            PrefillDecodeBalancer,
        )

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.is_mixed_chunk = True
        scheduler.prefill_decode_balancer = PrefillDecodeBalancer(
            overlap=True, clock=lambda: 0.0
        )
        scheduler.chunked_req = SimpleNamespace(
            full_untruncated_fill_ids=list(range(10000)),
            prefix_indices=list(range(2000)),
        )
        # 8000 tokens left: the next chunk launches before decode can run.
        self.assertFalse(scheduler._mixes_decode_rows(4096))
        scheduler.prefill_decode_balancer.on_batch_launched(
            is_prefill=True, num_tokens=4096
        )
        self.assertTrue(scheduler._mixes_decode_rows(4096))

        scheduler.prefill_decode_balancer = PrefillDecodeBalancer(
            overlap=True, clock=lambda: 0.0
        )
        scheduler.chunked_req.prefix_indices = list(range(6000))
        self.assertTrue(scheduler._mixes_decode_rows(4096))
        scheduler.is_mixed_chunk = False
        self.assertFalse(scheduler._mixes_decode_rows(4096))


class TestEvictionThrottlePrefixMatch(unittest.TestCase):
    def test_fcfs_throttle_does_not_hold_a_conversation_resident_on_device(self):
        """Under FCFS nothing matches the waiting queue before admission, so the
        throttle must refresh the match itself; with an empty match a returning
        conversation whose context is on device was held as if it were cold."""
        import torch

        from sglang.srt.managers import scheduler as scheduler_module
        from sglang.srt.managers.scheduler_components.eviction_throttle import (
            EvictionThrottle,
        )

        throttle = EvictionThrottle(
            device_tokens=1000,
            host_tokens=0,
            prefill_seconds_per_token=lambda: 1.0,
            clock=lambda: 0.0,
        )
        # Another live conversation already holds most of the pool.
        throttle.on_request_queued(rid="other", token_ids=list(range(10000, 10900)))
        req = SimpleNamespace(
            rid="returning",
            origin_input_ids=list(range(700)),
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            time_stats=SimpleNamespace(wait_queue_entry_time=0.0),
        )
        throttle.on_request_queued(rid=req.rid, token_ids=req.origin_input_ids)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.eviction_throttle = throttle
        scheduler.waiting_queue = [req]
        scheduler.tree_cache = object()
        scheduler.policy = SimpleNamespace(waiting_queue_prefix_matched=lambda q: False)
        adder = SimpleNamespace(
            admission_tokens=lambda r: 700 - len(r.prefix_indices) + 100,
            needs_eviction=lambda total: total >= 150,
        )

        def resident_match(tree_cache, r, include_req):
            # 600 of the request's tokens are cached on device.
            r.prefix_indices = torch.arange(600)

        original = scheduler_module.match_prefix_for_req
        scheduler_module.match_prefix_for_req = resident_match
        try:
            verdict = scheduler._eviction_throttle_holds(adder, req)
        finally:
            scheduler_module.match_prefix_for_req = original

        self.assertIsNotNone(verdict, "a resident conversation was held back")


if __name__ == "__main__":
    unittest.main()
