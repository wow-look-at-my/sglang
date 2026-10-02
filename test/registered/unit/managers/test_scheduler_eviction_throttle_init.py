"""The eviction throttle is set up before init_disaggregation runs."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scheduler_before_disaggregation() -> Scheduler:
    """A scheduler at the point __init__ reaches maybe_init_eviction_throttle:
    the balancer exists and self.disaggregation_mode does not yet."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_balancer = SimpleNamespace(init=lambda **kwargs: None)
    scheduler.tree_cache = SimpleNamespace(disable=True)
    scheduler.enable_hierarchical_cache = False
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(size_full=0)
    scheduler.chunked_prefill_size = 0
    return scheduler


class TestEvictionThrottleInitOrder(unittest.TestCase):
    def _init_with_mode(self, mode: str) -> Scheduler:
        scheduler = _scheduler_before_disaggregation()
        self.assertFalse(hasattr(scheduler, "disaggregation_mode"))
        with mock.patch.object(
            scheduler_module,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=mode),
        ):
            scheduler.maybe_init_eviction_throttle()
        return scheduler

    def test_colocated_server_reads_the_mode_from_config(self):
        self.assertIsNone(self._init_with_mode("null").eviction_throttle)

    def test_disaggregated_server_gets_no_throttle(self):
        self.assertIsNone(self._init_with_mode("prefill").eviction_throttle)


if __name__ == "__main__":
    unittest.main()
