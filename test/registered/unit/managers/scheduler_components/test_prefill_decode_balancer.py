"""Tests for the prefill/decode time-share balancer."""

import unittest

from sglang.srt.managers.scheduler_components.prefill_decode_balancer import (
    PrefillDecodeBalancer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _simulate(balancer, clock, *, steps, prefill_secs, decode_secs, contended=True):
    """Run a scheduler loop; returns the launched class sequence."""
    launched = []
    for _ in range(steps):
        defer = balancer.should_defer_prefill(
            prefill_pending=contended, decode_runnable=True
        )
        is_prefill = contended and not defer
        balancer.on_batch_launched(is_prefill)
        clock.now += prefill_secs if is_prefill else decode_secs
        launched.append(is_prefill)
    return launched


class TestPrefillDecodeBalancer(unittest.TestCase):
    def test_long_chunked_prefill_no_longer_starves_decode(self):
        # 0.8 s prefill chunks vs 20 ms decode steps: the old policy ran 100%
        # prefill; now each chunk is followed by ~0.8 s of decode.
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        launched = _simulate(
            balancer, clock, steps=2000, prefill_secs=0.8, decode_secs=0.02
        )
        prefill_time = launched.count(True) * 0.8
        decode_time = launched.count(False) * 0.02
        self.assertAlmostEqual(
            prefill_time / (prefill_time + decode_time), 0.5, delta=0.02
        )
        # No decode gap longer than one prefill chunk.
        longest_prefill_run = max(
            len(run) for run in "".join("P" if p else "D" for p in launched).split("D")
        )
        self.assertEqual(longest_prefill_run, 1)

    def test_never_defers_without_running_decode(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        for _ in range(10):
            self.assertFalse(
                balancer.should_defer_prefill(
                    prefill_pending=True, decode_runnable=False
                )
            )
            balancer.on_batch_launched(True)
            clock.now += 1.0
        self.assertEqual(balancer.debt, 0.0)

    def test_contention_ending_resets_balance(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        balancer.on_batch_launched(True)
        clock.now += 1.0
        self.assertTrue(
            balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        )
        self.assertFalse(
            balancer.should_defer_prefill(prefill_pending=False, decode_runnable=True)
        )
        self.assertEqual(balancer.debt, 0.0)

    def test_decode_does_not_bank_credit(self):
        # Decode ran for a long time while prefill was blocked (e.g. no
        # memory); that must not buy a later burst of back-to-back prefills.
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        for _ in range(100):
            balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
            balancer.on_batch_launched(False)
            clock.now += 0.02
        self.assertEqual(balancer.debt, 0.0)
        launched = _simulate(
            balancer, clock, steps=50, prefill_secs=0.8, decode_secs=0.02
        )
        self.assertNotIn("PP", "".join("P" if p else "D" for p in launched))

    def test_each_launch_is_charged_once(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        balancer.on_batch_launched(True)
        clock.now += 0.5
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        clock.now += 5.0
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.5)

    def test_consensus_value_drives_the_decision(self):
        # Ranks must agree; the balance follows the agreed (rank 0) value.
        clock = _Clock()
        balancer = PrefillDecodeBalancer(
            clock=clock, consensus_elapsed=lambda elapsed: 0.25
        )
        balancer.on_batch_launched(True)
        clock.now += 9.0
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.25)


if __name__ == "__main__":
    unittest.main()
