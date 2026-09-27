"""Tests for the prefill/decode time-share balancer."""

import unittest
from collections import deque

from sglang.srt.managers.scheduler_components.prefill_decode_balancer import (
    PrefillDecodeBalancer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFILL_SECS = 0.8
DECODE_SECS = 0.02


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _run(balancer, clock, *, steps, overlap, prefill_pending=True):
    """Drive the scheduler loop shape; returns the launched classes in order.

    The GPU runs batches back to back. With overlap the loop picks and launches
    batch N+1 before waiting on batch N's result, as event_loop_overlap does."""
    launched = []
    in_flight = deque()
    gpu_free_at = 0.0
    for _ in range(steps):
        defer = balancer.should_defer_prefill(
            prefill_pending=prefill_pending, decode_runnable=True
        )
        is_prefill = prefill_pending and not defer
        balancer.on_batch_launched()
        start = max(clock.now, gpu_free_at)
        gpu_free_at = start + (PREFILL_SECS if is_prefill else DECODE_SECS)
        in_flight.append((gpu_free_at, is_prefill))
        launched.append(is_prefill)
        if len(in_flight) > (1 if overlap else 0):
            done_at, done_is_prefill = in_flight.popleft()
            clock.now = max(clock.now, done_at)
            balancer.on_batch_finished(done_is_prefill)
    return launched


def _prefill_time_share(launched):
    prefill = launched.count(True) * PREFILL_SECS
    decode = launched.count(False) * DECODE_SECS
    return prefill / (prefill + decode)


def _longest_prefill_run(launched):
    return max(
        len(run) for run in "".join("P" if p else "D" for p in launched).split("D")
    )


class TestPrefillDecodeBalancer(unittest.TestCase):
    def test_long_chunked_prefill_no_longer_starves_decode(self):
        """Back-to-back prefill chunks used to lock out every running decode;
        each chunk must now be followed by a decode slice of equal GPU time."""
        clock = _Clock()
        launched = _run(
            PrefillDecodeBalancer(clock=clock), clock, steps=2000, overlap=False
        )
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.02)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_overlap_scheduler_charges_the_batch_that_ran(self):
        """With overlap, batch N+1 is chosen while N still runs; the long
        prefill's GPU time must not be charged to the decode launched after it
        (which would floor the balance at zero and never defer prefill)."""
        clock = _Clock()
        launched = _run(
            PrefillDecodeBalancer(clock=clock), clock, steps=2000, overlap=True
        )
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)
        self.assertLessEqual(_longest_prefill_run(launched), 2)

    def test_never_defers_without_running_decode(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        for _ in range(10):
            self.assertFalse(
                balancer.should_defer_prefill(
                    prefill_pending=True, decode_runnable=False
                )
            )
            balancer.on_batch_launched()
            clock.now += 1.0
            balancer.on_batch_finished(True)
        self.assertEqual(balancer.debt, 0.0)

    def test_decode_does_not_bank_credit(self):
        """Decode that ran while prefill was blocked (e.g. no memory) must not
        buy a later burst of back-to-back prefills."""
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        _run(balancer, clock, steps=100, overlap=False, prefill_pending=False)
        for _ in range(100):
            balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
            balancer.on_batch_launched()
            clock.now += DECODE_SECS
            balancer.on_batch_finished(False)
        self.assertEqual(balancer.debt, 0.0)
        launched = _run(balancer, clock, steps=50, overlap=False)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_idle_gap_is_not_charged(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(clock=clock)
        balancer.on_batch_launched()
        clock.now += 0.5
        balancer.on_batch_finished(True)
        clock.now += 60.0  # GPU idle, nothing in flight.
        balancer.on_batch_launched()
        clock.now += 0.02
        balancer.on_batch_finished(False)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.48)

    def test_consensus_value_drives_the_decision(self):
        """Ranks must agree; the balance follows the agreed (rank 0) value."""
        clock = _Clock()
        balancer = PrefillDecodeBalancer(
            clock=clock, consensus_elapsed=lambda elapsed: 0.25
        )
        balancer.on_batch_launched()
        clock.now += 9.0
        balancer.on_batch_finished(True)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.25)


if __name__ == "__main__":
    unittest.main()
