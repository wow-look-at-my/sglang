"""Tests for the prefill/decode time-share balancer."""

import unittest
from collections import deque

from sglang.srt.managers.scheduler_components.prefill_decode_balancer import (
    PrefillDecodeBalancer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK_TOKENS = 4096
PREFILL_SECS = 0.8
DECODE_SECS = 0.02


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _balancer(clock, **kwargs):
    return PrefillDecodeBalancer(burst_tokens=CHUNK_TOKENS, clock=clock, **kwargs)


def _run(balancer, clock, *, steps, overlap, prefill_pending=True, reqs_per_prefill=1):
    """Drive the scheduler loop shape with a long chunked prompt pending, each
    prefill batch serving ``reqs_per_prefill`` requests; returns the launched
    classes in order.

    The GPU runs batches back to back. With overlap the loop picks and launches
    batch N+1 before waiting on batch N's result, as event_loop_overlap does."""
    launched = []
    in_flight = deque()
    gpu_free_at = 0.0
    for _ in range(steps):
        defer = balancer.should_defer_prefill(
            prefill_pending=prefill_pending,
            decode_runnable=True,
            continues_chunk=prefill_pending,
        )
        is_prefill = prefill_pending and not defer
        balancer.on_batch_launched(
            is_prefill=is_prefill,
            num_tokens=CHUNK_TOKENS if is_prefill else 0,
            num_reqs=reqs_per_prefill if is_prefill else 5,
        )
        start = max(clock.now, gpu_free_at)
        gpu_free_at = start + (PREFILL_SECS if is_prefill else DECODE_SECS)
        in_flight.append(gpu_free_at)
        launched.append(is_prefill)
        if len(in_flight) > (1 if overlap else 0):
            clock.now = max(clock.now, in_flight.popleft())
            balancer.on_batch_finished()
    return launched


def _run_one(balancer, clock, *, is_prefill, seconds, num_tokens=0):
    balancer.on_batch_launched(is_prefill=is_prefill, num_tokens=num_tokens, num_reqs=1)
    clock.now += seconds
    balancer.on_batch_finished()


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
        launched = _run(_balancer(clock), clock, steps=2000, overlap=False)
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.02)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_overlap_scheduler_stalls_decode_for_one_chunk_at_most(self):
        """With overlap, batch N+1 is chosen while N still runs, before N's
        time is known; a second chunk must not be launched behind the first,
        which would double every decode stall."""
        clock = _Clock()
        launched = _run(_balancer(clock), clock, steps=2000, overlap=True)
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_waiting_requests_that_cannot_be_admitted_take_no_decode_time(self):
        """A long waiting queue whose requests cannot run (no memory, no slot,
        or behind the one unfinished chunk) must not shrink decode's half: the
        prefill batches it leaves each serve one request."""
        clock = _Clock()
        launched = _run(_balancer(clock), clock, steps=4000, overlap=True)
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)

    def test_batch_serving_three_requests_gets_three_quarters(self):
        """Each request a prefill batch serves weighs as much as the whole
        decode batch, so admitted work that shares a batch drains faster."""
        clock = _Clock()
        launched = _run(
            _balancer(clock), clock, steps=4000, overlap=True, reqs_per_prefill=3
        )
        self.assertAlmostEqual(_prefill_time_share(launched), 0.75, delta=0.05)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_short_prefills_share_one_chunk_budget_before_decode_repays(self):
        """A short prefill arriving right after another must not wait out a
        decode slice; together they may use one chunk's worth of tokens."""
        clock = _Clock()
        balancer = _balancer(clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.3, num_tokens=1000)
        self.assertFalse(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=False
            )
        )
        self.assertGreater(balancer.debt, 0.0)
        self.assertEqual(balancer.prefill_token_budget, CHUNK_TOKENS - 1000)
        _run_one(
            balancer,
            clock,
            is_prefill=True,
            seconds=0.9,
            num_tokens=CHUNK_TOKENS - 1000,
        )
        self.assertTrue(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=False
            )
        )

    def test_chunked_continuation_waits_for_decode_to_catch_up(self):
        """Leftover budget after a short prefill must not be spent on a sliver
        of a long prompt's next chunk; that chunk waits for its full turn."""
        clock = _Clock()
        balancer = _balancer(clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.3, num_tokens=1000)
        self.assertTrue(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=True
            )
        )
        _run_one(balancer, clock, is_prefill=False, seconds=0.3)
        self.assertFalse(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=True
            )
        )
        self.assertIsNone(balancer.prefill_token_budget)

    def test_without_chunked_prefill_defers_on_debt_alone(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(burst_tokens=None, clock=clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.3, num_tokens=1000)
        self.assertTrue(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=False
            )
        )
        self.assertIsNone(balancer.prefill_token_budget)

    def test_never_defers_without_running_decode(self):
        clock = _Clock()
        balancer = _balancer(clock)
        for _ in range(10):
            self.assertFalse(
                balancer.should_defer_prefill(
                    prefill_pending=True, decode_runnable=False, continues_chunk=True
                )
            )
            _run_one(
                balancer, clock, is_prefill=True, seconds=1.0, num_tokens=CHUNK_TOKENS
            )
        self.assertEqual(balancer.debt, 0.0)

    def test_decode_does_not_bank_credit(self):
        """Decode that ran while prefill was blocked (e.g. no memory) must not
        buy a later burst of back-to-back prefills."""
        clock = _Clock()
        balancer = _balancer(clock)
        _run(balancer, clock, steps=100, overlap=False, prefill_pending=False)
        for _ in range(100):
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=True
            )
            _run_one(balancer, clock, is_prefill=False, seconds=DECODE_SECS)
        self.assertEqual(balancer.debt, 0.0)
        launched = _run(balancer, clock, steps=50, overlap=False)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_idle_gap_is_not_charged(self):
        clock = _Clock()
        balancer = _balancer(clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.5, num_tokens=1)
        clock.now += 60.0  # GPU idle, nothing in flight.
        _run_one(balancer, clock, is_prefill=False, seconds=0.02)
        balancer.should_defer_prefill(
            prefill_pending=True, decode_runnable=True, continues_chunk=True
        )
        self.assertAlmostEqual(balancer.debt, 0.48)

    def test_consensus_value_drives_the_decision(self):
        """Ranks must agree; the balance follows the agreed (rank 0) values."""
        clock = _Clock()
        balancer = _balancer(clock, consensus_elapsed=lambda pair: (0.25, 0.05))
        _run_one(balancer, clock, is_prefill=True, seconds=9.0, num_tokens=1)
        balancer.should_defer_prefill(
            prefill_pending=True, decode_runnable=True, continues_chunk=True
        )
        self.assertAlmostEqual(balancer.debt, 0.20)


if __name__ == "__main__":
    unittest.main()
