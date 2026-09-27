"""Tests for the prefill/decode time-share balancer."""

import unittest
from collections import deque

from sglang.srt.managers.scheduler_components.prefill_decode_balancer import (
    PrefillDecodeBalancer,
    batch_class,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


def _balancer(clock, *, overlap=True, **kwargs):
    return PrefillDecodeBalancer(overlap=overlap, clock=clock, **kwargs)


def _run(
    balancer,
    clock,
    *,
    steps,
    overlap,
    prefill_pending=True,
    decode_rows=0,
):
    """Drive the scheduler loop shape with a long chunked prompt pending, each
    prefill batch, when mixed, decoding ``decode_rows`` running requests;
    returns the launched classes.

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
        rows = decode_rows if is_prefill else 0
        balancer.on_batch_launched(
            is_prefill=is_prefill,
            num_tokens=CHUNK_TOKENS + rows if is_prefill else 0,
            num_decode_rows=rows,
        )
        start = max(clock.now, gpu_free_at)
        seconds = PREFILL_SECS + rows * PREFILL_SECS / CHUNK_TOKENS
        gpu_free_at = start + (seconds if is_prefill else DECODE_SECS)
        in_flight.append(gpu_free_at)
        launched.append(is_prefill)
        if len(in_flight) > (1 if overlap else 0):
            clock.now = max(clock.now, in_flight.popleft())
            balancer.on_batch_finished()
    return launched


def _run_one(balancer, clock, *, is_prefill, seconds, num_tokens=0):
    balancer.on_batch_launched(is_prefill=is_prefill, num_tokens=num_tokens)
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
        launched = _run(
            _balancer(clock, overlap=False), clock, steps=2000, overlap=False
        )
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.02)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_overlap_scheduler_stalls_decode_for_two_chunks_at_most(self):
        """With overlap, batch N+1 is chosen while N still runs, before N's
        time is charged, so a burst is two chunks; a third would stretch the
        stall past what the overlap scheduler already imposes."""
        clock = _Clock()
        launched = _run(_balancer(clock), clock, steps=2000, overlap=True)
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)
        self.assertEqual(_longest_prefill_run(launched), 2)

    def test_only_the_last_chunk_of_a_burst_carries_decode_rows(self):
        """A decode token inside the first chunk of an overlap burst would split
        one stall into two, doubling the long gaps a stream sees."""
        clock = _Clock()
        balancer = _balancer(clock)
        self.assertTrue(balancer.chunk_follows(chunk_continues=True))
        self.assertFalse(balancer.chunk_follows(chunk_continues=False))
        balancer.on_batch_launched(is_prefill=True, num_tokens=CHUNK_TOKENS)
        self.assertFalse(balancer.chunk_follows(chunk_continues=True))
        self.assertFalse(
            _balancer(clock, overlap=False).chunk_follows(chunk_continues=True)
        )

    def test_waiting_requests_that_cannot_be_admitted_take_no_decode_time(self):
        """A long waiting queue whose requests cannot run (no memory, no slot,
        or behind the one unfinished chunk) must not shrink decode's half: the
        prefill batches it leaves each serve one request."""
        clock = _Clock()
        launched = _run(_balancer(clock), clock, steps=4000, overlap=True)
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)

    def test_piggybacked_decode_does_not_replace_the_decode_share(self):
        """A mixed chunk gives each running request one token, far below what
        a decode step of the same GPU time yields; it must still be followed
        by pure decode worth the chunk's time, or streams collapse to one
        token per chunk while a long prompt prefills."""
        self.assertIs(batch_class(ForwardMode.MIXED), True)
        self.assertIs(batch_class(ForwardMode.DECODE), False)
        clock = _Clock()
        launched = _run(
            _balancer(clock), clock, steps=2000, overlap=True, decode_rows=5
        )
        self.assertAlmostEqual(_prefill_time_share(launched), 0.5, delta=0.05)
        self.assertEqual(_longest_prefill_run(launched), 2)

    def test_mixed_batch_is_charged_without_its_decode_rows(self):
        clock = _Clock()
        balancer = _balancer(clock)
        # Measures 1 ms per extend token.
        _run_one(balancer, clock, is_prefill=True, seconds=1.0, num_tokens=1000)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=False)
        balancer.on_batch_launched(
            is_prefill=True, num_tokens=1500, num_decode_rows=500
        )
        clock.now += 1.5
        balancer.on_batch_finished()
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 1.0)

    def test_measures_prefill_seconds_per_token(self):
        clock = _Clock()
        balancer = _balancer(clock)
        self.assertEqual(balancer.prefill_seconds_per_token, 0.0)
        _run_one(balancer, clock, is_prefill=True, seconds=0.5, num_tokens=1000)
        _run_one(balancer, clock, is_prefill=False, seconds=0.02)
        _run_one(balancer, clock, is_prefill=False, seconds=0.3, num_tokens=500)
        self.assertAlmostEqual(balancer.prefill_seconds_per_token, 0.8 / 1500)

    def test_never_defers_without_running_decode(self):
        clock = _Clock()
        balancer = _balancer(clock)
        for _ in range(10):
            self.assertFalse(
                balancer.should_defer_prefill(
                    prefill_pending=True, decode_runnable=False
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
        balancer = _balancer(clock, overlap=False)
        _run(balancer, clock, steps=100, overlap=False, prefill_pending=False)
        for _ in range(100):
            balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
            _run_one(balancer, clock, is_prefill=False, seconds=DECODE_SECS)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertEqual(balancer.debt, 0.0)
        launched = _run(balancer, clock, steps=50, overlap=False)
        self.assertEqual(_longest_prefill_run(launched), 1)

    def test_idle_gap_is_not_charged(self):
        clock = _Clock()
        balancer = _balancer(clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.5, num_tokens=1)
        clock.now += 60.0  # GPU idle, nothing in flight.
        _run_one(balancer, clock, is_prefill=False, seconds=0.02)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.48)

    def test_consensus_value_drives_the_decision(self):
        """Ranks must agree; the balance follows the agreed (rank 0) values."""
        clock = _Clock()
        balancer = _balancer(clock, consensus_elapsed=lambda _: (0.25, 0.05))
        _run_one(balancer, clock, is_prefill=True, seconds=9.0, num_tokens=1)
        balancer.should_defer_prefill(prefill_pending=True, decode_runnable=True)
        self.assertAlmostEqual(balancer.debt, 0.20)


if __name__ == "__main__":
    unittest.main()
