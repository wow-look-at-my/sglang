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
    continues_chunk=None,
    prefill_tokens=CHUNK_TOKENS,
):
    """Drive the scheduler loop shape with a long chunked prompt pending, each
    prefill batch, when mixed, decoding ``decode_rows`` running requests;
    returns the launched classes.

    The GPU runs batches back to back. With overlap the loop picks and launches
    batch N+1 before waiting on batch N's result, as event_loop_overlap does.
    ``continues_chunk`` defaults to ``prefill_pending``, the chunked-prompt case;
    pass False for work that arrives fresh. A prefill batch costs its
    ``prefill_tokens`` at the chunk's measured rate."""
    if continues_chunk is None:
        continues_chunk = prefill_pending
    launched = []
    in_flight = deque()
    gpu_free_at = 0.0
    for _ in range(steps):
        defer = balancer.should_defer_prefill(
            prefill_pending=prefill_pending,
            decode_runnable=True,
            continues_chunk=continues_chunk,
        )
        is_prefill = prefill_pending and not defer
        rows = decode_rows if is_prefill else 0
        tokens = prefill_tokens + rows if is_prefill else 0
        balancer.on_batch_launched(
            is_prefill=is_prefill,
            num_tokens=tokens,
            num_decode_rows=rows,
        )
        start = max(clock.now, gpu_free_at)
        seconds = PREFILL_SECS * tokens / CHUNK_TOKENS
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
        """A DECODE batch reports extend_num_tokens equal to its batch size,
        milliseconds per token against prefill's microseconds, so it cannot join a
        rate that prices prefill work."""
        clock = _Clock()
        balancer = _balancer(clock)
        self.assertEqual(balancer.prefill_seconds_per_token, 0.0)
        self.assertEqual(balancer.marginal_prefill_seconds_per_token, 0.0)
        _run_one(balancer, clock, is_prefill=True, seconds=0.5, num_tokens=1000)
        _run_one(balancer, clock, is_prefill=False, seconds=0.02, num_tokens=6)
        _run_one(balancer, clock, is_prefill=False, seconds=0.03, num_tokens=8)
        self.assertAlmostEqual(balancer.prefill_seconds_per_token, 0.5 / 1000)
        self.assertAlmostEqual(balancer.marginal_prefill_seconds_per_token, 0.5 / 1000)

    def test_continuation_chunk_is_capped_by_its_measured_seconds(self):
        """Cost per prefill token rises with the context attention reads, so a
        token bound lets a long prompt's later chunks stall decode for longer than
        the seconds one chunk is worth; the continuation's cap holds the seconds."""
        clock = _Clock()
        balancer = _balancer(clock)
        pending = dict(prefill_pending=True, decode_runnable=True)
        _run_one(balancer, clock, is_prefill=True, seconds=0.8, num_tokens=4000)
        _run_one(balancer, clock, is_prefill=False, seconds=0.8)
        self.assertFalse(balancer.should_defer_prefill(continues_chunk=True, **pending))
        self.assertAlmostEqual(balancer.prefill_seconds_per_token, 0.0002)
        # The next chunk reads a longer context: the same 0.8 s for half the
        # tokens, twice the prefill rate measured so far.
        _run_one(balancer, clock, is_prefill=True, seconds=0.8, num_tokens=2000)
        _run_one(balancer, clock, is_prefill=False, seconds=1.6)
        self.assertFalse(balancer.should_defer_prefill(continues_chunk=True, **pending))
        self.assertAlmostEqual(balancer.marginal_prefill_seconds_per_token, 0.0004)
        budget = balancer.prefill_token_budget(continues_chunk=True)
        promised = CHUNK_TOKENS * balancer.prefill_seconds_per_token
        self.assertLessEqual(
            budget * balancer.marginal_prefill_seconds_per_token, promised
        )
        self.assertGreater(
            (budget + 1) * balancer.marginal_prefill_seconds_per_token, promised
        )
        self.assertLess(budget, CHUNK_TOKENS)

    def test_fresh_prefill_is_capped_in_tokens_not_by_the_marginal_rate(self):
        """Fresh work's cost is bounded by its own request, so it takes what is
        left of the burst in tokens; only a continuation is re-priced at the rate
        its own context now costs."""
        clock = _Clock()
        balancer = _balancer(clock)
        pending = dict(prefill_pending=True, decode_runnable=True)
        # A cheap batch then an expensive one: the marginal rate is 20x the cheap
        # batch's while the burst is still unspent.
        _run_one(balancer, clock, is_prefill=True, seconds=0.02, num_tokens=1000)
        _run_one(balancer, clock, is_prefill=True, seconds=0.8, num_tokens=2000)
        self.assertFalse(
            balancer.should_defer_prefill(continues_chunk=False, **pending)
        )
        self.assertEqual(
            balancer.prefill_token_budget(continues_chunk=False), CHUNK_TOKENS - 3000
        )
        continuation = balancer.prefill_token_budget(continues_chunk=True)
        self.assertLess(continuation, CHUNK_TOKENS)
        self.assertLessEqual(
            continuation * balancer.marginal_prefill_seconds_per_token,
            CHUNK_TOKENS * balancer.prefill_seconds_per_token,
        )

    def test_fresh_prefills_back_to_back_stop_at_the_chunk_bound(self):
        """Four 1,024-token requests arriving one after another share one chunk's
        tokens and run back to back; the fifth waits for decode, however fast the
        debt repays."""
        clock = _Clock()
        launched = _run(
            _balancer(clock),
            clock,
            steps=500,
            overlap=False,
            continues_chunk=False,
            prefill_tokens=CHUNK_TOKENS // 4,
        )
        self.assertEqual(_longest_prefill_run(launched), 4)

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
        self.assertEqual(
            balancer.prefill_token_budget(continues_chunk=False), CHUNK_TOKENS - 1000
        )
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
        self.assertEqual(
            balancer.prefill_token_budget(continues_chunk=True), CHUNK_TOKENS
        )

    def test_without_chunked_prefill_defers_on_debt_alone(self):
        clock = _Clock()
        balancer = PrefillDecodeBalancer(burst_tokens=None, clock=clock)
        _run_one(balancer, clock, is_prefill=True, seconds=0.3, num_tokens=1000)
        self.assertTrue(
            balancer.should_defer_prefill(
                prefill_pending=True, decode_runnable=True, continues_chunk=False
            )
        )
        self.assertIsNone(balancer.prefill_token_budget(continues_chunk=False))

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
