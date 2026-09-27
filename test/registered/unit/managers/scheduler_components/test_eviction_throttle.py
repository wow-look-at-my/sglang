"""Tests for the eviction throttle that keeps conversations from thrashing the
prefix cache."""

import unittest

from sglang.srt.managers.scheduler_components.eviction_throttle import (
    EvictionThrottle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DEVICE_TOKENS = 1000
# 1 ms per token: recomputing the whole device pool takes 1 s.
SECONDS_PER_TOKEN = 1e-3


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _throttle(clock, *, host_tokens=0, consensus=lambda verdict: verdict):
    return EvictionThrottle(
        device_tokens=DEVICE_TOKENS,
        host_tokens=host_tokens,
        prefill_seconds_per_token=lambda: SECONDS_PER_TOKEN,
        consensus=consensus,
        clock=clock,
    )


def _serve_turn(throttle, clock, rid, context):
    """A conversation turn arrives, runs, and finishes with ``context`` tokens."""
    throttle.on_request_queued(rid=rid, token_ids=context[:-10])
    clock.now += 1.0
    throttle.on_request_finished(
        rid=rid, input_ids=context[:-10], output_ids=context[-10:]
    )


def _hold(throttle, rid, *, input_len, device_hit=0, would_evict=True, queued_at):
    return throttle.should_hold(
        rid=rid,
        input_len=input_len,
        device_hit=device_hit,
        total_tokens=input_len + 50,
        would_evict=would_evict,
        queued_at=queued_at,
    )


class TestEvictionThrottle(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        # Two conversations of 400 tokens each: together they fill the pool.
        self.a = list(range(0, 400))
        self.b = list(range(10_000, 10_400))

    def _two_live_conversations(self, throttle):
        _serve_turn(throttle, self.clock, "a1", self.a)
        _serve_turn(throttle, self.clock, "b1", self.b)
        # "a" returns 1 s after finishing: live conversations stay live that long.
        throttle.on_request_queued(rid="a2", token_ids=self.a + [7] * 20)

    def test_new_conversation_is_held_when_it_would_evict_a_live_one(self):
        """Admitting a conversation that does not fit beside the live ones
        evicts one of them, which then evicts another on its next turn: the
        thrash that recomputed whole prefixes turn after turn."""
        throttle = _throttle(self.clock)
        self._two_live_conversations(throttle)
        cold = list(range(50_000, 50_300))
        throttle.on_request_queued(rid="c1", token_ids=cold)
        throttle.begin_pass(present_rids={"a2", "c1"})
        self.assertTrue(_hold(throttle, "c1", input_len=300, queued_at=self.clock.now))

    def test_admitted_once_it_fits_what_host_memory_can_restore(self):
        throttle = _throttle(self.clock, host_tokens=4 * DEVICE_TOKENS)
        self._two_live_conversations(throttle)
        throttle.on_request_queued(rid="c1", token_ids=list(range(50_000, 50_300)))
        throttle.begin_pass(present_rids={"a2", "c1"})
        self.assertFalse(_hold(throttle, "c1", input_len=300, queued_at=self.clock.now))

    def test_never_holds_without_eviction_or_for_a_resident_conversation(self):
        throttle = _throttle(self.clock)
        self._two_live_conversations(throttle)
        throttle.begin_pass(present_rids={"a2"})
        self.assertFalse(
            _hold(
                throttle,
                "a2",
                input_len=420,
                device_hit=0,
                would_evict=False,
                queued_at=self.clock.now,
            )
        )
        self.assertFalse(
            _hold(
                throttle, "a2", input_len=420, device_hit=410, queued_at=self.clock.now
            )
        )

    def test_idle_conversation_past_the_longest_return_gap_is_not_live(self):
        throttle = _throttle(self.clock)
        self._two_live_conversations(throttle)
        self.clock.now += 5.0  # "b" idle 6 s, longest observed return gap 1 s.
        throttle.on_request_queued(rid="c1", token_ids=list(range(50_000, 50_300)))
        throttle.begin_pass(present_rids={"a2", "c1"})
        self.assertFalse(_hold(throttle, "c1", input_len=300, queued_at=self.clock.now))

    def test_only_the_oldest_held_request_ages_in_to_evict(self):
        """FIFO with aging: the head evicts once it has waited as long as
        recomputing the device pool takes; later requests wait behind it."""
        throttle = _throttle(self.clock)
        self._two_live_conversations(throttle)
        for rid, base in (("c1", 50_000), ("d1", 60_000)):
            throttle.on_request_queued(rid=rid, token_ids=list(range(base, base + 300)))
        queued_at = self.clock.now
        pool_rebuild = DEVICE_TOKENS * SECONDS_PER_TOKEN

        self.clock.now = queued_at + 0.5 * pool_rebuild
        throttle.begin_pass(present_rids={"a2", "c1", "d1"})
        self.assertTrue(_hold(throttle, "c1", input_len=300, queued_at=queued_at))
        self.assertTrue(_hold(throttle, "d1", input_len=300, queued_at=queued_at))

        self.clock.now = queued_at + pool_rebuild
        throttle.begin_pass(present_rids={"a2", "c1", "d1"})
        self.assertFalse(_hold(throttle, "c1", input_len=300, queued_at=queued_at))
        self.assertTrue(_hold(throttle, "d1", input_len=300, queued_at=queued_at))
        throttle.on_admitted(evicted=True)

        # The next eviction waits a full pool rebuild after the previous one.
        throttle.begin_pass(present_rids={"a2", "d1"})
        self.assertTrue(_hold(throttle, "d1", input_len=300, queued_at=queued_at))

    def test_rank0_verdict_decides(self):
        throttle = _throttle(self.clock, consensus=lambda verdict: 0)
        self._two_live_conversations(throttle)
        throttle.on_request_queued(rid="c1", token_ids=list(range(50_000, 50_300)))
        throttle.begin_pass(present_rids={"a2", "c1"})
        self.assertFalse(_hold(throttle, "c1", input_len=300, queued_at=self.clock.now))

    def test_aborted_request_stops_keeping_its_conversation_live(self):
        throttle = _throttle(self.clock)
        self._two_live_conversations(throttle)
        self.clock.now += 5.0
        throttle.on_request_queued(rid="c1", token_ids=list(range(50_000, 50_600)))
        # With "a2" queued, "a" stays live and 420 + 650 tokens do not fit.
        throttle.begin_pass(present_rids={"a2", "c1"})
        self.assertTrue(_hold(throttle, "c1", input_len=600, queued_at=self.clock.now))
        # "a2" was aborted in the queue: it is absent from the next pass.
        throttle.begin_pass(present_rids={"c1"})
        self.assertFalse(_hold(throttle, "c1", input_len=600, queued_at=self.clock.now))


if __name__ == "__main__":
    unittest.main()
