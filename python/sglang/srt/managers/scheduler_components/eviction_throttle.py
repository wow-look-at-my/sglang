"""Hold a conversation back instead of letting it thrash the prefix cache.

Admission's capacity predicate is the allocator's free tokens plus the tree's
evictable tokens. A running request's prefix is locked and never evictable, so
admission cannot take memory from running work; what it can take is the cached
prefix of a conversation that is between turns (an agent waiting on a tool, a
user reading). When the conversations in use need more than the cache can
keep, admitting a new or returning one evicts an idle one, which on its next
turn evicts another, and every turn reloads its prefix from host or, without
HiCache, recomputes it from scratch: tok/s collapses while the GPU refills the
cache.

This throttle keeps a ledger of the conversations the scheduler has served,
recognizes a returning one by the tail of its previous context, and holds a
request back when admitting it would evict cached prefixes while the live
conversations plus it do not fit what the cache can keep. Nothing is tuned:

* A conversation is live while a request of it is queued or running, or while
  its idle time is within the longest gap between one of its turns finishing
  and the next arriving observed so far.
* What the cache can keep is the device pool, or the HiCache host tier when
  larger: write-through mirrors every cached prefix on host, and an evicted
  prefix reloads with one H2D copy instead of a recompute.
* A request whose own conversation is resident on device is never held: the
  live set has outgrown the pool and the least recently used must go.
* Held requests keep their FIFO order. The oldest may evict once it has waited,
  since its arrival or the previous evicting admission, as long as recomputing
  the whole device pool takes at the measured prefill rate, so rebuilding
  displaced conversations costs at most 1/K of the GPU (K conversations fit the
  pool) and nothing waits forever.

Ranks share every input but the clock; each decision that reads the clock is
rank 0's, broadcast through ``consensus``.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from sglang.srt.managers.scheduler_components.rank0_consensus import (
    SITE_EVICTION_THROTTLE,
    rank0_broadcast,
)

# Tokens compared to recognize a returning conversation: the tail of its
# previous context at the same offset. Arbitrary; long enough that unrelated
# contexts do not collide.
_TAIL_TOKENS = 64

_ADMIT, _ADMIT_AGED, _HOLD = 0, 1, 2


class _Conversation:
    __slots__ = ("length", "tail", "last_finish", "finished_once", "active_rids")

    def __init__(self, length: int, tail: Tuple[int, ...]) -> None:
        self.length = length
        self.tail = tail
        self.last_finish = 0.0
        self.finished_once = False
        self.active_rids: Set[str] = set()


class ConversationLedger:
    """Conversations the scheduler has served, independent of what is cached."""

    def __init__(self, *, retained_tokens: int) -> None:
        # Conversations beyond what the cache can keep would miss anyway.
        self._retained_tokens = retained_tokens
        self._by_rid: Dict[str, _Conversation] = {}
        # length -> tail -> conversation, for the returning-conversation lookup.
        self._index: Dict[int, Dict[Tuple[int, ...], _Conversation]] = {}
        self._conversations: List[_Conversation] = []
        self.max_gap = 0.0

    def conversation_of(self, rid: str) -> Optional[_Conversation]:
        return self._by_rid.get(rid)

    def on_queued(self, rid: str, token_ids: Sequence[int], now: float) -> None:
        if rid in self._by_rid:
            return  # retracted and requeued
        conv = self._find_returning(token_ids)
        if conv is None:
            conv = _Conversation(len(token_ids), _tail(token_ids, len(token_ids)))
            self._conversations.append(conv)
        elif not conv.active_rids:
            self.max_gap = max(self.max_gap, now - conv.last_finish)
        conv.active_rids.add(rid)
        conv.length = max(conv.length, len(token_ids))
        self._by_rid[rid] = conv

    def on_finished(
        self, rid: str, length: int, tail: Tuple[int, ...], now: float
    ) -> None:
        """``length`` and ``tail`` describe the request's full context."""
        conv = self._by_rid.pop(rid, None)
        if conv is None:
            return
        conv.active_rids.discard(rid)
        self._unindex(conv)
        conv.length = length
        conv.tail = tail
        conv.last_finish = now
        conv.finished_once = True
        self._index.setdefault(conv.length, {})[conv.tail] = conv
        self._prune()

    def drop_absent(self, present_rids: Set[str]) -> None:
        for rid in [rid for rid in self._by_rid if rid not in present_rids]:
            self._by_rid.pop(rid).active_rids.discard(rid)

    def is_live(self, conv: _Conversation, now: float) -> bool:
        return bool(conv.active_rids) or now - conv.last_finish <= self.max_gap

    def live_tokens(self, now: float, *, exclude: Optional[_Conversation]) -> int:
        return sum(
            conv.length
            for conv in self._conversations
            if conv is not exclude and self.is_live(conv, now)
        )

    def _find_returning(self, token_ids: Sequence[int]) -> Optional[_Conversation]:
        best = None
        for length, by_tail in self._index.items():
            if length > len(token_ids) or (best is not None and length <= best.length):
                continue
            conv = by_tail.get(_tail(token_ids, length))
            if conv is not None:
                best = conv
        return best

    def _unindex(self, conv: _Conversation) -> None:
        by_tail = self._index.get(conv.length)
        if by_tail is not None and by_tail.get(conv.tail) is conv:
            del by_tail[conv.tail]
            if not by_tail:
                del self._index[conv.length]

    def _prune(self) -> None:
        kept, total = [], 0
        for conv in sorted(
            self._conversations, key=lambda c: c.last_finish, reverse=True
        ):
            total += conv.length
            if conv.active_rids or total <= self._retained_tokens:
                kept.append(conv)
            else:
                self._unindex(conv)
        self._conversations = kept


def _tail(token_ids: Sequence[int], length: int) -> Tuple[int, ...]:
    return tuple(token_ids[max(0, length - _TAIL_TOKENS) : length])


class EvictionThrottle:
    def __init__(
        self,
        *,
        device_tokens: int,
        host_tokens: int,
        prefill_seconds_per_token: Callable[[], float],
        consensus: Callable[[int], int] = lambda verdict: verdict,
        # The scheduler stamps wait-queue entry with perf_counter.
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._device_tokens = device_tokens
        self._capacity = max(device_tokens, host_tokens)
        self._prefill_seconds_per_token = prefill_seconds_per_token
        self._consensus = consensus
        self._clock = clock
        self.ledger = ConversationLedger(retained_tokens=self._capacity)
        self._last_evicting_admit = float("-inf")
        self._head_taken = False

    def on_request_queued(self, *, rid: str, token_ids: Sequence[int]) -> None:
        self.ledger.on_queued(rid, token_ids, self._clock())

    def on_request_finished(
        self, *, rid: str, input_ids: Sequence[int], output_ids: Sequence[int]
    ) -> None:
        tail = tuple(input_ids[max(0, len(input_ids) - _TAIL_TOKENS) :]) + tuple(
            output_ids[max(0, len(output_ids) - _TAIL_TOKENS) :]
        )
        self.ledger.on_finished(
            rid, len(input_ids) + len(output_ids), tail[-_TAIL_TOKENS:], self._clock()
        )

    def begin_pass(self, *, present_rids: Set[str]) -> None:
        """Start one scan of the waiting queue. ``present_rids`` are the
        requests queued, running or in flight; any other request the ledger
        holds active left without finishing (an abort) and is let go."""
        self._head_taken = False
        self.ledger.drop_absent(present_rids)

    def should_hold(
        self,
        *,
        rid: str,
        input_len: int,
        device_hit: int,
        total_tokens: int,
        would_evict: bool,
        queued_at: float,
    ) -> bool:
        """``total_tokens`` is what admission reserves for the request;
        ``queued_at`` its wait-queue entry time on the scheduler clock."""
        if self._head_taken:
            return True
        # would_evict and device_hit read rank-local allocator state, so they feed
        # the verdict instead of gating the collective every rank must enter.
        # Half the input separates a conversation's own resident context from
        # a hit on nothing but a shared system prompt.
        if not would_evict or 2 * device_hit >= input_len:
            local = _ADMIT
        else:
            local = self._local_verdict(rid, total_tokens, queued_at)
        verdict = self._consensus(local)
        self._head_taken = verdict != _ADMIT
        return verdict == _HOLD

    def on_admitted(self, *, evicted: bool) -> None:
        if evicted:
            self._last_evicting_admit = self._clock()

    def _local_verdict(self, rid: str, total_tokens: int, queued_at: float) -> int:
        now = self._clock()
        conv = self.ledger.conversation_of(rid)
        if self.ledger.live_tokens(now, exclude=conv) + total_tokens <= self._capacity:
            return _ADMIT
        waited = now - max(queued_at, self._last_evicting_admit)
        if waited < self._device_tokens * self._prefill_seconds_per_token():
            return _HOLD
        return _ADMIT_AGED


def rank0_verdict_consensus(cpu_group) -> Callable[[int], int]:
    """Agree on the first rank's verdict across ``cpu_group``."""
    broadcast = rank0_broadcast(cpu_group, site=SITE_EVICTION_THROTTLE)
    return lambda verdict: int(broadcast((verdict,))[0])
