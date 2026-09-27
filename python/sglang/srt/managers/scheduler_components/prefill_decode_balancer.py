"""Share GPU time between prefill and decode while both have work.

The scheduler used to run a prefill batch whenever one could be formed, and
decode only when none could. A single long prompt is split into many
chunked-prefill batches, and each of those won its scheduling turn, so every
running request's decode stalled until the whole prompt was prefilled. On a
400K-token cold prompt that is a minute or more of zero generated tokens.

This controller measures how long each batch occupied the GPU, from the
previous completion (or its own launch, if the GPU was idle) to its own
completion, and keeps a running balance, ``debt``, accumulated only while the
two classes contend (a prefill is pending *and* running requests can decode).
There is nothing to tune:

* **Share.** Each request a prefill batch serves and the decode batch as a
  whole get equal GPU time: a prefill batch's seconds are charged divided by
  the number of requests in it, decode seconds in full. A chunk of one long
  prompt against running streams is a 50/50 split, so neither is slowed more
  than 2x. Only requests that were actually admitted count: a queue blocked on
  memory, request slots or the one-unfinished-chunk rule forms one-request
  batches and never takes decode below half.
* **Stall bound.** Between two points where decode has caught up (balance
  repaid, no prefill in flight), at most ``chunked_prefill_size`` prefill
  tokens are launched: the stall chunked prefill already promises. With the
  overlap scheduler this stops two chunks from going back to back. Short
  prefills that arrive one after another use what is left of that budget
  instead of each waiting out a decode slice, as they would have shared one
  batch had they arrived together. A chunked prompt's next chunk waits for
  decode to catch up rather than taking a sliver of the leftover.
* **Mixed batches are decode service.** With mixed chunked prefill every
  running request decodes a token inside each prefill chunk, so a mixed batch
  already bounds the stall to one chunk and is charged as decode: a long
  prompt then prefills at full speed. The exception is a batch that rebuilds a
  prefix its conversation lost to eviction, which is overhead rather than new
  work and stays time-shared with decode (the caller classifies the batch).
* Decode never banks credit (the balance is floored at zero), so when a
  prefill could not run anyway (memory, batch full) no burst follows later.

Charging at completion rather than between scheduling decisions matters with
the overlap scheduler, which picks batch N+1 while batch N still runs.

Decisions must be identical on every rank that shares a batch. Measured times
differ slightly per rank, so the caller supplies ``consensus_elapsed``, which
maps the local (prefill, decode) seconds to one agreed pair (rank 0's). Every
other input is replicated scheduler state: batch classes, their request and
token counts.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

ElapsedPair = Tuple[float, float]


class PrefillDecodeBalancer:
    def __init__(
        self,
        *,
        burst_tokens: Optional[int],
        consensus_elapsed: Callable[[ElapsedPair], ElapsedPair] = lambda pair: pair,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # None when chunked prefill is off: no token bound, defer on debt alone.
        self._burst_tokens = burst_tokens
        self._consensus_elapsed = consensus_elapsed
        self._clock = clock
        self._debt = 0.0
        # GPU seconds finished since the last decision, per class; prefill
        # seconds already divided by their batch's request count.
        self._unsettled_prefill = 0.0
        self._unsettled_decode = 0.0
        # Gates the consensus call on an event count every rank shares, not a float.
        self._unsettled_batches = 0
        # (batch class, requests, extend tokens) per launched, unfinished batch.
        self._in_flight: Deque[Tuple[Optional[bool], int, int]] = deque()
        self._busy_since = 0.0
        # Prefill tokens launched since decode last caught up.
        self._burst_used = 0
        # Local GPU seconds and tokens of every batch that extended tokens.
        self._extend_seconds = 0.0
        self._extend_tokens = 0

    @property
    def debt(self) -> float:
        return self._debt

    @property
    def prefill_seconds_per_token(self) -> float:
        """Measured prefill cost so far; 0 before any prefill finished."""
        if self._extend_tokens == 0:
            return 0.0
        return self._extend_seconds / self._extend_tokens

    @property
    def prefill_token_budget(self) -> Optional[int]:
        """Cap on the next prefill batch's new tokens; None means uncapped."""
        if self._burst_tokens is None or self._burst_used == 0:
            return None
        return self._burst_tokens - self._burst_used

    def should_defer_prefill(
        self,
        *,
        prefill_pending: bool,
        decode_runnable: bool,
        continues_chunk: bool,
    ) -> bool:
        if not (prefill_pending and decode_runnable):
            # No contention: whichever class has work runs at full speed.
            self._debt = 0.0
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
            self._burst_used = 0
            return False

        if self._unsettled_batches:
            prefill_s, decode_s = self._consensus_elapsed(
                (self._unsettled_prefill, self._unsettled_decode)
            )
            self._debt = max(self._debt + prefill_s - decode_s, 0.0)
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
        if self._debt == 0.0 and not self._prefill_in_flight():
            self._burst_used = 0

        if self._burst_tokens is None:
            return self._debt > 0.0
        if continues_chunk:
            return self._burst_used > 0
        return self._burst_used >= self._burst_tokens

    def on_batch_launched(
        self, *, is_prefill: Optional[bool], num_tokens: int, num_reqs: int
    ) -> None:
        """``is_prefill`` is how the batch is charged (``batch_class``, or True
        for a mixed batch rebuilding a lost prefix); ``num_tokens`` and
        ``num_reqs`` its extend tokens and requests."""
        if not self._in_flight:
            self._busy_since = self._clock()
        self._in_flight.append((is_prefill, num_reqs, num_tokens))
        if is_prefill:
            self._burst_used += num_tokens

    def on_batch_finished(self) -> None:
        """Charge the oldest launched batch, whose result was just processed."""
        if not self._in_flight:
            # Batch launched outside get_next_batch_to_run (disaggregation loops).
            return
        now = self._clock()
        elapsed = now - self._busy_since
        self._busy_since = now
        is_prefill, num_reqs, num_tokens = self._in_flight.popleft()
        if num_tokens > 0:
            self._extend_seconds += elapsed
            self._extend_tokens += num_tokens
        if is_prefill is None:
            return
        if is_prefill:
            self._unsettled_prefill += elapsed / max(num_reqs, 1)
        else:
            self._unsettled_decode += elapsed
        self._unsettled_batches += 1

    def _prefill_in_flight(self) -> bool:
        return any(is_prefill for is_prefill, _, _ in self._in_flight)


def batch_class(forward_mode) -> Optional[bool]:
    """True for prefill, False for decode (a mixed batch included, since its
    decode rows serve every running request), None for anything else."""
    if forward_mode.is_mixed() or forward_mode.is_decode():
        return False
    if forward_mode.is_extend():
        return True
    return None


def rank0_consensus(cpu_group) -> Callable[[ElapsedPair], ElapsedPair]:
    """Agree on the first rank's measurements across ``cpu_group``.

    Called only while prefill and decode contend, which every rank of the
    group derives from the same replicated scheduler state, so all ranks
    enter the broadcast together."""
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_world_size(group=cpu_group) == 1:
        return lambda pair: pair

    src = dist.get_global_rank(cpu_group, 0)
    buffer = torch.zeros(2, dtype=torch.float64)

    def consensus(pair: ElapsedPair) -> ElapsedPair:
        buffer[0], buffer[1] = pair
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0]), float(buffer[1])

    return consensus
