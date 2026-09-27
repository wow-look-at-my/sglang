<<<<<<< HEAD
"""Equal GPU time for prefill and decode while both have work.
=======
"""Share GPU time between prefill and decode while both have work.
>>>>>>> origin/master

The scheduler used to run a prefill batch whenever one could be formed, and
decode only when none could. A single long prompt is split into many
chunked-prefill batches, and each of those won its scheduling turn, so every
running request's decode stalled until the whole prompt was prefilled. On a
400K-token cold prompt that is a minute or more of zero generated tokens.

This controller measures how long each batch occupied the GPU, from the
previous completion (or its own launch, if the GPU was idle) to its own
<<<<<<< HEAD
completion, and keeps a running balance, ``debt`` = prefill seconds - decode
seconds, accumulated only while the two classes contend (a prefill is pending
*and* running requests can decode). A prefill is deferred while the balance
is positive, so under contention the GPU alternates between the two classes
in equal time slices; without contention nothing is ever deferred. There is
nothing to tune:

* A short prefill is followed by a correspondingly short decode slice, and a
  long chunk by a long one, so the split adapts to chunk size, context length,
  model and hardware without a hand-picked interval.
* Decode never banks credit (the balance is floored at zero), so when a
  prefill could not run anyway (memory, batch full) no burst of prefill
  follows later.
* Neither class can be slowed by more than 2x by the other.
=======
completion, and keeps a running balance, ``debt``, accumulated only while the
two classes contend (a prefill is pending *and* running requests can decode).
There is nothing to tune:

* **Share.** Prefill batches and decode batches get equal GPU time, however
  many requests a prefill batch serves: that half is decode's floor, so a
  long prompt against running streams slows neither by more than 2x. Only
  batches that actually ran are charged, so a queue blocked on memory or
  request slots takes nothing from decode.
* **Stall bound.** Between two points where decode has caught up (balance
  repaid, no prefill in flight), at most ``chunked_prefill_size`` prefill
  tokens are launched: the stall chunked prefill already promises. With the
  overlap scheduler this stops two chunks from going back to back. Short
  prefills that arrive one after another use what is left of that budget
  instead of each waiting out a decode slice, as they would have shared one
  batch had they arrived together. A chunked prompt's next chunk waits for
  decode to catch up rather than taking a sliver of the leftover.
* **Piggybacked decode is a bonus, not the share.** With mixed chunked
  prefill every running request decodes one token inside each prefill chunk.
  That batch is charged as prefill, minus only the marginal cost of its decode
  rows, so pure decode steps still get their equal share and the piggybacked
  tokens come on top. A mixed batch counts toward the stall bound like any
  other prefill.
* Decode banks at most the last decode batch's time: with the overlap
  scheduler that batch was launched before the balance showed repaid, so its
  overshoot counts toward the next prefill. Nothing more carries over, so
  decode that ran while a prefill could not (memory, batch full) buys no
  later burst.
>>>>>>> origin/master

Charging at completion rather than between scheduling decisions matters with
the overlap scheduler, which picks batch N+1 while batch N still runs.

Decisions must be identical on every rank that shares a batch. Measured times
differ slightly per rank, so the caller supplies ``consensus_elapsed``, which
<<<<<<< HEAD
maps the local measurement to one agreed value (rank 0's).
=======
maps the local (prefill, decode, last decode batch) seconds to one agreed
triple (rank 0's). Every
other input is replicated scheduler state: batch classes, their request and
token counts.
>>>>>>> origin/master
"""

from __future__ import annotations

import time
<<<<<<< HEAD
from typing import Callable, Optional
=======
from collections import deque
from typing import Callable, Deque, Optional, Tuple

# (prefill seconds, decode seconds, seconds of the latest decode batch).
Elapsed = Tuple[float, float, float]
>>>>>>> origin/master


class PrefillDecodeBalancer:
    def __init__(
        self,
        *,
<<<<<<< HEAD
        consensus_elapsed: Callable[[float], float] = lambda elapsed: elapsed,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._consensus_elapsed = consensus_elapsed
        self._clock = clock
        self._debt = 0.0
        # Prefill minus decode GPU seconds finished since the last decision.
        self._unsettled = 0.0
        # Gates the consensus call on an event count every rank shares, not a float.
        self._unsettled_batches = 0
        self._in_flight = 0
        self._busy_since = 0.0
=======
        burst_tokens: Optional[int],
        consensus_elapsed: Callable[[Elapsed], Elapsed] = lambda elapsed: elapsed,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # None when chunked prefill is off: no token bound, defer on debt alone.
        self._burst_tokens = burst_tokens
        self._consensus_elapsed = consensus_elapsed
        self._clock = clock
        self._debt = 0.0
        # GPU seconds finished since the last decision, per class.
        self._unsettled_prefill = 0.0
        self._unsettled_decode = 0.0
        self._last_decode = 0.0
        # Gates the consensus call on an event count every rank shares, not a float.
        self._unsettled_batches = 0
        # (batch class, extend tokens, piggybacked decode rows) per launched,
        # unfinished batch.
        self._in_flight: Deque[Tuple[Optional[bool], int, int]] = deque()
        self._busy_since = 0.0
        # Prefill tokens launched since decode last caught up.
        self._burst_used = 0
        # Local GPU seconds and tokens of every batch that extended tokens.
        self._extend_seconds = 0.0
        self._extend_tokens = 0
>>>>>>> origin/master

    @property
    def debt(self) -> float:
        return self._debt

<<<<<<< HEAD
    def should_defer_prefill(
        self, *, prefill_pending: bool, decode_runnable: bool
=======
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
>>>>>>> origin/master
    ) -> bool:
        if not (prefill_pending and decode_runnable):
            # No contention: whichever class has work runs at full speed.
            self._debt = 0.0
<<<<<<< HEAD
            self._unsettled = 0.0
            self._unsettled_batches = 0
            return False

        if self._unsettled_batches:
            self._debt = max(self._debt + self._consensus_elapsed(self._unsettled), 0.0)
            self._unsettled = 0.0
            self._unsettled_batches = 0
        return self._debt > 0.0

    def on_batch_launched(self) -> None:
        if self._in_flight == 0:
            self._busy_since = self._clock()
        self._in_flight += 1

    def on_batch_finished(self, is_prefill: Optional[bool]) -> None:
        """Charge the batch whose result was just processed, in launch order."""
        now = self._clock()
        elapsed = now - self._busy_since
        self._busy_since = now
        self._in_flight = max(self._in_flight - 1, 0)
        if is_prefill is None:
            return
        self._unsettled += elapsed if is_prefill else -elapsed
        self._unsettled_batches += 1


def batch_class(forward_mode) -> Optional[bool]:
    """True for prefill, False for decode, None for anything else."""
    if forward_mode.is_extend():
        return True
    if forward_mode.is_decode():
        return False
    return None


def rank0_consensus(cpu_group) -> Callable[[float], float]:
    """Agree on the first rank's measurement across ``cpu_group``.
=======
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
            self._burst_used = 0
            return False

        if self._unsettled_batches:
            prefill_s, decode_s, last_decode_s = self._consensus_elapsed(
                (self._unsettled_prefill, self._unsettled_decode, self._last_decode)
            )
            self._debt = max(self._debt + prefill_s - decode_s, -last_decode_s)
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
        if self._debt <= 0.0 and not self._prefill_in_flight():
            self._burst_used = 0

        if self._burst_tokens is None:
            return self._debt > 0.0
        if continues_chunk:
            return self._burst_used > 0
        return self._burst_used >= self._burst_tokens

    def on_batch_launched(
        self,
        *,
        is_prefill: Optional[bool],
        num_tokens: int,
        num_decode_rows: int = 0,
    ) -> None:
        """``is_prefill`` is ``batch_class``; ``num_tokens`` the extend tokens,
        ``num_decode_rows`` the running requests a mixed batch decodes
        alongside its prefill."""
        if not self._in_flight:
            self._busy_since = self._clock()
        self._in_flight.append((is_prefill, num_tokens, num_decode_rows))
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
        is_prefill, num_tokens, num_decode_rows = self._in_flight.popleft()
        # A decode row adds one token to the extend pass; its attention over its
        # own context is not counted, so the estimate errs toward decode time.
        piggyback = min(num_decode_rows * self.prefill_seconds_per_token, elapsed)
        if num_tokens > 0:
            self._extend_seconds += elapsed
            self._extend_tokens += num_tokens
        if is_prefill is None:
            return
        if is_prefill:
            self._unsettled_prefill += elapsed - piggyback
        else:
            self._unsettled_decode += elapsed
            self._last_decode = elapsed
        self._unsettled_batches += 1

    def _prefill_in_flight(self) -> bool:
        return any(is_prefill for is_prefill, _, _ in self._in_flight)


def batch_class(forward_mode) -> Optional[bool]:
    """True for prefill (a mixed batch included), False for decode, None for
    anything else."""
    if forward_mode.is_decode():
        return False
    if forward_mode.is_extend():
        return True
    return None


def rank0_consensus(cpu_group) -> Callable[[Elapsed], Elapsed]:
    """Agree on the first rank's measurements across ``cpu_group``.
>>>>>>> origin/master

    Called only while prefill and decode contend, which every rank of the
    group derives from the same replicated scheduler state, so all ranks
    enter the broadcast together."""
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_world_size(group=cpu_group) == 1:
        return lambda elapsed: elapsed

    src = dist.get_global_rank(cpu_group, 0)
<<<<<<< HEAD
    buffer = torch.zeros(1, dtype=torch.float64)

    def consensus(elapsed: float) -> float:
        buffer[0] = elapsed
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0])
=======
    buffer = torch.zeros(3, dtype=torch.float64)

    def consensus(elapsed: Elapsed) -> Elapsed:
        buffer[0], buffer[1], buffer[2] = elapsed
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0]), float(buffer[1]), float(buffer[2])
>>>>>>> origin/master

    return consensus
