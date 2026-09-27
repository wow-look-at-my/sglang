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
  decode to catch up rather than taking a sliver of the leftover, and is
  capped in seconds rather than tokens: cost per prefill token rises with the
  context attention reads, so the token bound alone lets the same tokens stall
  decode for several times the seconds they promise.
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

Charging at completion rather than between scheduling decisions matters with
the overlap scheduler, which picks batch N+1 while batch N still runs.

Decisions must be identical on every rank that shares a batch. Measured times
differ slightly per rank, so the caller supplies ``consensus_elapsed``, which
maps the local (prefill, decode, last decode batch) seconds to one agreed
triple (rank 0's). Every
other input is replicated scheduler state: batch classes, their request and
token counts.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

# (prefill seconds, decode seconds, seconds of the latest decode batch).
Elapsed = Tuple[float, float, float]


class PrefillDecodeBalancer:
    def __init__(
        self,
        *,
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
        # Local GPU seconds and tokens of every prefill batch that extended
        # tokens. A decode batch also reports extend tokens (one per request) and
        # costs an order of magnitude more per token, so it cannot join a rate
        # that prices prefill work.
        self._extend_seconds = 0.0
        self._extend_tokens = 0
        # GPU seconds and tokens of the last finished prefill batch: the marginal
        # rate, which is what the next chunk of the same prompt will pay.
        self._last_prefill_seconds = 0.0
        self._last_prefill_tokens = 0

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
    def marginal_prefill_seconds_per_token(self) -> float:
        """Cost of one more prefill token at the last finished prefill batch.

        Attention reads the whole context, so this rate rises along a chunked
        prompt and is what its next chunk actually pays. 0 before a prefill has
        finished.
        """
        if self._last_prefill_tokens == 0:
            return 0.0
        return self._last_prefill_seconds / self._last_prefill_tokens

    def prefill_token_budget(self, *, continues_chunk: bool) -> Optional[int]:
        """Cap on the next prefill batch's new tokens; None means uncapped.

        A batch continuing a chunked prompt is capped to the GPU seconds one
        chunk is worth at the measured average rate, priced at the marginal rate
        this prompt's context is now paying. A fresh request cannot drift that
        far -- its cost is bounded by its own length -- so fresh work keeps the
        token form, which is what lets several short prefills share one burst
        instead of each waiting out a decode slice.
        """
        if self._burst_tokens is None:
            return None
        average = self.prefill_seconds_per_token
        marginal = self.marginal_prefill_seconds_per_token
        if continues_chunk and average > 0.0 and marginal > 0.0:
            return max(0, int(self._burst_tokens * average / marginal))
        if self._burst_used == 0:
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
        if is_prefill and num_tokens > 0:
            self._extend_seconds += elapsed
            self._extend_tokens += num_tokens
            self._last_prefill_seconds, self._last_prefill_tokens = elapsed, num_tokens
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

    Called only while prefill and decode contend, which every rank of the
    group derives from the same replicated scheduler state, so all ranks
    enter the broadcast together."""
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_world_size(group=cpu_group) == 1:
        return lambda elapsed: elapsed

    src = dist.get_global_rank(cpu_group, 0)
    buffer = torch.zeros(3, dtype=torch.float64)

    def consensus(elapsed: Elapsed) -> Elapsed:
        buffer[0], buffer[1], buffer[2] = elapsed
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0]), float(buffer[1]), float(buffer[2])

    return consensus
