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
Prefill runs while the balance is repaid and waits while it is not. There is
nothing to tune:

* **Share.** Prefill batches and decode batches get equal GPU time, however
  many requests a prefill batch serves: that half is decode's floor, so a
  long prompt against running streams slows neither by more than 2x. Only
  batches that actually ran are charged, so a queue blocked on memory or
  request slots takes nothing from decode.
* **Stall.** A decision reads the balance charged so far. With the overlap
  scheduler the batch after a prefill is chosen while that prefill runs,
  before it is charged, so a prefill burst is two batches: a stream waits
  two chunks, and waits half as often as it would for one chunk at a time.
  Without overlap a burst is one batch.
* **Piggybacked decode.** With mixed chunked prefill the running requests
  decode one token inside a prefill batch. Only the batch that ends a burst
  carries them (``chunk_follows``): a token inside the first chunk of a pair
  would split one stall into two. That token ends the stall at the burst's
  last chunk instead of one decode step after it. The batch is charged as
  prefill minus the marginal cost of its decode rows, so pure decode keeps
  its equal share and the piggybacked tokens come on top.
* Decode banks nothing: the balance never goes below zero, so decode that ran
  while a prefill could not (memory, batch full) buys no later burst.

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

# (prefill seconds, decode seconds).
Elapsed = Tuple[float, float]


class PrefillDecodeBalancer:
    def __init__(
        self,
        *,
        overlap: bool,
        consensus_elapsed: Callable[[Elapsed], Elapsed] = lambda elapsed: elapsed,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._overlap = overlap
        self._consensus_elapsed = consensus_elapsed
        self._clock = clock
        self._debt = 0.0
        # GPU seconds finished since the last decision, per class.
        self._unsettled_prefill = 0.0
        self._unsettled_decode = 0.0
        # Gates the consensus call on an event count every rank shares, not a float.
        self._unsettled_batches = 0
        # (batch class, extend tokens, piggybacked decode rows) per launched,
        # unfinished batch.
        self._in_flight: Deque[Tuple[Optional[bool], int, int]] = deque()
        self._busy_since = 0.0
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

    def should_defer_prefill(
        self, *, prefill_pending: bool, decode_runnable: bool
    ) -> bool:
        if not (prefill_pending and decode_runnable):
            # No contention: whichever class has work runs at full speed.
            self._debt = 0.0
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
            return False

        if self._unsettled_batches:
            prefill_s, decode_s = self._consensus_elapsed(
                (self._unsettled_prefill, self._unsettled_decode)
            )
            self._debt = max(self._debt + prefill_s - decode_s, 0.0)
            self._unsettled_prefill = self._unsettled_decode = 0.0
            self._unsettled_batches = 0
        return self._debt > 0.0

    def chunk_follows(self, *, chunk_continues: bool) -> bool:
        """Whether the chunked request's next chunk launches right after the
        prefill batch being formed, before decode can run: with overlap, when
        no prefill is in flight yet and the request has more than this chunk left."""
        return self._overlap and chunk_continues and not self._prefill_in_flight()

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
    buffer = torch.zeros(2, dtype=torch.float64)

    def consensus(elapsed: Elapsed) -> Elapsed:
        buffer[0], buffer[1] = elapsed
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0]), float(buffer[1])

    return consensus
