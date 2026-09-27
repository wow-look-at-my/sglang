"""Equal GPU time for prefill and decode while both have work.

The scheduler used to run a prefill batch whenever one could be formed, and
decode only when none could. A single long prompt is split into many
chunked-prefill batches, and each of those won its scheduling turn, so every
running request's decode stalled until the whole prompt was prefilled. On a
400K-token cold prompt that is a minute or more of zero generated tokens.

This controller measures how long each batch occupied the GPU, from the
previous completion (or its own launch, if the GPU was idle) to its own
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

Charging at completion rather than between scheduling decisions matters with
the overlap scheduler, which picks batch N+1 while batch N still runs.

Decisions must be identical on every rank that shares a batch. Measured times
differ slightly per rank, so the caller supplies ``consensus_elapsed``, which
maps the local measurement to one agreed value (rank 0's).
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class PrefillDecodeBalancer:
    def __init__(
        self,
        *,
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

    @property
    def debt(self) -> float:
        return self._debt

    def should_defer_prefill(
        self, *, prefill_pending: bool, decode_runnable: bool
    ) -> bool:
        if not (prefill_pending and decode_runnable):
            # No contention: whichever class has work runs at full speed.
            self._debt = 0.0
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

    Called only while prefill and decode contend, which every rank of the
    group derives from the same replicated scheduler state, so all ranks
    enter the broadcast together."""
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_world_size(group=cpu_group) == 1:
        return lambda elapsed: elapsed

    src = dist.get_global_rank(cpu_group, 0)
    buffer = torch.zeros(1, dtype=torch.float64)

    def consensus(elapsed: float) -> float:
        buffer[0] = elapsed
        dist.broadcast(buffer, src=src, group=cpu_group)
        return float(buffer[0])

    return consensus
