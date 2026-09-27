"""Equal GPU time for prefill and decode while both have work.

The scheduler used to run a prefill batch whenever one could be formed, and
decode only when none could. A single long prompt is split into many
chunked-prefill batches, and each of those won its scheduling turn, so every
running request's decode stalled until the whole prompt was prefilled. On a
400K-token cold prompt that is a minute or more of zero generated tokens.

This controller measures how long each launched batch occupied the scheduler
and keeps a running balance, ``debt`` = prefill seconds - decode seconds,
accumulated only while the two classes contend (a prefill is pending *and*
running requests can decode). A prefill is deferred while the balance is
positive, so under contention the GPU alternates between the two classes in
equal time slices; without contention nothing is ever deferred. There is
nothing to tune:

* A short prefill is followed by a correspondingly short decode slice, and a
  long chunk by a long one, so the split adapts to chunk size, context length,
  model and hardware without a hand-picked interval.
* Decode never banks credit (the balance is floored at zero), so when a
  prefill could not run anyway (memory, batch full) no burst of prefill
  follows later.
* Neither class can be slowed by more than 2x by the other.

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
        # True: prefill, False: decode, None: nothing launched (idle/other).
        self._launched_is_prefill: Optional[bool] = None
        self._launched_at = 0.0

    @property
    def debt(self) -> float:
        return self._debt

    def should_defer_prefill(
        self, *, prefill_pending: bool, decode_runnable: bool
    ) -> bool:
        if not (prefill_pending and decode_runnable):
            # No contention: whichever class has work runs at full speed.
            self._debt = 0.0
            return False

        if self._launched_is_prefill is not None:
            elapsed = self._consensus_elapsed(self._clock() - self._launched_at)
            if self._launched_is_prefill:
                self._debt += elapsed
            else:
                self._debt = max(self._debt - elapsed, 0.0)
            # Charge each launch once, even if no new batch follows it.
            self._launched_is_prefill = None
        return self._debt > 0.0

    def on_batch_launched(self, is_prefill: Optional[bool]) -> None:
        self._launched_is_prefill = is_prefill
        self._launched_at = self._clock()


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
