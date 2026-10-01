"""Rank 0's values broadcast across a TP CPU group.

Every rank must enter each consensus in the same order. A rank that skips one,
or enters a different one, pairs its broadcast with the wrong collective; each
broadcast therefore carries its call site and a per-process sequence number,
and a receiving rank that disagrees raises instead of computing on garbage or
waiting forever.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

SITE_PREFILL_DECODE_BALANCER = 1
SITE_EVICTION_THROTTLE = 2

# Every site broadcasts the same shape so a mismatched pair completes and is
# caught by the header check rather than by the transport.
_WIDTH = 3
_HEADER = 2

_sequence = 0


class ConsensusDivergence(RuntimeError):
    pass


def rank0_broadcast(
    cpu_group, *, site: int
) -> Callable[[Sequence[float]], Tuple[float, ...]]:
    """Return a function that replaces up to ``_WIDTH`` values with rank 0's."""
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_world_size(group=cpu_group) == 1:
        return lambda values: tuple(float(v) for v in values)

    src = dist.get_global_rank(cpu_group, 0)
    buffer = torch.zeros(_HEADER + _WIDTH, dtype=torch.float64)

    def broadcast(values: Sequence[float]) -> Tuple[float, ...]:
        global _sequence
        if len(values) > _WIDTH:
            raise ValueError(f"consensus carries at most {_WIDTH} values")
        _sequence += 1
        buffer.zero_()
        buffer[0] = site
        buffer[1] = _sequence
        for i, value in enumerate(values):
            buffer[_HEADER + i] = value
        dist.broadcast(buffer, src=src, group=cpu_group)
        got_site, got_sequence = int(buffer[0]), int(buffer[1])
        if (got_site, got_sequence) != (site, _sequence):
            raise ConsensusDivergence(
                f"rank {dist.get_rank()} entered consensus site {site} call "
                f"{_sequence}, but rank {src} broadcast site {got_site} call "
                f"{got_sequence}: ranks took different scheduling paths"
            )
        return tuple(float(v) for v in buffer[_HEADER : _HEADER + len(values)])

    return broadcast
