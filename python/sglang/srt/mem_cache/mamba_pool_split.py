"""Default split of a hybrid model's pool budget between recurrent state and KV.

A hybrid (attention + Mamba/GDN/KDA) model serves from two pools fixed at
startup: the state pool caps how many requests run at once, the KV pool caps
their total tokens. A request of ``L`` tokens costs ``state_bytes_per_request``
in one and ``L * kv_bytes_per_token`` in the other, so both pools fill at the
same concurrency when the state pool gets
``state_bytes_per_request / (state_bytes_per_request + L * kv_bytes_per_token)``
of the budget (the ``compute-mamba-ratio`` skill's ``r*``, as a share).
"""

from __future__ import annotations

from typing import Optional

import msgspec

# Arbitrary; bounds the KV given up for short-request concurrency (and for idle
# conversations' state checkpoints) when context-length requests need less.
MIN_STATE_POOL_SHARE = 0.125


class MambaPoolSplit(msgspec.Struct, frozen=True, kw_only=True):
    state_share: float
    # Share at which both pools fill with context_len-token requests.
    balanced_share: float
    state_requests: float
    kv_context_len_requests: float
    # One of: "max_running_requests", "context_len", "short_request_floor",
    # "default_share_cap".
    sized_by: str

    @property
    def mamba_full_memory_ratio(self) -> float:
        return self.state_share / (1 - self.state_share)


def derive_mamba_pool_split(
    *,
    budget_bytes: float,
    state_bytes_per_request: float,
    fixed_state_bytes: float,
    kv_bytes_per_token: float,
    context_len: int,
    max_running_requests: Optional[int],
    default_share: float,
) -> MambaPoolSplit:
    """Pick the state pool's share of ``budget_bytes``.

    With ``max_running_requests`` the state pool holds exactly that many
    requests (never more than ``default_share``, the undecided default).
    Otherwise it holds as many requests as the KV pool fits at ``context_len``
    tokens each, but never less than ``MIN_STATE_POOL_SHARE`` of the budget.
    ``fixed_state_bytes`` is the per-pool overhead (padding slots) on top of
    the per-request cost.
    """
    kv_bytes_per_request = context_len * kv_bytes_per_token
    balanced_requests = (budget_bytes - fixed_state_bytes) / (
        state_bytes_per_request + kv_bytes_per_request
    )
    balanced_bytes = balanced_requests * state_bytes_per_request + fixed_state_bytes

    if max_running_requests is not None:
        state_bytes = max_running_requests * state_bytes_per_request
        state_bytes += fixed_state_bytes
        sized_by = "max_running_requests"
        if state_bytes > default_share * budget_bytes:
            state_bytes = default_share * budget_bytes
            sized_by = "default_share_cap"
    elif balanced_bytes >= MIN_STATE_POOL_SHARE * budget_bytes:
        state_bytes = balanced_bytes
        sized_by = "context_len"
    else:
        state_bytes = MIN_STATE_POOL_SHARE * budget_bytes
        sized_by = "short_request_floor"

    return MambaPoolSplit(
        state_share=state_bytes / budget_bytes,
        balanced_share=balanced_bytes / budget_bytes,
        state_requests=(state_bytes - fixed_state_bytes) / state_bytes_per_request,
        kv_context_len_requests=(budget_bytes - state_bytes) / kv_bytes_per_request,
        sized_by=sized_by,
    )
