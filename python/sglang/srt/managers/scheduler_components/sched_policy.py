"""Client of the sglang-sched-policy process (tools/sched-policy).

The process decides prefill/decode time sharing and eviction throttling for
one TP group. Every rank sends it each call; it checks the ranks agree, decides
from rank 0's values and answers all ranks alike, or fails loudly. Nothing here
decides anything.
"""

from __future__ import annotations

import array
import os
import secrets
import shutil
import subprocess
import time
from typing import Optional, Sequence

import goipc

from sglang.srt.managers.scheduler_components import sched_policy_messages as msg

# Must equal ProtocolVersion in tools/sched-policy/internal/schedpolicy/version.go.
PROTOCOL_VERSION = 2

BINARY_ENV = "SGLANG_SCHED_POLICY_BIN"
CLASS_OTHER, CLASS_PREFILL, CLASS_DECODE = 0, 1, 2
_TAIL_TOKENS = 64


def binary_path() -> str:
    path = os.environ.get(BINARY_ENV) or shutil.which("sglang-sched-policy")
    if not path:
        raise RuntimeError(
            f"sglang-sched-policy not found: install it on PATH or set {BINARY_ENV}"
        )
    return path


def spawn(*, name: str, ranks: int, lockstep_timeout: float) -> subprocess.Popen:
    """Rank 0 starts the process. A rank that connects first parks until it serves."""
    return subprocess.Popen(
        [
            binary_path(),
            "-name",
            name,
            "-ranks",
            str(ranks),
            "-lockstep-timeout",
            f"{lockstep_timeout}s",
        ]
    )


def new_name() -> str:
    return f"sglang-sched-{secrets.token_hex(8)}"


def batch_class(forward_mode) -> int:
    if forward_mode.is_decode():
        return CLASS_DECODE
    if forward_mode.is_extend():
        return CLASS_PREFILL
    return CLASS_OTHER


def _tokens(ids: Sequence[int]) -> bytes:
    return array.array("i", ids).tobytes()


class SchedPolicy:
    def __init__(self, *, name: str, rank: int, world: int, timeout: float) -> None:
        self._client = goipc.service.connect(
            name, timeout=timeout, messages=msg.MESSAGES
        )
        self._call(msg.Hello(protocol=PROTOCOL_VERSION, rank=rank, world=world))

    def init(
        self,
        *,
        burst_tokens: Optional[int],
        device_tokens: int,
        host_tokens: int,
        throttle: bool,
    ) -> None:
        self._call(
            msg.Init(
                burst_tokens=-1 if burst_tokens is None else burst_tokens,
                device_tokens=device_tokens,
                host_tokens=host_tokens,
                throttle=throttle,
            )
        )

    def should_defer_prefill(
        self, *, prefill_pending: bool, decode_runnable: bool, continues_chunk: bool
    ) -> bool:
        return self._call(
            msg.ShouldDeferPrefill(
                prefill_pending=prefill_pending,
                decode_runnable=decode_runnable,
                continues_chunk=continues_chunk,
            )
        ).value

    def prefill_token_budget(self, *, continues_chunk: bool) -> Optional[int]:
        reply = self._call(msg.PrefillTokenBudget(continues_chunk=continues_chunk))
        return reply.value if reply.present else None

    def on_batch_launched(
        self, *, batch_class: int, num_tokens: int, num_decode_rows: int = 0
    ) -> None:
        self._call(
            msg.BatchLaunched(
                batch_class=batch_class,
                tokens=num_tokens,
                decode_rows=num_decode_rows,
                now=time.perf_counter(),
            )
        )

    def on_batch_finished(self) -> None:
        self._call(msg.BatchFinished(now=time.perf_counter()))

    def on_request_queued(self, *, rid: str, token_ids: Sequence[int]) -> None:
        self._call(
            msg.RequestQueued(
                now=time.perf_counter(), rid=rid, tokens=_tokens(token_ids)
            )
        )

    def on_request_finished(
        self, *, rid: str, input_ids: Sequence[int], output_ids: Sequence[int]
    ) -> None:
        tail = list(input_ids[-_TAIL_TOKENS:]) + list(output_ids[-_TAIL_TOKENS:])
        self._call(
            msg.RequestFinished(
                now=time.perf_counter(),
                length=len(input_ids) + len(output_ids),
                rid=rid,
                tail=_tokens(tail[-_TAIL_TOKENS:]),
            )
        )

    def begin_pass(self, *, present_rids: Sequence[str]) -> None:
        self._call(msg.BeginPass(present="\n".join(sorted(present_rids))))

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
        return self._call(
            msg.ShouldHold(
                input_len=input_len,
                device_hit=device_hit,
                total_tokens=total_tokens,
                would_evict=would_evict,
                queued_at=queued_at,
                now=time.perf_counter(),
                rid=rid,
            )
        ).value

    def on_admitted(self, *, evicted: bool) -> None:
        self._call(msg.Admitted(evicted=evicted, now=time.perf_counter()))

    def close(self) -> None:
        self._client.close()

    def _call(self, request):
        try:
            return self._client.call_typed(request)
        except goipc.CallError as e:
            raise RuntimeError(f"sglang-sched-policy: {e}") from None
