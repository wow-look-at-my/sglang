# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for mixed chunked prefill."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

logger = logging.getLogger(__name__)


def handle_mixed_chunk_auto(server_args: Any):
    """Resolve an unset ``enable_mixed_chunk``.

    Runs after every hook that reads the flag, which saw it unset (off), so
    only configurations those hooks would neither reject nor turn off qualify.
    """
    cfg = resolving_view(server_args)
    if cfg.enable_mixed_chunk is not None:
        return
    reason = auto_mixed_chunk_blocker(cfg)
    if reason is not None:
        logger.info("Mixed chunked prefill auto: off, %s.", reason)
        declare_resolution(
            server_args, "handle_mixed_chunk_auto", enable_mixed_chunk=False
        )
        return
    logger.info(
        "Mixed chunked prefill auto: on, running requests decode inside each "
        "prefill chunk; --no-enable-mixed-chunk opts out."
    )
    declare_resolution(server_args, "handle_mixed_chunk_auto", enable_mixed_chunk=True)


def auto_mixed_chunk_blocker(cfg: Any) -> Optional[str]:
    """Why mixed chunked prefill must stay off when the flag is unset, or None."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_platform
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if not get_platform().is_cuda:
        return "the automatic setup is validated on CUDA only"
    if cfg.chunked_prefill_size is None or cfg.chunked_prefill_size <= 0:
        return "chunked prefill is off"
    if (
        cfg.speculative_algorithm is not None
        and not (
            SpeculativeAlgorithm.from_string(cfg.speculative_algorithm)
        ).supports_mixed_chunk()
    ):
        return f"speculative algorithm {cfg.speculative_algorithm} does not support it"
    if cfg.dllm_algorithm is not None:
        return "diffusion LLM inference has its own prefill loop"
    if cfg.pp_size > 1:
        return "pipeline parallelism is not covered by the automatic setup"
    if cfg.enable_dp_attention:
        return "DP attention is not covered by the automatic setup"
    if cfg.disaggregation_mode != "null":
        return "PD disaggregation separates prefill from decode"
    if cfg.enable_lora:
        return "LoRA is not covered by the automatic setup"
    if cfg.enable_encoder_swa_bounded_replay:
        return "--enable-encoder-swa-bounded-replay does not support it"
    if cfg.cuda_graph_config.prefill.backend not in (
        Backend.DISABLED,
        Backend.BREAKABLE,
    ):
        # PrefillCudaGraphRunner asserts the breakable backend under mixed chunk.
        return "the prefill CUDA graph backend is not breakable"
    return None
