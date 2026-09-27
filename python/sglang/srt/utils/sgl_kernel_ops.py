"""Startup check that the installed sglang-kernel wheel registers the ops this tree calls.

A wheel published before an op was added to ``python/sglang/kernels/aot`` imports
cleanly and reports a matching version, but the op is absent from
``torch.ops.sgl_kernel``; callers then fall back or fail far from the cause. This
check names every such op in one log line at startup.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# Ops registered by each backend's main sgl_kernel library (the one `import
# sgl_kernel` loads) that python/sglang/srt reaches, directly or through the
# sgl_kernel Python wrappers it imports. Derived from the TORCH_LIBRARY schema in
# python/sglang/kernels/aot/csrc; test/registered/unit/utils/test_sgl_kernel_ops.py
# fails with the regenerated literal whenever this drifts from the source.
EXPECTED_SGL_KERNEL_OPS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "cpu": frozenset(
            {
                "apply_multidimensional_rope_cpu",
                "apply_rotary_pos_emb_cpu",
                "assign_draft_cache_locs_contiguous_cpu",
                "assign_extend_cache_locs_cpu",
                "assign_req_to_token_pool_cpu",
                "biased_grouped_topk_cpu",
                "bmm_cpu",
                "build_draft_decode_metadata_cpu",
                "build_tree_kernel_efficient_cpu",
                "causal_conv1d_fwd_cpu",
                "causal_conv1d_update_cpu",
                "causal_conv1d_weight_pack",
                "chunk_gated_delta_rule_cpu",
                "conv3d_embed_cpu",
                "conv3d_embed_weight_pack",
                "convert_scale_packed",
                "convert_weight_packed",
                "convert_weight_packed_scale_zp",
                "decode_attention_cpu",
                "extend_attention_cpu",
                "flash_attn_varlen_func",
                "fp8_per_tensor_scaled_mm_cpu",
                "fp8_scaled_mm_cpu",
                "fused_add_rmsnorm_cpu",
                "fused_experts_cpu",
                "fused_gdn_gating_cpu",
                "fused_input_proj_cpu",
                "fused_linear_sigmoid_mul",
                "fused_qk_gemma_rmsnorm_cpu",
                "fused_qk_gemma_rmsnorm_with_gate_cpu",
                "fused_qk_norm_cpu",
                "fused_qk_norm_rope_cpu",
                "fused_qk_rmsnorm_apply_from_stats_cpu",
                "fused_qk_rmsnorm_cpu",
                "fused_qk_rmsnorm_sumsq_cpu",
                "fused_qkvzba_split_reshape_cat_contiguous_cpu",
                "fused_qkvzba_split_reshape_cat_cpu",
                "fused_sigmoid_gating_delta_rule_update_cpu",
                "fused_sigmoid_mul_cpu",
                "gelu_and_mul_cpu",
                "gelu_tanh_and_mul_cpu",
                "gemma3_rmsnorm_cpu",
                "gemma4_rmsnorm_cpu",
                "gemma_fused_add_rmsnorm_cpu",
                "gemma_rmsnorm_cpu",
                "grouped_topk_cpu",
                "image_preprocess_cpu",
                "init_cpu_threads_env",
                "initialize",
                "int4_scaled_mm_cpu",
                "int8_scaled_mm_with_quant",
                "layernorm_cpu",
                "multimodal_rotary_embedding_cpu",
                "qkv_proj_with_rope_fused_weight",
                "reconstruct_indices_from_tree_mask_cpu",
                "rmsnorm_cpu",
                "rotary_embedding_cpu",
                "shared_expert_cpu",
                "shm_allgather",
                "shm_allreduce",
                "silu_and_mul_cpu",
                "store_cache_cpu",
                "topk_sigmoid_cpu",
                "topk_softmax_cpu",
                "verify_tree_greedy_cpu",
                "weight_packed_linear",
            }
        ),
        "cuda": frozenset(
            {
                "all_reduce",
                "apply_shuffle_mul_sum",
                "apply_token_bitmask_inplace_cuda",
                "concat_mla_k",
                "copy_to_gpu_no_ce",
                "cutlass_w4a8_moe_mm",
                "dispose",
                "es_fp8_blockwise_scaled_grouped_mm",
                "es_sm100_mxfp8_blockscaled_grouped_mm",
                "es_sm100_mxfp8_blockscaled_grouped_quant",
                "fast_topk",
                "fast_topk_transform_fused",
                "fast_topk_transform_ragged_fused",
                "fp8_blockwise_scaled_grouped_mm",
                "gelu_and_mul",
                "gelu_tanh_and_mul",
                "get_cutlass_w4a8_moe_mm_data",
                "get_device_accessible_ptr",
                "get_graph_buffer_ipc_meta",
                "ggml_dequantize",
                "ggml_moe_a8",
                "ggml_moe_a8_vec",
                "ggml_moe_get_block_size",
                "ggml_mul_mat_a8",
                "ggml_mul_mat_vec_a8",
                "infllm_v2_max_pooling_1d_varlen",
                "init_custom_ar",
                "int8_scaled_mm",
                "merge_state_v2",
                "meta_size",
                "moe_align_block_size",
                "moe_sum",
                "moe_sum_reduce",
                "prepare_moe_input",
                "register_buffer",
                "register_graph_buffers",
                "rmsnorm",
                "shuffle_rows",
                "silu_and_mul",
                "top_k_renorm_probs",
                "top_p_renorm_probs",
                "transfer_embedding_ranges_direct",
                "transfer_kv_all_layer",
                "transfer_kv_all_layer_direct_lf_pf",
                "transfer_kv_all_layer_lf_pf",
                "transfer_kv_all_layer_lf_ph",
                "transfer_kv_all_layer_mla",
                "transfer_kv_all_layer_mla_lf_pf",
                "transfer_kv_direct",
                "transfer_kv_per_layer",
                "transfer_kv_per_layer_direct_pf_lf",
                "transfer_kv_per_layer_mla",
                "transfer_kv_per_layer_mla_pf_lf",
                "transfer_kv_per_layer_pf_lf",
                "transfer_kv_per_layer_ph_lf",
            }
        ),
        "musa": frozenset(
            {
                "all_reduce",
                "apply_token_bitmask_inplace_cuda",
                "build_tree_kernel_efficient",
                "concat_mla_k",
                "dispose",
                "gelu_and_mul",
                "gelu_tanh_and_mul",
                "get_graph_buffer_ipc_meta",
                "ggml_dequantize",
                "ggml_moe_a8",
                "ggml_moe_a8_vec",
                "ggml_moe_get_block_size",
                "ggml_mul_mat_a8",
                "ggml_mul_mat_vec_a8",
                "init_custom_ar",
                "merge_state_v2",
                "meta_size",
                "min_p_sampling_from_probs",
                "moe_align_block_size",
                "moe_sum",
                "moe_sum_reduce",
                "musa_top_k_top_p_sampling_from_probs",
                "reconstruct_indices_from_tree_mask",
                "register_buffer",
                "register_graph_buffers",
                "rmsnorm",
                "silu_and_mul",
                "top_k_renorm_probs",
                "top_p_renorm_probs",
                "top_p_sampling_from_probs",
                "transfer_kv_all_layer",
                "transfer_kv_all_layer_direct_lf_pf",
                "transfer_kv_all_layer_lf_pf",
                "transfer_kv_all_layer_lf_ph",
                "transfer_kv_all_layer_mla",
                "transfer_kv_all_layer_mla_lf_pf",
                "transfer_kv_direct",
                "transfer_kv_per_layer",
                "transfer_kv_per_layer_direct_pf_lf",
                "transfer_kv_per_layer_mla",
                "transfer_kv_per_layer_mla_pf_lf",
                "transfer_kv_per_layer_pf_lf",
                "transfer_kv_per_layer_ph_lf",
                "tree_speculative_sampling_target_only",
                "verify_tree_greedy",
            }
        ),
        "rocm": frozenset(
            {
                "all_reduce_reg",
                "all_reduce_unreg",
                "allocate_meta_buffer",
                "apply_token_bitmask_inplace_cuda",
                "deterministic_all_reduce_reg",
                "deterministic_all_reduce_unreg",
                "dispose",
                "fast_topk",
                "fast_topk_transform_fused",
                "fast_topk_transform_ragged_fused",
                "gelu_and_mul",
                "gelu_quick",
                "gelu_tanh_and_mul",
                "get_device_accessible_ptr",
                "get_graph_buffer_ipc_meta",
                "get_meta_buffer_ipc_handle",
                "init_custom_ar",
                "init_custom_qr",
                "meta_size",
                "moe_align_block_size",
                "qr_all_reduce",
                "qr_destroy",
                "qr_get_handle",
                "qr_max_size",
                "qr_open_handles",
                "register_buffer",
                "register_graph_buffers",
                "silu_and_mul",
                "transfer_kv_all_layer",
                "transfer_kv_all_layer_direct_lf_pf",
                "transfer_kv_all_layer_lf_pf",
                "transfer_kv_all_layer_lf_ph",
                "transfer_kv_all_layer_mla",
                "transfer_kv_all_layer_mla_lf_pf",
                "transfer_kv_direct",
                "transfer_kv_per_layer",
                "transfer_kv_per_layer_direct_pf_lf",
                "transfer_kv_per_layer_mla",
                "transfer_kv_per_layer_mla_pf_lf",
                "transfer_kv_per_layer_pf_lf",
                "transfer_kv_per_layer_ph_lf",
            }
        ),
    }
)


def find_missing_sgl_kernel_ops(
    *, expected: Iterable[str], has_op: Callable[[str], bool]
) -> list[str]:
    return sorted(name for name in expected if not has_op(name))


def _current_backend() -> Optional[str]:
    from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_musa

    # sgl_kernel on XPU / NPU comes from separately versioned packages whose
    # schema is not in this tree.
    if is_cuda():
        return "cuda"
    if is_hip():
        return "rocm"
    if is_musa():
        return "musa"
    if is_cpu():
        return "cpu"
    return None


def warn_missing_sgl_kernel_ops() -> None:
    """Log one warning naming the ops the installed sglang-kernel lacks.

    Call it where the device is already initialized: importing sgl_kernel
    queries the current GPU.
    """
    backend = _current_backend()
    if backend is None:
        return
    try:
        import sgl_kernel
        import torch
    except Exception:
        # A missing or unloadable wheel is reported by the version check at launch.
        return

    missing = find_missing_sgl_kernel_ops(
        expected=EXPECTED_SGL_KERNEL_OPS[backend],
        has_op=lambda name: hasattr(torch.ops.sgl_kernel, name),
    )
    if missing:
        logger.warning(
            "sglang-kernel %s at %s lacks %d op(s) this SGLang calls: %s. The wheel "
            "predates them, so their callers fall back or fail; install sglang-kernel "
            "%s built from python/sglang/kernels/aot of this checkout.",
            sgl_kernel.__version__,
            sgl_kernel.__file__,
            len(missing),
            ", ".join(missing),
            _pinned_kernel_version(),
        )


def _pinned_kernel_version() -> str:
    from importlib.metadata import PackageNotFoundError, requires

    try:
        requirements = requires("sglang") or []
    except PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        name, sep, version = requirement.partition("==")
        if sep and name.strip() == "sglang-kernel":
            return version.split(";")[0].strip()
    return "(the version python/pyproject.toml pins)"
